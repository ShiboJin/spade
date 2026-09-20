"""Run with torchrun: real Qwen/LoRA/FSDP2 probability and gradient parity."""
import os
import argparse
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reentrant-checkpoint", action="store_true",
                        help="Reproduce Swift's outer no-grad checkpoint forward")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from trl.trainer.utils import selective_log_softmax
        from spade.swift_backend.memory_efficient_grpo import GRPOTrainer, checkpoint_decoder_layers

        torch.manual_seed(42)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=97, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, layer_types=["linear_attention", "full_attention"],
                             linear_num_key_heads=2, linear_num_value_heads=4,
                             linear_key_head_dim=16, linear_value_head_dim=16,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                              "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
                             max_position_embeddings=1024, pad_token_id=0),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                               num_heads=4, out_hidden_size=64))
        model = Qwen3_5ForConditionalGeneration(config)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
            task_type="CAUSAL_LM"))
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.02)
        model.cuda().train()
        if args.reentrant_checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.enable_input_require_grads()
        layers = model.base_model.model.model.language_model.layers
        for i, layer in enumerate(layers):
            fully_shard(layers[i], reshard_after_forward=True)
        fully_shard(model.get_output_embeddings(), reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(unwrap_model=lambda m: m, is_main_process=dist.get_rank() == 0),
            args=SimpleNamespace(report_to=[]), is_multimodal=True,
            model_kwarg_keys={"use_cache", "logits_to_keep"}, temperature=0.6)
        # Match non-Liger trainers: Swift does not initialize this attribute.
        assert not hasattr(trainer, "_forward_redirection")
        trainer._get_last_hidden_state = MethodType(GRPOTrainer._get_last_hidden_state, trainer)
        tokens = torch.randint(1, 97, (2, 529), device="cuda")
        mask = torch.ones_like(tokens)
        mask[0, :5] = 0
        inputs = {"input_ids": tokens, "attention_mask": mask, "use_cache": False}
        keep = 517
        scale = torch.randn(2, keep, device="cuda")
        torch.manual_seed(91)
        reference = model(**inputs, logits_to_keep=keep + 1).logits[:, :-1] / trainer.temperature
        reference = selective_log_softmax(reference, tokens[:, -keep:])
        (reference * scale).mean().backward()
        expected_grads = {name: p.grad.full_tensor().detach().clone()
                          for name, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        checkpoint_decoder_layers(model)
        if args.reentrant_checkpoint:
            # Swift can re-enable HF checkpointing after our FSDP setup.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            assert all(layer.gradient_checkpointing for layer in layers)
        torch.manual_seed(91)
        actual, _ = GRPOTrainer._get_logps_via_local_forward(trainer, model, inputs, keep, tokens)
        assert all(not layer.gradient_checkpointing for layer in layers)
        torch.testing.assert_close(actual, reference.detach(), rtol=2e-5, atol=2e-5)
        (actual * scale).mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad.full_tensor(), expected_grads[name], rtol=2e-4, atol=2e-5,
                                           msg=lambda message: f"{name}: {message}")
        assert sum(g.abs().sum().item() for g in expected_grads.values()) > 0
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        optimizer.step()
        with torch.no_grad():
            updated, _ = GRPOTrainer._get_logps_via_local_forward(trainer, model, inputs, keep, tokens)
        assert torch.isfinite(updated).all()
        assert not torch.equal(updated, actual.detach()), "optimizer did not change policy"
        print(f"rank {dist.get_rank()}: Qwen LoRA FSDP2 logps, gradients and optimizer update passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
