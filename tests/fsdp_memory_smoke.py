"""Explicit two-GPU tiny-model check; run with torchrun, never in unit discovery."""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from safetensors.torch import load_file
from transformers import BertConfig, BertModel

from spade.swift_backend.fsdp_ram_loader import install_transformers_fsdp_ram_loader


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        checkpoint = [tempfile.mkdtemp(prefix="spade-memory-smoke-") if rank == 0 else None]
        dist.broadcast_object_list(checkpoint, src=0)
        if rank == 0:
            torch.manual_seed(1234)
            source = BertModel(BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                                           num_attention_heads=4, intermediate_size=64))
            source.save_pretrained(checkpoint[0])
            del source
        dist.barrier()
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
        install_transformers_fsdp_ram_loader()
        accelerator = Accelerator(mixed_precision="bf16", fsdp_plugin=FullyShardedDataParallelPlugin(
            fsdp_version=2, auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["BertLayer"], cpu_ram_efficient_loading=True,
            reshard_after_forward=True,
        ))
        model = BertModel.from_pretrained(checkpoint[0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer = accelerator.prepare(model, optimizer)
        expected = load_file(str(Path(checkpoint[0]) / "model.safetensors"))
        for name, parameter in model.named_parameters():
            value = parameter.full_tensor().detach().cpu()
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4]], device=accelerator.device)
        output = model(input_ids=tokens).last_hidden_state
        assert torch.isfinite(output).all(), "nonfinite forward output"
        loss = output.float().square().mean()
        accelerator.backward(loss)
        optimizer.step()
        dist.barrier()
        print(f"rank {rank}: all parameters match checkpoint exactly; forward/backward/update passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
