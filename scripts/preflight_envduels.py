"""Read-only CPU preflight; --runtime additionally checks the GPU software imports.

Never loads weights or initializes CUDA. All paths come from the run profile.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct


def check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(os.environ.get("HF_CHECKPOINT", root / "checkpoints/Qwen3.8-27B"))
    export = Path(os.environ.get("ENVDUELS_EXPORT_DIR", root.parent / "exports/duel_harness_004_rl"))
    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    expected = dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
                    num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                    vocab_size=248320, linear_num_key_heads=16, linear_num_value_heads=48,
                    linear_key_head_dim=128, linear_value_head_dim=128)
    assert config["model_type"] == "qwen3_5"
    for key, value in expected.items():
        assert text[key] == value, (key, text[key], value)
    assert text["output_gate_type"] == "swish"
    assert text["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * 16
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = checkpoint / shard
        with path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            assert header_len < 100_000_000
            header = json.loads(stream.read(header_len))
        data_len = path.stat().st_size - 8 - header_len
        for name, entry in header.items():
            if name != "__metadata__":
                assert 0 <= entry["data_offsets"][0] <= entry["data_offsets"][1] <= data_len
                headers[name] = (shard, entry)
    for name, shard in index["weight_map"].items():
        assert headers[name][0] == shard, name
    assert headers["model.language_model.embed_tokens.weight"][1]["shape"] == [248320, 5120]
    print(f"Weights: {len(headers)} indexed tensors, headers and file bounds OK")

    from spade.core.envs.envduels_adapter import EnvDuelsAdapter
    adapter = EnvDuelsAdapter(export, env_ids_file=os.environ.get("ENVDUELS_IDS_FILE"))
    expected_count = os.environ.get("ENVDUELS_EXPECTED_COUNT")
    if expected_count and len(adapter.list_environments()) != int(expected_count):
        raise ValueError(f"Expected {expected_count} environments, got {len(adapter.list_environments())}")
    for env_id in adapter.list_environments():
        row = adapter.rows[env_id]
        path = (export / row["source"]).resolve()
        assert path.is_relative_to(export.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"], env_id
    print(f"Environments: {len(adapter.list_environments())} source hashes verified (no execution)")

    from transformers import AutoTokenizer
    from spade.core.utils.token_utils import get_observation_delta
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for thinking in (False, True):
        kwargs = dict(enable_thinking=thinking, preserve_thinking=True)
        messages = [{"role": "user", "content": "Choose an action."}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                add_generation_prompt=True, **kwargs)["input_ids"]
        response = ("Consider the lock.\n</think>\n\n" if thinking else "") + r"\boxed{READ}"
        prefix += tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        assistant = {"role": "assistant", "content": response}
        if thinking:
            assistant = {"role": "assistant", "content": r"\boxed{READ}",
                         "reasoning_content": "Consider the lock."}
        messages += [assistant, {"role": "user", "content": "The lamps are lit."}]
        delta, mask = get_observation_delta(tokenizer, messages, prefix, kwargs)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        assert tokenizer.decode(prefix + delta, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) == rendered
        assert delta and not any(mask)
    print("Tokenizer: both thinking modes preserve sampled prefixes and zero-mask observations")

    if args.runtime:
        for module in ("torch", "ray", "sglang", "megatron.core", "fla", "mbridge",
                       "spade.slime.fixed_env_rollout", "slime.ray.placement_group"):
            imported = importlib.import_module(module)
            print("Runtime import:", module, getattr(imported, "__version__", "OK"))
        import train_spade_slime
        train_spade_slime.load_train()
        importlib.import_module("slime_plugins.models.qwen3_5")
        importlib.import_module("slime_plugins.mbridge.qwen3_5")
        print("Training entrypoint and Qwen model/bridge imports OK (no CUDA execution)")


if __name__ == "__main__":
    check()
