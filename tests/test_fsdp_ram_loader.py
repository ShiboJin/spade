"""Contracts for the Transformers 5.12.1 FSDP host-RAM loading patch."""
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from unittest import mock

import torch


try:
    import transformers  # noqa: F401
except ImportError:
    transformers = None


class CheckpointSlice:
    def __init__(self):
        self.was_read = False

    def get_shape(self):
        return [4, 8]

    def get_dtype(self):
        return "BF16"

    def __getitem__(self, key):
        del key
        self.was_read = True
        raise AssertionError("worker rank must not read checkpoint payload")


@unittest.skipIf(transformers is None, "transformers is not installed")
class FsdpRamLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from spade.swift_backend import fsdp_ram_loader

        cls.patch_module = fsdp_ram_loader

    def test_worker_rank_uses_empty_tensor_without_reading_checkpoint(self):
        from transformers import core_model_loading

        checkpoint_slice = CheckpointSlice()
        environment = {
            "ACCELERATE_USE_FSDP": "true",
            "FSDP_CPU_RAM_EFFICIENT_LOADING": "true",
        }
        with (
            mock.patch.dict("os.environ", environment, clear=False),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "get_rank", return_value=1),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            result = core_model_loading.spawn_materialize(
                pool, checkpoint_slice, device="cpu", dtype=torch.bfloat16
            ).result()

        self.assertEqual(result.shape, (4, 8))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(checkpoint_slice.was_read)

    def test_patch_is_idempotent(self):
        from transformers import core_model_loading

        before = core_model_loading.spawn_materialize
        self.patch_module.install_transformers_fsdp_ram_loader()
        self.assertIs(core_model_loading.spawn_materialize, before)

    def test_complete_worker_load_does_not_zero_fill_parameters(self):
        from transformers import BertConfig, BertModel

        with tempfile.TemporaryDirectory() as directory:
            source = BertModel(BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                                          num_attention_heads=4, intermediate_size=64))
            source.save_pretrained(directory)
            with (
                mock.patch.dict("os.environ", {"ACCELERATE_USE_FSDP": "true",
                                               "FSDP_CPU_RAM_EFFICIENT_LOADING": "true", "LOCAL_RANK": "1"}),
                mock.patch.object(torch.distributed, "is_initialized", return_value=True),
                mock.patch.object(torch.distributed, "get_rank", return_value=1),
                mock.patch.object(torch, "zeros_like", wraps=torch.zeros_like) as zeros,
            ):
                worker = BertModel.from_pretrained(directory)
            # The upstream post-load zeros_like pass used to touch every parameter.
            self.assertTrue(all(not isinstance(call.args[0], torch.nn.Parameter)
                                for call in zeros.call_args_list))
            self.assertTrue(all(p.device.type == "cpu" for p in worker.parameters()))
            self.assertEqual({k: v.shape for k, v in worker.state_dict().items()},
                             {k: v.shape for k, v in source.state_dict().items()})
            # Rank zero still loads exactly the checkpoint values.
            with (
                mock.patch.dict("os.environ", {"ACCELERATE_USE_FSDP": "true",
                                               "FSDP_CPU_RAM_EFFICIENT_LOADING": "true", "LOCAL_RANK": "0"}),
                mock.patch.object(torch.distributed, "is_initialized", return_value=True),
                mock.patch.object(torch.distributed, "get_rank", return_value=0),
            ):
                main = BertModel.from_pretrained(directory)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(main.state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
