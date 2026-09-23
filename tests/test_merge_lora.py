"""CPU-only checks for the safe LoRA merge launcher."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.merge_lora import docker_command, validate_inputs


class MergeLoraTests(unittest.TestCase):
    def test_command_keeps_sources_read_only_and_uses_bfloat16(self):
        command = docker_command(
            Path("/models/base"), Path("/models/adapter"),
            Path("/models/merged"), "image:test", 96,
        )
        self.assertIn("type=bind,src=/models/base,dst=/model,readonly", command)
        self.assertIn("type=bind,src=/models/adapter,dst=/adapter,readonly", command)
        self.assertIn(
            "type=bind,src=/models/.merged.merge-tmp,dst=/output", command
        )
        self.assertEqual(command[command.index("--output_dir") + 1], "/output/merged")
        self.assertEqual(command[command.index("--torch_dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--model_type") + 1], "qwen3_5")
        self.assertEqual(command[command.index("--template") + 1], "qwen3_8")
        self.assertEqual(command[command.index("--device_map") + 1], "cpu")
        self.assertEqual(command[command.index("--load_args") + 1], "false")
        self.assertIn("USER=envduels", command)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                validate_inputs(path / "base", path / "adapter", path)

    def test_output_cannot_be_inside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "separate"):
                validate_inputs(path / "base", path / "adapter", path / "base/merged")

if __name__ == "__main__":
    unittest.main()
