"""Worker device binding must precede Swift pipeline import and execution."""
import builtins
import os
from types import ModuleType
import unittest
from unittest.mock import patch

from spade.swift_backend.rlhf_entry import main


class EntryTests(unittest.TestCase):
    def test_bind_before_pipeline_import(self):
        events = []
        torch = ModuleType("torch")
        torch.cuda = type("Cuda", (), {
            "set_device": staticmethod(lambda rank: events.append(("bind", rank))),
            "current_device": staticmethod(lambda: 3),
        })
        pipelines = ModuleType("swift.pipelines")
        pipelines.rlhf_main = lambda: events.append("train")
        original_import = builtins.__import__

        def import_module(name, *args, **kwargs):
            if name == "torch":
                return torch
            if name == "swift.pipelines":
                events.append("import_pipeline")
                return pipelines
            return original_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"LOCAL_RANK": "3", "RANK": "3"}), \
                patch("builtins.__import__", side_effect=import_module), patch("builtins.print"):
            main()
        self.assertEqual(events, [("bind", 3), "import_pipeline", "train"])
