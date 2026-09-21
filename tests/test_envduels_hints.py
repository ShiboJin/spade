import json
from pathlib import Path
import tempfile
import unittest

from spade.core.envduels_hints import load_hint_levels


class HintExportTests(unittest.TestCase):
    def test_single_and_graded_hints_preserve_author_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(id="env", privileged="privileged.json")
            for hints, expected in [(dict(hint=" hint exactly \n"), (" hint exactly \n",)),
                                    (dict(hint_1="weak", hint_2="strong"), ("weak", "strong"))]:
                (root / "privileged.json").write_text(json.dumps(dict(environment_id="env", hints=hints)))
                self.assertEqual(load_hint_levels(root, row), expected)
            for document in [dict(environment_id="wrong", hints=dict(hint="secret")),
                             dict(environment_id="env", hints=dict(hint="")),
                             dict(environment_id="env", hints=dict(hint_2="missing first"))]:
                (root / "privileged.json").write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    load_hint_levels(root, row)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_hint_levels(root, dict(id="env", privileged="../secret.json"))
            self.assertEqual(load_hint_levels(root, dict(id="no-hints")), ())


if __name__ == "__main__":
    unittest.main()
