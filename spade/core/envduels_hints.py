"""Read author-provided hints without importing environment or ML dependencies."""
import json
from pathlib import Path


def load_hint_levels(root, row):
    env_id = row["id"]
    root = Path(root).resolve()
    if not row.get("privileged"):
        return ()
    path = (root / row["privileged"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest privileged path escapes export root")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("environment_id") != env_id:
        raise ValueError(f"Privileged environment ID mismatch: {env_id}")
    hints = document.get("hints")
    if not isinstance(hints, dict):
        raise ValueError(f"Missing hint mapping: {env_id}")
    fields = ("hint",) if set(hints) == {"hint"} else ("hint_1", "hint_2")
    if set(hints) != set(fields) or any(
            not isinstance(hints.get(key), str) or not hints[key].strip() for key in fields):
        raise ValueError(f"Expected one hint or both graded hints: {env_id}")
    return tuple(hints[key] for key in fields)
