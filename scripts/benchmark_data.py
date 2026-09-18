"""Explicit evaluation protocols; extend here for new benchmark semantics."""
import hashlib
import json
import re
from scripts.prepare_aime26 import load_data as load_aime26
from spade.core.envs.envduels_adapter import extract_action

BENCHMARKS = ("aime2026", "boxed_integer", "boxed_exact_match")

def validate_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {BENCHMARKS}")
    return name

def load_benchmark(path, name):
    validate_benchmark(name)
    if name == "aime2026":
        return load_aime26(path)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not {"id", "problem", "answer"} <= row.keys():
            raise ValueError("Each JSONL row requires id, problem, answer")
        if type(row["id"]) not in (str, int) or not str(row["id"]).strip() or str(row["id"]) in ids:
            raise ValueError("IDs must be nonempty unique strings or integers")
        ids.add(str(row["id"]))
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError("problem must be a nonempty string")
        if type(row["answer"]) not in (str, int) or not str(row["answer"]).strip():
            raise ValueError("answer must be a nonempty string or integer")
        if name == "boxed_integer" and not re.fullmatch(r"[+-]?[0-9]+", str(row["answer"]).strip()):
            raise ValueError("boxed_integer requires integer labels")
    return rows, hashlib.sha256(raw).hexdigest()

def grade(text, answer, name):
    validate_benchmark(name)
    predicted = extract_action(text)
    valid = predicted is not None and bool(predicted.strip())
    if name == "boxed_exact_match":
        return predicted, bool(valid and predicted.strip() == str(answer).strip()), bool(valid)
    valid = bool(valid and re.fullmatch(r"[+-]?[0-9]+", predicted))
    if name == "aime2026":
        valid = bool(valid and predicted.isascii() and predicted.isdigit() and 0 <= int(predicted) <= 999)
    return predicted, bool(valid and int(predicted) == int(answer)), valid
