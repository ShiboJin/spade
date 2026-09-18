"""Fetch one pinned community AIME26 snapshot, without modifying the questions."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "79037aebdb6580008fb960d17cb21fd3099083e3"
EXPECTED_SHA256 = "52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee"
URL = f"https://huggingface.co/datasets/math-ai/aime26/resolve/{REVISION}/aime2026.jsonl"


def validate_rows(rows):
    if len(rows) != 30:
        raise ValueError(f"Expected 30 AIME26 problems, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != 30 or len({row["problem"].strip() for row in rows}) != 30:
        raise ValueError("Duplicate problem IDs/text")
    for row in rows:
        if type(row["id"]) is not int or row["id"] < 1:
            raise ValueError("Problem IDs must be positive integers")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError("Empty problem")
        answer = str(row["answer"])
        if not answer.isascii() or not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError("AIME answers must be integers 0..999")
    return rows


def load_data(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("AIME26 data does not match the pinned snapshot; do not mix benchmark revisions")
    rows = validate_rows([json.loads(line) for line in data.splitlines() if line.strip()])
    return rows, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/aime26/aime2026.jsonl")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        rows, digest = load_data(target)
        print(f"Existing file validated, not overwritten: {len(rows)} problems; sha256={digest}")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Downloaded AIME26 checksum mismatch")
    validate_rows([json.loads(line) for line in data.splitlines() if line.strip()])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    provenance = {"url": URL, "revision": REVISION, "sha256": hashlib.sha256(data).hexdigest(),
                  "source": "math-ai/aime26 (community copy, not an official MAA distribution)", "count": 30}
    with target.with_suffix(".source.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
