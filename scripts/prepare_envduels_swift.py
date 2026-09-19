"""Create deterministic ms-swift Gym dataset rows from an EnvDuels manifest."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stable_seed(master_seed, env_id, index):
    material = f"{master_seed}\0{env_id}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def build_rows(export_dir, container_export_dir, master_seed):
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported EnvDuels manifest version")
    environments = manifest.get("environments")
    if not isinstance(environments, list) or not environments:
        raise ValueError("EnvDuels manifest contains no environments")
    ids = [row.get("id") for row in environments]
    if any(not isinstance(env_id, str) or not env_id for env_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest environment IDs must be unique nonempty strings")
    rows = []
    for env_id in ids:
        seed = stable_seed(master_seed, env_id, 0)
        rows.append({
            "messages": [{"role": "user", "content": "<envduels-reset>"}],
            "env_config": {
                "name": "envduels",
                "export_dir": container_export_dir,
                "env_id": env_id,
                "seed": seed,
            },
            "env_id": env_id,
            "seed": seed,
        })
    return rows, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def serialize_rows(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=ROOT.parent / "exports/duel_harness_004_rl")
    parser.add_argument("--container-export-dir", default="/workspace/envduels/exports/duel_harness_004_rl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/envduels/fixed90-swift.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_dir = args.export_dir.expanduser().resolve()
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")
    rows, digest = build_rows(export_dir, args.container_export_dir, args.seed)
    content = serialize_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != content:
            parser.error(
                f"{args.output} already exists with different content; "
                "choose another --output or remove it explicitly"
            )
        status = "validated-existing"
    else:
        args.output.write_text(content, encoding="utf-8")
        status = "created"
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "status": status,
                      "environments": len(rows),
                      "fixed_instances_per_environment": 1,
                      "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
