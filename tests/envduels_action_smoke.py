"""Compare adapter actions to direct boxed actions across the trusted export.

Run in the unified image; no model weights or GPUs are needed.
"""
from pathlib import Path
import argparse

from spade.core.envs.envduels_adapter import EnvDuelsAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=(
        Path(__file__).resolve().parents[2] / "exports/duel_harness_004_rl"))
    args = parser.parse_args()
    adapter = EnvDuelsAdapter(args.export_dir)
    cases = {env_id: "help" for env_id in adapter.list_environments()}
    cases.update({
        "environments/gpt-5.6-sol/env_005/harden_01": "heat 3 purge 0",
        "environments/grok-4.6/env_006/harden_01": "read north",
    })
    for env_id, action in cases.items():
        direct = adapter.create_instance_with_seed(env_id, 42)
        wrapped = adapter.create_instance_with_seed(env_id, 42)
        try:
            assert direct.reset() == wrapped.reset(), env_id
            expected = direct.env.env.step(r"\boxed{" + action + "}")
            # A reasoning example must not override the final chosen action.
            actual = wrapped.step(r"Example: \boxed{WRONG}. Final: \boxed{" + action + "}")
            assert actual[0] == expected[0], env_id
            assert actual[1] == float(bool(expected[2]) and expected[1] > 0), env_id
            assert actual[2:4] == expected[2:4], env_id
            if action == "heat 3 purge 0":
                assert "entered heat 3, purge 0" in actual[0], actual[0]
                assert "Invalid command" not in actual[0], actual[0]
            elif action == "read north":
                assert "No command was found" not in actual[0], actual[0]
            print(f"PASS {env_id}: {action}", flush=True)
        finally:
            direct.close()
            wrapped.close()
    print(f"PASS: all {len(cases)} exported environments receive the final boxed action")


if __name__ == "__main__":
    main()
