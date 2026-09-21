"""CPU contracts for the ms-swift EnvDuels bridge; ms-swift is stubbed."""
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

from scripts.prepare_envduels_swift import build_rows, stable_seed


SOURCE = r'''
class Env:
    def __init__(self, max_turns): self.turns = 0
    def reset(self, seed=None): self.turns = 0; return f"problem-{seed}", {"reset": True}
    def step(self, action):
        self.turns += 1
        assert action.startswith("\\boxed{") and action.endswith("}"), action
        action = action[len("\\boxed{"):-1]
        return ("done" if action == "WIN" else "try again",
                1.0 if action == "WIN" else 0.0,
                action == "WIN", False, {"action": action})
'''


def install_swift_stubs():
    swift = ModuleType("swift")
    infer_engine = ModuleType("swift.infer_engine")
    protocol = ModuleType("swift.infer_engine.protocol")
    rollout = ModuleType("swift.rollout")
    gym_env = ModuleType("swift.rollout.gym_env")
    template = ModuleType("swift.template")

    class Env:
        def __init__(self, env_config): self.env_config = env_config

    protocol.RolloutInferRequest = object
    gym_env.Env = Env
    gym_env.envs = {}
    template.Messages = list
    sys.modules.update({"swift": swift, "swift.infer_engine": infer_engine,
                        "swift.infer_engine.protocol": protocol, "swift.rollout": rollout,
                        "swift.rollout.gym_env": gym_env, "swift.template": template})
    return gym_env.envs


class SwiftEnvDuelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "env.py").write_text(SOURCE)
        self.env_id = "test/env_001"
        manifest = {"format_version": 1, "environments": [{
            "id": self.env_id, "source": "env.py", "class_name": "Env",
            "max_turns": 3, "domain": "reasoning",
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        }]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_dataset_has_one_deterministic_instance_per_environment(self):
        rows, digest = build_rows(self.root, "/exports", 42)
        again, again_digest = build_rows(self.root, "/exports", 42)
        self.assertEqual(rows, again)
        self.assertEqual(digest, again_digest)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["env_config"]["name"], "envduels")
        self.assertEqual(rows[0]["seed"], stable_seed(42, self.env_id, 0))

    def test_swift_environment_reset_step_reward_and_close(self):
        registry = install_swift_stubs()
        sys.modules.pop("spade.swift_backend.envduels_gym", None)
        plugin = importlib.import_module("spade.swift_backend.envduels_gym")
        self.assertIs(registry["envduels"], plugin.SwiftEnvDuelsEnv)
        config = {"export_dir": str(self.root), "env_id": self.env_id, "seed": 123}
        first, second = plugin.SwiftEnvDuelsEnv(config), plugin.SwiftEnvDuelsEnv(config)

        observation, info, system = asyncio.run(first.reset(object()))
        self.assertIn("problem-123", observation)
        self.assertIn(r"\boxed{}", observation)
        self.assertEqual(info["problem_id"], f"{self.env_id}@seed=123")
        self.assertIn("interactive", system)
        asyncio.run(second.reset(object()))
        observation, reward, done, info = asyncio.run(
            first.step([{"role": "assistant", "content": r"thinking \boxed{READ}"}]))
        self.assertEqual((observation, reward, done), ("try again", 0.0, False))
        self.assertEqual(second.instance.env.env.turns, 0)
        _, reward, done, info = asyncio.run(
            first.step([{"role": "assistant", "content": r"\boxed{WIN}"}]))
        self.assertEqual((reward, done, info["terminated"]), (1.0, True, True))
        asyncio.run(first.close())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            asyncio.run(first.reset(object()))

    def test_hint_is_only_player_context_and_reward_is_unchanged(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["environments"][0]["privileged"] = "privileged.json"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "privileged.json").write_text(json.dumps(
            dict(environment_id=self.env_id, hints=dict(hint="Choose WIN."))))
        install_swift_stubs()
        sys.modules.pop("spade.swift_backend.envduels_gym", None)
        plugin = importlib.import_module("spade.swift_backend.envduels_gym")
        base = dict(export_dir=str(self.root), env_id=self.env_id, seed=123)
        plain = plugin.SwiftEnvDuelsEnv(base)
        hinted = plugin.SwiftEnvDuelsEnv({**base, "hint_level": 1})
        observation0, _, system0 = asyncio.run(plain.reset(object()))
        observation1, info1, system1 = asyncio.run(hinted.reset(object()))
        self.assertNotIn("Choose WIN.", observation0 + system0)
        self.assertEqual(system0, system1)
        self.assertEqual(observation1, observation0 + "\n\nPlayer hint:\nChoose WIN.\n")
        self.assertEqual(info1["hint_level"], 1)
        for env in (plain, hinted):
            _, reward, done, _ = asyncio.run(env.step([dict(role="assistant", content=r"\boxed{WIN}")]))
            self.assertEqual((reward, done), (1.0, True))
            asyncio.run(env.close())


if __name__ == "__main__":
    unittest.main()
