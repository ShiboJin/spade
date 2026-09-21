"""Manifest-backed EnvDuels environments for actor-only fixed-environment RL.

Only load trusted exports: source hashes check integrity, not execution safety.
Environment code currently executes in the rollout process.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

from spade.core.envs.env_adapter import EnvironmentAdapter, EnvInstance


def select_complete_problem_groups(trajectories, batch_size, group_size):
    """Keep complete task groups; never duplicate episodes to fill a batch."""
    if group_size < 2 or batch_size < 1 or batch_size % group_size:
        raise ValueError("EnvDuels GRPO needs group_size >= 2 and a divisible batch_size")
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.metadata["problem_id"], []).append(trajectory)
    selected = []
    for group in groups.values():
        if len(group) == group_size:
            selected.extend(group)
        if len(selected) == batch_size:
            return selected
    raise RuntimeError(
        f"Not enough complete EnvDuels groups: {len(selected)}/{batch_size} episodes; "
        "inspect environment/generation failures before retrying"
    )


def extract_action(response: str):
    """Extract the final balanced boxed action, including nested JSON braces."""
    marker = response.rfind("\\boxed{")
    if marker < 0:
        return None
    start = marker + len("\\boxed{")
    depth = 1
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[start:index].strip() or None
    return None


class BoundEnvDuelsEnv:
    def __init__(self, env, seed, max_turns):
        self.env = env
        self.seed = seed
        self.max_turns = max_turns
        self.turns = 0

    def reset(self, seed=None):
        if seed is not None and seed != self.seed:
            raise ValueError("Cannot change the seed of a bound EnvDuels problem")
        self.turns = 0
        return self.env.reset(seed=self.seed)

    def step(self, response):
        self.turns += 1
        action = extract_action(response)
        if action is None:
            # Formatting mistakes are trainable failures, not environment crashes.
            return (
                "Invalid action format. Use \\boxed{ACTION}.",
                0.0, False, self.turns >= self.max_turns,
                {"format_error": True},
            )
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        if not math.isfinite(float(raw_reward)):
            raise ValueError("Non-finite EnvDuels reward")
        reward = float(bool(terminated) and raw_reward > 0)
        truncated = bool(truncated or (self.turns >= self.max_turns and not terminated))
        return obs, reward, terminated, truncated, {
            **info, "raw_reward": raw_reward, "success": bool(reward),
        }

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class EnvDuelsAdapter(EnvironmentAdapter):
    requires_same_problem_groups = True

    def __init__(self, export_dir, env_ids_file=None, seed=42):
        self.root = Path(export_dir).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported EnvDuels manifest version")
        rows = manifest["environments"]
        self.rows = {row["id"]: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("Duplicate EnvDuels environment IDs")
        if env_ids_file is not None:
            selected = [line.strip() for line in Path(env_ids_file).read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
            if len(selected) != len(set(selected)) or not set(selected) <= self.rows.keys():
                raise ValueError("Duplicate or unknown IDs in EnvDuels selection")
            self.rows = {env_id: self.rows[env_id] for env_id in selected}
        if not self.rows:
            raise ValueError("Empty EnvDuels environment selection")
        self.rng = random.Random(seed)
        self.classes = {}

    def list_environments(self):
        return list(self.rows)

    def get_difficulty_range(self, env_id):
        return (0, 0)

    def get_category(self, env_id):
        return self.rows[env_id]["domain"]

    def get_hint_levels(self, env_id):
        """Read author hints explicitly; ordinary environment resets never read them.

        Level 0 is always unhinted. A single-hint export supplies level 1;
        graded exports supply hint_1, then hint_2 (not concatenated).
        """
        from spade.core.envduels_hints import load_hint_levels
        return load_hint_levels(self.root, self.rows[env_id])

    def _create(self, env_id, seed):
        row = self.rows[env_id]
        if env_id not in self.classes:
            source = (self.root / row["source"]).resolve()
            if not source.is_relative_to(self.root):
                raise ValueError("Manifest source escapes export root")
            code = source.read_bytes()
            if hashlib.sha256(code).hexdigest() != row["source_sha256"]:
                raise ValueError(f"Environment source hash mismatch: {env_id}")
            name = "envduels_export_" + row["source_sha256"]
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                exec(compile(code, str(source), "exec"), module.__dict__)
                self.classes[env_id] = getattr(module, row["class_name"])
            except Exception:
                sys.modules.pop(name, None)
                raise
        env = self.classes[env_id](max_turns=row["max_turns"])
        return EnvInstance(
            env=BoundEnvDuelsEnv(env, seed, row["max_turns"]),
            env_id=env_id, category=self.get_category(env_id), source="envduels",
            metadata={"seed": seed, "problem_id": f"{env_id}@seed={seed}",
                      "source_sha256": row["source_sha256"]},
        )

    def create_instance(self, env_id, difficulty=0):
        return self._create(env_id, self.rng.randrange(2**63))

    def create_instance_with_seed(self, env_id, seed):
        """Create one reproducible instance for external rollout schedulers.

        GRPO backends duplicate one dataset row ``num_generations`` times.  A
        seed carried by that row therefore gives every member of the group the
        same problem while keeping independent environment state.
        """
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("EnvDuels seed must be an integer in [0, 2**63)")
        if env_id not in self.rows:
            raise ValueError(f"Unknown EnvDuels environment: {env_id}")
        return self._create(env_id, seed)

    def create_instances_same_problem(self, env_id, difficulty=0, n=1):
        seed = self.rng.randrange(2**63)
        instances = []
        try:
            for _ in range(n):
                instances.append(self._create(env_id, seed))
        except Exception:
            for instance in instances:
                instance.close()
            raise
        return instances
