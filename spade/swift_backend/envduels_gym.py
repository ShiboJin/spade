"""Register EnvDuels as an ms-swift multi-turn Gym environment.

Load this file with ``swift rlhf --external_plugins ... --gym_env envduels``.
The dataset row supplies ``export_dir``, ``env_id`` and ``seed`` in
``env_config``.  ms-swift duplicates the row for ``num_generations``; each
duplicate receives a separate environment object bound to the same seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from spade.core.envs.envduels_adapter import EnvDuelsAdapter
from swift.infer_engine.protocol import RolloutInferRequest
from swift.rollout.gym_env import Env, envs
from swift.template import Messages


SYSTEM_PROMPT = (
    "You are playing an interactive language game. Make one valid action per turn. "
    "Reason carefully and put the action for that turn inside \\boxed{}."
)

# One adapter per export directory caches checked source modules.  Each rollout
# still gets a new, independently mutable EnvInstance.
_ADAPTERS: Dict[str, EnvDuelsAdapter] = {}


def _get_adapter(export_dir: str) -> EnvDuelsAdapter:
    root = str(Path(export_dir).expanduser().resolve())
    adapter = _ADAPTERS.get(root)
    if adapter is None:
        adapter = EnvDuelsAdapter(root)
        _ADAPTERS[root] = adapter
    return adapter


class SwiftEnvDuelsEnv(Env):
    """Thin async wrapper around SPADE's manifest-verified EnvDuels adapter."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        export_dir = env_config.get("export_dir")
        env_id = env_config.get("env_id")
        seed = env_config.get("seed")
        if not isinstance(export_dir, str) or not export_dir:
            raise ValueError("env_config.export_dir must be a nonempty path")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("env_config.env_id must be a nonempty string")
        if type(seed) is not int:
            raise ValueError("env_config.seed must be an integer")
        adapter = _get_adapter(export_dir)
        self.hint_level = env_config.get("hint_level", 0)
        if type(self.hint_level) is not int or self.hint_level < 0:
            raise ValueError("hint_level must be a nonnegative integer")
        self.hint = None
        if self.hint_level:
            levels = adapter.get_hint_levels(env_id)
            if self.hint_level > len(levels):
                raise ValueError(f"No hint level {self.hint_level} for {env_id}")
            self.hint = levels[self.hint_level - 1]
        self.instance = adapter.create_instance_with_seed(env_id, seed)
        self.seed = seed
        self.closed = False

    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        del config
        if self.closed:
            raise RuntimeError("Cannot reset a closed EnvDuels environment")
        observation, info = self.instance.reset()
        initial = (
            f"Observation: {observation}\n\n"
            "Respond with exactly one action for this turn inside \\boxed{}."
        )
        if self.hint is not None:
            # Match the external CLI harness: guidance in player context only.
            initial += "\n\nPlayer hint:\n" + self.hint + "\n"
        metadata = {
            **info,
            "env_id": self.instance.env_id,
            "category": self.instance.category,
            "seed": self.seed,
            "problem_id": self.instance.metadata["problem_id"],
            "hint_level": self.hint_level,
        }
        return initial, metadata, SYSTEM_PROMPT

    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.closed:
            raise RuntimeError("Cannot step a closed EnvDuels environment")
        completion = action[-1].get("content", "") if action else ""
        if not isinstance(completion, str):
            completion = ""
        observation, reward, terminated, truncated, info = self.instance.step(completion)
        return observation, float(reward), bool(terminated or truncated), {
            **info,
            "env_id": self.instance.env_id,
            "seed": self.seed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    async def close(self):
        if not self.closed:
            self.instance.close()
            self.closed = True


envs["envduels"] = SwiftEnvDuelsEnv
