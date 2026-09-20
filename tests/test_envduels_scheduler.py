"""Exercise the real ms-swift Gym lifecycle at rollout limits (CPU only)."""
import asyncio
import importlib.util
from types import SimpleNamespace
import unittest


if importlib.util.find_spec("swift") is not None:
    from spade.swift_backend.envduels_scheduler import EnvDuelsScheduler
else:
    EnvDuelsScheduler = None


class Template:
    max_length = 100

    def set_mode(self, mode):
        self.mode = mode

    def encode(self, inputs):
        # Character tokens with a generation prefix: the observation must count.
        length = sum(len(m["content"]) for m in inputs["messages"]) + 5
        return {"input_ids": list(range(length))}


class Env:
    closed = False

    def __init__(self, observation, done):
        self.observation, self.done = observation, done

    async def step(self, messages):
        return self.observation, 0.25, self.done, {}

    async def close(self):
        self.closed = True


@unittest.skipIf(EnvDuelsScheduler is None, "Requires the ms-swift training image")
class EnvDuelsSchedulerTests(unittest.TestCase):
    def run_turn(self, *, finish="stop", turn=1, observation="next", done=False):
        template = Template()
        scheduler = EnvDuelsScheduler(template=template, max_turns=24)
        env = Env(observation, done)
        request = SimpleNamespace(uuid="test", messages=[{"role": "assistant", "content": "action"}])
        response = SimpleNamespace(finish_reason=finish)
        scheduler._envs["test"] = env
        scheduler._total_rewards["test"] = 0.5
        scheduler._step_rewards["test"] = [0.5]
        result = asyncio.run(scheduler.on_turn_end(request, response, turn))
        self.assertEqual(result["rollout_infos"]["total_reward"], 0.75)
        self.assertEqual(result["rollout_infos"]["step_rewards"], [0.5, 0.25])
        self.assertEqual(template.max_length, 100)
        return scheduler, env, request, response, result

    def test_per_round_generation_limit_can_continue(self):
        scheduler, env, _, _, result = self.run_turn(finish="length")
        self.assertFalse(result["done"])
        self.assertFalse(result["rollout_infos"]["gym_done"])
        self.assertFalse(env.closed)
        self.assertIn("test", scheduler._envs)

    def test_next_observation_cannot_fill_or_overflow_context(self):
        for size in (89, 90, 200):
            with self.subTest(size=size):
                scheduler, env, request, response, result = self.run_turn(observation="x" * size)
                self.assertTrue(result["done"])
                self.assertEqual(response.finish_reason, "length")
                self.assertEqual(len(request.messages), 1)
                self.assertTrue(env.closed)
                self.assertFalse(scheduler._envs)
                self.assertFalse(scheduler._pending_obs)

    def test_prompt_with_room_continues(self):
        scheduler, env, request, response, result = self.run_turn(observation="x" * 88)
        self.assertFalse(result["done"])
        self.assertFalse(env.closed)
        scheduler.step(request, response, 1)
        self.assertEqual(request.messages[-1]["content"], "x" * 88)

    def test_max_turns_closes_unfinished_environment(self):
        _, env, _, _, result = self.run_turn(turn=24)
        self.assertTrue(result["done"])
        self.assertTrue(env.closed)
        self.assertEqual(result["rollout_infos"]["truncation_reason"], "max_turns")

    def test_natural_terminal_reward_is_preserved(self):
        _, env, _, _, result = self.run_turn(done=True)
        self.assertTrue(result["rollout_infos"]["gym_done"])
        self.assertNotIn("gym_truncated", result["rollout_infos"])
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
