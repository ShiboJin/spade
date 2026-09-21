"""CPU group-selection contracts, including collective calls from empty ranks."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace
import unittest

from spade.swift_backend.hint_resampling import (
    RefillUnavailable, SkipHintBatch, env_config, refill_constant_groups, resample_with_hints,
)


def samples(env="hard", n=4, seed=17):
    return [SimpleNamespace(messages=[dict(role="user", content="<envduels-reset>")],
                            extra=dict(env_config=dict(name="envduels", export_dir="/export",
                                                       env_id=env, seed=seed)),
                            request_id=f"{env}-{i}", prompt_id="same-placeholder",
                            response_token_ids=[], rollout_logprobs=[], rollout_infos={}) for i in range(n)]


def hints_for(config):
    return ("author strategy", "stronger author strategy") if config["env_id"] == "graded" else ("author strategy",)


class Generator:
    def __init__(self, rewards):
        self.rewards = rewards
        self.calls = []

    def __call__(self, batch):
        self.calls.append(deepcopy(batch))
        seen = defaultdict(int)
        output = []
        for sample in deepcopy(batch):
            cfg = env_config(sample)
            env, level = cfg["env_id"], cfg["hint_level"]
            index = seen[env]
            seen[env] += 1
            sample.messages = [dict(role="system", content="game"),
                               dict(role="user", content=f"Observation: {cfg['seed']}")]
            if level:
                sample.messages[1]["content"] += "\nPlayer hint:\n" + hints_for(cfg)[level - 1] + "\n"
            sample.messages.append(dict(role="assistant", content="new rollout"))
            sample.response_token_ids = [[level + 10, index + 20]]
            sample.rollout_logprobs = [[-0.1, -0.2]]
            sample.rollout_infos = dict(total_reward=self.rewards[env, level][index])
            output.append(sample)
        return output


class HintResamplingTests(unittest.TestCase):
    def run_sage(self, batch, generator, **kwargs):
        return resample_with_hints(batch, generate=generator, gather=lambda x: x,
                                   group_size=4, hints_for=hints_for, **kwargs)

    def test_rescues_whole_group_preserves_hint_and_selected_tokens(self):
        batch = samples()
        generator = Generator({("hard", 0): [0]*4, ("hard", 1): [0, 1, 0, 0]})
        output = self.run_sage(batch, generator)
        self.assertEqual(len(generator.calls), 2)
        self.assertTrue(all(env_config(s)["hint_level"] == 1 for s in output))
        self.assertEqual([s.rollout_infos["total_reward"] for s in output], [0, 1, 0, 0])
        for original, selected, retry in zip(batch, output, generator.calls[1]):
            self.assertNotIn("hint_level", env_config(original))
            self.assertIn("author strategy", selected.messages[1]["content"])
            self.assertEqual(selected.response_token_ids[0][0], 11)
            self.assertEqual(selected.rollout_logprobs, [[-0.1, -0.2]])
            self.assertEqual(retry.response_token_ids, [])
            self.assertEqual(env_config(retry)["seed"], 17)
            self.assertNotEqual(selected.request_id, original.request_id)

    def test_successful_no_hint_group_is_not_regenerated(self):
        generator = Generator({("easy", 0): [0, 1, 0, 0],
                               ("hard", 0): [0]*4, ("hard", 1): [1, 0, 0, 0]})
        output = self.run_sage(samples("easy") + samples(), generator)
        self.assertEqual(len(generator.calls[1]), 4)
        self.assertTrue(all(env_config(s)["env_id"] == "hard" for s in generator.calls[1]))
        self.assertTrue(all(env_config(s)["hint_level"] == 0 for s in output[:4]))
        self.assertTrue(all("Player hint" not in s.messages[1]["content"] for s in output[:4]))

    def test_graded_hints_escalate_and_exhaustion_is_finite(self):
        for final in ([0, 1, 0, 0], [0]*4):
            generator = Generator({("graded", 0): [0]*4, ("graded", 1): [0]*4,
                                   ("graded", 2): final})
            output = self.run_sage(samples("graded"), generator)
            self.assertEqual(len(generator.calls), 3)
            self.assertTrue(all(env_config(s)["hint_level"] == 2 for s in output))

    def test_bad_group_or_nonbinary_reward_fails(self):
        for batch in (samples()[:3], samples()[:3] + samples("different")[:1]):
            with self.assertRaises(ValueError):
                self.run_sage(batch, Generator({}))
        with self.assertRaisesRegex(ValueError, "binary"):
            self.run_sage(samples(), Generator({("hard", 0): [0, 0.5, 0, 0]}))

    def test_missing_hint_from_returned_messages_fails(self):
        base = Generator({("hard", 0): [0]*4, ("hard", 1): [0, 1, 0, 0]})
        def broken(batch):
            output = base(batch)
            for sample in output:
                sample.messages[1]["content"] = "hint removed"
            return output
        with self.assertRaisesRegex(ValueError, "Hint missing"):
            self.run_sage(samples(), broken)

    def test_both_constant_reward_types_refilled_without_touching_pool(self):
        for constant in (0, 1):
            generator = Generator({("hard", 0): [constant]*4, ("hard", 1): [constant]*4,
                                   ("other", 0): [0, 1, 0, 0]})
            pool_row = samples("other")[0]
            output = refill_constant_groups(
                samples(), sage_generate=lambda b, a, **kw: self.run_sage(b, generator, **kw),
                gather=lambda x: x, group_size=4, max_attempts=3,
                refill=lambda pending, current, attempt: {g: pool_row for g in pending})
            self.assertTrue(all(env_config(s)["env_id"] == "other" for s in output))
            self.assertNotIn("hint_level", env_config(pool_row))
            self.assertEqual(pool_row.response_token_ids, [])

    def test_one_additional_attempt_then_skip(self):
        generator = Generator({("hard", 0): [0]*4, ("hard", 1): [0]*4,
                               ("other", 0): [1]*4})
        refills, stats = [], []
        def refill(pending, current, attempt):
            refills.append(attempt)
            return {g: samples("other")[0] for g in pending}
        with self.assertRaisesRegex(SkipHintBatch, "max_rollout_attempts=2"):
            refill_constant_groups(
                samples(), sage_generate=lambda b, a: self.run_sage(b, generator),
                gather=lambda x: x, group_size=4, max_attempts=2,
                refill=refill, metrics=stats.append)
        self.assertEqual(refills, [1])
        self.assertEqual(len(generator.calls), 3)  # initial, hint, one refill (all 1)
        self.assertEqual(stats[-1]["refill_attempts"], 1)
        self.assertEqual(stats[-1]["discarded_groups"], 2)

    def test_incomplete_batch_skips_even_with_accepted_groups(self):
        generator = Generator({("easy", 0): [0, 1, 0, 0],
                               ("hard", 0): [0]*4, ("hard", 1): [0, 1, 0, 0],
                               ("blocked", 0): [0]*4, ("blocked", 1): [0]*4,
                               ("other", 0): [1]*4})
        initial = samples("easy") + samples() + samples("blocked")
        before = deepcopy(initial)
        def refill(pending, current, attempt):
            self.assertEqual(pending, [2])
            return {2: samples("other")[0]}
        with self.assertRaises(SkipHintBatch):
            refill_constant_groups(
                initial, sage_generate=lambda b, a: self.run_sage(b, generator),
                gather=lambda x: x, group_size=4, max_attempts=2, refill=refill)
        self.assertEqual(initial, before)
        self.assertEqual([env_config(s)["env_id"] for s in generator.calls[-1]], ["other"]*4)

    def test_no_other_environments_skips_but_unrelated_errors_propagate(self):
        for error in (RefillUnavailable("pool exhausted"), ValueError("bad dataset")):
            generator = Generator({("hard", 0): [0]*4, ("hard", 1): [0]*4})
            def refill(*args):
                raise error
            with self.assertRaisesRegex(SkipHintBatch if isinstance(error, RefillUnavailable)
                                        else ValueError, str(error)):
                refill_constant_groups(
                    samples(), sage_generate=lambda b, a: self.run_sage(b, generator),
                    gather=lambda x: x, group_size=4, max_attempts=2, refill=refill)
            self.assertEqual(len(generator.calls), 2)

    def test_six_requested_accepts_four_five_six_and_skips_three(self):
        for valid in (3, 4, 5, 6):
            with self.subTest(valid=valid):
                initial = [s for g in range(6) for s in samples(f"env{g}")]
                rewards = {}
                for g in range(6):
                    rewards[f"env{g}", 0] = [0, 1, 0, 0] if g < valid else [g % 2]*4
                    rewards[f"env{g}", 1] = [0]*4
                for g in range(6):
                    rewards[f"replacement{g}", 0] = [1]*4
                generator = Generator(rewards)
                stats, records, refills = [], [], []
                def refill(pending, current, attempt):
                    refills.append((pending, attempt))
                    return {g: samples(f"replacement{g}")[0] for g in pending}
                def run():
                    return refill_constant_groups(
                        initial, sage_generate=lambda b, a: self.run_sage(b, generator),
                        gather=lambda x: x, group_size=4, max_attempts=2,
                        min_valid_groups=4, refill=refill, metrics=stats.append, audit=records.append)
                if valid < 4:
                    with self.assertRaisesRegex(SkipHintBatch, "min_valid_groups=4"):
                        run()
                else:
                    output = run()
                    self.assertEqual(len(output), 24)
                    self.assertEqual([s.rollout_infos["sage_valid_group"] for s in output],
                                     [True]*(valid*4) + [False]*((6-valid)*4))
                    self.assertTrue(all(s.rollout_infos["sage_valid_groups"] == valid for s in output))
                    self.assertEqual([s.request_id for s in output[:valid*4]],
                                     [s.request_id for s in initial[:valid*4]])
                self.assertEqual(stats[-1]["valid_groups"], valid)
                self.assertEqual(stats[-1]["effective_trajectories"], valid*4 if valid >= 4 else 0)
                self.assertEqual(records[-1]["accepted"], valid >= 4)
                self.assertEqual(len(refills), int(valid != 6))

    def test_pool_exhaustion_accepts_partial_if_threshold_met(self):
        generator = Generator({("easy", 0): [0, 1, 0, 0],
                               ("hard", 0): [0]*4, ("hard", 1): [0]*4})
        def unavailable(*args):
            raise RefillUnavailable("no other environments")
        output = refill_constant_groups(
            samples("easy") + samples(), sage_generate=lambda b, a: self.run_sage(b, generator),
            gather=lambda x: x, group_size=4, max_attempts=2, min_valid_groups=1, refill=unavailable)
        self.assertEqual([s.rollout_infos["sage_valid_group"] for s in output], [True]*4 + [False]*4)

    def test_threshold_is_validated(self):
        for minimum in (0, -1, True, 1.5, 2):
            with self.assertRaisesRegex(ValueError, "min_valid_groups"):
                refill_constant_groups(samples(), sage_generate=None, gather=lambda x: x, group_size=4,
                                       max_attempts=2, min_valid_groups=minimum, refill=None)

    def test_distributed_group_spans_ranks_and_idle_rank_joins_retry(self):
        # Three local samples per rank: G=4 groups cross rank boundaries, and
        # rank 2 has only easy-group samples and must still enter hint generation.
        all_samples = samples() + samples("easy")
        partitions = [all_samples[:3], all_samples[3:6], all_samples[6:]]
        barrier = Barrier(3, timeout=10)
        buffers = [None]*3
        calls = [[], [], []]
        def worker(rank):
            def gather(items):
                buffers[rank] = deepcopy(items)
                barrier.wait()
                result = deepcopy([item for buffer in buffers for item in buffer])
                barrier.wait()
                return result
            def generate(batch):
                calls[rank].append(len(batch))
                output = deepcopy(batch)
                for s in output:
                    cfg = env_config(s)
                    level = cfg["hint_level"]
                    # Only rank 0's first hard sample wins after receiving a hint.
                    s.rollout_infos = dict(total_reward=int(
                        (cfg["env_id"] == "hard" and level == 1 and rank == 0 and s is output[0])
                        or (cfg["env_id"] == "easy" and s.request_id == "easy-0")))
                    s.messages = [dict(role="user", content=("\nPlayer hint:\n" + hints_for(cfg)[0] + "\n") if level else "obs")]
                return output
            return resample_with_hints(partitions[rank], generate=generate, gather=gather,
                                       group_size=4, hints_for=hints_for)
        with ThreadPoolExecutor(3) as executor:
            outputs = list(executor.map(worker, range(3)))
        self.assertEqual(calls, [[3, 3], [3, 1], [2, 0]])
        flat = [s for output in outputs for s in output]
        self.assertEqual([env_config(s)["hint_level"] for s in flat], [1]*4 + [0]*4)

    def distributed_refill(self, rescue, min_valid_groups=None):
        batch = samples() + samples("easy")
        parts = [batch[:3], batch[3:6], batch[6:]]
        barrier = Barrier(3, timeout=10)
        buffers = [None]*3
        def worker(rank):
            def gather(rows):
                buffers[rank] = deepcopy(rows)
                barrier.wait()
                out = deepcopy([r for b in buffers for r in b])
                barrier.wait()
                return out
            def generate(local):
                out = deepcopy(local)
                for i, sample in enumerate(out):
                    cfg = env_config(sample)
                    sample.rollout_infos = dict(total_reward=int(
                        sample.request_id == "easy-0" or (rescue and cfg["env_id"] == "other" and rank == 0 and i == 0)))
                    sample.messages = [dict(role="user", content="\nPlayer hint:\nauthor strategy\n"
                                            if cfg["hint_level"] else "obs")]
                return out
            try:
                return refill_constant_groups(
                    parts[rank], gather=gather, group_size=4, max_attempts=2, process_index=rank,
                    min_valid_groups=min_valid_groups,
                    sage_generate=lambda local, attempt, **kw: resample_with_hints(
                        local, generate=generate, gather=gather, group_size=4, hints_for=hints_for, **kw),
                    refill=lambda pending, current, attempt: {g: samples("other")[0] for g in pending})
            except SkipHintBatch:
                return ["skipped"]
        with ThreadPoolExecutor(3) as executor:
            outputs = list(executor.map(worker, range(3)))
        return [s for output in outputs for s in output]

    def test_distributed_refill_preserves_successful_slots(self):
        flat = self.distributed_refill(rescue=True)
        self.assertEqual([env_config(s)["env_id"] for s in flat], ["other"]*4 + ["easy"]*4)
        self.assertEqual([s.request_id for s in flat[4:]], [f"easy-{i}" for i in range(4)])

    def test_distributed_partial_group_mask_agrees_across_ranks(self):
        flat = self.distributed_refill(rescue=False, min_valid_groups=1)
        self.assertEqual([s.rollout_infos["sage_valid_group"] for s in flat], [False]*4 + [True]*4)
        self.assertTrue(all(s.rollout_infos["sage_valid_groups"] == 1 for s in flat))

    def test_distributed_exhaustion_skips_on_every_rank(self):
        self.assertEqual(self.distributed_refill(rescue=False), ["skipped"]*3)


if __name__ == "__main__":
    unittest.main()
