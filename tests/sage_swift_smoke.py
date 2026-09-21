"""Exercise installed ms-swift Gym rollout -> hint selection -> training encoding.

Uses scripted actions and the real Qwen tokenizer/template, never loads weights.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from swift.model import get_model_processor
from swift.template import get_template
from swift.rl_core.data import GRPOSample
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin
from swift.rlhf_trainers.utils import encode_sample
from swift.infer_engine.protocol import (
    ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, RequestConfig, RolloutOutput, UsageInfo,
)
from swift.rollout.multi_turn import GYMScheduler

from spade.swift_backend import envduels_gym
from spade.swift_backend.envduels_sage import install_hint_resampling
from spade.swift_backend.hint_resampling import SkipHintBatch
from unittest.mock import patch


SOURCE = r'''
class Env:
    def __init__(self, max_turns): pass
    def reset(self, seed=None): return f"Problem {seed}. Choose an action.", {}
    def step(self, action): return "Done", float(action == r"\boxed{WIN}"), True, False, {}
'''


class Harness:
    _infer_single_or_multi_turn = RolloutTrainerMixin._infer_single_or_multi_turn
    _colocate_multi_turn_infer = RolloutTrainerMixin._colocate_multi_turn_infer
    _postprocess_rollout_outputs = RolloutTrainerMixin._postprocess_rollout_outputs
    samples2requests = RolloutTrainerMixin.samples2requests
    _has_teacher_explicit = RolloutTrainerMixin._has_teacher_explicit
    _setup_teacher = RolloutTrainerMixin._setup_teacher

    def training_step(self, *args, **kwargs):
        raise AssertionError("This smoke never trains")

    def _postprocess_batch(self, *args):
        raise AssertionError("This rollout smoke never prepares loss")

    def compute_loss(self, *args, **kwargs):
        raise AssertionError("This rollout smoke never computes loss")

    def to_samples(self, rows):
        return [GRPOSample.from_row(row) for row in rows]

    def _rollout(self, samples, request_config, is_global_inputs=False):
        requests = self.samples2requests(samples)
        outputs = []
        self.rounds += bool(requests)
        counts = defaultdict(int)
        for req in requests:
            cfg = req.data_dict["env_config"]
            index = counts[cfg["env_id"]]
            counts[cfg["env_id"]] += 1
            success = (cfg.get("hint_level", 0) > 0 or cfg["env_id"] == "easy") and index == 0 and cfg["env_id"] != "blocked"
            text = r"\boxed{WIN}" if success else r"\boxed{LOSE}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            choice = ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop",
                token_ids=tokens, logprobs={"content": [{"logprob": -0.25} for _ in tokens]})
            outputs.append(RolloutOutput(response=ChatCompletionResponse(
                model="scripted", choices=[choice], usage=UsageInfo(0, len(tokens), len(tokens)), id=req.uuid)))
        return outputs


@patch.dict("os.environ", SPADE_MIN_VALID_GROUPS="1")
def main():
    _, processor = get_model_processor("checkpoints/Qwen3.8-27B", model_type="qwen3_5", load_model=False)
    template = get_template(processor, template_type="qwen3_5", enable_thinking=False)
    template.set_mode("train")
    install_hint_resampling(Harness, lambda x: x)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "env.py").write_text(SOURCE)
        rows = []
        for env in ("hard", "easy", "blocked"):
            (root / f"{env}.json").write_text(json.dumps(
                dict(environment_id=env, hints=dict(hint="Use WIN to finish the game."))))
            rows.append(dict(id=env, source="env.py", class_name="Env", max_turns=2, domain="test",
                             source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(), privileged=f"{env}.json"))
        (root / "manifest.json").write_text(json.dumps(dict(format_version=1, environments=rows)))
        harness = Harness()
        harness.tokenizer = template.tokenizer
        harness.template = template
        harness.model = torch.nn.Linear(1, 1)
        harness.args = SimpleNamespace(max_turns=2, output_dir=directory, seed=42,
                                       steps_per_generation=1, gradient_accumulation_steps=1)
        harness.state = SimpleNamespace(global_step=0)
        harness.accelerator = SimpleNamespace(is_main_process=True, process_index=0)
        harness._metrics = {"train": defaultdict(list)}
        harness._step = 0
        harness.num_generations = 4
        harness.num_iterations = 1
        harness.loss_type = "grpo"
        harness.scale_rewards = "none"
        harness.use_liger_loss = False
        harness.kl_in_reward = False
        # Use real Swift initialization instead of hardcoding _has_teacher=False.
        harness._setup_teacher()
        assert harness._has_teacher is True
        assert harness._has_teacher_explicit() is False
        assert harness.teacher_model is None
        harness.chord_sft_iterator = None
        harness.use_gym_env = True
        harness.use_fast_infer = True
        harness.vllm_mode = "colocate"
        harness.enable_server_multi_turn = False
        harness.dynamic_num_samples = False
        harness.async_generate = False
        harness.rounds = 0
        harness.multi_turn_scheduler = GYMScheduler(gym_env="envduels", max_turns=2,
                                                    tokenizer=template.tokenizer, template=template)
        def row(env):
            return dict(messages=[dict(role="user", content="<envduels-reset>")],
                        env_config=dict(name="envduels", export_dir=directory, env_id=env, seed=17),
                        chat_template_kwargs=dict(enable_thinking=False))
        harness.train_dataset = [row("hard"), row("easy"), row("blocked")]
        batch = harness.to_samples([row("hard")] * 4)
        for i, sample in enumerate(batch):
            sample.request_id = f"sample-{i}"
        for changes, expected in ((dict(teacher_prompt="privileged teacher input"), "teacher_input=True"),
                                  (dict(teacher_images=[]), "teacher_input=True")):
            unsupported = deepcopy(batch)
            for key, value in changes.items():
                setattr(unsupported[0], key, value)
            try:
                harness._infer_single_or_multi_turn(unsupported, RequestConfig())
            except ValueError as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("Actual teacher input must still be rejected")
        assert harness.rounds == 0
        outputs = harness._infer_single_or_multi_turn(batch, RequestConfig())
        assert harness.rounds == 2
        assert [s.rollout_infos["total_reward"] for s in outputs] == [1, 0, 0, 0]
        for sample in outputs:
            encoded = encode_sample(sample, template)
            decoded = template.tokenizer.decode(encoded["input_ids"])
            assert "Player hint:" in decoded and "Use WIN to finish the game." in decoded
            # Hint tokens are context, while only assistant actions carry loss.
            train_ids = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
            assert "Use WIN to finish" not in template.tokenizer.decode(train_ids)
            assert sample.rollout_logprobs and sample.response_token_ids
        # Eval must clear the level and must not invoke rescue, even for hard tasks.
        harness.model.eval()
        outputs_eval = harness._infer_single_or_multi_turn(deepcopy(outputs), RequestConfig())
        assert harness.rounds == 3
        assert all(s.rollout_infos["total_reward"] == 0 for s in outputs_eval)
        assert all("Player hint:" not in str(s.messages) for s in outputs_eval)
        # Even with other pool rows and a stale refill setting, never replace envs.
        harness.model.train()
        blocked = harness.to_samples([row("blocked")] * 4)
        for i, sample in enumerate(blocked):
            sample.request_id = f"blocked-{i}"
        before = harness.rounds
        with patch.dict("os.environ", SPADE_MAX_ROLLOUT_ATTEMPTS="2"):
            try:
                harness._infer_single_or_multi_turn(blocked, RequestConfig())
            except SkipHintBatch:
                pass
            else:
                raise AssertionError("All-zero group must skip without environment refill")
        assert harness.rounds == before + 2  # no hint, one hinted retry
        assert all("hint_level" not in r["env_config"] for r in harness.train_dataset)
        with patch.dict("os.environ", SPADE_MAX_ROLLOUT_ATTEMPTS="1", SPADE_MIN_VALID_GROUPS="2"):
            initial = deepcopy(batch) + deepcopy(blocked)
            try:
                harness._infer_single_or_multi_turn(initial, RequestConfig())
            except SkipHintBatch:
                pass
            else:
                raise AssertionError("Incomplete batch must skip, never restore constant groups")
        with patch.dict("os.environ", SPADE_MAX_ROLLOUT_ATTEMPTS="1"):
            partial = harness._infer_single_or_multi_turn(deepcopy(batch) + deepcopy(blocked), RequestConfig())
            assert len(partial) == 8
            assert [s.rollout_infos["sage_valid_group"] for s in partial] == [True]*4 + [False]*4
            assert all(s.rollout_infos["sage_valid_groups"] == 1 for s in partial)
        harness.train_dataset = [row("blocked")]
        try:
            harness._infer_single_or_multi_turn(blocked, RequestConfig())
        except SkipHintBatch:
            pass
        else:
            raise AssertionError("All-zero window must skip")
        print("PASS: real Swift Gym rollout, Qwen training encoding retains hint, masks hint loss, eval stays unhinted")
        print("PASS: no-refill selection and bounded skip use the installed Swift sample/request interfaces")


if __name__ == "__main__":
    main()
