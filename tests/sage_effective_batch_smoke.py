"""Three CPU DDP ranks: full-batch Swift GRPO loss and gradients, including constant groups."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from spade.swift_backend.envduels_sage_loss import install_effective_batch_loss
from spade.swift_backend.hint_resampling import refill_constant_groups
from test_hint_resampling import samples


def gather_rows(rows):
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, rows)
    return [row for part in parts for row in part]


def gather_tensor(tensor):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.detach().contiguous())
    return torch.stack(parts) if tensor.ndim == 0 else torch.cat(parts)


def worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from swift.rl_core.data import GRPOBatch

    class Harness:
        _postprocess_batch = GRPOTrainer._postprocess_batch
        _compute_advantages = GRPOTrainer._compute_advantages
        compute_loss = GRPOTrainer.compute_loss
        _compute_loss = GRPOTrainer._compute_loss
        _compute_loss_single = GRPOTrainer._compute_loss_single
        _compute_loss_and_metrics = GRPOTrainer._compute_loss_and_metrics
        _update_metrics = GRPOTrainer._update_metrics

        def split_by_mini_batches(self, rows):
            return [rows[i:i+2] for i in range(0, len(rows), 2)]

        def _get_per_token_logps_and_entropies(self, model, model_inputs, grpo_batch, compute_entropy=False):
            return model(model_inputs["features"]).squeeze(-1), None

    install_effective_batch_loss(Harness, gather_rows)
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.03, -0.02]], dtype=torch.float64))
    ddp = DDP(model)
    trainer = Harness()
    trainer.model = ddp
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=world,
                                          process_index=rank, is_main_process=rank == 0, gather_for_metrics=gather_tensor)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                   report_to=[], delta=None, top_entropy_quantile=1.0,
                                   steps_per_generation=4, gradient_accumulation_steps=4)
    trainer.template = SimpleNamespace(padding_free=False, sequence_parallel_size=1)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._logs = defaultdict(list)
    trainer._logs["rewards"] = defaultdict(list)
    trainer.num_generations = trainer.num_generations_eval = 4
    trainer.loss_type = "grpo"
    trainer.scale_rewards = "none"
    trainer.advantage_estimator = "grpo"
    trainer.reward_weights = torch.ones(1, dtype=torch.float64)
    trainer.reward_func_names = ["terminal"]
    trainer.dynamic_num_samples = False
    trainer.kl_in_reward = trainer.use_liger_loss = trainer.compute_entropy = False
    # Swift exposes dynamic OPSD capability even when no teacher input is present.
    trainer._has_teacher = True
    trainer.use_teacher_api = trainer.overlong_filter = False
    trainer.log_rollout_offpolicy_metrics = False
    trainer.rollout_importance_sampling_mode = None
    trainer.disable_rollout_importance_sampling = True
    trainer.importance_sampling_level = "token"
    trainer.off_policy_sequence_mask_delta = None
    trainer.advantage_reweight = None
    trainer.sdar_loss_coef = 0.0
    trainer.chord_sft_iterator = None
    trainer.epsilon_low = trainer.epsilon_high = 0.2

    features = (torch.arange(24*3*2, dtype=torch.float64).view(24, 3, 2) % 17 - 8) / 10
    lengths = torch.tensor([1 + i % 3 for i in range(24)])
    mask = torch.arange(3)[None, :] < lengths[:, None]
    start, end = rank * 8, (rank+1)*8
    for valid in (6, 5, 4, 3, 2, 1, 0, 6):
        initial = [s for g in range(6) for s in samples(f"env{g}")]
        rewards = torch.tensor([int(i % 4 == 0) if i//4 < valid else int(i//4 % 2)
                                for i in range(24)], dtype=torch.float64)
        def generate(local, attempt):
            output = deepcopy(local)
            for row in output:
                env, offset = row.request_id.split("-")
                row.rollout_infos = dict(total_reward=int(rewards[int(env[3:])*4 + int(offset)]))
            return output
        selection_stats = []
        selected = refill_constant_groups(initial[start:end], sage_generate=generate,
                                          gather=gather_rows, group_size=4, max_attempts=1,
                                          min_valid_groups=99, process_index=rank, refill=None,
                                          keep_constant_groups=True, metrics=selection_stats.append)
        assert selection_stats[-1]["valid_groups"] == valid
        assert selection_stats[-1]["effective_trajectories"] == 24
        assert selection_stats[-1]["masked_groups"] == selection_stats[-1]["discarded_groups"] == 0
        for beta in (0.0, 0.1):
            trainer.beta = beta
            ddp.zero_grad(set_to_none=True)
            with torch.no_grad():
                old = model(features).squeeze(-1) + torch.linspace(-0.3, 0.3, 24)[:, None]
                reference = old + 0.15
                # Constant groups retain their KL loss when beta is nonzero.
                reference[valid*4:] += 2.0
            trainer._rewards_per_func = rewards[:, None]
            batches = []
            for i in range(start, end, 2):
                grpo = GRPOBatch(completion_mask=mask[i:i+2].clone(),
                                 truncated_mask=torch.zeros(2, dtype=torch.bool), seq_lengths=lengths[i:i+2],
                                 old_per_token_logps=old[i:i+2], ref_per_token_logps=reference[i:i+2])
                batches.append(dict(model_inputs=dict(input_ids=torch.ones(2, 3, dtype=torch.long),
                                                      features=features[i:i+2]), grpo_batch=grpo))
            trainer._postprocess_batch(selected, batches)
            total_loss = torch.zeros((), dtype=torch.float64)
            for i, batch in enumerate(batches):
                loss = trainer.compute_loss(ddp, batch)
                assert torch.isfinite(loss)
                if start + i*2 >= valid*4:
                    if beta == 0:
                        assert loss.item() == 0.0
                    assert batch["grpo_batch"].completion_mask.any()
                (loss / 4).backward()
                total_loss += loss.detach() / 4
            dist.all_reduce(total_loss)
            total_loss /= world
            actual_grad = model.weight.grad.clone()
            # Independent full-batch GRPO objective; constant slots stay in the denominator.
            weight = model.weight.detach().clone().requires_grad_(True)
            n = 24
            logps = torch.nn.functional.linear(features[:n], weight).squeeze(-1)
            adv = rewards[:n].view(-1, 4)
            adv = (adv - adv.mean(1, keepdim=True)).flatten()[:, None]
            ratio = (logps - old[:n]).exp()
            per_token = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
            difference = (reference[:n] - logps).clamp(-20, 20)
            per_token += beta * (difference.exp() - difference - 1).clamp(-10, 10)
            baseline = ((per_token * mask[:n]).sum(1) / mask[:n].sum(1)).mean()
            baseline.backward()
            torch.testing.assert_close(total_loss, baseline.detach(), atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(actual_grad, weight.grad, atol=1e-10, rtol=1e-10)
            assert trainer._metrics["train"]["sage/effective_reward"][-1] == rewards.mean().item()
    # Eval must not require training selection metadata or apply its scaling.
    ddp.eval()
    raw = batches[0]
    assert "_sage_loss_scale" not in raw
    assert torch.isfinite(trainer.compute_loss(ddp, raw))
    dist.barrier()
    if rank == 0:
        print("PASS: 3 CPU DDP ranks: 0–6 mixed groups match full-batch Swift GRPO loss and gradients")
        print("PASS: constant groups retain masks and KL; 0–1 mixed groups still train; subsequent full window and eval work")
    dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    with TemporaryDirectory() as directory:
        mp.start_processes(worker, args=(3, "file://" + str(Path(directory) / "rendezvous")),
                           nprocs=3, start_method="fork", join=True)


if __name__ == "__main__":
    main()
