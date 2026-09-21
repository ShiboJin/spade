"""CPU check of installed Swift sampler + Accelerate's four-rank batch sharding."""
from collections import Counter
from types import SimpleNamespace

from torch.utils.data import DataLoader
from accelerate.data_loader import prepare_data_loader
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


def identity(rows):
    return rows


def main():
    for shuffle in (False, True):
        loaders = []
        trainers = []
        for rank in range(4):
            trainer = object.__new__(GRPOTrainer)
            trainer.train_dataset = list(range(90))
            trainer.template = SimpleNamespace(sequence_parallel_size=1)
            trainer.args = SimpleNamespace(generation_batch_size=32, steps_per_generation=8, seed=42)
            trainer.num_generations = 4
            trainer.num_iterations = 1
            trainer.shuffle_dataset = shuffle
            sampler = trainer._get_train_sampler()
            assert len(sampler) == 88 * 4 * 8
            loader = DataLoader(trainer.train_dataset, batch_size=8, sampler=sampler, collate_fn=identity)
            loaders.append(prepare_data_loader(loader, num_processes=4, process_index=rank,
                                               split_batches=False, put_on_device=False))
            trainers.append(trainer)
        omitted = []
        for epoch in range(3):
            shards = [list(loader) for loader in loaders]
            assert all(len(loader) == len(shard) == 88 for loader, shard in zip(loaders, shards))
            seen = set()
            for start in range(0, 88, 8):
                group_batch = [i for shard in shards for i in shard[start]]
                counts = Counter(group_batch)
                assert len(group_batch) == 32 and len(counts) == 8
                assert set(counts.values()) == {4}
                assert seen.isdisjoint(counts)
                seen.update(counts)
                # The same complete generation batch is reused across all 8 microsteps.
                for offset in range(8):
                    assert [i for shard in shards for i in shard[start+offset]] == group_batch
            assert len(seen) == 88
            omitted.append(set(range(90)) - seen)
            assert all(len(t.train_dataset) == 90 for t in trainers)
        if shuffle:
            assert omitted[0] != omitted[1]
        else:
            assert all(tail == {88, 89} for tail in omitted)
    print("PASS: 90 envs / 8 groups -> 11 complete windows on all 4 ranks; no partial window or padding")
    print("PASS: shuffled tails vary across passes; unshuffled tail repeats; full refill pool stays intact")


if __name__ == "__main__":
    main()
