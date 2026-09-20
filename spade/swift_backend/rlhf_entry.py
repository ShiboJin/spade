"""Bind each Accelerate worker before importing the training stack."""
import os


def main():
    import torch

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[device-guard] pid={os.getpid()} rank={os.environ.get('RANK', local_rank)} "
          f"local_rank={local_rank} current_device={torch.cuda.current_device()}", flush=True)
    from swift.pipelines import rlhf_main

    rlhf_main()


if __name__ == "__main__":
    main()
