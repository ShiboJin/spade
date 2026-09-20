"""CPU-only offline reproduction of Swift's completions table logging.

Run in the training image as a non-root user with no writable home directory.
"""
import os
from pathlib import Path
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix="spade-wandb-smoke-") as directory:
        root = Path(directory)
        for variable, path in (("WANDB_DIR", root / "wandb"),
                               ("WANDB_DATA_DIR", root / "wandb/data"),
                               ("WANDB_CACHE_DIR", root / "wandb/cache")):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
        os.environ["WANDB_MODE"] = "offline"
        import wandb
        from wandb.sdk.artifacts.staging import get_staging_dir

        with wandb.init(project="spade-artifact-smoke", mode="offline"):
            wandb.log({"completions": wandb.Table(
                columns=["prompt", "completion", "reward"],
                data=[["test prompt", "test completion", 1.0]])})
            staging = Path(get_staging_dir())
            assert staging.is_relative_to(root / "wandb/data"), staging
            assert list((root / "wandb").rglob("*.table.json")), "table was not serialized"
        print("PASS: offline completions table and artifact staging with writable run directories")


if __name__ == "__main__":
    main()
