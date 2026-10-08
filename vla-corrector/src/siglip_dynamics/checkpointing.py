"""Retain periodic record-low archives and one epoch-labelled current best."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def should_archive(epoch: int) -> bool:
    """Called only after the epoch strictly improves the global validation best."""
    return epoch > 10 and epoch % 5 == 0


def capture_rng_state(rank: int, cuda_devices=()) -> dict:
    """Use builtins/tensors so training archives load with weights_only=True.

    Capture only the CUDA devices used by this process; visiting every visible
    GPU in DDP would create unnecessary contexts on the other ranks' devices.
    """
    generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "rank": rank,
        "python": random.getstate(),
        "numpy": {
            "generator": generator,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": {str(device): torch.cuda.get_rng_state(device) for device in cuda_devices},
    }


def _atomic_write(path: Path, payload, *, json_file=False) -> None:
    """Publish a complete file; a failed write cannot replace the previous best."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        if json_file:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointWriter:
    """Own only files from this run; previously saved model directories are protected."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "checkpoint_manifest.json"
        if self.manifest_path.exists() or any(self.directory.glob("*.pt")):
            raise FileExistsError(
                f"Checkpoint directory already contains saved models: {self.directory}. "
                "Use a fresh --checkpoint-dir; existing checkpoints will not be overwritten."
            )
        self.current_best: Path | None = None
        self.archives: list[str] = []

    def save_improvement(self, checkpoint: dict, *, training_state: dict | None = None):
        """The trainer calls this only for a strict validation cosine improvement."""
        epoch = int(checkpoint["best_epoch"])
        best_path = self.directory / f"best_cosine_model_epoch_{epoch}.pt"
        archive_path = self.directory / f"checkpoint_epoch_{epoch}.pt" if should_archive(epoch) else None
        if archive_path is not None and training_state is None:
            raise ValueError("A periodic training archive requires optimizer/scheduler/RNG state")
        for path in (best_path, archive_path):
            if path is not None and path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {path}")

        best = {**checkpoint, "format_version": 2, "epoch": epoch, "checkpoint_role": "current_best"}
        _atomic_write(best_path, best)
        archives = list(self.archives)
        if archive_path is not None:
            archive = {**best, **training_state, "checkpoint_role": "training_archive"}
            _atomic_write(archive_path, archive)
            archives.append(archive_path.name)
        _atomic_write(self.directory / "config.json", checkpoint["train_cfg"], json_file=True)
        _atomic_write(
            self.manifest_path,
            {
                "format_version": 2,
                "current_best": best_path.name,
                "current_best_epoch": epoch,
                "current_best_cosine_loss": checkpoint["best_metric_value"],
                "archives": archives,
            },
            json_file=True,
        )
        # Only remove the previous best owned by this instance, after publishing
        # the new model and manifest. Periodic archives and old runs are untouched.
        if self.current_best is not None:
            self.current_best.unlink()
        self.current_best = best_path
        self.archives = archives
        return best_path, archive_path
