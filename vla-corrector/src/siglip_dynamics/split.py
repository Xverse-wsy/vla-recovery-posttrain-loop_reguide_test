"""Reproducible train/validation splits for cached dynamics samples."""

import numpy as np
import torch
from torch.utils.data import Subset, random_split


def split_dataset(dataset, *, val_ratio: float, seed: int, unit: str):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    if unit == "sample":
        count = len(dataset)
        if count < 2:
            raise ValueError("Splitting requires at least two samples")
        val_count = min(max(1, int(count * val_ratio)), count - 1)
        train, val = random_split(dataset, [count - val_count, val_count], generator=generator)
        manifest = {"unit": unit, "seed": seed, "train_samples": len(train), "val_samples": len(val)}
    elif unit == "episode":
        if dataset.mode != "quantized_frames":
            raise ValueError("Episode splitting requires a frame cache with episode_index.npy")
        sample_episodes = np.asarray([int(dataset.ep_idx[t]) for t, _ in dataset.pairs])
        episodes = np.unique(sample_episodes)
        if len(episodes) < 2:
            raise ValueError("Episode splitting requires at least two episodes with valid samples")
        shuffled = episodes[torch.randperm(len(episodes), generator=generator).numpy()]
        val_count = min(max(1, int(len(episodes) * val_ratio)), len(episodes) - 1)
        val_episodes, train_episodes = shuffled[:val_count], shuffled[val_count:]
        val_mask = np.isin(sample_episodes, val_episodes)
        train = Subset(dataset, np.flatnonzero(~val_mask).tolist())
        val = Subset(dataset, np.flatnonzero(val_mask).tolist())
        manifest = {
            "unit": unit,
            "seed": seed,
            "train_episodes": sorted(int(x) for x in train_episodes),
            "val_episodes": sorted(int(x) for x in val_episodes),
            "train_samples": len(train),
            "val_samples": len(val),
        }
    else:
        raise ValueError(f"Unknown split unit: {unit}")
    return train, val, manifest
