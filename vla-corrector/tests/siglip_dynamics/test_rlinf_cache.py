"""Standalone contract tests: python -m unittest discover -s tests/siglip_dynamics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from siglip_dynamics.dataset import SiglipDynamicsDataset
from siglip_dynamics.extract_rlinf import quantize_tokens
from siglip_dynamics.split import split_dataset


class CacheContractTests(unittest.TestCase):
    def test_episode_split_keeps_history_and_targets_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text(json.dumps({"format": "siglip_frame_cache_v2"}))
            z = np.arange(24 * 2 * 3, dtype=np.float32).reshape(24, 2, 3) / 25
            q, scales = quantize_tokens(z)
            for name, array in {
                "z_q": q,
                "z_scale": scales,
                "episode_index": np.repeat([10, 20, 30], 8),
                "actions": np.tile([0, 0, 0, 0, 0, 0, 1.0003991], (24, 1)).astype(np.float32),
            }.items():
                np.save(path / (name + ".npy"), array)
            dataset = SiglipDynamicsDataset(str(path), k_step=2, h_window=3)
            train, val, manifest = split_dataset(dataset, val_ratio=0.34, seed=42, unit="episode")
            self.assertEqual(len(dataset), 12)
            self.assertFalse(set(manifest["train_episodes"]) & set(manifest["val_episodes"]))
            frame_sets = []
            for subset in (train, val):
                frames = set()
                for index in subset.indices:
                    t, target = dataset.pairs[index]
                    frames.update(range(t - 2, t + 1))
                    frames.add(target)
                    self.assertEqual(dataset.ep_idx[t], dataset.ep_idx[target])
                frame_sets.append(frames)
            self.assertFalse(frame_sets[0] & frame_sets[1])
            _, _, again = split_dataset(dataset, val_ratio=0.34, seed=42, unit="episode")
            self.assertEqual(manifest, again)
            self.assertGreater(dataset[0]["a_t"][6].item(), 1)

    def test_quantization_error_and_zero_tokens(self):
        z = np.random.default_rng(0).normal(size=(4, 3, 16)).astype(np.float32)
        z[0, 0] = 0
        q, scales = quantize_tokens(z)
        restored = q.astype(np.float32) * scales.astype(np.float32)[..., None]
        self.assertTrue(np.isfinite(restored).all())
        self.assertLessEqual(float(np.max(np.abs(restored - z))), float(scales.max()) * 0.501)
        self.assertTrue((q[0, 0] == 0).all())

    def test_reject_single_episode(self):
        class SingleEpisode:
            mode = "quantized_frames"
            ep_idx = np.zeros(4)
            pairs = [(0, 1), (1, 2)]

        with self.assertRaisesRegex(ValueError, "at least two episodes"):
            split_dataset(SingleEpisode(), val_ratio=0.1, seed=0, unit="episode")


if __name__ == "__main__":
    unittest.main()
