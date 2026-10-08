"""Exercise the full CLI rollout/queue logic with a deterministic CPU simulator."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class CPUPolicy(nn.Module):
    action_horizon = 10

    def to(self, *args, **kwargs):
        return self

    def run_suffix(self, observation, x_t, time, cache=None, mask=None):
        return x_t * 0.1 + 0.2

    def velocity_from_suffix(self, suffix):
        return suffix

    @torch.no_grad()
    def predict_action_batch(self, observation, mode="eval", noise=None):
        x = torch.randn(1, 10, 32) if noise is None else noise.clone()
        for time in [1.0, 2 / 3, 1 / 3]:
            x = x - self.velocity_from_suffix(self.run_suffix(observation, x, torch.tensor([time]))) / 3
        return x[:, :, :7], {"model_actions": x}


class CPUCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 4)
        self.requires_grad_(False)

    def forward(self, z, action):
        return self.projection(action)[:, None].expand_as(z) + z * 0.1


class CPUEnv:
    def __init__(self, cfg, **kwargs):
        self.task_suite = types.SimpleNamespace(
            get_num_tasks=lambda: 10,
            get_task=lambda task_id: types.SimpleNamespace(name=f"task_{task_id}"),
        )
        self.trial_id_bins = np.full(10, 50)
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)
        self.env = types.SimpleNamespace(close=lambda: None)

    def reset(self, reset_state_ids):
        reset_id = int(reset_state_ids[0])
        self.task_ids = [reset_id // 50]
        self.trial_ids = [reset_id % 50]
        self.step_count = 0
        self.z = torch.arange(8).reshape(1, 2, 4).float() / 10
        return {"z": self.z.clone()}, {}

    def step(self, action, **kwargs):
        self.step_count += 1
        self.z = self.z + 0.01 * action[:, None, :4]
        return {"z": self.z.clone()}, 0, torch.tensor([False]), torch.tensor([self.step_count >= 40]), {}


class EvalModeTests(unittest.TestCase):
    def test_full_loop_monitor_preserves_actions_and_ogg_discards_only_unexecuted_actions(self):
        script = Path(__file__).resolve().parents[2] / "scripts/eval_rlinf_libero.py"
        spec = importlib.util.spec_from_file_location("corrector_eval_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def config(_):
            return types.SimpleNamespace(
                max_episode_steps=40, video_cfg=types.SimpleNamespace(save_video=False)
            )

        modules = {
            "omegaconf": types.SimpleNamespace(
                OmegaConf=types.SimpleNamespace(create=lambda value: value, load=config)
            ),
            "rlinf.envs.sim.libero.libero_env": types.SimpleNamespace(LiberoEnv=CPUEnv),
            "rlinf.models.embodiment.openpi": types.SimpleNamespace(get_model=lambda _: CPUPolicy()),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, modules),
            patch.object(module, "load_corrector", side_effect=lambda *args: (CPUCorrector(), 10, {})),
            patch.object(module, "encode_features", side_effect=lambda policy, obs: obs["z"]),
            patch.object(torch.cuda, "reset_peak_memory_stats"),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            patch.object(torch.cuda, "max_memory_reserved", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            records = {}
            for mode in ["off", "monitor", "ogg"]:
                output = Path(directory) / mode
                argv = [
                    str(script),
                    "--project-root",
                    directory,
                    "--suites",
                    "libero_spatial",
                    "--task-ids",
                    "0",
                    "--trials-per-task",
                    "1",
                    "--action-chunk",
                    "10",
                    "--seed",
                    "42",
                    "--corrector-mode",
                    mode,
                    "--corrector-checkpoint",
                    "unused.pt",
                    "--output-dir",
                    str(output),
                    "--threshold-bootstrap-queue-size",
                    "2",
                    "--threshold-k-on",
                    "0",
                    "--threshold-k-off",
                    "0",
                    "--threshold-ewma-alpha",
                    "1",
                    "--threshold-trigger-consecutive-steps",
                    "1",
                    "--threshold-trigger-margin",
                    "0",
                    "--threshold-jump-trigger",
                    "0",
                    "--meltdown-cooldown-steps",
                    "1",
                ]
                with patch.object(sys, "argv", argv):
                    module.main()
                records[mode] = json.loads((output / "episodes.jsonl").read_text())
                self.assertTrue(json.loads((output / "summary.json").read_text())["complete"])
            off, monitor, ogg = [records[mode] for mode in ["off", "monitor", "ogg"]]
            self.assertEqual(off["executed_actions_sha256"], monitor["executed_actions_sha256"])
            self.assertEqual(off["queries"], monitor["queries"])
            self.assertEqual(monitor["corrector_comparisons"], 30)
            self.assertEqual(monitor["discarded_actions"], 0)
            self.assertEqual(monitor["guidance_requests"], 0)
            self.assertGreater(ogg["guidance_requests"], 0)
            self.assertGreater(ogg["discarded_actions"], 0)
            self.assertGreater(ogg["effective_guidance_requests"], 0)
            self.assertEqual(ogg["guidance_applied_steps"], 3 * ogg["guidance_requests"])
            self.assertNotEqual(off["executed_actions_sha256"], ogg["executed_actions_sha256"])
            events = [
                json.loads(line)
                for line in (Path(directory) / "ogg/corrector_events.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(row["step"] - row["origin_step"] == 10 for row in events))


if __name__ == "__main__":
    unittest.main()
