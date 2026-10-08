"""CPU checks for timestep alignment, local gradients and original threshold behavior."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from siglip_dynamics.inference.rlinf_pi05 import (
    CorrectorMonitor,
    GuidanceTarget,
    cache_feature_roundtrip,
    guide_velocity,
    predict_guided,
    sha256_file,
    validate_contract,
)
from siglip_dynamics.inference.robust_threshold import RobustThresholdState, robust_threshold_step


class ToyCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 4)
        self.requires_grad_(False)
        self.last_action = None

    def forward(self, z, action):
        self.last_action = action.detach().clone()
        return self.projection(action)[:, None, :].expand_as(z) + z * 0.1


class ToyPolicy(nn.Module):
    def run_suffix(self, observation, x_t, time, cache=None, mask=None):
        return x_t * 0.1 + 0.2

    def velocity_from_suffix(self, suffix):
        return suffix

    @torch.no_grad()
    def predict_action_batch(self, observation, mode="eval", noise=None):
        x = noise.clone() if noise is not None else torch.randn(1, 10, 32)
        for time in [1.0, 2 / 3, 1 / 3]:
            suffix = self.run_suffix(observation, x, torch.tensor([time]))
            x = x - self.velocity_from_suffix(suffix) / 3
            # Mimic a native sampler that consumes randomness even with zero variance.
            x = x + torch.randn_like(x) * 0
        return x[:, :, :7], {"model_actions": x}


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.predictor = ToyCorrector()
        self.z = torch.randn(1, 2, 4)

    def test_alignment_and_reset(self):
        monitor = CorrectorMonitor(self.predictor, k_step=2)
        model_action = torch.zeros(32)
        model_action[6] = 1.0003991
        event, _ = monitor.observe(0, self.z)
        self.assertIsNone(event)
        monitor.queue_action(0, self.z, model_action)
        self.assertGreater(self.predictor.last_action[0, 6].item(), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            monitor.queue_action(0, self.z, model_action)
        self.assertIsNone(monitor.observe(1, self.z + 1)[0])
        monitor.queue_action(1, self.z + 1, model_action)
        event, target = monitor.observe(2, self.z + 2)
        self.assertEqual(event["origin_step"], 0)
        self.assertAlmostEqual(event["real_norm"], float(torch.full_like(self.z, 2).norm()))
        self.assertIsNone(target)  # original bootstrap threshold has no early triggers
        self.assertEqual(len(monitor.pending), 1)
        monitor.reset()
        self.assertEqual(len(monitor.pending), 0)
        self.assertEqual(monitor.last_step, -1)
        self.assertFalse(monitor.threshold.bootstrapped)

    def test_online_features_match_training_cache_roundtrip(self):
        from siglip_dynamics.extract_rlinf import quantize_tokens

        z = torch.randn(3, 5, 17) * 30
        z[0, 0] = 0
        q, scales = quantize_tokens(z.numpy())
        expected = torch.from_numpy(q).float() * torch.from_numpy(scales).float()[..., None]
        self.assertTrue(torch.equal(cache_feature_roundtrip(z), expected))

    def test_skipped_timestep_is_rejected(self):
        monitor = CorrectorMonitor(self.predictor, k_step=2)
        monitor.observe(0, self.z)
        monitor.queue_action(0, self.z, torch.zeros(7))
        with self.assertRaisesRegex(ValueError, "Skipped"):
            monitor.observe(3, self.z)

    def test_local_gradient_only_first_action_seven_dims_and_loss_decreases(self):
        x = torch.randn(1, 10, 32)
        x[0, 0, 6] = 1.8
        v = torch.zeros_like(x)
        target = GuidanceTarget(self.z, torch.randn_like(self.z), torch.randn_like(self.z))
        with torch.no_grad():
            original_loss = (
                1
                - torch.nn.functional.cosine_similarity(
                    self.predictor(self.z, x[:, 0, :7]).flatten(1), target.correction.flatten(1)
                ).mean()
            )
            guided, stats = guide_velocity(self.predictor, x, v, 1.0, target, 0.01)
        self.assertGreater(stats["grad_norm"], 0)
        self.assertGreater(self.predictor.last_action[0, 6].item(), 1)
        self.assertTrue(torch.equal(guided[:, 1:], v[:, 1:]))
        self.assertTrue(torch.equal(guided[:, 0, 7:], v[:, 0, 7:]))
        self.assertTrue(all(parameter.grad is None for parameter in self.predictor.parameters()))
        with torch.no_grad():
            new_loss = (
                1
                - torch.nn.functional.cosine_similarity(
                    self.predictor(self.z, (x - guided)[:, 0, :7]).flatten(1), target.correction.flatten(1)
                ).mean()
            )
        self.assertLess(float(new_loss), float(original_loss))

    def test_zero_guidance_calls_native_sampler(self):
        class Policy:
            def predict_action_batch(self, observation, **kwargs):
                self.observation = observation
                self.kwargs = kwargs
                return "native actions", {"model_actions": "native normalized actions"}

        policy = Policy()
        observation = object()
        noise = torch.zeros(1, 10, 32)
        actions, aux, stats = predict_guided(policy, observation, self.predictor, None, eta=0, noise=noise)
        self.assertEqual(actions, "native actions")
        self.assertEqual(aux["model_actions"], "native normalized actions")
        self.assertIs(policy.observation, observation)
        self.assertIs(policy.kwargs["noise"], noise)
        self.assertEqual(stats, [])

    def test_native_sampler_rng_and_hooks_restored(self):
        policy = ToyPolicy()
        target = GuidanceTarget(self.z, torch.randn_like(self.z), torch.randn_like(self.z))
        torch.manual_seed(42)
        native, _ = policy.predict_action_batch({})
        native_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        guided, aux, stats = predict_guided(policy, {}, self.predictor, target, eta=0.01)
        self.assertTrue(torch.equal(native_rng, torch.get_rng_state()))
        self.assertEqual(len(stats), 3)
        self.assertEqual(aux["model_actions"].shape, (1, 10, 32))
        self.assertFalse(torch.equal(native, guided))
        self.assertNotIn("run_suffix", policy.__dict__)
        self.assertNotIn("velocity_from_suffix", policy.__dict__)
        torch.manual_seed(42)
        restored, _ = policy.predict_action_batch({})
        self.assertTrue(torch.equal(native, restored))

    def test_hooks_restored_after_gradient_error(self):
        policy = ToyPolicy()
        target = GuidanceTarget(torch.zeros(1, 2, 5), torch.zeros(1, 2, 5), torch.ones(1, 2, 5))
        with self.assertRaises(RuntimeError):
            predict_guided(policy, {}, self.predictor, target)
        self.assertNotIn("run_suffix", policy.__dict__)
        self.assertNotIn("velocity_from_suffix", policy.__dict__)

    def test_original_threshold_bootstrap_jump_and_reset(self):
        state = RobustThresholdState(bootstrap_queue_size=3, reset_consecutive_steps=2)
        for _ in range(3):
            _, triggered = robust_threshold_step(0.1, state)
            self.assertFalse(triggered)
        self.assertTrue(state.bootstrapped)
        threshold, triggered = robust_threshold_step(1.0, state)
        self.assertEqual(threshold, 0.25)
        self.assertTrue(triggered)
        self.assertTrue(state.alert_active)
        for _ in range(12):
            robust_threshold_step(0.0, state)
        self.assertFalse(state.alert_active)

    def test_contract_rejects_stats_checkpoint_camera_and_clipping_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)
            stats = policy / "physical-intelligence/libero/norm_stats.json"
            stats.parent.mkdir(parents=True)
            stats.write_text("{}")
            checkpoint = policy / "model.safetensors"
            checkpoint.write_bytes(b"test weights")
            metadata = {
                "encoder_backend": "rlinf_pi05",
                "l_tokens": 512,
                "d_dim": 2048,
                "action_dim": 7,
                "action_representation": "rlinf_normalized_environment_action",
                "dtype_z_q": "int8",
                "dtype_z_scale": "float16",
                "encoder_info": {
                    "feature_stage": "projected_siglip_before_language_model",
                    "cameras": ["base_0_rgb", "left_wrist_0_rgb"],
                    "checkpoint": {"sha256": sha256_file(checkpoint)},
                },
                "action_normalization": {
                    "type": "checkpoint_quantiles",
                    "clip": False,
                    "gripper_index": 6,
                    "epsilon": 1e-6,
                    "stats_sha256": sha256_file(stats),
                },
            }
            validate_contract(metadata, policy)
            changed = json.loads(json.dumps(metadata))
            changed["encoder_info"]["cameras"].reverse()
            with self.assertRaisesRegex(ValueError, "camera"):
                validate_contract(changed, policy)
            changed = json.loads(json.dumps(metadata))
            changed["action_normalization"]["clip"] = True
            with self.assertRaisesRegex(ValueError, "normalization"):
                validate_contract(changed, policy)
            checkpoint.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "checkpoints differ"):
                validate_contract(metadata, policy)
            checkpoint.write_bytes(b"test weights")
            stats.write_text("changed")
            with self.assertRaisesRegex(ValueError, "statistics differ"):
                validate_contract(metadata, policy)


if __name__ == "__main__":
    unittest.main()
