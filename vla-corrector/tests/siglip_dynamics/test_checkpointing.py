"""Exercise real trainer retention, early stopping, state recovery and inference."""

import dataclasses
import importlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor, _pick_checkpoint
from siglip_dynamics.checkpointing import CheckpointWriter, capture_rng_state
from siglip_dynamics.config import LossType, ModelScale, SiglipMLPConfig, TrainConfig
from siglip_dynamics.MLP import SiglipResidualMLP


def restore_cpu_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["generator"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["position"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.original_rng = capture_rng_state(0)
        torch.manual_seed(17)
        self.cfg = TrainConfig(
            device="cpu",
            model=SiglipMLPConfig(
                token_dim=4, action_dim=2, action_embed_dim=3, scale=ModelScale.CUSTOM, custom_widths=(8, 8)
            ),
            batch_size=2,
            num_workers=0,
            train_loss_type=LossType.COSINE,
            patience=5,
        )

    def tearDown(self):
        restore_cpu_rng(self.original_rng)
        torch.set_num_threads(self.threads)

    def trainer(self):
        stub = types.SimpleNamespace(
            init=lambda **kwargs: None,
            watch=lambda *args, **kwargs: None,
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
        )
        with patch.dict(sys.modules, {"wandb": stub}):
            module = importlib.import_module("siglip_dynamics.train")
        return module, stub

    def run_training(self, directory, losses, *, epochs):
        module, stub = self.trainer()
        data = directory / "tiny.pt"
        z = torch.randn(6, 2, 4)
        torch.save({"Z_t": z, "Z_next": z + torch.randn_like(z), "A_t": torch.randn(6, 2)}, data)
        cfg = dataclasses.replace(
            self.cfg,
            dataset_path=str(data),
            checkpoint_dir=str(directory / "run"),
            epochs=epochs,
            val_ratio=1 / 3,
        )
        original_loss = module._combined_loss
        observed = []

        def controlled_validation(pred, target, *args):
            # Training/optimizer steps remain real; control only the scalar
            # validation signal to exercise exact selection and stopping cases.
            result = original_loss(pred, target, *args)
            if torch.is_grad_enabled():
                return result
            value = losses[len(observed)]
            observed.append(value)
            cosine = torch.tensor(value, device=pred.device)
            return cosine, result[1], cosine

        with (
            patch.object(module, "wandb", stub),
            patch.object(module, "_combined_loss", controlled_validation),
            patch.dict("os.environ", {"WORLD_SIZE": "1"}),
            self.assertLogs("siglip_dynamics.train", level="INFO") as logs,
        ):
            module.train(cfg)
        return directory / "run", observed, logs.output

    def test_real_trainer_archives_15_20_and_best_21_then_stops_after_five_non_improvements(self):
        with tempfile.TemporaryDirectory() as tmp:
            # All epochs through 21 improve; then five epochs fail to improve.
            losses = [1 - epoch * 0.01 for epoch in range(1, 22)] + [0.80, 0.81, 0.82, 0.83, 0.84]
            folder, observed, logs = self.run_training(Path(tmp), losses, epochs=30)
            self.assertEqual(len(observed), 26)
            self.assertTrue(any("Early stop triggered at epoch 26" in line for line in logs))
            self.assertEqual(
                {p.name for p in folder.glob("*.pt")},
                {"checkpoint_epoch_15.pt", "checkpoint_epoch_20.pt", "best_cosine_model_epoch_21.pt"},
            )
            manifest = json.loads((folder / "checkpoint_manifest.json").read_text())
            self.assertEqual(manifest["current_best_epoch"], 21)
            self.assertEqual(_pick_checkpoint(folder).name, "best_cosine_model_epoch_21.pt")
            for epoch in [15, 20]:
                archive = torch.load(folder / f"checkpoint_epoch_{epoch}.pt", weights_only=True)
                self.assertEqual(archive["epoch"], epoch)
                self.assertEqual(archive["scheduler_state_dict"]["last_epoch"], epoch)
                self.assertTrue(archive["optimizer_state_dict"]["state"])
                self.assertEqual(archive["early_stopping_state"]["no_improve_epochs"], 0)
                self.assertEqual([r["rank"] for r in archive["rng_states"]], [0])
                self.assertEqual(archive["world_size"], 1)
            best = torch.load(folder / "best_cosine_model_epoch_21.pt", weights_only=True)
            self.assertNotIn("optimizer_state_dict", best)
            # Directory selection chooses current best; explicit selection must
            # honor the older training archive even when the best differs.
            latest = SiglipDynamicsPredictor(folder, device="cpu")
            older = SiglipDynamicsPredictor(folder / "checkpoint_epoch_15.pt", device="cpu")
            self.assertEqual(latest.checkpoint_path.name, "best_cosine_model_epoch_21.pt")
            self.assertEqual(older.checkpoint_path.name, "checkpoint_epoch_15.pt")
            for name, value in best["state_dict"].items():
                torch.testing.assert_close(latest.model.state_dict()[name], value, rtol=0, atol=0)

    def test_periodic_epoch_must_beat_nonperiodic_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            losses = [1 - epoch * 0.01 for epoch in range(1, 21)]
            losses[14] = losses[13] + 0.005  # 15 beats 10 but loses to 14: skip.
            folder, observed, logs = self.run_training(Path(tmp), losses, epochs=20)
            self.assertEqual(len(observed), 20)
            self.assertFalse(any("Early stop triggered" in line for line in logs))
            self.assertEqual(
                {p.name for p in folder.glob("*.pt")},
                {"checkpoint_epoch_20.pt", "best_cosine_model_epoch_20.pt"},
            )

    def test_training_archive_replays_next_adamw_step_and_rng_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = SiglipResidualMLP(self.cfg.model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

            def step(net, opt, sched):
                z, action = torch.randn(2, 2, 4), torch.randn(2, 2)
                loss = net(z, action).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

            for _ in range(15):
                step(model, optimizer, scheduler)
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_cfg": {},
                "train_cfg": {},
                "best_epoch": 15,
                "best_metric_value": 0.4,
            }
            writer = CheckpointWriter(Path(tmp))
            _, archive_path = writer.save_improvement(
                checkpoint,
                training_state={
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_states": [capture_rng_state(0)],
                },
            )
            expected_random = (random.random(), float(np.random.random()))
            step(model, optimizer, scheduler)
            expected_rng = torch.get_rng_state().clone()
            state = torch.load(archive_path, weights_only=True)
            restored = SiglipResidualMLP(self.cfg.model)
            opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
            restored.load_state_dict(state["state_dict"])
            opt.load_state_dict(state["optimizer_state_dict"])
            sched.load_state_dict(state["scheduler_state_dict"])
            restore_cpu_rng(state["rng_states"][0])
            self.assertEqual((random.random(), float(np.random.random())), expected_random)
            step(restored, opt, sched)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual(sched.state_dict(), scheduler.state_dict())

    def test_existing_weights_and_legacy_loader_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            old = folder / "best_cosine_model.pt"
            old.write_bytes(b"original checkpoint")
            self.assertEqual(_pick_checkpoint(folder), old)
            with self.assertRaisesRegex(FileExistsError, "fresh --checkpoint-dir"):
                CheckpointWriter(folder)
            self.assertEqual(old.read_bytes(), b"original checkpoint")

    def test_rlinf_loader_honors_training_archive_when_manifest_selects_newer_best(self):
        from siglip_dynamics.inference.rlinf_pi05 import load_corrector

        module, _ = self.trainer()
        cfg = dataclasses.replace(
            self.cfg,
            model=dataclasses.replace(self.cfg.model, token_dim=2048, action_dim=7),
        )
        model = SiglipResidualMLP(cfg.model)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "cache_metadata.json").write_text("{}")
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_cfg": module._to_serializable(dataclasses.asdict(cfg.model)),
                "train_cfg": module._to_serializable(dataclasses.asdict(cfg)),
                "best_epoch": 15,
                "best_metric_value": 0.4,
            }
            writer = CheckpointWriter(folder)
            _, archive = writer.save_improvement(
                checkpoint, training_state={"rng_states": [capture_rng_state(0)]}
            )
            with torch.no_grad():
                next(model.parameters()).add_(1)
            writer.save_improvement({**checkpoint, "best_epoch": 16, "best_metric_value": 0.3})
            # Normalization/SHA contracts have separate tests; exercise the real
            # architecture loader and explicit-vs-directory selection here.
            with patch("siglip_dynamics.inference.rlinf_pi05.validate_contract"):
                restored, k, info = load_corrector(archive, folder, device="cpu")
            self.assertEqual(k, 10)
            self.assertEqual(info["best_epoch"], 15)
            saved = torch.load(archive, weights_only=True)
            for name, value in saved["state_dict"].items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_failed_save_keeps_previous_best_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            writer = CheckpointWriter(folder)
            first, _ = writer.save_improvement({"best_epoch": 1, "best_metric_value": 0.5, "train_cfg": {}})
            with (
                patch("siglip_dynamics.checkpointing.torch.save", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                writer.save_improvement({"best_epoch": 2, "best_metric_value": 0.4, "train_cfg": {}})
            self.assertTrue(first.is_file())
            self.assertEqual(_pick_checkpoint(folder), first)
            self.assertFalse(list(folder.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
