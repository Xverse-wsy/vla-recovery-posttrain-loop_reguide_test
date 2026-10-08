"""Extract LIBERO frame caches through RLinf's actual PI0.5 input pipeline.

This entry point intentionally uses source checkouts, not LeRobot's PI policy.
Run in the isolated corrector-rlinf environment (see docs/rlinf_libero.md).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")
LOG = logging.getLogger(__name__)


def quantize_tokens(z):
    """Use the existing int8/per-token cache format, accounting for fp16 scales."""
    scale = np.maximum(np.max(np.abs(z), axis=-1) / 127, np.finfo(np.float16).tiny).astype(np.float16)
    q = np.clip(np.rint(z / scale.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return q, scale


def source_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def checkpoint_provenance(path):
    # This entry point supports the downloaded base checkpoint, not RL resume weights.
    if any(
        (path / name).exists()
        for name in (
            "full_weights.pt",
            "model_state_dict/full_weights.pt",
            "actor/model_state_dict/full_weights.pt",
        )
    ):
        raise ValueError("Use a base model.safetensors directory without RL resume weights")
    weights = path / "model.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(weights.resolve()), "bytes": weights.stat().st_size, "sha256": digest}


def load_policy(args):
    for path in (args.rlinf_src, args.openpi_src / "src", args.openpi_src / "packages/openpi-client/src"):
        sys.path.insert(0, str(path.resolve()))
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(args.policy_path.resolve()),
            "precision": None,
            "pi05": True,
            "action_dim": 7,
            "num_action_chunks": 10,
            "num_steps": 5,
            "add_value_head": False,
            "openpi_data": {"norm_stats_path": str(args.norm_stats.resolve())},
            "openpi": {
                "task": "eval",
                "config_name": "pi05_libero",
                "model_action_dim": 32,
                "action_env_dim": 7,
                "action_chunk": 10,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "torch_compile": False,
            },
        }
    )
    policy = get_model(cfg).to(args.device).eval().requires_grad_(False)
    if policy.action_horizon != 10:
        raise ValueError(f"Expected pi05_libero horizon 10, got {policy.action_horizon}")
    return policy


def make_batch(rows, tasks):
    from PIL import Image

    def decode(value):
        if not isinstance(value, dict) or not value.get("bytes"):
            raise ValueError("Expected embedded image bytes in physical-intelligence/libero parquet")
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return np.asarray(image.convert("RGB"))

    state = np.asarray([r["state"] for r in rows], dtype=np.float32)
    actions = np.asarray([r["actions"] for r in rows], dtype=np.float32)
    if state.shape != (len(rows), 8) or actions.shape != (len(rows), 7):
        raise ValueError(f"Unexpected state/actions shapes: {state.shape}/{actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state/actions")
    return {
        "observation/image": np.stack([decode(r["image"]) for r in rows]),
        "observation/wrist_image": np.stack([decode(r["wrist_image"]) for r in rows]),
        "observation/state": state,
        "actions": actions[:, None, :],
        "prompt": [tasks[int(r["task_index"])] for r in rows],
    }


@torch.no_grad()
def encode_batch(policy, batch):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

    processed = policy.input_transform(batch)
    actions = processed["actions"][:, 0, :7].float().numpy()
    observation = policy._observation_dict_to_device(processed)
    observation = preprocess_observation(observation, train=False)
    for name in CAMERAS:
        if not observation.image_masks[name].all():
            raise ValueError(f"Missing active camera: {name}")
    features = torch.cat([policy.img(observation.images[name])[0] for name in CAMERAS], dim=1)
    if features.shape[1:] != (512, 2048):
        raise ValueError(f"Unexpected visual features: {features.shape}")
    z = features.float().cpu().numpy()
    if not np.isfinite(z).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite features/normalized actions")
    return z, actions, processed, observation, features


@torch.no_grad()
def verify(policy, batch, processed, observation, features, stats_path):
    """Check the real policy prefix, output decoding, unclipped quantiles, and ODE path."""
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    q01, q99 = np.asarray(stats["q01"][:7]), np.asarray(stats["q99"][:7])
    expected = 2 * (batch["actions"] - q01) / (q99 - q01 + 1e-6) - 1
    actual = processed["actions"][..., :7].numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    decoded = policy.decode_actions(processed["actions"].to(policy.device), observation.state)
    np.testing.assert_allclose(decoded.cpu().numpy(), batch["actions"], atol=3e-6, rtol=2e-6)
    prefix, _, _ = policy.embed_prefix(observation)
    torch.testing.assert_close(prefix[:, :512].float(), features.float(), atol=0, rtol=0)
    # Probe gripper endpoints AND out-of-quantile translation; catches hidden clipping.
    probes = np.stack([q01 - 0.1 * (q99 - q01), q99 + 0.1 * (q99 - q01)]).astype(np.float32)
    probes[:, 6] = [-1, 1]
    probe_batch = {
        k: ([v[0]] * 2 if isinstance(v, list) else np.repeat(v[:1], 2, axis=0)) for k, v in batch.items()
    }
    probe_batch["actions"] = probes[:, None, :]
    probe_processed = policy.input_transform(probe_batch)
    probe_expected = 2 * (probes - q01) / (q99 - q01 + 1e-6) - 1
    np.testing.assert_allclose(probe_processed["actions"][:, 0, :7].numpy(), probe_expected, atol=2e-6)
    noise = torch.randn(
        (len(batch["actions"]), 10, 32),
        device=policy.device,
        generator=torch.Generator(device=policy.device).manual_seed(42),
    )
    env_actions, aux = policy._predict_eval(observation, noise=noise)
    torch.testing.assert_close(env_actions, policy.decode_actions(aux["model_actions"], observation.state))
    return {
        "action_normalization_max_error": float(np.max(np.abs(expected - actual))),
        "demonstration_roundtrip_max_error": float(np.max(np.abs(decoded.cpu().numpy() - batch["actions"]))),
        "visual_prefix_max_error": 0.0,
        "policy_output_decode_verified": True,
        "unclipped_probe_verified": True,
        "gripper_normalized_endpoints": probe_expected[:, 6].tolist(),
        "policy_model_action_min": aux["model_actions"][..., :7].amin((0, 1)).tolist(),
        "policy_model_action_max": aux["model_actions"][..., :7].amax((0, 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-path", "policy-path", "output-path", "rlinf-src", "openpi-src"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-episodes", type=int, default=0, help="0: all; otherwise first complete episodes"
    )
    parser.add_argument("--max-output-gb", type=float, default=500)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.batch_size < 1 or args.max_episodes < 0 or args.max_output_gb <= 0:
        parser.error("Invalid batch-size/max-episodes/max-output-gb")
    args.norm_stats = args.norm_stats or args.policy_path / "physical-intelligence/libero/norm_stats.json"
    if not args.norm_stats.is_file():
        raise FileNotFoundError(args.norm_stats)
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {args.output_path}")
    import pyarrow.parquet as pq

    files = sorted(args.dataset_path.glob("data/chunk-*/episode_*.parquet"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError("No episode parquet files found")
    counts = [pq.read_metadata(f).num_rows for f in files]
    n = sum(counts)
    estimated = n * (512 * 2048 + 512 * 2 + 7 * 4 + 2 * 4)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        estimated > args.max_output_gb * 1e9
        or estimated * 1.05 > shutil.disk_usage(args.output_path.parent).free
    ):
        raise RuntimeError(f"Insufficient space or cache budget: need about {estimated / 1e9:.2f} GB")
    tasks = {
        int(r["task_index"]): r["task"]
        for r in (
            json.loads(line) for line in (args.dataset_path / "meta/tasks.jsonl").read_text().splitlines()
        )
    }
    LOG.info("Extracting %d episodes/%d frames (~%.2f GB)", len(files), n, estimated / 1e9)
    checkpoint = checkpoint_provenance(args.policy_path)
    policy = load_policy(args)
    args.output_path.mkdir()
    arrays = {
        name: np.lib.format.open_memmap(
            args.output_path / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
        )
        for name, dtype, shape in [
            ("z_q", "int8", (n, 512, 2048)),
            ("z_scale", "float16", (n, 512)),
            ("actions", "float32", (n, 7)),
            ("episode_index", "int32", (n,)),
            ("frame_index", "int32", (n,)),
        ]
    }
    offset, episodes, report, max_error = 0, [], {}, 0.0
    amin, amax = np.full(7, np.inf), np.full(7, -np.inf)
    for file, count in zip(files, counts, strict=True):
        rows = pq.read_table(file).to_pylist()
        episode_ids = {int(r["episode_index"]) for r in rows}
        frames = np.asarray([int(r["frame_index"]) for r in rows])
        if len(episode_ids) != 1 or not np.array_equal(frames, np.arange(count)):
            raise ValueError(f"Invalid episode boundaries/frame order in {file}")
        ep = episode_ids.pop()
        if ep in episodes:
            raise ValueError(f"Repeated episode {ep}")
        episodes.append(ep)
        for start in range(0, count, args.batch_size):
            part = rows[start : start + args.batch_size]
            batch = make_batch(part, tasks)
            z, actions, processed, observation, features = encode_batch(policy, batch)
            if args.verify and not report:
                report = verify(policy, batch, processed, observation, features, args.norm_stats)
            q, scale = quantize_tokens(z)
            max_error = max(
                max_error,
                float(np.max(np.abs(q.astype(np.float32) * scale.astype(np.float32)[..., None] - z))),
            )
            sl = slice(offset, offset + len(part))
            arrays["z_q"][sl], arrays["z_scale"][sl], arrays["actions"][sl] = q, scale, actions
            arrays["episode_index"][sl] = ep
            arrays["frame_index"][sl] = frames[start : start + len(part)]
            amin, amax = np.minimum(amin, actions.min(0)), np.maximum(amax, actions.max(0))
            offset += len(part)
        LOG.info("Episode %d done; %d/%d frames", ep, offset, n)
    for array in arrays.values():
        array.flush()
    if offset != n:
        raise RuntimeError(f"Frame count changed: {offset} != {n}")
    stats = json.loads(args.norm_stats.read_text())["norm_stats"]["actions"]
    metadata = {
        "format": "siglip_frame_cache_v2",
        "dataset_format": "libero",
        "dataset_path": str(args.dataset_path.resolve()),
        "encoder_backend": "rlinf_pi05",
        "encoder_info": {
            "policy_path": str(args.policy_path.resolve()),
            "checkpoint": checkpoint,
            "rlinf_revision": source_revision(args.rlinf_src),
            "openpi_revision": source_revision(args.openpi_src),
            "feature_stage": "projected_siglip_before_language_model",
            "cameras": list(CAMERAS),
            "precision": "rlinf_selective_bfloat16",
        },
        "num_frames": n,
        "l_tokens": 512,
        "d_dim": 2048,
        "action_dim": 7,
        "dtype_z_q": "int8",
        "dtype_z_scale": "float16",
        "dtype_actions": "float32",
        "image_key": "image",
        "wrist_image_key": "wrist_image",
        "use_wrist_image": True,
        "action_key": "actions",
        "action_representation": "rlinf_normalized_environment_action",
        "action_normalization": {
            "type": "checkpoint_quantiles",
            "clip": False,
            "epsilon": 1e-6,
            "stats_path": str(args.norm_stats.resolve()),
            "stats_sha256": hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
            "q01": stats["q01"][:7],
            "q99": stats["q99"][:7],
            "gripper_index": 6,
        },
        "episode_ids": episodes,
        "complete_episodes": True,
        "action_min": amin.tolist(),
        "action_max": amax.tolist(),
        "quantization_max_absolute_error": max_error,
        "verification": report,
    }
    # Publishing metadata last prevents the training reader from accepting partial caches.
    (args.output_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    LOG.info("Cache ready: %s; verification=%s", args.output_path, report)


if __name__ == "__main__":
    main()
