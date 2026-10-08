"""Frozen RLinf PI0.5 features, aligned corrector monitoring and local OGG gradients."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from .robust_threshold import RobustThresholdState, robust_threshold_step

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(metadata: dict, policy_path: Path) -> None:
    """Reject incompatible features, action statistics or base policy weights."""
    info = metadata.get("encoder_info", {})
    norm = metadata.get("action_normalization", {})
    expected = {
        "encoder_backend": "rlinf_pi05",
        "l_tokens": 512,
        "d_dim": 2048,
        "action_dim": 7,
        "action_representation": "rlinf_normalized_environment_action",
        "dtype_z_q": "int8",
        "dtype_z_scale": "float16",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Corrector cache contract mismatch: {key}")
    if info.get("feature_stage") != "projected_siglip_before_language_model" or info.get("cameras") != list(
        CAMERAS
    ):
        raise ValueError("Corrector feature stage/camera order mismatch")
    if (
        norm.get("type") != "checkpoint_quantiles"
        or norm.get("clip") is not False
        or norm.get("gripper_index") != 6
        or norm.get("epsilon") != 1e-6
    ):
        raise ValueError("Corrector action normalization mismatch")
    if any(
        (policy_path / name).exists()
        for name in (
            "full_weights.pt",
            "model_state_dict/full_weights.pt",
            "actor/model_state_dict/full_weights.pt",
        )
    ):
        raise ValueError("Corrector requires the recorded base checkpoint, not RL resume weights")
    stats = policy_path / "physical-intelligence/libero/norm_stats.json"
    if sha256_file(stats) != norm.get("stats_sha256"):
        raise ValueError("Corrector and policy normalization statistics differ")
    checkpoint = policy_path / "model.safetensors"
    if sha256_file(checkpoint) != info.get("checkpoint", {}).get("sha256"):
        raise ValueError("Corrector and policy checkpoints differ")


def load_corrector(checkpoint_path: Path, policy_path: Path, device="cuda:0"):
    """Use the repository's original strict checkpoint loader and cached contract."""
    from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or checkpoint_path.suffix != ".pt":
        raise ValueError("--corrector-checkpoint must name a .pt file")
    metadata = json.loads((checkpoint_path.parent / "cache_metadata.json").read_text())
    validate_contract(metadata, policy_path)
    predictor = SiglipDynamicsPredictor(str(checkpoint_path), device=str(device))
    # Validate the explicit checkpoint against its directory's architecture.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_cfg") != dataclass_model_config(predictor):
        raise ValueError("Checkpoint architecture differs from its directory configuration")
    predictor.model.load_state_dict(checkpoint["state_dict"], strict=True)
    if predictor.model_type != "mlp" or predictor.h_window != 1 or predictor.k_step != 10:
        raise ValueError("This RLinf adapter requires an h=1, k=10 MLP")
    if predictor.token_dim != 2048 or predictor.action_dim != 7:
        raise ValueError("Corrector dimensions must be token_dim=2048, action_dim=7")
    predictor.model.float().eval().requires_grad_(False)
    return (
        predictor.model,
        predictor.k_step,
        {
            "checkpoint": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_cosine_loss": checkpoint.get("best_metric_value"),
            "cache_contract_verified": True,
            "feature_quantization": "same per-token int8/fp16-scale roundtrip as training cache",
        },
    )


def dataclass_model_config(predictor):
    """Compare only architecture fields understood by the original MLP loader."""
    # Training includes two unused transformer fields; retain them for equality.
    import dataclasses

    cfg = dataclasses.asdict(predictor.model_cfg)
    cfg["custom_widths"] = list(cfg["custom_widths"])
    cfg.update(rope_theta=predictor.rope_theta, ada_rmsnorm_eps=predictor.ada_rmsnorm_eps)
    return cfg


@torch.no_grad()
def cache_feature_roundtrip(z):
    """Match quantize_tokens + dataset dequantization on-device, without touching actions."""
    scales = (z.abs().amax(dim=-1) / 127).clamp_min(torch.finfo(torch.float16).tiny).half()
    q = (z / scales.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scales.float()[..., None]


@torch.no_grad()
def encode_features(policy, env_obs, *, quantize=True):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

    observation = preprocess_observation(policy.env_obs_to_observation(env_obs), train=False)
    for name in CAMERAS:
        if not observation.image_masks[name].all():
            raise ValueError(f"Missing active camera: {name}")
    z = torch.cat([policy.img(observation.images[name])[0] for name in CAMERAS], dim=1).float()
    if z.shape != (1, 512, 2048) or not torch.isfinite(z).all():
        raise ValueError(f"Invalid online SigLIP features: {tuple(z.shape)}")
    return cache_feature_roundtrip(z) if quantize else z


@dataclass
class GuidanceTarget:
    z_current: torch.Tensor
    delta_real: torch.Tensor
    delta_pred: torch.Tensor

    @property
    def correction(self):
        return (self.delta_pred - self.delta_real).detach()


class CorrectorMonitor:
    """Queue predictions only for executed actions and compare exactly k steps later."""

    def __init__(self, predictor, k_step=10, threshold_config=None):
        self.predictor = predictor
        self.k_step = k_step
        self.threshold_config = threshold_config or {}
        self.reset()

    def reset(self):
        self.pending = deque()
        self.threshold = RobustThresholdState(**self.threshold_config)
        self.last_step = -1
        self.errors = []
        self.trigger_count = 0

    @torch.no_grad()
    def observe(self, step, z):
        if step <= self.last_step:
            raise ValueError("Monitor observations must advance by environment step")
        self.last_step = step
        if not self.pending:
            return None, None
        due, origin, pred = self.pending[0]
        if due < step:
            raise ValueError("Skipped a corrector target timestep")
        if due != step:
            return None, None
        self.pending.popleft()
        real = z - origin
        error = float((1 - F.cosine_similarity(pred.flatten()[None], real.flatten()[None], dim=1)).item())
        threshold, triggered = robust_threshold_step(error, self.threshold)
        if not torch.isfinite(real).all() or not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite corrector feature change")
        self.errors.append(error)
        self.trigger_count += int(triggered)
        event = {
            "step": step,
            "origin_step": step - self.k_step,
            "error": error,
            "threshold": threshold,
            "triggered": triggered,
            "real_norm": float(real.norm()),
            "pred_norm": float(pred.norm()),
        }
        target = GuidanceTarget(z.detach(), real.detach(), pred.detach()) if triggered else None
        return event, target

    @torch.no_grad()
    def queue_action(self, step, z, model_action):
        if step != self.last_step:
            raise ValueError("Observe before queuing the executed action")
        if self.pending and self.pending[-1][0] >= step + self.k_step:
            raise ValueError("Duplicate action prediction for a timestep")
        action = model_action.reshape(1, -1)[:, :7].float()
        if action.shape != (1, 7) or not torch.isfinite(action).all():
            raise ValueError("Corrector needs seven finite normalized action dimensions")
        pred = self.predictor(z, action)
        if pred.shape != z.shape or not torch.isfinite(pred).all():
            raise RuntimeError("Invalid corrector prediction")
        self.pending.append((step + self.k_step, z.detach().clone(), pred.detach()))


def guide_velocity(predictor, x_t, v_t, time, target, eta):
    """Original attract_delta_z_correction objective; gradients stop at detached v_t."""
    if eta == 0:
        return v_t, {"applied": False, "grad_norm": 0.0, "delta_v_norm": 0.0}
    with torch.enable_grad():
        velocity = v_t.detach().float().requires_grad_(True)
        x0_hat = x_t.detach().float() - time * velocity
        prediction = predictor(target.z_current.detach().float(), x0_hat[:, 0, :7])
        correction = target.correction.float()
        loss = (1 - F.cosine_similarity(prediction.flatten(1), correction.flatten(1), dim=-1)).mean()
        (gradient,) = torch.autograd.grad(loss, velocity)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite OGG loss/gradient")
        delta_v = eta * gradient
        guided = (velocity - delta_v).detach()
    return guided, {
        "applied": True,
        "loss": float(loss.detach()),
        "grad_norm": float(gradient.norm()),
        "delta_v_norm": float(delta_v.norm()),
        "correction_norm": float(correction.norm()),
    }


@torch.no_grad()
def predict_guided(policy, env_obs, predictor, target, eta=1.0, apply_every=1, noise=None):
    """Temporarily guide native velocities; preserve sampling RNG, time grid and decoding."""
    if eta == 0:
        actions, aux = policy.predict_action_batch(env_obs, mode="eval", noise=noise)
        return actions, aux, []
    if target is None or apply_every < 1:
        raise ValueError("OGG requires a target and a positive application interval")
    run_suffix = policy.run_suffix
    velocity_from_suffix = policy.velocity_from_suffix
    signature = inspect.signature(run_suffix)
    names = list(signature.parameters)
    if len(names) < 3:
        raise TypeError("Unsupported RLinf run_suffix signature")
    context = {}
    stats = []
    index = 0

    def wrapped_suffix(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        context["x_t"] = bound.arguments[names[1]]
        context["time"] = bound.arguments[names[2]]
        return run_suffix(*args, **kwargs)

    def wrapped_velocity(*args, **kwargs):
        nonlocal index
        velocity = velocity_from_suffix(*args, **kwargs)
        if not context:
            raise RuntimeError("Velocity computed without a matching RLinf suffix")
        if index % apply_every == 0:
            time = float(context["time"].reshape(-1)[0].item())
            velocity, detail = guide_velocity(predictor, context["x_t"], velocity, time, target, eta)
            detail.update(flow_step=index, time=time)
            stats.append(detail)
        index += 1
        return velocity

    # Single-threaded evaluation only. Restore descriptors, even after an exception.
    old_suffix = policy.__dict__.get("run_suffix")
    old_velocity = policy.__dict__.get("velocity_from_suffix")
    policy.run_suffix = wrapped_suffix
    policy.velocity_from_suffix = wrapped_velocity
    try:
        actions, aux = policy.predict_action_batch(env_obs, mode="eval", noise=noise)
        if not stats:
            raise RuntimeError("Native RLinf sampler bypassed the OGG velocity hooks")
        if not torch.isfinite(actions).all():
            raise RuntimeError("OGG returned non-finite environment actions")
        return actions, aux, stats
    finally:
        if old_suffix is None:
            del policy.run_suffix
        else:
            policy.run_suffix = old_suffix
        if old_velocity is None:
            del policy.velocity_from_suffix
        else:
            policy.velocity_from_suffix = old_velocity
