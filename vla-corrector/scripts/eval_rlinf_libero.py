"""Evaluate the RLinf PI0.5 checkpoint over explicit LIBERO task/trial pairs."""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from siglip_dynamics.inference.rlinf_pi05 import (
    CorrectorMonitor,
    encode_features,
    load_corrector,
    predict_guided,
)
from siglip_dynamics.inference.robust_threshold import RobustThresholdState


def summarize(records, settings, expected):
    suites = {}
    for suite in settings["suites"]:
        subset = [row for row in records if row["suite"] == suite]
        tasks = {}
        for task_id in sorted({row["task_id"] for row in subset}):
            trials = [row for row in subset if row["task_id"] == task_id]
            successes = sum(row["success"] for row in trials)
            tasks[str(task_id)] = {
                "task_name": trials[0]["task_name"],
                "episodes": len(trials),
                "successes": successes,
                "success_rate": successes / len(trials),
            }
        successes = sum(row["success"] for row in subset)
        suites[suite] = {
            "episodes": len(subset),
            "successes": successes,
            "success_rate": successes / len(subset) if subset else None,
            "tasks": tasks,
        }
    return {
        "settings": settings,
        "complete": len(records) == expected,
        "expected_episodes": expected,
        "completed_episodes": len(records),
        "diagnostic_only": bool(settings.get("max_steps_override", 0)),
        "overall_success_rate": (sum(row["success"] for row in records) / len(records) if records else None),
        "corrector_totals": {
            key: sum(row.get(key, 0) for row in records)
            for key in [
                "corrector_comparisons",
                "trigger_count",
                "guidance_requests",
                "guidance_applied_steps",
                "effective_guidance_requests",
                "discarded_actions",
            ]
        },
        "suites": suites,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["libero_spatial", "libero_object", "libero_goal"],
        default=["libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num-steps", type=int, default=3, help="Flow denoising steps")
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corrector-mode", choices=["off", "monitor", "ogg"], default="off")
    parser.add_argument("--corrector-checkpoint", type=Path)
    parser.add_argument("--guidance-eta", type=float, default=1.0)
    parser.add_argument("--guidance-apply-every", type=int, default=1)
    parser.add_argument("--meltdown-cooldown-steps", type=int, default=10)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0: suite limit; positive: diagnostic short rollout"
    )
    threshold_defaults = RobustThresholdState()
    threshold_fields = []
    for field in dataclasses.fields(threshold_defaults):
        value = getattr(threshold_defaults, field.name)
        if field.name != "cooldown_steps" and isinstance(value, (int, float)) and not isinstance(value, bool):
            # Counters are runtime state, not configurable algorithm parameters.
            if field.name in {"over_count", "under_count", "hard_over_count", "cooldown_remaining"}:
                continue
            threshold_fields.append(field.name)
            parser.add_argument(
                "--threshold-" + field.name.replace("_", "-"), type=type(value), default=value
            )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials_per_task <= 50:
        parser.error("--trials-per-task must be between 1 and 50")
    if not 1 <= args.action_chunk <= 10 or args.num_steps < 1:
        parser.error("action chunk must be 1..10; num steps must be positive")
    if any(task_id not in range(10) for task_id in args.task_ids):
        parser.error("task ids must be 0..9")
    if args.corrector_mode != "off" and args.corrector_checkpoint is None:
        parser.error("monitor/ogg requires --corrector-checkpoint")
    if not np.isfinite(args.guidance_eta) or args.guidance_eta < 0 or args.guidance_apply_every < 1:
        parser.error("guidance eta must be finite and nonnegative; apply every must be positive")
    if args.meltdown_cooldown_steps < 0 or args.max_steps < 0:
        parser.error("cooldown and max steps must be nonnegative")
    for name in threshold_fields:
        value = getattr(args, "threshold_" + name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"threshold {name} must be finite and nonnegative")
        if name.endswith(("size", "steps")) and value < 1:
            parser.error(f"threshold {name} must be positive")
    if not 0 < args.threshold_ewma_alpha <= 1:
        parser.error("threshold ewma alpha must be in (0,1]")
    if len(set(args.suites)) != len(args.suites) or len(set(args.task_ids)) != len(args.task_ids):
        parser.error("suites and task ids must not contain duplicates")
    root = args.project_root.resolve()
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.models.embodiment.openpi import get_model

    policy_dir = root / "models/RLinf-Pi05-LIBERO-SFT"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model_cfg = OmegaConf.create(
        {
            "model_path": str(policy_dir),
            "precision": None,
            "pi05": True,
            "action_dim": 7,
            "num_action_chunks": args.action_chunk,
            "num_steps": args.num_steps,
            "add_value_head": False,
            "openpi_data": {
                "norm_stats_path": str(policy_dir / "physical-intelligence/libero/norm_stats.json")
            },
            "openpi": {
                "task": "eval",
                "config_name": "pi05_libero",
                "model_action_dim": 32,
                "action_env_dim": 7,
                "action_chunk": args.action_chunk,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "torch_compile": False,
            },
        }
    )
    print("Loading PI0.5 checkpoint", str(policy_dir), flush=True)
    policy = get_model(model_cfg).to("cuda:0").eval().requires_grad_(False)
    assert policy.action_horizon == 10
    threshold_config = {name: getattr(args, "threshold_" + name) for name in threshold_fields}
    threshold_config["cooldown_steps"] = args.meltdown_cooldown_steps
    monitor, corrector_info = None, None
    if args.corrector_mode != "off":
        print("Verifying corrector contract and loading checkpoint", flush=True)
        predictor, k_step, corrector_info = load_corrector(args.corrector_checkpoint, policy_dir)
        monitor = CorrectorMonitor(predictor, k_step, threshold_config)
    settings = {
        "checkpoint": str(policy_dir),
        "suites": args.suites,
        "task_ids": args.task_ids,
        "trials_per_task": args.trials_per_task,
        "seed": args.seed,
        "model_action_horizon": 10,
        "executed_chunk": args.action_chunk,
        "flow_steps": args.num_steps,
        "corrector_enabled": args.corrector_mode != "off",
        "corrector_mode": args.corrector_mode,
        "corrector": corrector_info,
        "guidance_eta": args.guidance_eta,
        "guidance_apply_every": args.guidance_apply_every,
        "threshold_config": threshold_config,
        "max_steps_override": args.max_steps,
        "protocol": "RLinf LiberoEnv (suite step limits, 15 settling steps)",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "libero_config": os.environ.get("LIBERO_CONFIG_PATH"),
    }
    expected = len(args.suites) * len(args.task_ids) * args.trials_per_task
    records = []

    def write_summary():
        temporary = args.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summarize(records, settings, expected), indent=2) + "\n")
        temporary.replace(args.output_dir / "summary.json")

    write_summary()
    with (
        (args.output_dir / "episodes.jsonl").open("x", buffering=1) as output,
        (args.output_dir / "corrector_events.jsonl").open("x", buffering=1) as event_output,
    ):
        for suite_index, suite in enumerate(args.suites):
            cfg = OmegaConf.load(root / "RLinf/examples/embodiment/config/env" / f"{suite}.yaml")
            cfg.total_num_envs = 1
            cfg.auto_reset = False
            cfg.ignore_terminations = False
            cfg.is_eval = True
            cfg.seed = args.seed
            cfg.specific_reset_id = None
            cfg.use_fixed_reset_state_ids = True
            cfg.video_cfg.save_video = False
            env = LiberoEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
            try:
                assert env.task_suite.get_num_tasks() == 10
                if any(int(env.trial_id_bins[task_id]) < args.trials_per_task for task_id in args.task_ids):
                    raise ValueError("Not enough initial states for requested trials")
                # Disable the first-reset override so every requested id is explicit.
                env.is_start = False
                for task_id in args.task_ids:
                    offset = int(env.cumsum_trial_id_bins[task_id - 1]) if task_id else 0
                    task = env.task_suite.get_task(task_id)
                    for trial_id in range(args.trials_per_task):
                        reset_id = offset + trial_id
                        episode_seed = args.seed + suite_index * 1000 + reset_id
                        torch.manual_seed(episode_seed)
                        observation, _info = env.reset(reset_state_ids=np.array([reset_id], dtype=np.int64))
                        assert int(env.task_ids[0]) == task_id and int(env.trial_ids[0]) == trial_id
                        steps, queries, success, truncated = 0, 0, False, False
                        action_plan = deque()
                        if monitor is not None:
                            monitor.reset()
                        guidance_requests, guidance_steps, effective_guidance, discarded_actions = 0, 0, 0, 0
                        monitor_seconds, policy_seconds = 0.0, 0.0
                        action_digest = hashlib.sha256()
                        torch.cuda.reset_peak_memory_stats()
                        max_steps = (
                            min(int(cfg.max_episode_steps), args.max_steps)
                            if args.max_steps
                            else int(cfg.max_episode_steps)
                        )
                        started = time.monotonic()
                        while steps < max_steps and not (success or truncated):
                            target, z = None, None
                            event = None
                            if monitor is not None:
                                stage_start = time.monotonic()
                                z = encode_features(policy, observation)
                                event, target = monitor.observe(steps, z)
                                monitor_seconds += time.monotonic() - stage_start
                            run_guidance = target is not None and args.corrector_mode == "ogg"
                            if run_guidance:
                                discarded_actions += len(action_plan)
                                action_plan.clear()
                            if not action_plan:
                                torch.cuda.synchronize()
                                stage_start = time.monotonic()
                                with torch.no_grad():
                                    if run_guidance:
                                        actions, aux, guidance_stats = predict_guided(
                                            policy,
                                            observation,
                                            monitor.predictor,
                                            target,
                                            eta=args.guidance_eta,
                                            apply_every=args.guidance_apply_every,
                                        )
                                        guidance_requests += 1
                                        applied = sum(int(row["applied"]) for row in guidance_stats)
                                        guidance_steps += applied
                                        effective_guidance += int(
                                            any(row["delta_v_norm"] > 0 for row in guidance_stats)
                                        )
                                        event["guidance_stats"] = guidance_stats
                                        print(
                                            f"OGG suite={suite} task={task_id} trial={trial_id} step={steps} "
                                            f"error={event['error']:.4f} threshold={event['threshold']:.4f} "
                                            f"applied_steps={applied}",
                                            flush=True,
                                        )
                                    else:
                                        actions, aux = policy.predict_action_batch(observation, mode="eval")
                                torch.cuda.synchronize()
                                policy_seconds += time.monotonic() - stage_start
                                assert actions.shape == (1, args.action_chunk, 7)
                                assert aux["model_actions"].shape == (1, 10, 32)
                                if not torch.isfinite(actions).all():
                                    raise RuntimeError("PI0.5 returned non-finite actions")
                                queries += 1
                                for index in range(args.action_chunk):
                                    action_plan.append((actions[0, index], aux["model_actions"][0, index]))
                            if event is not None:
                                event.update(
                                    suite=suite,
                                    task_id=task_id,
                                    trial_id=trial_id,
                                    mode=args.corrector_mode,
                                    guidance_requested=run_guidance,
                                )
                                event_output.write(json.dumps(event) + "\n")
                            action, model_action = action_plan.popleft()
                            if monitor is not None:
                                stage_start = time.monotonic()
                                monitor.queue_action(steps, z, model_action)
                                monitor_seconds += time.monotonic() - stage_start
                            action_digest.update(action.detach().float().cpu().numpy().tobytes())
                            observation, _reward, terminated, truncation, _info = env.step(
                                action[None], auto_reset=False
                            )
                            steps += 1
                            success = bool(terminated.any().item())
                            truncated = bool(truncation.any().item())
                        record = {
                            "suite": suite,
                            "task_id": task_id,
                            "task_name": task.name,
                            "trial_id": trial_id,
                            "reset_id": reset_id,
                            "episode_seed": episode_seed,
                            "success": success,
                            "truncated": truncated,
                            "steps": steps,
                            "queries": queries,
                            "max_episode_steps": int(cfg.max_episode_steps),
                            "elapsed_rollout_seconds": time.monotonic() - started,
                            "corrector_mode": args.corrector_mode,
                            "diagnostic_step_limit": args.max_steps,
                            "diagnostic_cutoff": bool(
                                args.max_steps and steps >= max_steps and not (success or truncated)
                            ),
                            "executed_actions_sha256": action_digest.hexdigest(),
                            "corrector_comparisons": len(monitor.errors) if monitor else 0,
                            "mean_corrector_error": float(np.mean(monitor.errors))
                            if monitor and monitor.errors
                            else None,
                            "trigger_count": monitor.trigger_count if monitor else 0,
                            "guidance_requests": guidance_requests,
                            "guidance_applied_steps": guidance_steps,
                            "effective_guidance_requests": effective_guidance,
                            "discarded_actions": discarded_actions,
                            "monitor_seconds": monitor_seconds,
                            "policy_seconds": policy_seconds,
                            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                        }
                        output.write(json.dumps(record) + "\n")
                        records.append(record)
                        write_summary()
                        total_successes = sum(row["success"] for row in records)
                        print(
                            f"[{len(records)}/{expected}] {suite} task={task_id} trial={trial_id} "
                            f"success={success} steps={steps} total_SR={total_successes / len(records):.2%}",
                            flush=True,
                        )
            finally:
                env.env.close()
    print(json.dumps(summarize(records, settings, expected), indent=2), flush=True)


if __name__ == "__main__":
    main()
