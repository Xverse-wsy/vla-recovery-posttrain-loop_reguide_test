# LIBERO evaluation environment

The server's `corrector-eval` Conda environment is ready for PI0.5 baseline
evaluation. It is separate from the feature-extraction and corrector-training
environments. The repository now includes online corrector monitoring, action
queue truncation and OGG integration for this same RLinf baseline.

## Versions and resources

- Python 3.12, PyTorch 2.6.0+cu124, numpy 1.26.4, transformers 4.53.2.
- `rlinf-libero==0.1.3` (the package recommended by RLinf's official installer),
  robosuite 1.4.1, MuJoCo 3.3.7, gym 0.26.2, gymnasium 1.0.0.
- CPU JAX for OpenPI transforms; `ml-dtypes==0.5.1` and `numpydantic==1.6.11`
  preserve numpy 1.x compatibility.
- Independent resources under
  `/home/wenchaoxu/wsy/vla-recovery-posttrain-loop/libero-resources/`.
  They contain copied, verified task definitions, initial states and assets.
  `manifest.json` records SHA256 for all 699 files, including 585 runtime assets.
- Tasks/states match official `Lifelong-Robot-Learning/LIBERO` revision
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`; assets match
  `lerobot/libero-assets` revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
- The installed LIBERO package's assets link points to this independent resource
  directory. `LIBERO_CONFIG_PATH` selects its own config, leaving the shared
  `~/.libero/config.yaml` and other projects' caches in place.

Conda activation sets the resource configuration, `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl`, `JAX_PLATFORMS=cpu`, and source import paths.
Select GPU devices explicitly for each run. Robosuite expects
`MUJOCO_EGL_DEVICE_ID` to use the physical GPU number in `CUDA_VISIBLE_DEVICES`.

## Validated single-episode baseline

```bash
cd /home/wenchaoxu/wsy/vla-recovery-posttrain-loop
conda activate corrector-eval
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_EGL_DEVICE_ID=2
export OMP_NUM_THREADS=4
python environment-setup/eval_libero_baseline.py \
  --suite libero_spatial --reset-id 0 \
  --output environment-setup/my-libero-spatial-baseline.json
```

Recheck GPU availability before selecting a device. The helper uses RLinf's
actual `LiberoEnv` and `Pi0Eval` wrappers, the downloaded checkpoint and its exact
normalization pipeline. It generates 10 actions in 32 model dimensions, executes
the first 5 actions in 7 environment dimensions, and uses 3 flow steps, matching
the RLinf spatial evaluation example. Supported suites are `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`.

This is a one-episode readiness check, not a full benchmark. The initial check
completed task 0, trial 0 successfully in 73 environment steps and 15 policy
queries, with about 6.65 GiB peak PyTorch allocated GPU memory. The rollout took
7.05 seconds, excluding initialization. JSON/logs are in
`environment-setup/corrector-eval-smoke.{json,log}`.

## Three-suite baseline benchmark

`scripts/eval_rlinf_libero.py` loads the checkpoint once and evaluates explicit
task/trial pairs sequentially using the same RLinf policy/environment wrappers.
By default it runs tasks 0..9 and initial states 0..49 for spatial, object and
goal (1500 episodes), with no corrector. The RLinf protocol uses 240/240/320
action steps respectively, 15 settling steps, 3 flow steps, and by default executes
5 of the 10 generated actions. The completed server baseline explicitly used
chunk 10 and seed 42 instead. Its 1,500-episode success rates were spatial 82.0%,
object 97.2%, goal 80.2%, and overall 86.4667%. Preserve chunk 10 and seed 42 for
comparisons to that result. Other protocol settings should also be held fixed;
this is not a claim to reproduce the upstream OpenPI paper evaluation protocol.

```bash
cd /home/wenchaoxu/wsy/vla-recovery-posttrain-loop
conda activate corrector-eval
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=4
python -u vla-corrector/scripts/eval_rlinf_libero.py \
  --suites libero_spatial libero_object libero_goal \
  --trials-per-task 50 --num-steps 3 --action-chunk 5 --seed 0 \
  --output-dir vla-corrector/outputs/pi05_libero_baseline_chunk5_seed0_new
```

Recheck GPU availability first. For a smaller all-task preliminary evaluation,
use `--trials-per-task 5` and a different output directory (150 episodes).
Output directories must be new. Each completed episode is written immediately
to `episodes.jsonl`; `summary.json` is updated after every episode with task and
suite success rates and a completion flag. Runtime exceptions abort evaluation
instead of counting as policy failures. No full benchmark is launched by setup.

Validated on 2026-10-05 with task IDs 0 and 1, trial 0 in all three suites:
six completed episodes (five successes and one goal-task timeout), correct
explicit reset IDs, per-task/suite statistics and completed summary. This small
check verifies the evaluation loop, not the checkpoint's benchmark success rate.

## Corrector monitoring and OGG

The same `scripts/eval_rlinf_libero.py` entry supports `--corrector-mode off`
(default), `monitor` (detection only), and `ogg` (detection, action queue
truncation, and guided replanning). It keeps RLinf's native PI0.5 sampler and
environment transforms. The original LeRobot modified-evaluation threshold
implementation is shared through `siglip_dynamics.inference.robust_threshold`.
No RLinf or OpenPI source files are modified.

For the existing chunk-10, flow-3, seed-42 baseline, use:

```bash
cd /home/wenchaoxu/wsy/vla-recovery-posttrain-loop
conda activate corrector-eval
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=4
python -u vla-corrector/scripts/eval_rlinf_libero.py \
  --suites libero_spatial libero_object libero_goal \
  --trials-per-task 50 --action-chunk 10 --num-steps 3 --seed 42 \
  --corrector-mode ogg \
  --corrector-checkpoint vla-corrector/outputs/rlinf_pi05_libero/h1_k10/best_cosine_model.pt \
  --guidance-eta 1 --guidance-apply-every 1 --meltdown-cooldown-steps 10 \
  --output-dir vla-corrector/outputs/pi05_libero_ogg_chunk10_seed42
```

Use `monitor` and a new directory for detection-only comparisons. For a short
diagnostic rollout, add `--task-ids 0 --trials-per-task 1 --max-steps 40` and select
one suite. Positive `--max-steps` values cap the rollout for debugging and must
not be used as a benchmark success-rate result. The output settings record this
override. `--help` lists the original threshold parameters as `--threshold-*`
options; do not silently change them between experiments.

The adapter verifies SHA256 of the exact base checkpoint and normalization
statistics against `cache_metadata.json`. It requires the trained h=1, k=10 MLP
and exact camera order (base then left wrist), projected SigLIP features
`[1,512,2048]`, and seven normalized action dimensions. Visual features use the
same int8/per-token fp16-scale quantize/dequantize roundtrip as the training
cache. Actions remain float32 and are never clipped or normalized again.

Previously trained runs retain `best_cosine_model.pt` and their existing epoch
archives. New runs use `best_cosine_model_epoch_<epoch>.pt` for the current best
and `checkpoint_epoch_<epoch>.pt` for permanent record-low periodic archives.
Pass the desired file explicitly to `--corrector-checkpoint`; training archives
also work for inference, using their model `state_dict`. The underlying predictor
uses `checkpoint_manifest.json` for directory-based current-best selection and
honors explicit file paths. See `rlinf_libero.md` for the retention/state policy.

Every environment step is observed. A forecast is queued only for an action
actually executed, and compared to the feature change exactly ten steps later.
Detector bootstrap, history and cooldown reset each episode. A trigger in `ogg`
discards unexecuted actions and requests a new chunk using the original
`attract_delta_z_correction` objective. The correction target is the previous
predicted change minus its observed change. The corrector receives
`x0_hat[:,0,:7]`, with `x0_hat = x_t - t * v_t`. Only a detached velocity leaf and
the frozen corrector's action-input graph participate in differentiation; the
VLA, images and prefix KV cache never receive gradients. The temporary velocity
hooks are restored even on failure. This adapter is for single-threaded,
single-environment evaluation.

`corrector_events.jsonl` stores each aligned comparison, trigger and per-flow-step
OGG loss/gradient/update norms. `episodes.jsonl` additionally records comparisons,
triggers, guided requests, nonzero updates, discarded actions, action-sequence
hashes, inference/monitoring time, and process-local peak allocated/reserved GPU
memory in GiB. The timing includes feature extraction for monitoring. Runtime
errors abort the run instead of being reported as policy failures. Eta zero
disables velocity updates but OGG-mode detector triggers can still truncate the
queue; use `off` for the unchanged baseline.

CPU regression checks:

```bash
PYTHONPATH=vla-corrector/src python -m unittest discover \
  -s vla-corrector/tests/siglip_dynamics -v
```

They cover unchanged original threshold code, training/live feature roundtrip,
normalization/checkpoint mismatches, exact k-step alignment, episode resets,
native RNG consumption and hook restoration, local finite gradients limited to
the first action's seven dimensions, and full CLI rollout logic in a CPU test
simulator. Off/monitor action hashes match there; OGG triggers produce nonzero
updates and discard only queued unexecuted actions.

### Real-model checks on 2026-10-06

The trained epoch-20 corrector was verified against the downloaded base policy
and its normalization statistics. All 13 CPU regression checks and the new
adapter's Ruff checks passed locally and on the server.

- Real projected SigLIP features matched the actual PI0.5 prefix exactly.
- Fixed-input/fixed-noise monitoring and eta-zero sampling matched native
  actions exactly. A synthetic correction target exercised all three native
  flow steps with finite nonzero gradients; policy/corrector parameters received
  no gradients and the original sampler methods were restored.
- A further 36 real policy queries were replayed with identical observations
  and RNG states before/after the frozen monitoring forwards. Both normalized
  model actions and decoded environment actions had maximum error exactly zero;
  monitoring consumed neither CPU nor CUDA RNG.
- Real CLI short rollouts completed task 0 / trial 0 in spatial, object and goal.
  Off and monitor each executed 40 steps with four policy queries. Monitoring
  produced 30 comparisons per episode with exact ten-step target alignment.
- Default-threshold OGG checks (at most 80 steps per suite) produced no triggers
  in these three samples. A separate 40-step spatial diagnostic used bootstrap
  queue 2, k-on/k-off 0, EWMA alpha 1, trigger margin/jump 0, trigger consecutive
  steps 1 and cooldown 1. It triggered six guided requests, all 18 flow updates
  had nonzero gradients, and 47 queued unexecuted actions were discarded.
  These lowered thresholds are only a wiring check and are not benchmark
  defaults or evidence of improved success rates.
- Peak PyTorch allocated memory for the real guided rollout was 6.90 GiB,
  reserved memory 7.02 GiB. Allow about 12 GiB free GPU memory for CUDA/EGL and
  runtime overhead; the allocator figures exclude other GPU users and contexts.

Separate seeded rollouts did not reproduce bitwise-identical trajectories in
spatial/goal even when both runs used `off`; object matched exactly. A same-process
audit likewise found different observations on repeated `off` rollouts, despite
identical policy RNG states and exact fixed-input replay parity. Consequently,
cross-process action hashes cannot alone establish a monitor regression. Use
fixed observations/noise to check policy parity, and retain identical evaluation
settings for statistical comparisons.

Machine-readable reports and logs are in the server project's
`environment-setup/rlinf-corrector-ogg-runtime-check.{json,log}` and
`environment-setup/rlinf-corrector-ogg-rollout-check.{json,log}`, with per-query
observations/RNG hashes and action replay errors in
`environment-setup/rlinf-corrector-ogg-action-parity.{json,log}`. Diagnostic rollout
directories start with `vla-corrector/outputs/ogg_adapter_check_20261006_`.
No full corrector/OGG benchmark was launched by these checks.

## Official RLinf evaluation entry

The official entry imports and composes its Hydra configuration successfully in
this environment. A full Ray-based benchmark run has not been launched. To inspect
the resolved config without running evaluation:

```bash
export EMBODIED_PATH="$PWD/RLinf/examples/embodiment"
python RLinf/evaluations/eval_embodied_agent.py \
  --config-path "$PWD/RLinf/evaluations/libero" \
  --config-name libero_spatial_openpi_pi05_eval --cfg job \
  rollout.model.model_path="$PWD/models/RLinf-Pi05-LIBERO-SFT" \
  +rollout.model.openpi_data.norm_stats_path="$PWD/models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json" \
  env.eval.total_num_envs=1
```

The supplied official example defaults to 500 simultaneous environments; choose
parallelism and reset-state coverage deliberately before launching a benchmark.
Environment snapshots and setup records live in `environment-setup/` alongside
the server project, including `corrector-eval-freeze.txt`,
`corrector-eval-env.yml`, and `corrector-eval-README.md`.
