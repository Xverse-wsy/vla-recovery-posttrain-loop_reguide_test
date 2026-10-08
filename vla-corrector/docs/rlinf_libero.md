# RLinf PI0.5 LIBERO corrector training

This adapter trains the existing dynamics/corrector component on the combined
four-suite `physical-intelligence/libero` dataset. Training keeps the original
policy/environment output path. Online monitoring and OGG are
available through the separate evaluation adapter documented in
`rlinf_libero_eval.md`.

## Data and feature contract

- Load the actual RLinf policy factory, including legacy checkpoint-key conversion.
- Use `pi05_libero` and explicitly load the checkpoint's
  `physical-intelligence/libero/norm_stats.json`; dataset `stats.json` is unused.
- Pass state, task text, images and demonstration actions through the real RLinf
  input transforms. Do not rotate the already converted offline images again.
- Extract frozen projected SigLIP features before the language model: base camera
  then left wrist, 256 tokens each, 2048 dimensions. Exclude the masked dummy camera.
- Save normalized actions as float32, seven dimensions, gripper at index 6.
  Quantile normalization is `2*(a-q01)/(q99-q01+1e-6)-1`, without clipping.
  With this checkpoint gripper -1/+1 becomes -1/1.000399146.
- Visual features use the existing int8/per-token float16-scale cache format.
  Metadata records normalization, source revisions, cameras, errors and episode IDs.
- Read complete episodes and split training/validation by episode. History windows
  and future targets remain inside the same episode. This is a dynamics validation
  split, not a holdout of entire tasks or LIBERO suites.

The feature representation changes from the LeRobot extraction path; extract a
new cache and train a new corrector. Do not reuse a corrector trained with another
feature or action contract.

## Server environments

Use `corrector-rlinf` (Python 3.12, torch 2.6/cu124, numpy 1.26.4,
transformers 4.53.2, CPU JAX 0.5.3) for extraction. Use the existing
`vla-corrector` environment for corrector training. Their transformer dependencies
differ. Do not install this project's full dependency list in `corrector-rlinf`.

Source checkouts live alongside this repo. RLinf was pinned at
`c70606f08cdca259b8dec03d4430926b5b8fac9d` for validation; use the environment
snapshot under the server's `environment-setup` directory when reproducing.

## Commands

Run from `/home/wenchaoxu/wsy/vla-recovery-posttrain-loop`.
Choose one GPU with sufficient free memory before either command. GPU 1 was used
for the smoke extraction; availability can change.

```bash
conda activate corrector-rlinf
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/vla-corrector/src${PYTHONPATH:+:$PYTHONPATH}"
python -m siglip_dynamics.extract_rlinf \
  --dataset-path dataset \
  --policy-path models/RLinf-Pi05-LIBERO-SFT \
  --rlinf-src RLinf --openpi-src openpi \
  --output-path dataset/corrector_rlinf_pi05_full \
  --batch-size 4 --verify
```

For a small extraction use a fresh output path and `--max-episodes 3`; this keeps
three whole episodes. Existing output directories are rejected. Partial caches
have no `metadata.json` and cannot be used for training. The full cache is about
287 GB; the extractor checks free space and its default 500 GB limit first.

```bash
conda activate vla-corrector
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/vla-corrector/src${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc_per_node=1 -m siglip_dynamics.train \
  --dataset-path dataset/corrector_rlinf_pi05_full \
  --checkpoint-dir vla-corrector/outputs/rlinf_pi05_libero \
  --model-type mlp --scale 20m \
  --split-unit episode --val-ratio 0.1 --seed 42 \
  --h-window 1 --k-step-list 10 \
  --batch-size 512 --epochs 30 --train-loss-type cosine --num-workers 0
```

The batch size, epoch limit and cosine objective above match the official README.
`--scale 20m` is a preset name: with these MLP input dimensions it contains
42,510,336 parameters (approximately 40M). Default early stopping patience is five
epochs, so training may stop before epoch 30. The completed server run used
batch size 512 per GPU on three GPUs for 20 epochs. Reduce the batch size if a
different GPU cannot accommodate it.
CPU batch pinning is disabled by default for both training and validation.
Large feature tensors can otherwise fail during NVIDIA host-page locking when
the server's memory is fragmented, even with ample available RAM. Use
`--no-pin-memory` to make the disabled setting explicit; `--pin-memory` opts
back in on CUDA. Pageable-memory transfers may be slower, without changing the
model, objective, data split or learning-rate schedule.
`--k-step-list 10` writes the run under `outputs/rlinf_pi05_libero/h1_k10/`.

The trainer saves `split_manifest.json` and `cache_metadata.json` with checkpoints.
Checkpoint selection and early stopping use validation cosine loss, tracked on
every epoch. On each strict improvement, a lightweight current-best file is
written as `best_cosine_model_epoch_<epoch>.pt`; the previous current-best file
from this run is removed after the new one is published. Its weights,
configuration, epoch and best cosine loss remain compatible with evaluation.

A permanent `checkpoint_epoch_<epoch>.pt` training archive is created only if
the epoch is greater than 10, divisible by 5, and strictly improves the global
validation cosine best. Thus eligible epochs are 15, 20, 25, ...; an improvement
at epoch 16 counts when deciding whether epoch 20 is a record low. Archives also
contain AdamW and scheduler state, epoch, early-stopping state, split manifest,
world size and Python/numpy/torch CPU/CUDA RNG state for every training rank.
CUDA state capture visits only devices used by that rank. The scheduler state
is captured after its end-of-epoch step. These archives load with
`torch.load(..., weights_only=True)` and retain the same `state_dict` for inference.

For example, improvements at 15, 20 and 21 leave `checkpoint_epoch_15.pt`,
`checkpoint_epoch_20.pt` and `best_cosine_model_epoch_21.pt`. If 22 improves,
the current-best 21 file is replaced by 22; both periodic archives remain.
`checkpoint_manifest.json` identifies the current best and retained archives,
so directory-based evaluation selects the latest best without relying on
alphabetical filenames. Explicit checkpoint paths are honored as well.

MSE remains a logged metric and does not select checkpoints. Early stopping is
unchanged: every strict cosine improvement resets the counter, and patience
defaults to five epochs without improvement. Reaching `--epochs` also stops.
Use a fresh output directory: training refuses directories containing `.pt`
checkpoints or an existing checkpoint manifest, preserving old runs. All models
saved before this change remain untouched and load through the legacy path.
This change stores training state for future restoration; the training CLI does
not yet provide a resume option, and existing epoch-20 files still contain only
the previously saved weights/configuration/metrics.

Validated on 2026-10-07: all 19 regression checks passed locally and in the
server training environment. The real training-loop tests cover global-best
selection across nonperiodic epochs, archive/current-best retention, unchanged
five-epoch early stopping, legacy/new inference loading and atomic-write failure.
A small AdamW update replay restored from archive state matched the uninterrupted
next update exactly. A two-GPU synthetic DDP check retained archives 15/20 and
current best 21, with CPU/CUDA RNG state for both ranks. Its validation values
were controlled to exercise save boundaries; it did not train the full dataset.
The server report/log are
`environment-setup/checkpoint-rule-ddp-check-20261007.{json,log}`.
The MLP learns the original objective from current-frame action to visual change
at `t+k`; `k=10` does not mean it consumes a full 10-action chunk.

## Validation and OGG action coordinates

`--verify` checks the first real batch: normalized demonstrations against the
checkpoint quantiles, inverse-transform roundtrip, visual tokens against the
actual policy prefix, gripper endpoints and unclipped outliers, and a real ODE
sample through the policy's original output decoding.

Server smoke validation completed on 2026-10-05:

- Three complete episodes, 843 frames, 813 valid `k=10` training pairs.
- Episodes 1/2 train (609 pairs), episode 0 validation (204 pairs).
- First-batch normalization, inverse roundtrip and visual-prefix maximum errors: 0.
- A custom 32/32-width MLP trained for two epochs; all three saved checkpoints
  reloaded with strict parameter matching and produced finite `[1,512,2048]` outputs.
- Three standalone cache/split contract tests and dependency compatibility passed.

Smoke cache: `dataset/corrector_rlinf_pi05_smoke`; checkpoints:
`vla-corrector/outputs/rlinf_pi05_smoke`; machine-readable report:
`environment-setup/rlinf-corrector-smoke-report.json` (all paths relative to the
server project parent). These small-model weights validate the pipeline and are
not benchmark-trained corrector weights. That initial readiness check did not
launch full extraction/training.

The subsequent full cache contains 273,465 frames and 1,693 episodes across all
four suites. The completed 20-epoch run selected epoch 20 with validation cosine
loss 0.336331282103799. Its checkpoint is
`vla-corrector/outputs/rlinf_pi05_libero/h1_k10/best_cosine_model.pt`.

```bash
PYTHONPATH=src python -m unittest discover -s tests/siglip_dynamics -v
```

OGG uses the same feature stage/camera order and the first seven
dimensions of the clean normalized action estimate (`x0_hat`). Do not normalize
that estimate twice, use noisy `x_t`/velocity as the action, or clamp it to [-1,1].
Leave the policy's original inverse normalization and environment transformations
in place. Shared coordinates do not guarantee equal numeric support or good OGG
performance: PI0.5 can predict outside the demonstration quantiles. The online
adapter enforces this contract; guidance effectiveness still requires benchmark
evaluation. See `rlinf_libero_eval.md` for modes, diagnostics and commands.
