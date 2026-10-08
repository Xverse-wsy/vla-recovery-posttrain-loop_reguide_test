# LIBERO PI0.5 corrector evaluation — epoch 50

This repository contains the adapted VLA corrector evaluation code and the
epoch-50 corrector trained with RLinf PI0.5 LIBERO features. The original upstream
license and documentation remain in `vla-corrector/`. RLinf and OpenPI are pinned
as submodules to the revisions recorded during feature extraction.

## Contents

- `vla-corrector/`: source snapshot, including RLinf baseline/monitor/OGG evaluation.
- `models/corrector/libero_epoch50/`: the epoch-50 weight via Git LFS, plus exact
  training `config.json` and `cache_metadata.json`. Always keep them together.
- `environment-setup/`: RTX 5090 compatible dependency pins and environment notes.
- `source_provenance.json`: framework revisions and checkpoint checksum.

The corrector is an h=1, k=10 MLP with 7 normalized action dimensions. Its
epoch-50 validation cosine loss was 0.3031339021648817. See checkpoint_info.json
for SHA256 and size. This is the corrector weight; the 7 GiB base policy and
LIBERO simulation resources are external prerequisites. No training datasets,
feature caches, optimizer archives, or other corrector epochs are included.

## Download

Install Git LFS, then clone this repository with its submodules:

```bash
git lfs install
git clone --recurse-submodules https://github.com/Xverse-wsy/vla-recovery-posttrain-loop_reguide_test.git
cd vla-recovery-posttrain-loop_reguide_test
git lfs pull --include='models/corrector/libero_epoch50/*.pt'
```

On a machine that already has prepared framework sources, a normal clone can be
used to obtain the adapted code and weight. Preserve and verify the recorded
framework revisions before using those existing sources. Verify the checkpoint
SHA256 against `models/corrector/libero_epoch50/checkpoint_info.json`.

## Evaluation

Prepare `models/RLinf-Pi05-LIBERO-SFT/model.safetensors` and its
`physical-intelligence/libero/norm_stats.json`. They must exactly match the
checkpoint and normalization hashes in the corrector's cache_metadata.json.
Prepare LIBERO assets, BDDL definitions, and initial states, with configuration
paths pointing to this machine. The adapter checks the original action and
feature contract and does not renormalize or clip corrector action inputs.

After environment and GPU/EGL validation, the three-suite OGG protocol is:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=4
python -u vla-corrector/scripts/eval_rlinf_libero.py \
  --suites libero_spatial libero_object libero_goal \
  --trials-per-task 50 --num-steps 3 --action-chunk 10 --seed 42 \
  --corrector-mode ogg \
  --corrector-checkpoint models/corrector/libero_epoch50/best_cosine_model_epoch_50.pt \
  --guidance-eta 1 --guidance-apply-every 1 --meltdown-cooldown-steps 10 \
  --output-dir vla-corrector/outputs/pi05_libero_ogg_epoch50
```

This runs 1,500 episodes across spatial, object, and goal. Keep suite step limits
(240/240/320) and threshold defaults unchanged when comparing with the previous
protocol. Rerun a matching baseline on new GPU/software if making comparisons.
