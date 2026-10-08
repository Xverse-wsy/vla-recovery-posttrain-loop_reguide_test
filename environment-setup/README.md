# Evaluation environment

On the agilex machine, `corrector-eval` is already installed under
`/home/agilex/miniconda3/envs/corrector-eval`. Pip dependency checks, actual
RLinf/OpenPI/LIBERO/corrector imports, evaluation CLI help, and 19 CPU regression
tests passed. No CUDA context, GPU policy inference, or EGL render was started.

For a fresh RTX 5090 environment, use Python 3.12, PyTorch 2.7.1 with CUDA 12.8
and matching torchvision 0.22.1. Install PyTorch using its official cu128 wheel
index, then install the requirements with the supplied constraints. These files
describe the selective evaluation dependencies, not the full RL training stack.

Use the pinned submodules in the repository. Apply the OpenPI
`src/openpi/models_pytorch/transformers_replace/` files to this environment's
Transformers 4.53.2 installation, preserving originals. Set PYTHONPATH to the
project's vla-corrector/src, RLinf, openpi/src, and
openpi/packages/openpi-client/src directories. Set JAX_PLATFORMS=cpu,
MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, LIBERO_TYPE=standard, and
ROBOT_PLATFORM=LIBERO. Use an isolated LIBERO_CONFIG_PATH with paths for this
machine; do not copy another server's absolute paths.

The original PI0.5 checkpoint, its normalization stats, verified LIBERO resources,
and tokenizer assets are external prerequisites. Complete these and a GPU/EGL
smoke check before running a full benchmark. See the adapted repository's
docs/rlinf_libero_eval.md for the action/feature contract and evaluation protocol.
