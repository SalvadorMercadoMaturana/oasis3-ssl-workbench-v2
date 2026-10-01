# Recommended environment for the RTX 4090 runs (sourced by bootstrap_runpod.sh).
# Safe to source more than once; every value can be overridden beforehand.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# Fewer allocator splits when several processes share the 24 GB (MPS / all-at-once).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Persistent Inductor caches: the 2nd+ run of a model reuses compiled kernels.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$REPO_DIR/.inductor-cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
# ImageNet / DeiT weights downloaded once per pod.
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"
# Main processes do little CPU math; DataLoader workers parallelize augmentation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
# SimpleITK (TorchIO elastic deformation, resampling) sizes its thread pool from the
# host's cores (192 on the RunPod 5090 host): one thread per DataLoader worker.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-1}"
export OASIS3_CSV="${OASIS3_CSV:-$REPO_DIR/data/dementia/mni-dataset.csv}"
# CUDA MPS (run_all.py --parallel mps sets these too).
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
