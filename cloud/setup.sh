#!/usr/bin/env bash
# Install the workbench on a CUDA 12 + PyTorch >= 2.4 host (e.g. the RunPod image
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) without touching the
# preinstalled torch/torchvision: they are pinned through a constraints file.
set -euo pipefail
cd "$(dirname "$0")/.."  # the repository

python - <<'PY'
import sys
import torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 4):
    sys.exit(f"PyTorch {torch.__version__} < 2.4: use a newer image or the Dockerfile")
if not (torch.version.cuda or "").startswith("12"):
    sys.exit(f"CUDA runtime {torch.version.cuda} is not 12.x")
print(f"torch {torch.__version__} / CUDA {torch.version.cuda}")
PY

# Every campaign ran on torch 2.8 (CUDA 12.8 wheels): install it when the image's
# PyTorch is older (2.4's fused AdamW rejects the channels_last BYOL parameters) or
# predates the GPU (no sm_120 kernels for an RTX 5090).
NEED_TORCH=$(python - <<'PY'
import torch
version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if version < (2, 8):
    print(1)
elif not torch.cuda.is_available():
    print(0)
else:
    major, minor = torch.cuda.get_device_capability(0)
    # cubins run on later minor versions of the same major (sm_86 on an sm_89 4090)
    archs = [a[3:] for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    print(0 if any(int(a[:-1]) == major and int(a[-1]) <= minor for a in archs) else 1)
PY
)
if [[ "$NEED_TORCH" == "1" ]]; then
  SPEC="${TORCH_UPGRADE_SPEC:-torch==2.8.0 torchvision==0.23.0}"
  echo "Installing $SPEC (cu128): the image's PyTorch is older than 2.8 or lacks this GPU"
  # shellcheck disable=SC2086
  python -m pip install --quiet $SPEC --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
fi

CONSTRAINTS="$(mktemp)"
python - > "$CONSTRAINTS" <<'PY'
import importlib.metadata as md
for name in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -c "$CONSTRAINTS" -e ".[analysis]" torchio
rm -f "$CONSTRAINTS"
mkdir -p runs data/roi3d data/dementia campaign_full50_pretrained_balanced
python -c "import lightning, hydra, torchio, sklearn, t5_ssl; print('setup ok: lightning', lightning.__version__, '| torchio', torchio.__version__)"
