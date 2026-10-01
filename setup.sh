#!/usr/bin/env bash
# Install the workbench on a CUDA 12 + PyTorch >= 2.4 host (e.g. the RunPod image
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) without touching the
# preinstalled torch/torchvision: they are pinned through a constraints file.
set -euo pipefail
cd "$(dirname "$0")"

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
