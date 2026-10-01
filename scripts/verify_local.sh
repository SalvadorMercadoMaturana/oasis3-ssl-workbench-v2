#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m ruff check src tests scripts tools/dataset_oasis3/scripts
python -m pytest
python -m t5_ssl.cli data-info --source auto
python tools/dataset_oasis3/scripts/quick_start.py --check
python - <<'PY'
from t5_ssl.data.oasis3_lmdb_datamodule import OASIS3LMDBDataModule

dm = OASIS3LMDBDataModule(batch_size=1, num_workers=0)
dm.setup()
sample = dm.train_dataset[0]
if isinstance(sample, dict):
    print("LMDB sample:", tuple(sample["image"].shape), sample.get("subject"))
else:
    image, metadata = sample
    print("LMDB sample:", tuple(image.shape), metadata)
PY

echo "Verificación local completada (sin entrenamiento)."
