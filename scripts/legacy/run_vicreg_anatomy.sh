#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p campaign_anatomy
if [[ -f campaign_anatomy/experiment.txt ]]; then
  experiment="$(cat campaign_anatomy/experiment.txt)"
  if [[ -f "$experiment/checkpoints/last.ckpt" ]]; then
    printf 'SKIP_COMPLETED %s\n' "$experiment"
    exit 0
  fi
fi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/t5-ssl train \
  data=oasis3_full50 trainer=full50 augmentation=mri_anatomy \
  trainer.accelerator=cpu trainer.enable_progress_bar=false \
  > campaign_anatomy/vicreg.log 2>&1
experiment="$(sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' \
  campaign_anatomy/vicreg.log | tail -1)"
if [[ -z "$experiment" || ! -f "$experiment/checkpoints/last.ckpt" ]]; then
  printf 'Training did not produce a final checkpoint; inspect campaign_anatomy/vicreg.log\n' >&2
  exit 1
fi
printf '%s\n' "$experiment" > campaign_anatomy/experiment.txt
printf 'COMPLETE %s\n' "$experiment"
