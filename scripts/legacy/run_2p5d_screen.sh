#!/usr/bin/env bash
# Update-matched GPU screening for the SSL-input-diagnostic follow-up:
# 2.5D axial stacks + intermediate augmentation, across four candidates
# (VICReg with a feature queue, MAE/SimMIM, MAE+VICReg hybrid, and a BYOL
# GroupNorm control). Sequential, not parallel: the local GPU has 4GB VRAM,
# and the machine has under 2GB of RAM actually free, so this deliberately
# does not raise data.num_workers above the data config's safe default (each
# worker duplicates the volume cache; 3 workers x a large cache OOM'd once).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p campaign_2p5d_screen
touch campaign_2p5d_screen/status.log

run_method() {
  local method="$1"
  shift
  local existing
  existing="$(awk -v name="$method" '($2 == "COMPLETE") && $3 == name {print $4}' \
    campaign_2p5d_screen/status.log 2>/dev/null | tail -1)"
  if [[ -n "$existing" && -f "$existing/checkpoints/last.ckpt" ]]; then
    printf '%s SKIP_COMPLETED %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$existing"
    return 0
  fi
  local logfile="campaign_2p5d_screen/${method}.log"
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_2p5d_screen/status.log
  if OMP_NUM_THREADS=4 .venv/Scripts/python.exe -m t5_ssl.train \
      data=oasis3_2p5d_screen trainer=gpu_screen augmentation=mri_2p5d_intermediate \
      trainer.enable_progress_bar=false "$@" >"$logfile" 2>&1; then
    local experiment
    experiment="$(sed -n 's/.*Artifacts saved in: \(logs[\\/][^ ]*\).*/\1/p' "$logfile" | tail -1)"
    experiment="${experiment//\\//}"
    if [[ -n "$experiment" && -f "$experiment/checkpoints/last.ckpt" ]]; then
      printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" >> campaign_2p5d_screen/status.log
      return 0
    fi
  fi
  printf '%s FAILED %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_2p5d_screen/status.log
  return 1
}

run_method vicreg_queue model=convnext_2p5d loss=vicreg loss.queue_size=1024
run_method byol_groupnorm model=byol_convnext_groupnorm loss=byol model.backbone.in_channels=5
run_method mae model=vit_2p5d loss=mae model.backbone.pool=mean
run_method mae_vicreg model=vit_2p5d loss=mae_vicreg loss.queue_size=1024
