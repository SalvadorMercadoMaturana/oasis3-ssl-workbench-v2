#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p campaign_100ep

run_method() {
  local method="$1"
  shift
  local existing=""
  if [[ -f campaign_100ep/status.log ]]; then
    existing="$(awk -v name="$method" '$2 == "COMPLETE" && $3 == name {print $4}' \
      campaign_100ep/status.log | tail -1)"
  fi
  if [[ -n "$existing" && -f "$existing/checkpoints/last.ckpt" ]]; then
    printf '%s SKIP_COMPLETED %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
      "$method" "$existing"
    return
  fi
  local logfile="campaign_100ep/${method}.log"
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_100ep/status.log
  OMP_NUM_THREADS=2 .venv/bin/t5-ssl train data=oasis3_lmdb \
    trainer.max_epochs=100 trainer.accelerator=cpu \
    trainer.enable_progress_bar=false "$@" >"$logfile" 2>&1
  local experiment
  experiment="$(sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' "$logfile" | tail -1)"
  if [[ -z "$experiment" || ! -f "$experiment/checkpoints/last.ckpt" ]]; then
    printf '%s FAILED_CHECKPOINT %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_100ep/status.log
    exit 1
  fi
  printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" | tee -a campaign_100ep/status.log
}

run_method vicreg
run_method byol model=byol_convnext loss=byol
run_method dinov2 model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop
