#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p campaign_full50
touch campaign_full50/status.log

run_method() {
  local method="$1"
  shift
  local existing
  existing="$(awk -v name="$method" '($2 == "COMPLETE" || $2 == "EARLY_STOP") && $3 == name {print $4}' \
    campaign_full50/status.log 2>/dev/null | tail -1)"
  if [[ -n "$existing" && ( -f "$existing/checkpoints/last.ckpt" || \
      -n "$(find "$existing/checkpoints" -maxdepth 1 -name 'best-loss-epoch=*.ckpt' -print -quit 2>/dev/null)" ) ]]; then
    printf '%s SKIP_COMPLETED %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$existing"
    return 0
  fi
  local logfile="campaign_full50/${method}.log"
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_full50/status.log
  if OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/t5-ssl train \
      data=oasis3_full50 trainer=full50 trainer.accelerator=cpu \
      trainer.enable_progress_bar=false "$@" >"$logfile" 2>&1; then
    local experiment
    experiment="$(sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' "$logfile" | tail -1)"
    if [[ -n "$experiment" && -f "$experiment/checkpoints/last.ckpt" ]]; then
      printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" >> campaign_full50/status.log
      return 0
    fi
  fi
  printf '%s FAILED %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_full50/status.log
  return 1
}

run_method vicreg & p1=$!
run_method byol model=byol_convnext loss=byol & p2=$!
run_method dinov2 model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop & p3=$!
result=0
wait "$p1" || result=1
wait "$p2" || result=1
wait "$p3" || result=1
exit "$result"
