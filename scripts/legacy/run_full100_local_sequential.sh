#!/usr/bin/env bash
# Local single-GPU variant of run_full100_gpu.sh for a memory-constrained
# machine (here: GTX 1650, 4GB VRAM, ~7.4GB system RAM with very little
# free). Same campaign folder, same model/loss choices, same manifest as
# run_full100_gpu.sh, but the three methods run ONE AT A TIME instead of
# concurrently: benchmarking on this machine showed each process alone
# already leaves under 1GB of free system RAM (LMDB cache + Python + CUDA
# context), so three at once risks exhausting RAM even though GPU VRAM
# usage per process is tiny (~600MB of 4096MB).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p campaign_full100_gpu
touch campaign_full100_gpu/status.log

run_method() {
  local method="$1"
  shift
  local existing
  existing="$(awk -v name="$method" '($2 == "COMPLETE" || $2 == "EARLY_STOP") && $3 == name {print $4}' \
    campaign_full100_gpu/status.log 2>/dev/null | tail -1)"
  if [[ -n "$existing" && ( -f "$existing/checkpoints/last.ckpt" || \
      -n "$(find "$existing/checkpoints" -maxdepth 1 -name 'best-loss-epoch=*.ckpt' -print -quit 2>/dev/null)" ) ]]; then
    printf '%s SKIP_COMPLETED %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$existing"
    return 0
  fi
  local logfile="campaign_full100_gpu/${method}.log"
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_full100_gpu/status.log
  if .venv/Scripts/python.exe -m t5_ssl.train \
      data=oasis3_full50 trainer=full100 data.train_cache_capacity=60 \
      +data.eval_cache_capacity=16 trainer.enable_progress_bar=false "$@" \
      >"$logfile" 2>&1; then
    local experiment
    # Windows prints "logs\<dir>" (backslash); normalize to "/" before matching.
    experiment="$(tr '\\' '/' < "$logfile" | sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' | tail -1)"
    if [[ -n "$experiment" && -f "$experiment/checkpoints/last.ckpt" ]]; then
      printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" | tee -a campaign_full100_gpu/status.log
      return 0
    fi
  fi
  printf '%s FAILED %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_full100_gpu/status.log
  return 1
}

result=0
run_method vicreg model.backbone.pretrained=true || result=1
run_method byol model=byol_resnet18 loss=byol || result=1
run_method dinov2 model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop || result=1
exit "$result"
