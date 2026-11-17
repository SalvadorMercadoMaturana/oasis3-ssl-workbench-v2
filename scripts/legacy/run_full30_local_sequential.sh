#!/usr/bin/env bash
# Reproduce the 2026-09-20 "good" pretrained VICReg/BYOL results on this
# machine's GPU: trainer.max_epochs=30, exactly like the session that got
# VICReg 0.829 AUC and BYOL 0.822 AUC. Only addition vs that CPU session:
# trainer.deterministic="warn" (DINOv2's bicubic positional-embedding
# interpolation has no deterministic CUDA backward; see full100.yaml).
#
# Why this script exists instead of reusing run_full100_local_sequential.sh:
# this codebase's cosine LR schedule horizon defaults to trainer.max_epochs
# (src/t5_ssl/train.py). Raising the ceiling to 100 "just in case, early
# stopping will cut it short" silently stretches the whole decay curve —
# at epoch 26 the LR was ~10% of peak on a 30-epoch horizon but ~92% of
# peak on a 100-epoch horizon. That mismatch, not a wrong model choice, is
# almost certainly why the 100-epoch run's VICReg diverged immediately and
# BYOL scored lower despite more epochs. Fixed properly in train.py via the
# new optimizer.schedule_epochs override (defaults to trainer.max_epochs,
# so every existing config/result is unaffected); this script just uses the
# already-proven max_epochs=30 for both the ceiling and the schedule.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p campaign_full30_gpu
touch campaign_full30_gpu/status.log

run_method() {
  local method="$1"
  shift
  local existing
  existing="$(awk -v name="$method" '($2 == "COMPLETE" || $2 == "EARLY_STOP") && $3 == name {print $4}' \
    campaign_full30_gpu/status.log 2>/dev/null | tail -1)"
  if [[ -n "$existing" && ( -f "$existing/checkpoints/last.ckpt" || \
      -n "$(find "$existing/checkpoints" -maxdepth 1 -name 'best-loss-epoch=*.ckpt' -print -quit 2>/dev/null)" ) ]]; then
    printf '%s SKIP_COMPLETED %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$existing"
    return 0
  fi
  local logfile="campaign_full30_gpu/${method}.log"
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_full30_gpu/status.log
  if .venv/Scripts/python.exe -m t5_ssl.train \
      data=oasis3_full50 trainer=full50 trainer.max_epochs=30 \
      trainer.deterministic="warn" data.train_cache_capacity=60 \
      +data.eval_cache_capacity=16 trainer.enable_progress_bar=false "$@" \
      >"$logfile" 2>&1; then
    local experiment
    experiment="$(tr '\\' '/' < "$logfile" | sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' | tail -1)"
    if [[ -n "$experiment" && -f "$experiment/checkpoints/last.ckpt" ]]; then
      printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" | tee -a campaign_full30_gpu/status.log
      return 0
    fi
  fi
  printf '%s FAILED %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" | tee -a campaign_full30_gpu/status.log
  return 1
}

result=0
run_method vicreg model.backbone.pretrained=true || result=1
run_method byol model=byol_resnet18 loss=byol || result=1
run_method dinov2 model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop || result=1
exit "$result"
