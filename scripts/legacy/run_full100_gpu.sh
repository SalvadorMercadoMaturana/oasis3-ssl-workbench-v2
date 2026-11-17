#!/usr/bin/env bash
# 100-epoch GPU campaign: same 981-subject OASIS3 split and manifest as
# campaign_full50, but with the backbones that came out ahead in the
# 2026-09-20 exploratory session (VICReg = ConvNeXt-tiny + ImageNet
# pretraining + projector dropout 0.1, BYOL = ResNet-18 + ImageNet
# pretraining) and DINOv2 unchanged (ViT-Small, patch 60). Meant to run on a
# single rented GPU (e.g. one RTX 4090): the three methods launch as three
# concurrent processes sharing that one device, exactly like the CPU
# campaigns before it, just faster.
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
  printf '%s START %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_full100_gpu/status.log
  # train_cache_capacity=500 caps each process's in-RAM LMDB cache at
  # ~2.1GB (500 volumes * ~4.2MB); with 3 concurrent processes that is
  # ~6.3GB total, safe even on a leaner RunPod RAM tier. Raise it (or drop
  # the override to use the config default, 1300) if the pod has plenty of
  # spare RAM and you want fewer repeat LMDB reads across epochs.
  if OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/t5-ssl train \
      data=oasis3_full50 trainer=full100 data.train_cache_capacity=500 \
      trainer.enable_progress_bar=false "$@" >"$logfile" 2>&1; then
    local experiment
    experiment="$(sed -n 's/.*Artifacts saved in: \(logs\/[^ ]*\).*/\1/p' "$logfile" | tail -1)"
    if [[ -n "$experiment" && -f "$experiment/checkpoints/last.ckpt" ]]; then
      printf '%s COMPLETE %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" "$experiment" >> campaign_full100_gpu/status.log
      return 0
    fi
  fi
  printf '%s FAILED %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$method" >> campaign_full100_gpu/status.log
  return 1
}

# All three attach to the same (only) GPU on a single-4090 pod. Uncomment to
# pin explicitly if the pod ever exposes more than one device:
# export CUDA_VISIBLE_DEVICES=0

run_method vicreg model.backbone.pretrained=true & p1=$!
run_method byol model=byol_resnet18 loss=byol & p2=$!
run_method dinov2 model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop & p3=$!
result=0
wait "$p1" || result=1
wait "$p2" || result=1
wait "$p3" || result=1
exit "$result"
