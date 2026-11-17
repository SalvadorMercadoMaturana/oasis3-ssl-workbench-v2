#!/usr/bin/env bash
# Single entry point on the pod: setup + preflight + launch.
#
#   bash cloud/bootstrap_runpod.sh [run_all.py args...]       # detached, log in runs/
#   BOOTSTRAP_FOREGROUND=1 bash cloud/bootstrap_runpod.sh ...  # stay attached
#   BOOTSTRAP_SKIP_SETUP=1 ...                           # already installed
#
# Examples:
#   bash cloud/bootstrap_runpod.sh --preset byol-screen --epochs 30 --seeds 1 2 3 --tag screen30
#   bash cloud/bootstrap_runpod.sh --parallel mps --epochs 50 --seeds 1 2 3 \
#        --byol-variant border_h3 --tag final50 --init-controls -- --spatial-probe
set -euo pipefail
cd "$(dirname "$0")/.."  # the repository
source cloud/env.sh

if [[ "${BOOTSTRAP_SKIP_SETUP:-0}" != "1" ]]; then
  bash cloud/setup.sh
fi
mkdir -p runs
python cloud/preflight.py --require-data --json runs/preflight.json

if [[ $# -eq 0 ]]; then
  echo "setup + preflight done; pass run_all.py arguments to launch training"
  exit 0
fi
mkdir -p runs
LOG="runs/run_all_$(date +%Y%m%d_%H%M%S).log"
if [[ "${BOOTSTRAP_FOREGROUND:-0}" == "1" ]]; then
  python cloud/run_all.py "$@" 2>&1 | tee "$LOG"
else
  nohup python cloud/run_all.py "$@" > "$LOG" 2>&1 &
  echo $! > runs/run_all.pid
  echo "run_all.py started (pid $(cat runs/run_all.pid)), log: $LOG"
fi
