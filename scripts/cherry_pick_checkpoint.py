"""Cherry-pick the best periodic SSL checkpoint for one experiment.

Motivated by the anti-collapse principle established for this project: never
select a model by SSL loss alone. This scans every periodic checkpoint saved
during training (``checkpoints/ckpt-epoch=NN.ckpt``), scores each one with the
frozen probe in ``--validation-only`` mode (so the held-out test set is never
touched during selection), reports the nearest anti-collapse health-check
record for context, and picks the epoch with the best validation AUROC. The
final test-set numbers are then computed once, for that epoch only, exactly
like the report's own "evaluate several epochs, pick the one before collapse"
approach for BYOL (Anexo F / section 4.1.1).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_RE = re.compile(r"^ckpt-epoch=(\d+)\.ckpt$")


def available_epochs(experiment_dir: Path) -> list[int]:
    epochs = []
    for path in (experiment_dir / "checkpoints").glob("ckpt-epoch=*.ckpt"):
        match = CKPT_RE.match(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def nearest_health_record(experiment_dir: Path, epoch: int) -> dict | None:
    health_path = experiment_dir / "representation_health.json"
    if not health_path.is_file():
        return None
    records = json.loads(health_path.read_text(encoding="utf-8"))
    if not records:
        return None
    return min(records, key=lambda r: abs(r["epoch"] - epoch))


def run_probe(experiment_dir: Path, epoch: int, *, validation_only: bool,
             batch_size: int) -> dict:
    command = [sys.executable, str(ROOT / "scripts/evaluate_frozen_probe.py"),
              str(experiment_dir), "--checkpoint-epoch", str(epoch),
              "--batch-size", str(batch_size)]
    if validation_only:
        command.append("--validation-only")
    result = subprocess.run(command, cwd=ROOT, check=True,
                            capture_output=True, text=True)
    # torch/lightning warnings can land on stdout ahead of the script's own
    # trailing JSON print; find where the JSON object actually starts.
    start = result.stdout.index("{")
    return json.loads(result.stdout[start:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    epochs = available_epochs(experiment_dir)
    if not epochs:
        raise RuntimeError(f"No periodic checkpoints found under {experiment_dir}/checkpoints")

    candidates = []
    for epoch in epochs:
        doc = run_probe(experiment_dir, epoch, validation_only=True, batch_size=args.batch_size)
        val_auc = doc["metrics"]["trained"]["val_auc"]
        health = nearest_health_record(experiment_dir, epoch)
        candidates.append({
            "epoch": epoch, "val_auc": val_auc,
            "health_epoch": health.get("epoch") if health else None,
            "effective_rank": health.get("effective_rank") if health else None,
            "mean_feature_std": health.get("mean_feature_std") if health else None,
            "mean_cosine_similarity_diff_samples":
                health.get("mean_cosine_similarity_diff_samples") if health else None,
            "ad_hc_centroid_distance": health.get("ad_hc_centroid_distance") if health else None,
        })
        print(f"epoch={epoch:02d} val_auc={val_auc:.4f} "
              f"(health@{candidates[-1]['health_epoch']}: "
              f"rank={candidates[-1]['effective_rank']}, "
              f"cos={candidates[-1]['mean_cosine_similarity_diff_samples']}, "
              f"centroid_dist={candidates[-1]['ad_hc_centroid_distance']})", flush=True)

    best = max(candidates, key=lambda c: c["val_auc"])
    print(f"\nSelected epoch={best['epoch']:02d} (val_auc={best['val_auc']:.4f})", flush=True)

    final = run_probe(experiment_dir, best["epoch"], validation_only=False,
                      batch_size=args.batch_size)
    summary = {"experiment_dir": str(experiment_dir), "candidates": candidates,
              "selected_epoch": best["epoch"], "selection_criterion": "max val_auc "
              "(validation-only probe; test set untouched during selection)",
              "final_test_metrics": final["metrics"], "final_comparison": final["comparison"]}
    (experiment_dir / "cherry_pick_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
