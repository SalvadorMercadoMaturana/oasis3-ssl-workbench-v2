"""Training curves of one run: the epoch-level series of Lightning's ``metrics.csv``.

Moved from ``scripts/legacy/run_ssl3d_v2.py`` (epoch_metrics).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def epoch_metrics(exp_dir: Path | str) -> dict[str, list[float]]:
    """Per-epoch series of the epoch-level metrics logged during training."""
    series: dict[str, dict[int, float]] = defaultdict(dict)
    with (Path(exp_dir) / "metrics.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value not in ("", None) and key not in ("epoch", "step", "train/loss_step"):
                    series[key][int(row["epoch"])] = float(value)
    return {key: [values[e] for e in sorted(values)] for key, values in series.items()}
