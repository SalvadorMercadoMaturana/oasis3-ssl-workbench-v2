"""Screen TorchIO acquisition-artifact augmentations with short BYOL runs.

Each experiment trains BYOL/ResNet-18 (the fastest of the three methods) for 30
epochs on the balanced split with one augmentation set, probes its best
val/loss checkpoint with ``evaluate_frozen_probe.py``, and appends one row to
``<campaign>/results.csv``. Sets are ranked on *validation* AUROC, F1-macro and
AUPR; test metrics are recorded for reference only, so the choice is not tuned
on the 56-subject test set. Finished experiments are skipped on rerun.

Every set except ``current`` starts from ``configs/augmentation/hospital_base.yaml``
(SSL crop + left-right flip + +/-10 degree rotation); ``current`` is the
unchanged ``ssl_multiview`` config used by the campaigns, as a reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/Scripts/python.exe")
CAMPAIGN = ROOT / "campaign_aug_screen"
MANIFEST = "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
COMMON = [
    "data=oasis3_full50", "trainer=full50", "trainer.max_epochs=30",
    "trainer.accelerator=gpu", "trainer.precision=32-true", "trainer.deterministic=false",
    "trainer.enable_progress_bar=false", "trainer.early_stopping_patience=null",
    "data.train_cache_capacity=80", "+data.eval_cache_capacity=16",
    f"data.manifest_path={MANIFEST}", "model=byol_resnet18", "loss=byol",
    # Screening only needs the probed checkpoint: skip periodic snapshots
    # (best-loss + last are still written) and the 5-epoch health report.
    "trainer.checkpoint_every_n_epochs=1000", "trainer.enable_health_check=false",
]

# Single acquisition artifacts; "p" is added per experiment. Parameters are in
# pixels of the ~2 mm slice (97x115), applied before the 96x96 crop/resize.
MOTION = {"name": "RandomMotion", "degrees": 5, "translation": 5, "num_transforms": 2}
GHOSTING = {"name": "RandomGhosting", "num_ghosts": [2, 6], "axes": [0, 1],
            "intensity": [0.2, 0.6]}
SPIKE = {"name": "RandomSpike", "num_spikes": 1, "intensity": [0.6, 1.2]}
BIAS = {"name": "RandomBiasField", "coefficients": 0.15, "order": 2}
NOISE = {"name": "RandomNoise", "std": [0, 0.04]}
BLUR = {"name": "RandomBlur", "std": [0, 1.5]}
GAMMA = {"name": "RandomGamma", "log_gamma": [-0.3, 0.3]}
ANISOTROPY = {"name": "RandomAnisotropy", "axes": [0, 1], "downsampling": [1.5, 3]}
ELASTIC = {"name": "RandomElasticDeformation", "num_control_points": 7,
           "max_displacement": [6, 6, 0]}


def p(spec: dict[str, Any], prob: float) -> dict[str, Any]:
    return {**spec, "p": prob}


# name -> TorchIO specs on top of hospital_base (None = the campaigns' ssl_multiview).
EXPERIMENTS: dict[str, list[dict[str, Any]] | None] = {
    "current": None,
    "base": [],
    "motion": [p(MOTION, 0.5)],
    "ghosting": [p(GHOSTING, 0.5)],
    "bias": [p(BIAS, 0.5)],
    "noise": [p(NOISE, 0.5)],
    "blur": [p(BLUR, 0.5)],
    "gamma": [p(GAMMA, 0.5)],
    "anisotropy": [p(ANISOTROPY, 0.5)],
    "spike": [p(SPIKE, 0.3)],
    "elastic": [p(ELASTIC, 0.3)],
    # Scanner/hardware: intensity inhomogeneity, contrast and thermal noise.
    "scanner": [p(BIAS, 0.5), p(GAMMA, 0.5), p(NOISE, 0.5)],
    # Uncooperative patient: movement during the acquisition.
    "patient": [p(MOTION, 0.5), p(GHOSTING, 0.5)],
    # Fast/low-resolution protocols.
    "lowres": [p(ANISOTROPY, 0.5), p(BLUR, 0.5)],
    # One acquisition artifact at a time, plus scanner intensity variation.
    "hospital_oneof": [
        {"name": "OneOf", "p": 0.7,
         "transforms": [MOTION, GHOSTING, SPIKE, ANISOTROPY, BLUR]},
        p(BIAS, 0.5), p(GAMMA, 0.5), p(NOISE, 0.5),
    ],
    # Everything, each rarely.
    "hospital_all": [p(MOTION, 0.25), p(GHOSTING, 0.25), p(SPIKE, 0.15),
                     p(ANISOTROPY, 0.25), p(BLUR, 0.25), p(BIAS, 0.25),
                     p(GAMMA, 0.25), p(NOISE, 0.25), p(ELASTIC, 0.15)],
}

FIELDS = ["name", "status", "experiment_dir", "minutes", "ssl_val_best",
          "val_auc", "val_f1_macro", "val_aupr", "val_mean",
          "test_auc", "test_f1_macro", "test_aupr",
          "random_val_auc", "random_test_auc", "specs"]


def to_hydra(value: Any) -> str:
    """Serialize nested lists/dicts into Hydra's override grammar."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{to_hydra(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(to_hydra(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def overrides(name: str) -> list[str]:
    specs = EXPERIMENTS[name]
    if specs is None:
        return ["augmentation=ssl_multiview"]
    return ["augmentation=hospital_base", f"augmentation.torchio={to_hydra(specs)}"]


def done_names(results: Path) -> set[str]:
    if not results.is_file():
        return set()
    with results.open(newline="", encoding="utf-8") as stream:
        return {row["name"] for row in csv.DictReader(stream) if row["status"] == "ok"}


def run(name: str, results: Path, lock: Lock) -> None:
    log_dir = CAMPAIGN / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}
    row: dict[str, Any] = {"name": name, "status": "failed",
                           "specs": json.dumps(EXPERIMENTS[name])}
    start = time.time()
    train_log = log_dir / f"{name}.log"
    with train_log.open("w", encoding="utf-8") as stream:
        code = subprocess.run([PY, "-m", "t5_ssl.train", *COMMON, *overrides(name)],
                              cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    match = re.findall(r"Artifacts saved in: (\S+)", train_log.read_text(encoding="utf-8"))
    if code == 0 and match:
        exp_dir = ROOT / match[-1].replace("\\", "/")
        row["experiment_dir"] = str(exp_dir.relative_to(ROOT))
        with (log_dir / f"{name}_probe.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run([PY, str(ROOT / "scripts/evaluate_frozen_probe.py"),
                                   str(exp_dir.relative_to(ROOT)), "--checkpoint", "best"],
                                  cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        probe = exp_dir / "ad_hc_subject_probe_best.json"
        if code == 0 and probe.is_file():
            doc = json.loads(probe.read_text(encoding="utf-8"))["results"]
            trained, random = doc["trained"], doc["random"]
            with (exp_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                val_losses = [float(r["val/loss"]) for r in csv.DictReader(stream)
                              if r.get("val/loss")]
            row.update(
                status="ok", ssl_val_best=min(val_losses),
                val_auc=trained["val_auc"], val_f1_macro=trained["val_f1_macro"],
                val_aupr=trained["val_average_precision"],
                val_mean=(trained["val_auc"] + trained["val_f1_macro"]
                          + trained["val_average_precision"]) / 3,
                test_auc=trained["test_auc"], test_f1_macro=trained["test_f1_macro"],
                test_aupr=trained["test_average_precision"],
                random_val_auc=random["val_auc"], random_test_auc=random["test_auc"])
    row["minutes"] = round((time.time() - start) / 60, 1)
    with lock:
        new_file = not results.is_file()
        with results.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    print(f"{time.strftime('%H:%M:%S')} {row['status'].upper()} {name} "
          f"({row['minutes']} min) val_mean={row.get('val_mean', float('nan')):.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Experiments to run (default: all, in order)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent trainings sharing the GPU")
    args = parser.parse_args()
    unknown = set(args.names) - set(EXPERIMENTS)
    if unknown:
        sys.exit(f"Unknown experiments: {sorted(unknown)}")
    results = CAMPAIGN / "results.csv"
    CAMPAIGN.mkdir(exist_ok=True)
    finished = done_names(results)
    todo = [name for name in (args.names or EXPERIMENTS) if name not in finished]
    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(run, name, results, lock) for name in todo]:
            future.result()


if __name__ == "__main__":
    main()
