"""Second augmentation screen: equal-strength artifacts, 10-epoch BYOL, robustness probe.

Changes from ``screen_augmentations.py``:

- Equal strength. Each tested set *replaces* the photometric part of the
  campaigns' augmentations (blur p=0.5, Gaussian noise 0.03, intensity x0.9-1.1)
  with one TorchIO artifact calibrated to the same mean RMSE per view
  (``calibrate``), so sets differ in the kind of degradation, not its size.
  Geometry is the same for every set (the campaign crop + left-right flip +
  +/-10 degree rotation); ``current`` keeps the campaign config unchanged.
- Short runs. 10 epochs on the campaign schedule compressed proportionally
  (LR warmup 10/50 -> 2/10 epochs), no checkpoints, two seeds per set, and an
  ``init`` row that skips SSL training altogether (ImageNet weights).
- Stabler score. The frozen encoder is scored in-process by 5x repeated
  stratified 5-fold CV of the logistic probe over every train+val AD/HC subject
  (~357 instead of 61 validation subjects). The test split is never touched.
- Hospital robustness. The probe is fit on clean embeddings and also predicts
  held-out subjects whose images carry each artifact at 2x the training
  strength (same corrupted images for every run), i.e. a model built on research
  scans applied to poorly acquired ones.

BYOL, not DINOv2: on the existing balanced-campaign checkpoints the CV probe of
DINOv2 stays at its random-init level (0.709 -> 0.708 at epoch 9 -> 0.712 at
epoch 49), so no augmentation could move it; BYOL does move (0.720 -> 0.694 -> 0.728).

Usage: ``calibrate`` once, then ``screen --workers 2``, then ``report``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from report_aug_screen import fast_mean3, mean3, scores
from screen_augmentations import MANIFEST, ROOT, to_hydra

CAMPAIGN = ROOT / "campaign_aug_screen2"
PY = str(ROOT / ".venv/Scripts/python.exe")
SEEDS = (1, 2)
IMAGE_SIZE = (96, 96)
MODEL_OVERRIDES = ["model=byol_resnet18", "loss=byol", "augmentation=ssl_multiview"]
COMMON = [
    "data=oasis3_full50", "trainer=full50", "trainer.max_epochs=10", "optimizer.warmup_epochs=2",
    "trainer.accelerator=gpu", "trainer.precision=32-true", "trainer.deterministic=false",
    "trainer.enable_progress_bar=false", "trainer.early_stopping_patience=null",
    "trainer.enable_checkpointing=false", "trainer.enable_health_check=false",
    "data.train_cache_capacity=80", "+data.eval_cache_capacity=16",
    f"data.manifest_path={MANIFEST}",
]
# The campaigns' photometric augmentations: the strength every artifact is matched to.
REFERENCE_PHOTOMETRIC = {"blur_p": 0.5, "noise_std": 0.03, "intensity_scale": [0.9, 1.1]}
NO_PHOTOMETRIC = {"blur_p": 0.0, "noise_std": 0.0, "intensity_scale": [1.0, 1.0]}
HOSPITAL_GEOMETRY = {"horizontal_flip_p": 0.5, "vertical_flip_p": 0.0, "affine_degrees": 10.0}
EVAL_SEVERITY = 2.0  # corrupted evaluation images: artifacts at 2x the training RMSE

# One strength knob per artifact (monotone in RMSE). Upper bounds keep the
# evaluation corruptions plausible (noise std <= 0.3, blur sigma <= 4 px).
KNOBS: dict[str, tuple[Any, float, float]] = {
    "noise": (lambda s: {"name": "RandomNoise", "std": [0, s]}, 1e-3, 0.3),
    "blur": (lambda s: {"name": "RandomBlur", "std": [0, s]}, 1e-2, 4.0),
    "gamma": (lambda s: {"name": "RandomGamma", "log_gamma": [-s, s]}, 1e-3, 2.0),
    "bias": (lambda s: {"name": "RandomBiasField", "coefficients": s, "order": 2}, 1e-3, 2.0),
    "motion": (lambda s: {"name": "RandomMotion", "degrees": s, "translation": s,
                          "num_transforms": 2}, 1e-2, 30.0),
    "ghosting": (lambda s: {"name": "RandomGhosting", "num_ghosts": [2, 6], "axes": [0, 1],
                            "intensity": [0, s]}, 1e-3, 3.0),
    "anisotropy": (lambda s: {"name": "RandomAnisotropy", "axes": [0, 1],
                              "downsampling": [1, s]}, 1.01, 10.0),
}
MIXES = {"intensity_mix": ("noise", "gamma", "bias"),
         "acquisition_mix": ("motion", "ghosting", "anisotropy", "blur")}
SET_NAMES = ["init", "current", "ref", "none", *KNOBS, *MIXES]
# Artifacts each set trained on, excluded from its "held-out" robustness score.
OWN_ARTIFACTS = {**{name: (name,) for name in KNOBS}, **MIXES,
                 "current": ("noise", "blur"), "ref": ("noise", "blur")}


# ---------------------------------------------------------------- calibration
def photometric_view(photometric: dict[str, Any], torchio: list[dict] | None) -> Any:
    """Build a view transform with the photometric part only (full field, no geometry)."""
    from t5_ssl.data.augmentations import SSLViewTransform
    return SSLViewTransform(image_size=IMAGE_SIZE, crop_scale=(1.0, 1.0),
                            horizontal_flip_p=0.0, vertical_flip_p=0.0, affine_degrees=0.0,
                            blur_p=photometric["blur_p"], noise_std=photometric["noise_std"],
                            intensity_scale=tuple(photometric["intensity_scale"]),
                            torchio_transforms=torchio)


def calibration_images(n: int = 48) -> list[torch.Tensor]:
    from t5_ssl.data.oasis3_lmdb_datamodule import OASIS3LMDBSliceDataset, select_rows
    from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_LMDB
    selected, _ = select_rows(DEFAULT_CSV, ROOT / MANIFEST, False, 42, 0.7, 0.15)
    rows = random.Random(0).sample(selected["train"], n)
    dataset = OASIS3LMDBSliceDataset(rows, DEFAULT_LMDB, slice_range=(0.3, 0.7))
    return [dataset[i]["image"].float() for i in range(n)]


def mean_rmse(images: list[torch.Tensor], photometric: dict[str, Any],
              torchio: list[dict] | None, draws: int = 2) -> float:
    """Mean RMSE between a photometric-only view and the clean resized slice."""
    from t5_ssl.data.augmentations import EvalTransform
    view, clean = photometric_view(photometric, torchio), EvalTransform(IMAGE_SIZE)
    random.seed(0), np.random.seed(0), torch.manual_seed(0)  # common random numbers
    return float(np.mean([float((view._make_view(img) - clean(img)).pow(2).mean().sqrt())
                          for img in images for _ in range(draws)]))


def solve_knob(images: list[torch.Tensor], make: Any, low: float, high: float,
               target: float) -> tuple[float, float]:
    """Bisect log(knob) until the artifact alone reaches `target` mean RMSE."""
    def rmse(s: float) -> float:
        return mean_rmse(images, NO_PHOTOMETRIC, [{**make(s), "p": 1.0}])
    if rmse(high) < target:  # unreachable: use the strongest allowed setting
        return high, rmse(high)
    lo, hi = np.log(low), np.log(high)
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rmse(float(np.exp(mid))) < target else (lo, mid)
    knob = float(np.exp((lo + hi) / 2))
    return knob, rmse(knob)


def calibrate() -> None:
    CAMPAIGN.mkdir(exist_ok=True)
    images = calibration_images()
    target = mean_rmse(images, REFERENCE_PHOTOMETRIC, None)
    result: dict[str, Any] = {"target_rmse": target, "eval_severity": EVAL_SEVERITY,
                              "artifacts": {}}
    print(f"target RMSE (campaign photometric) = {target:.4f}", flush=True)
    for name, (make, low, high) in KNOBS.items():
        knob, achieved = solve_knob(images, make, low, high, target)
        eval_knob, eval_rmse = solve_knob(images, make, low, high, EVAL_SEVERITY * target)
        result["artifacts"][name] = {"knob": knob, "rmse": achieved, "spec": make(knob),
                                     "eval_knob": eval_knob, "eval_rmse": eval_rmse,
                                     "eval_spec": make(eval_knob)}
        print(f"{name:11s} train knob={knob:8.4f} rmse={achieved:.4f} | "
              f"eval knob={eval_knob:8.4f} rmse={eval_rmse:.4f}", flush=True)
    (CAMPAIGN / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    preview(images, result)


def preview(images: list[torch.Tensor], calib: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from t5_ssl.data.augmentations import EvalTransform
    columns = ["original", "ref (campaña)", *calib["artifacts"]]
    row_labels = ["entrenamiento (1×)", "entrenamiento (1×)",
                  f"evaluación ({calib['eval_severity']:g}×)"]
    fig, axes = plt.subplots(3, len(columns), figsize=(1.9 * len(columns), 6.4))
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    for c, name in enumerate(columns):
        for r in range(3):
            img = images[r]
            if name == "original":
                out = EvalTransform(IMAGE_SIZE)(img)
            elif name.startswith("ref"):
                out = photometric_view(REFERENCE_PHOTOMETRIC, None)._make_view(img)
            else:
                info = calib["artifacts"][name]
                spec = info["eval_spec"] if r == 2 else info["spec"]
                out = photometric_view(NO_PHOTOMETRIC, [{**spec, "p": 1.0}])._make_view(img)
            axes[r, c].imshow(out[0].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=8)
        axes[0, c].set_title(name, fontsize=9)
    fig.suptitle("Artefactos calibrados: filas 1-2 a la RMSE media de las augmentations "
                 f"fotométricas de la campaña ({calib['target_rmse']:.3f}); fila 3 a "
                 f"{calib['eval_severity']:g}× para evaluar robustez", fontsize=10)
    fig.tight_layout()
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    fig.savefig(CAMPAIGN / "figures" / "calibrated_examples.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- one run
def aug_overrides(name: str, calib: dict) -> list[str]:
    if name in ("current", "init"):
        return []
    items: dict[str, Any] = dict(HOSPITAL_GEOMETRY)
    items.update(REFERENCE_PHOTOMETRIC if name == "ref" else NO_PHOTOMETRIC)
    specs: list[dict] = []
    if name in KNOBS:
        specs = [{**calib["artifacts"][name]["spec"], "p": 1.0}]
    elif name in MIXES:
        specs = [{"name": "OneOf", "p": 1.0,
                  "transforms": [calib["artifacts"][a]["spec"] for a in MIXES[name]]}]
    items["torchio"] = specs
    return [f"++augmentation.{key}={to_hydra(value)}" for key, value in items.items()]


class CorruptedEval:
    """Deterministic resize of one artifact-corrupted, full-field slice."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.view = photometric_view(NO_PHOTOMETRIC, [{**spec, "p": 1.0}])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.view._make_view(image)


def embed(backbone: torch.nn.Module, dm: Any, transform: Any
          ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mean central-slice embedding per unambiguous AD/HC subject of train+val."""
    from torch.utils.data import DataLoader, Subset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device).eval()
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "val"):
        dataset = dm.eval_dataset(split, transform)
        eligible = [i for i, (record, _) in enumerate(dataset.indices)
                    if dataset.records[record]["subject"] in dm.subject_labels]
        loader = DataLoader(Subset(dataset, eligible), batch_size=64, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                out = backbone(batch["image"].to(device)).float().cpu().numpy()
                for subject, vector in zip(batch["subject"], out, strict=True):
                    features[subject].append(vector)
    subjects = sorted(features)
    x = np.stack([np.mean(features[s], axis=0) for s in subjects])
    y = np.array([int(dm.subject_labels[s] == "AD") for s in subjects])
    return x, y, subjects


def cv_probe(x: np.ndarray, y: np.ndarray, variants: dict[str, np.ndarray],
             repeats: int = 5, folds: int = 5) -> dict[str, dict]:
    """Repeated stratified K-fold probe fit on clean `x`, scored on every variant."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = {name: np.zeros((repeats, len(y))) for name in variants}
    for repeat in range(repeats):
        for train_idx, test_idx in StratifiedKFold(folds, shuffle=True,
                                                   random_state=repeat).split(x, y):
            probe = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0], cv=3, scoring="roc_auc",
                class_weight="balanced", max_iter=2000))
            probe.fit(x[train_idx], y[train_idx])
            for name, values in variants.items():
                oof[name][repeat, test_idx] = probe.predict_proba(values[test_idx])[:, 1]
    out = {}
    for name, predictions in oof.items():
        per_repeat = [scores(y, predictions[r]) for r in range(repeats)]
        out[name] = {"metrics": {k: float(np.mean([m[k] for m in per_repeat]))
                                 for k in per_repeat[0]},
                     "oof_mean": predictions.mean(axis=0).tolist()}
    return out


def run_one(name: str, seed: int) -> None:
    from hydra import compose, initialize_config_dir

    from t5_ssl.data.augmentations import EvalTransform
    from t5_ssl.train import build_backbone, build_datamodule, train
    calib = json.loads((CAMPAIGN / "calibration.json").read_text(encoding="utf-8"))
    overrides = [*COMMON, *MODEL_OVERRIDES, f"seed={seed}", *aug_overrides(name, calib)]
    start = time.time()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    if name == "init":  # no SSL: the ImageNet-initialised encoder as-is
        torch.manual_seed(seed)
        backbone = build_backbone(cfg)
        dm = build_datamodule(cfg, EvalTransform(IMAGE_SIZE), EvalTransform(IMAGE_SIZE))
        dm.setup()
        exp_dir, val_loss = None, None
    else:
        result = train(cfg)
        backbone, dm, exp_dir = result["model"].backbone, result["datamodule"], result["exp_dir"]
        val_loss = result["trainer"].callback_metrics.get("val/loss")
    x, y, subjects = embed(backbone, dm, EvalTransform(IMAGE_SIZE))
    variants = {"clean": x}
    for artifact, info in calib["artifacts"].items():
        random.seed(0), np.random.seed(0), torch.manual_seed(0)  # same corruption every run
        x_art, _, art_subjects = embed(backbone, dm, CorruptedEval(info["eval_spec"]))
        if art_subjects != subjects:
            raise RuntimeError("Corrupted embeddings cover different subjects")
        variants[artifact] = x_art
    probe = cv_probe(x, y, variants)
    out = {"name": name, "seed": seed, "overrides": overrides,
           "experiment_dir": str(exp_dir) if exp_dir else None,
           "minutes": (time.time() - start) / 60,
           "ssl_val_loss_final": float(val_loss) if val_loss is not None else None,
           "subjects": subjects, "labels": y.tolist(), "probe": probe}
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "runs" / f"{name}_s{seed}.json").write_text(json.dumps(out) + "\n")


# ---------------------------------------------------------------- orchestration
def robust_mean(probe: dict, exclude: tuple[str, ...] = ()) -> float:
    names = [n for n in probe if n != "clean" and n not in exclude]
    return float(np.mean([mean3(probe[n]["metrics"]) for n in names]))


def screen(workers: int, names: list[str]) -> None:
    (CAMPAIGN / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(name, seed) for seed in SEEDS for name in names
            if not (name == "init" and seed != SEEDS[0])
            and not (CAMPAIGN / "runs" / f"{name}_s{seed}.json").is_file()]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}

    def launch(job: tuple[str, int]) -> None:
        name, seed = job
        start = time.time()
        with (CAMPAIGN / "logs" / f"{name}_s{seed}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run([PY, __file__, "run", name, str(seed)], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        path = CAMPAIGN / "runs" / f"{name}_s{seed}.json"
        detail = "FAILED"
        if code == 0 and path.is_file():
            probe = json.loads(path.read_text(encoding="utf-8"))["probe"]
            detail = (f"OK clean={mean3(probe['clean']['metrics']):.3f} "
                      f"hospital={robust_mean(probe):.3f}")
        print(f"{time.strftime('%H:%M:%S')} {name} s{seed} {detail} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, todo))


# ---------------------------------------------------------------- report
def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    calib = json.loads((CAMPAIGN / "calibration.json").read_text(encoding="utf-8"))
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((CAMPAIGN / "runs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        runs[doc["name"]].append(doc)
    if "ref" not in runs:
        raise RuntimeError("The reference set 'ref' has no finished runs")
    labels = np.array(runs["ref"][0]["labels"])
    subjects = runs["ref"][0]["subjects"]
    if any(d["subjects"] != subjects for docs in runs.values() for d in docs):
        raise RuntimeError("Runs were scored on different subjects")
    artifacts = list(calib["artifacts"])

    def hospital(doc: dict, idx: np.ndarray, exclude: tuple[str, ...] = ()) -> float:
        return float(np.mean([fast_mean3(labels[idx], np.asarray(doc["probe"][a]["oof_mean"])[idx])
                              for a in artifacts if a not in exclude]))

    def clean(doc: dict, idx: np.ndarray) -> float:
        return fast_mean3(labels[idx], np.asarray(doc["probe"]["clean"]["oof_mean"])[idx])

    rng = np.random.default_rng(42)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    boot = [np.concatenate((rng.choice(positive, len(positive)), rng.choice(negative, len(negative))))
            for _ in range(500)]
    rows = []
    for name, docs in runs.items():
        exclude = OWN_ARTIFACTS.get(name, ())

        row: dict[str, Any] = {"name": name, "n_seeds": len(docs),
                               "minutes": float(np.mean([d["minutes"] for d in docs]))}
        for key in ("auc", "f1", "aupr"):
            row[f"clean_{key}"] = float(np.mean([d["probe"]["clean"]["metrics"][key] for d in docs]))
            row[f"hosp_{key}"] = float(np.mean([np.mean([d["probe"][a]["metrics"][key]
                                                         for a in artifacts]) for d in docs]))
        row["clean"] = float(np.mean([mean3(d["probe"]["clean"]["metrics"]) for d in docs]))
        row["hospital"] = float(np.mean([robust_mean(d["probe"]) for d in docs]))
        row["heldout"] = float(np.mean([robust_mean(d["probe"], exclude) for d in docs]))
        row["hospital_seeds"] = [robust_mean(d["probe"]) for d in docs]
        row["clean_seeds"] = [mean3(d["probe"]["clean"]["metrics"]) for d in docs]
        for label, fn in (("hospital", hospital), ("clean", clean)):
            deltas = np.array([np.mean([fn(d, idx) for d in docs])
                               - np.mean([fn(d, idx) for d in runs["ref"]]) for idx in boot])
            row[f"{label}_delta"] = row[label] - float(np.mean(
                [robust_mean(d["probe"]) if label == "hospital" else mean3(d["probe"]["clean"]["metrics"])
                 for d in runs["ref"]]))
            row[f"{label}_ci"] = np.percentile(deltas, [2.5, 97.5]).tolist()
            row[f"{label}_p"] = float(np.mean(deltas > 0))
        row["per_artifact"] = {a: float(np.mean([mean3(d["probe"][a]["metrics"]) for d in docs]))
                               for a in artifacts}
        rows.append(row)
    rows.sort(key=lambda r: r["hospital"], reverse=True)
    spread = {key: float(np.mean([abs(r[f"{key}_seeds"][0] - r[f"{key}_seeds"][1])
                                  for r in rows if r["n_seeds"] == 2] or [np.nan]))
              for key in ("hospital", "clean")}
    (CAMPAIGN / "ranking.json").write_text(json.dumps(rows, indent=2) + "\n")

    # Figure: clean vs hospital-degraded score per set; seeds small, mean large.
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    order = rows[::-1]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.38 * len(order) + 1.7), sharey=True)
    ref_row = next(r for r in rows if r["name"] == "ref")
    for ax, key, title in ((axes[0], "hospital", f"Imágenes con artefactos ({EVAL_SEVERITY:g}×)"),
                           (axes[1], "clean", "Imágenes limpias")):
        ax.axvline(ref_row[key], color="#52514e", linewidth=1, linestyle=":")
        for i, row in enumerate(order):
            ax.scatter(row[f"{key}_seeds"], [i] * len(row[f"{key}_seeds"]), s=14,
                       color="#2a78d6", alpha=0.45, zorder=2)
            ax.scatter([row[key]], [i], s=46, color="#2a78d6", zorder=3)
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("Media de AUROC, F1-macro y AUPR (CV)", fontsize=9, color="#52514e")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(colors="#52514e", labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(y, [r["name"] for r in order], color="#0b0b0b")
    fig.suptitle("Screen 2 (BYOL, 10 épocas, artefactos a igual intensidad). "
                 "Punto grande = media de semillas; línea punteada = ref.", fontsize=11)
    fig.tight_layout()
    fig.savefig(CAMPAIGN / "figures" / "screen_ranking.png", dpi=160)
    plt.close(fig)

    def fmt_delta(row: dict, key: str) -> tuple[str, str]:
        if row["name"] == "ref":
            return "—", "—"
        lo, hi = row[f"{key}_ci"]
        return f"{row[f'{key}_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{row[f'{key}_p']:.2f}"

    lines = [
        "# Screen 2 de augmentations: artefactos a igual intensidad (BYOL, 10 épocas)", "",
        "BYOL con ResNet-18 ImageNet1K, 10 épocas con el calendario de la campaña comprimido "
        "proporcionalmente (warmup del LR 2 de 10 épocas), split balanceado, "
        f"{len(SEEDS)} semillas por set, sin checkpoints. `init` es el encoder con pesos de "
        "ImageNet sin entrenamiento SSL. Se descartó DINOv2 porque en los checkpoints de la "
        "campaña balanceada su probe no se mueve del nivel de pesos aleatorios (0,709 → 0,708 "
        "en la época 9 → 0,712 en la 49), así que ninguna augmentation podría notarse.", "",
        "**Igual intensidad.** Cada set reemplaza la parte fotométrica de las augmentations de "
        "la campaña (blur p=0,5, ruido 0,03, intensidad ×0,9–1,1) por un artefacto de TorchIO "
        f"calibrado a la misma RMSE media por vista ({calib['target_rmse']:.3f}; "
        "[ejemplos](figures/calibrated_examples.png)). La geometría es igual en todos: el crop "
        "de la campaña + flip izquierda-derecha + rotación ±10°. `ref` usa esa geometría con la "
        "fotometría de la campaña, `none` sin fotometría, y `current` es la config de la campaña "
        "sin cambios (flip vertical y rotación ±15°).", "",
        f"**Evaluación.** Encoder congelado; probe logístico con validación cruzada estratificada "
        f"5×5 sobre los {len(labels)} sujetos AD/HC de train+val ({int(labels.sum())} AD); el "
        "test no se usa. *Limpias*: sujetos de validación con sus imágenes originales. *Con "
        "artefactos*: el mismo probe (ajustado con imágenes limpias) sobre las imágenes de los "
        f"sujetos de validación con cada artefacto a {EVAL_SEVERITY:g}× la intensidad de "
        "entrenamiento, promediado sobre los 7 artefactos; son las mismas imágenes corruptas en "
        "todas las corridas. *Sin el propio* excluye de ese promedio los artefactos con los que "
        "entrenó el set, para no premiarlo por haber visto exactamente la misma corrupción. "
        "Δ vs `ref`: bootstrap sobre sujetos promediando semillas.", "",
        f"Ruido entre semillas (|semilla 1 − semilla 2| medio): {spread['hospital']:.3f} con "
        f"artefactos, {spread['clean']:.3f} en limpias. Diferencias menores que eso no son "
        "confiables.", "",
        "![Ranking](figures/screen_ranking.png)", "",
        "## Ranking por desempeño con artefactos (media de AUROC, F1-macro y AUPR)", "",
        "| # | Set | Con artefactos | Sin el propio | Δ vs ref [IC 95 %] | P(Δ>0) | Limpias | "
        "Δ limpias vs ref [IC 95 %] | P(Δ>0) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, start=1):
        hosp_delta, hosp_p = fmt_delta(row, "hospital")
        clean_delta, clean_p = fmt_delta(row, "clean")
        lines.append(f"| {rank} | `{row['name']}` | **{row['hospital']:.3f}** | "
                     f"{row['heldout']:.3f} | {hosp_delta} | {hosp_p} | {row['clean']:.3f} | "
                     f"{clean_delta} | {clean_p} |")
    lines += ["", "## Métricas por separado", "",
              "| Set | AUROC art. | F1 art. | AUPR art. | AUROC limpias | F1 limpias | "
              "AUPR limpias | Semillas (art.) | Min/corrida |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        lines.append(f"| `{row['name']}` | {row['hosp_auc']:.3f} | {row['hosp_f1']:.3f} | "
                     f"{row['hosp_aupr']:.3f} | {row['clean_auc']:.3f} | {row['clean_f1']:.3f} | "
                     f"{row['clean_aupr']:.3f} | "
                     f"{', '.join(f'{v:.3f}' for v in row['hospital_seeds'])} | "
                     f"{row['minutes']:.1f} |")
    lines += ["", "## Desempeño por artefacto de evaluación (media de las 3 métricas)", "",
              "| Set | " + " | ".join(artifacts) + " |",
              "|---|" + "---:|" * len(artifacts)]
    for row in rows:
        lines.append(f"| `{row['name']}` | "
                     + " | ".join(f"{row['per_artifact'][a]:.3f}" for a in artifacts) + " |")
    lines += ["", "## Calibración", "",
              "| Artefacto | Parámetro TorchIO (entrenamiento) | RMSE | RMSE evaluación |",
              "|---|---|---:|---:|",
              f"| ref (campaña) | blur p=0,5 + ruido 0,03 + ×0,9–1,1 | {calib['target_rmse']:.4f} | — |"]
    for name, info in calib["artifacts"].items():
        spec = {k: v for k, v in info["spec"].items() if k != "name"}
        lines.append(f"| {name} | `{info['spec']['name']}` {json.dumps(spec)} | "
                     f"{info['rmse']:.4f} | {info['eval_rmse']:.4f} |")
    lines += ["", "Mixes: `intensity_mix` = OneOf(noise, gamma, bias); `acquisition_mix` = "
              "OneOf(motion, ghosting, anisotropy, blur), cada hijo con su calibración.", "",
              "Regenerar: `python scripts/screen_aug_equal.py report`.", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {CAMPAIGN / 'RESULTADOS.md'} ({sum(len(d) for d in runs.values())} runs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    run = sub.add_parser("run")
    run.add_argument("name", choices=SET_NAMES)
    run.add_argument("seed", type=int)
    scr = sub.add_parser("screen")
    scr.add_argument("names", nargs="*", default=SET_NAMES)
    scr.add_argument("--workers", type=int, default=2)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate()
    elif args.command == "run":
        run_one(args.name, args.seed)
    elif args.command == "screen":
        screen(args.workers, args.names)
    else:
        report()


if __name__ == "__main__":
    sys.exit(main())
