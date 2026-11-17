"""3D hippocampal-ROI BYOL vs 2D full-slice BYOL, both with the screen-2 champion augmentations.

Runs, 30 epochs on the campaign schedule compressed proportionally (LR warmup 6/30):

- ``byol_2d``: BYOL ResNet-18 (ImageNet) on the campaigns' 2D axial slices, with the
  champion augmentation set of ``campaign_aug_screen2`` exactly as screened.
- ``byol_3d32`` / ``byol_3d48``: BYOL 3D ResNet-18 (ImageNet weights inflated to 3D)
  on the hippocampal ROI pair (``t5_ssl.data.roi3d``) of 32 (64 mm) or 48 (96 mm)
  voxels per side, with the champion recalibrated in 3D to the same mean RMSE as
  the campaign photometric augmentations.
- ``init_*``: the same encoders without SSL.

Every run is scored with a frozen encoder in two ways: (1) as in screen 2, a 5x5
cross-validated probe over the train+val AD/HC subjects on clean images and on
images with hospital artifacts at 2x the training strength (3D artifacts for 3D
runs); (2) as in the campaigns, a probe fit on train with C chosen on validation
AUROC and scored once on the test split.

Usage: ``preprocess``, ``calibrate``, ``all --workers 2``, ``report``.
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
import screen_aug_equal as screen2
import torch
from evaluate_frozen_probe import evaluate
from report_aug_screen import fast_mean3, mean3
from screen_augmentations import MANIFEST, ROOT, to_hydra

# Moved to the package (2026-11-16); re-exported for the legacy campaign scripts.
from t5_ssl.evaluation.embeddings import split_embeddings

CAMPAIGN = ROOT / "campaign_roi3d"
PY = str(ROOT / ".venv/Scripts/python.exe")
SIZES = (32, 48)
SEEDS = (1, 2)
EPOCHS, WARMUP = 30, 6
EVAL_SEVERITY = 2.0
# `_none`: same run with the screen's `none` set (geometry only, no photometric change).
RUNS = ["init_2d", "init_3d32", "init_3d48", "byol_2d", "byol_3d32", "byol_3d32_none",
        "byol_3d48"]
TRAINER = [f"trainer.max_epochs={EPOCHS}", f"optimizer.warmup_epochs={WARMUP}"]
COMMON_3D = [
    "data=oasis3_roi3d", "model=byol_resnet18_3d", "loss=byol", "trainer=full50",
    *TRAINER, "trainer.accelerator=gpu", "trainer.precision=32-true",
    "trainer.deterministic=false", "trainer.enable_progress_bar=false",
    "trainer.early_stopping_patience=null", "trainer.enable_checkpointing=false",
    "trainer.enable_health_check=false",
]


def champion() -> str:
    path = CAMPAIGN / "champion.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["name"]
    ranking = json.loads((screen2.CAMPAIGN / "ranking.json").read_text(encoding="utf-8"))
    best = next(row["name"] for row in ranking if row["name"] != "init")
    CAMPAIGN.mkdir(exist_ok=True)
    path.write_text(json.dumps({"name": best, "source": "campaign_aug_screen2/ranking.json, "
                                "best hospital-artifact score among trained sets"}) + "\n")
    return best


# ---------------------------------------------------------------- 3D calibration
def knob3d(name: str) -> tuple[Any, float, float]:
    make, low, high = screen2.KNOBS[name]
    if name == "motion":  # beyond ~10 mm/5 degrees 3D motion stops adding RMSE; keep it plausible
        high = 5.0

    def make3(s: float) -> dict:
        spec = make(s)
        if "axes" in spec:
            spec["axes"] = [0, 1, 2]
        return spec
    return make3, low, high


def photometric3d(size: int, photometric: dict[str, Any], torchio: list[dict] | None) -> Any:
    from t5_ssl.data.roi3d import ROIView3DTransform
    return ROIView3DTransform(size=size, blur_p=photometric["blur_p"],
                              noise_std=photometric["noise_std"],
                              intensity_scale=tuple(photometric["intensity_scale"]),
                              torchio_transforms=torchio)


def calibrate(size: int) -> None:
    from t5_ssl.data.roi3d import ROIEval3D
    cache = np.load(ROOT / f"data/roi3d/hippo_s{size}.npy", mmap_mode="r")
    picks = random.Random(0).sample(range(len(cache)), 16)
    cubes = [torch.from_numpy(cache[i].astype(np.float32)) for i in sorted(picks)]
    clean = ROIEval3D(size)

    def rmse(photometric: dict, torchio: list[dict] | None) -> float:
        view = photometric3d(size, photometric, torchio)
        random.seed(0), np.random.seed(0), torch.manual_seed(0)
        return float(np.mean([float((view.photometric_only(c) - clean(c)).pow(2).mean().sqrt())
                              for c in cubes for _ in range(2)]))

    def solve(make: Any, low: float, high: float, target: float) -> tuple[float, float]:
        def at(s: float) -> float:
            return rmse(screen2.NO_PHOTOMETRIC, [{**make(s), "p": 1.0}])
        if at(high) < target:
            return high, at(high)
        lo, hi = np.log(low), np.log(high)
        for _ in range(10):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if at(float(np.exp(mid))) < target else (lo, mid)
        knob = float(np.exp((lo + hi) / 2))
        return knob, at(knob)

    target = rmse(screen2.REFERENCE_PHOTOMETRIC, None)
    result: dict[str, Any] = {"size": size, "target_rmse": target, "eval_severity": EVAL_SEVERITY,
                              "artifacts": {}}
    print(f"[{size}] target RMSE (campaign photometric, 3D) = {target:.4f}", flush=True)
    for name in screen2.KNOBS:
        make, low, high = knob3d(name)
        knob, achieved = solve(make, low, high, target)
        eval_knob, eval_rmse = solve(make, low, high, EVAL_SEVERITY * target)
        result["artifacts"][name] = {"knob": knob, "rmse": achieved, "spec": make(knob),
                                     "eval_knob": eval_knob, "eval_rmse": eval_rmse,
                                     "eval_spec": make(eval_knob)}
        print(f"[{size}] {name:11s} train knob={knob:8.4f} rmse={achieved:.4f} | "
              f"eval knob={eval_knob:8.4f} rmse={eval_rmse:.4f}", flush=True)
    (CAMPAIGN / f"calibration_s{size}.json").write_text(json.dumps(result, indent=2) + "\n")


# ---------------------------------------------------------------- runs
def aug3d_overrides(name: str, calib: dict) -> list[str]:
    """Translate the screen-2 set `name` to overrides of the 3D ROI views."""
    items: dict[str, Any] = {"lr_swap_p": 0.5, "ap_flip_p": 0.0, "affine_degrees": 10.0}
    items.update(screen2.REFERENCE_PHOTOMETRIC if name in ("ref", "current")
                 else screen2.NO_PHOTOMETRIC)
    if name == "current":  # campaign geometry: front-back flip, +/-15 degrees, wider crops
        items.update(ap_flip_p=0.5, affine_degrees=15.0, side_range=[0.55, 1.0])
    specs: list[dict] = []
    if name in screen2.KNOBS:
        specs = [{**calib["artifacts"][name]["spec"], "p": 1.0}]
    elif name in screen2.MIXES:
        specs = [{"name": "OneOf", "p": 1.0,
                  "transforms": [calib["artifacts"][a]["spec"] for a in screen2.MIXES[name]]}]
    items["torchio"] = specs
    return ["augmentation=roi3d", *(f"++augmentation.{k}={to_hydra(v)}" for k, v in items.items())]


def overrides_for(run: str, seed: int) -> tuple[list[str], int | None]:
    best = "none" if run.endswith("_none") else champion()
    if run.endswith("_2d"):
        calib = json.loads((screen2.CAMPAIGN / "calibration.json").read_text(encoding="utf-8"))
        common = [o for o in screen2.COMMON if not o.startswith(("trainer.max_epochs=",
                                                                   "optimizer.warmup_epochs="))]
        return [*common, *TRAINER, *screen2.MODEL_OVERRIDES, f"seed={seed}",
                *screen2.aug_overrides(best, calib)], None
    size = int(run.removesuffix("_none").removeprefix("init_3d").removeprefix("byol_3d"))
    calib = json.loads((CAMPAIGN / f"calibration_s{size}.json").read_text(encoding="utf-8"))
    return [*COMMON_3D, f"data.roi_size={size}", f"data.manifest_path={MANIFEST}",
            f"seed={seed}", *aug3d_overrides(best, calib)], size


def run_one(run: str, seed: int) -> None:
    import warnings

    from hydra import compose, initialize_config_dir

    from t5_ssl.data.augmentations import EvalTransform
    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_backbone, build_datamodule, train
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    overrides, size = overrides_for(run, seed)
    start = time.time()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    clean = EvalTransform((96, 96)) if size is None else ROIEval3D(size)
    if run.startswith("init"):
        torch.manual_seed(seed)
        backbone = build_backbone(cfg)
        dm = build_datamodule(cfg, clean, clean)
        dm.setup()
        exp_dir = None
    else:
        result = train(cfg)
        backbone, dm, exp_dir = result["model"].backbone, result["datamodule"], result["exp_dir"]
    train_minutes = (time.time() - start) / 60
    x, y, subjects = screen2.embed(backbone, dm, clean)
    variants = {"clean": x}
    if size is None:
        calib = json.loads((screen2.CAMPAIGN / "calibration.json").read_text(encoding="utf-8"))
    else:
        calib = json.loads((CAMPAIGN / f"calibration_s{size}.json").read_text(encoding="utf-8"))
    for artifact, info in calib["artifacts"].items():
        corrupt = (screen2.CorruptedEval(info["eval_spec"]) if size is None else
                   photometric3d(size, screen2.NO_PHOTOMETRIC, [{**info["eval_spec"], "p": 1.0}])
                   .photometric_only)
        random.seed(0), np.random.seed(0), torch.manual_seed(0)  # same corruption every run
        x_art, _, art_subjects = screen2.embed(backbone, dm, corrupt)
        if art_subjects != subjects:
            raise RuntimeError("Corrupted embeddings cover different subjects")
        variants[artifact] = x_art
    cv = screen2.cv_probe(x, y, variants)
    test = evaluate(*(split_embeddings(backbone, dm, clean, s) for s in ("train", "val", "test")))
    out = {"run": run, "seed": seed,
           "augmentation": "none" if run.endswith("_none") else champion(), "overrides": overrides,
           "experiment_dir": str(exp_dir) if exp_dir else None, "train_minutes": train_minutes,
           "minutes": (time.time() - start) / 60, "subjects": subjects, "labels": y.tolist(),
           "cv": cv, "test": test}
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "runs" / f"{run}_s{seed}.json").write_text(json.dumps(out) + "\n")


def run_all(workers: int, runs: list[str]) -> None:
    (CAMPAIGN / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(run, seed) for seed in SEEDS for run in runs
            if not (run.startswith("init") and seed != SEEDS[0])
            and not (CAMPAIGN / "runs" / f"{run}_s{seed}.json").is_file()]
    # Longest runs first so the two workers finish together.
    todo.sort(key=lambda job: ("init" in job[0], "3d48" not in job[0], job[1]))
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}

    def launch(job: tuple[str, int]) -> None:
        run, seed = job
        start = time.time()
        with (CAMPAIGN / "logs" / f"{run}_s{seed}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run([PY, __file__, "run", run, str(seed)], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        path = CAMPAIGN / "runs" / f"{run}_s{seed}.json"
        detail = "FAILED"
        if code == 0 and path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            t = doc["test"]
            detail = (f"OK cv_clean={mean3(doc['cv']['clean']['metrics']):.3f} test AUROC="
                      f"{t['test_auc']:.3f} F1={t['test_f1_macro']:.3f} "
                      f"AUPR={t['test_average_precision']:.3f}")
        print(f"{time.strftime('%H:%M:%S')} {run} s{seed} {detail} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, todo))


# ---------------------------------------------------------------- report
def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    docs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((CAMPAIGN / "runs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[doc["run"]].append(doc)
    order = [run for run in RUNS if run in docs]
    labels = np.array(docs[order[0]][0]["labels"])
    test_pred = {run: [d["test"]["test_predictions"] for d in ds] for run, ds in docs.items()}
    test_subjects = [p["subject"] for p in test_pred[order[0]][0]]
    test_labels = np.array([p["AD"] for p in test_pred[order[0]][0]])
    for run in order:
        for preds in test_pred[run]:
            if [p["subject"] for p in preds] != test_subjects:
                raise RuntimeError(f"{run} scored different test subjects")
        if any(d["subjects"] != docs[order[0]][0]["subjects"] for d in docs[run]):
            raise RuntimeError(f"{run} scored different CV subjects")

    def hospital(doc: dict) -> float:
        return float(np.mean([mean3(v["metrics"]) for k, v in doc["cv"].items() if k != "clean"]))

    rows = []
    for run in order:
        ds = docs[run]
        row: dict[str, Any] = {"run": run, "n": len(ds),
                               "minutes": float(np.mean([d["minutes"] for d in ds]))}
        for key in ("auc", "f1", "aupr"):
            row[f"cv_{key}"] = float(np.mean([d["cv"]["clean"]["metrics"][key] for d in ds]))
        row["cv_mean"] = float(np.mean([mean3(d["cv"]["clean"]["metrics"]) for d in ds]))
        row["hospital"] = float(np.mean([hospital(d) for d in ds]))
        for key, field in (("auc", "test_auc"), ("f1", "test_f1_macro"),
                           ("aupr", "test_average_precision")):
            row[f"test_{key}"] = float(np.mean([d["test"][field] for d in ds]))
            row[f"test_{key}_seeds"] = [d["test"][field] for d in ds]
            row[f"cv_{key}_seeds"] = [d["cv"]["clean"]["metrics"][key] for d in ds]
        row["test_mean"] = (row["test_auc"] + row["test_f1"] + row["test_aupr"]) / 3
        rows.append(row)

    # Paired bootstrap of each run against byol_2d, over subjects, seeds averaged.
    rng = np.random.default_rng(42)

    def boot(labels_: np.ndarray, n: int = 1000) -> list[np.ndarray]:
        pos, neg = np.flatnonzero(labels_ == 1), np.flatnonzero(labels_ == 0)
        return [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
                for _ in range(n)]

    cv_boot, test_boot = boot(labels), boot(test_labels)
    by_run = {row["run"]: row for row in rows}

    def compare(a: str, b: str) -> dict[str, tuple[float, list[float], float]]:
        """Seed-averaged mean-of-3-metrics difference a - b, paired bootstrap CI."""
        out = {}
        for label, draws, lab, getter in (
            ("cv", cv_boot, labels, lambda d: np.asarray(d["cv"]["clean"]["oof_mean"])),
            ("test", test_boot, test_labels,
             lambda d: np.array([p["score"] for p in d["test"]["test_predictions"]])),
        ):
            def seed_avg(run: str, idx: np.ndarray, lab: np.ndarray = lab, getter: Any = getter
                         ) -> float:
                return float(np.mean([fast_mean3(lab[idx], getter(d)[idx]) for d in docs[run]]))
            deltas = np.array([seed_avg(a, idx) - seed_avg(b, idx) for idx in draws])
            out[label] = (by_run[a][f"{label}_mean"] - by_run[b][f"{label}_mean"],
                          np.percentile(deltas, [2.5, 97.5]).tolist(), float(np.mean(deltas > 0)))
        return out

    base = "byol_2d"
    for row in rows:
        if base in docs and row["run"] != base:
            row["vs_2d"] = compare(row["run"], base)
    key_comparisons = [(a, b, text) for a, b, text in (
        ("byol_3d32", "byol_2d", "3D ROI 64 mm vs 2D, ambos BYOL con el set campeón"),
        ("byol_3d32", "byol_3d32_none", f"Efecto del set campeón (`{champion()}`) en 3D"),
        ("byol_3d32", "init_3d32", "Efecto del entrenamiento SSL en 3D (32)"),
        ("byol_2d", "init_2d", "Efecto del entrenamiento SSL en 2D"),
        ("init_3d32", "init_2d", "3D ROI 64 mm vs 2D, sin SSL"),
        ("byol_3d48", "byol_3d32", "Cubo de 96 mm vs 64 mm (BYOL)"),
        ("init_3d48", "init_3d32", "Cubo de 96 mm vs 64 mm (sin SSL)"),
    ) if a in docs and b in docs]
    comparisons = [(a, b, text, compare(a, b)) for a, b, text in key_comparisons]
    (CAMPAIGN / "results.json").write_text(json.dumps(rows, indent=2) + "\n")

    # Figure: AUROC / F1-macro / AUPR, CV (train+val) and test; seeds small, mean large.
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 0.5 * len(rows) * 2 + 2.2), sharey=True)
    for r, (prefix, row_title) in enumerate((("cv", "CV train+val (imágenes limpias)"),
                                             ("test", "Test (probe de la campaña)"))):
        for c, (key, title) in enumerate((("auc", "AUROC"), ("f1", "F1-macro"), ("aupr", "AUPR"))):
            ax = axes[r, c]
            for yi, row in zip(y, rows, strict=True):
                color = "#2a78d6" if row["run"].startswith("byol") else "#9c9a92"
                ax.scatter(row[f"{prefix}_{key}_seeds"], [yi] * row["n"], s=14, color=color,
                           alpha=0.45, zorder=2)
                ax.scatter([row[f"{prefix}_{key}"]], [yi], s=46, color=color, zorder=3)
            ax.set_title(f"{title} — {row_title}", fontsize=9, color="#0b0b0b")
            ax.grid(axis="x", alpha=0.25)
            ax.tick_params(colors="#52514e", labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[r, 0].set_yticks(y, [row["run"] for row in rows], color="#0b0b0b")
    fig.suptitle(f"3D ROI hipocampal vs 2D (BYOL, {EPOCHS} épocas, augmentations "
                 f"`{champion()}`). Azul = BYOL; gris = sin SSL.", fontsize=11)
    fig.tight_layout()
    fig.savefig(CAMPAIGN / "figures" / "roi3d_vs_2d.png", dpi=160)
    plt.close(fig)

    meta = {s: json.loads((ROOT / f"data/roi3d/hippo_s{s}.json").read_text(encoding="utf-8"))
            for s in SIZES if (ROOT / f"data/roi3d/hippo_s{s}.json").is_file()}
    lines = [
        "# 3D ROI hipocampal vs 2D, con las augmentations campeonas del screen 2", "",
        f"BYOL, {EPOCHS} épocas (calendario de la campaña comprimido: warmup del LR "
        f"{WARMUP} de {EPOCHS}), split balanceado, {len(SEEDS)} semillas por corrida, "
        f"augmentations campeonas `{champion()}` de `campaign_aug_screen2`.", "",
        "- `byol_2d`: ResNet-18 ImageNet sobre cortes axiales 2D de cerebro completo (la entrada "
        "de las campañas), con el set campeón tal como se evaluó en el screen 2.",
        "- `byol_3d32` / `byol_3d48`: ResNet-18 3D con pesos de ImageNet inflados (I3D) sobre el "
        "par de cubos hipocampales [izquierdo, derecho espejado] de 32 vóxeles (64 mm) o 48 "
        "(96 mm) por lado; el set campeón se recalibró en 3D a la misma RMSE media que las "
        "augmentations fotométricas de la campaña. Geometría 3D: sub-cubo aleatorio del 84–100 % "
        "del lado guardado, flip izquierda-derecha (= intercambio de canales) y rotación ±10° en "
        "los tres ejes.",
        "- `byol_3d32_none`: igual que `byol_3d32` pero sin augmentation fotométrica (el set "
        "`none` del screen, que empató con el campeón en imágenes con artefactos), para aislar "
        "el efecto del set campeón.",
        "- `init_*`: los mismos encoders sin entrenamiento SSL.", "",
        "**Preprocesamiento.** Cubos recortados una vez desde el LMDB (vóxeles de 2 mm, "
        "MNI 97×115×97) alrededor del centroide poblacional de los hipocampos (etiquetas 17 y 53 "
        "de la segmentación incluida en el LMDB, promediadas sobre las sesiones de train) y "
        "desplazados para quedar dentro del cerebro de todas las sesiones."]
    for s, m in meta.items():
        empty = sum(m["empty_slices_in_evaluated_cube"].values())
        lines.append(f"  Cubo {s}: centros (vóxel) izq {m['hippocampus_centers_voxel']['left']}, "
                     f"der {m['hippocampus_centers_voxel']['right']}; inicio izq "
                     f"{m['cube_starts']['left']}, der {m['cube_starts']['right']}; cortes sin "
                     f"cerebro en el cubo evaluado: {empty}.")
    lines += [
        "", "**Evaluación.** Encoder congelado. *CV*: probe logístico con validación cruzada "
        f"5×5 sobre los {len(labels)} sujetos AD/HC de train+val, en imágenes limpias y con "
        "artefactos hospitalarios a 2× (artefactos 3D en las corridas 3D, así que esa columna "
        "no es comparable entre 2D y 3D). *Test*: el protocolo de las campañas (probe ajustado "
        f"en train, C por AUROC de validación, test una vez; {len(test_labels)} sujetos, "
        f"{int(test_labels.sum())} AD). Δ vs `byol_2d`: bootstrap pareado sobre sujetos, "
        "promediando semillas, de la media de AUROC, F1-macro y AUPR.", "",
        "![3D vs 2D](figures/roi3d_vs_2d.png)", "",
        "| Corrida | CV AUROC | CV F1 | CV AUPR | CV media | Δ CV vs 2D [IC 95 %] | P(Δ>0) | "
        "Test AUROC | Test F1 | Test AUPR | Δ test vs 2D [IC 95 %] | P(Δ>0) | Con artefactos | "
        "Min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cells = []
        for label in ("cv", "test"):
            if "vs_2d" in row:
                delta, (lo, hi), prob = row["vs_2d"][label]
                cells.append((f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{prob:.2f}"))
            else:
                cells.append(("—", "—"))
        lines.append(
            f"| `{row['run']}` ({row['n']}) | {row['cv_auc']:.3f} | {row['cv_f1']:.3f} | "
            f"{row['cv_aupr']:.3f} | **{row['cv_mean']:.3f}** | {cells[0][0]} | {cells[0][1]} | "
            f"{row['test_auc']:.3f} | {row['test_f1']:.3f} | {row['test_aupr']:.3f} | "
            f"{cells[1][0]} | {cells[1][1]} | {row['hospital']:.3f} | {row['minutes']:.0f} |")
    lines += ["", "## Comparaciones clave (media de AUROC, F1-macro y AUPR)", "",
              "| Comparación | Δ CV [IC 95 %] | P(Δ>0) | Δ test [IC 95 %] | P(Δ>0) |",
              "|---|---:|---:|---:|---:|"]
    for a, b, text, result in comparisons:
        cv_d, (cv_lo, cv_hi), cv_p = result["cv"]
        te_d, (te_lo, te_hi), te_p = result["test"]
        lines.append(f"| {text}: `{a}` − `{b}` | {cv_d:+.3f} [{cv_lo:+.3f}, {cv_hi:+.3f}] | "
                     f"{cv_p:.2f} | {te_d:+.3f} [{te_lo:+.3f}, {te_hi:+.3f}] | {te_p:.2f} |")
    lines += ["", "Entre paréntesis, el número de semillas. Regenerar con "
              "`python scripts/legacy/run_roi3d.py report`; figuras de los cubos en "
              "[figures/](figures/).", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {CAMPAIGN / 'RESULTADOS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preprocess")
    pre.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    cal = sub.add_parser("calibrate")
    cal.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    one = sub.add_parser("run")
    one.add_argument("run", choices=RUNS)
    one.add_argument("seed", type=int)
    every = sub.add_parser("all")
    every.add_argument("runs", nargs="*", default=RUNS)
    every.add_argument("--workers", type=int, default=2)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "preprocess":
        from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_LMDB
        from t5_ssl.data.roi3d import build_cache
        for size in args.sizes:
            meta = build_cache(ROOT / f"data/roi3d/hippo_s{size}.npy", csv_path=DEFAULT_CSV,
                               lmdb_path=DEFAULT_LMDB, manifest_path=str(ROOT / MANIFEST), size=size)
            print(json.dumps(meta), flush=True)
    elif args.command == "calibrate":
        CAMPAIGN.mkdir(exist_ok=True)
        for size in args.sizes:
            calibrate(size)
    elif args.command == "run":
        run_one(args.run, args.seed)
    elif args.command == "all":
        run_all(args.workers, args.runs)
    else:
        report()


if __name__ == "__main__":
    sys.exit(main())
