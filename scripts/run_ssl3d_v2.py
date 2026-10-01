"""3D SSL campaign, second pass: DINOv2 and BYOL after the 2026-09-30 diagnosis.

Same protocol as ``run_ssl3d.py`` (30 epochs on the campaign schedule compressed
proportionally — LR warmup 6/30, DINOv2 teacher-temperature warmup 9/30 — the
balanced split, the screen-2 champion augmentation ``bias`` recalibrated in 3D,
3 seeds, frozen-encoder probes), changing only what the diagnosis blamed:

- DINOv2 (``dinov2_3d``): DINO heads with an L2-normalized bottleneck and cosine
  prototypes, with batch norm in the head (the old head let the CLS loss sit at
  ln K in every run), teacher temperature 0.01 held constant (at 0.04 -> 0.07,
  and still at 0.04 or 0.02, the teacher drifted to uniform targets here),
  Sinkhorn-Knopp centering, KoLeo on the backbone CLS,
  DINOv2 block masking (half of the global views, ratios 0.1-0.5, no forced
  patch), prototypes frozen in the first epoch, weight decay 0.04 -> 0.2, EMA
  0.99, 8-voxel patches (64 tokens per view instead of 8), a ViT-S starting from
  DeiT-S ImageNet-1k weights inflated to 3D like the ImageNet CNNs of VICReg and
  BYOL, and views of its own (``augmentation=roi3d_dinov2``): the two global
  views differ only by the bias field and the masks - no crop jitter, rotation
  or local crops, because any geometric change erased the subject from the
  ViT's features and left the student nothing to predict. Scored with the
  teacher, as in DINO/DINOv2; the student is scored too.
- BYOL (``byol_3d``): EMA 0.99 instead of 0.996 (with 690 steps the target kept
  25% of its ImageNet initialization at the end); nothing else changes.

VICReg, the ConvNeXt/ResNet controls and the previous BYOL/DINOv2 runs
(``*_v1_3d``) are imported from ``campaign_ssl3d``. The ViT controls are new
because the backbone changed: ``imagenet_vit_3d`` (DINOv2's start) and
``random_vit_3d``.

Usage: ``all --workers 2``, then ``report``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Heavy modules (torch, the probes, matplotlib) are imported inside the functions
# that need them: the `all` orchestrator only launches subprocesses and reads
# JSON, and importing them there costs ~1.7 GB of RAM per orchestrator.
from screen_augmentations import MANIFEST, ROOT

if TYPE_CHECKING:
    import numpy as np
    import torch

CAMPAIGN = ROOT / "campaign_ssl3d_v2"
V1 = ROOT / "campaign_ssl3d"
PY = str(ROOT / ".venv/Scripts/python.exe")
SEEDS = (1, 2, 3)
CHAMPION = "bias"
METHODS = ("vicreg", "byol", "dinov2")
LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
BACKBONE = {"vicreg": "convnext", "byol": "resnet", "dinov2": "vit"}
MODEL = {
    "vicreg": ["model=vicreg_convnext_3d", "loss=vicreg", "augmentation=roi3d"],
    "byol": ["model=byol_resnet18_3d", "loss=byol", "augmentation=roi3d", "loss.ema_base=0.99"],
    "dinov2": ["model=dinov2_vit3d", "loss=dinov2", "augmentation=roi3d_dinov2",
               "optimizer=adamw_dinov2", "loss.teacher_temp_warmup_epochs=9"],
}
RUNTIME = ["trainer.accelerator=gpu", "trainer.precision=32-true",
           "trainer.deterministic=false", "trainer.enable_progress_bar=false",
           "trainer.early_stopping_patience=null", "trainer.enable_checkpointing=false",
           "trainer.enable_health_check=false"]
COMMON = ["data=oasis3_roi3d", "data.roi_size=32", f"data.manifest_path={MANIFEST}",
          "trainer=full50", *RUNTIME]
# run -> (kind, method, campaign_ssl3d run it is imported from; None = run here)
RUNS: dict[str, tuple[str, str, str | None]] = {
    "dinov2_3d": ("ssl", "dinov2", None),
    "byol_3d": ("ssl", "byol", None),
    "vicreg_3d": ("ssl", "vicreg", "vicreg_3d"),
    "dinov2_v1_3d": ("ssl", "dinov2", "dinov2_3d"),
    "byol_v1_3d": ("ssl", "byol", "byol_3d"),
    "imagenet_vit_3d": ("imagenet", "dinov2", None),
    "random_vit_3d": ("random", "dinov2", None),
    "imagenet_resnet_3d": ("imagenet", "byol", "imagenet_resnet_3d"),
    "random_resnet_3d": ("random", "byol", "random_resnet_3d"),
    "imagenet_convnext_3d": ("imagenet", "vicreg", "imagenet_convnext_3d"),
    "random_convnext_3d": ("random", "vicreg", "random_convnext_3d"),
}
START = {"vicreg": "imagenet_convnext_3d", "byol": "imagenet_resnet_3d",
         "dinov2": "imagenet_vit_3d"}
COMPARISONS = [
    ("dinov2", "Aporte SSL: DINOv2 − su inicio (`imagenet_vit_3d`)",
     [(1, "dinov2_3d"), (-1, "imagenet_vit_3d")]),
    ("dinov2", "DINOv2 vs pesos aleatorios (`random_vit_3d`)",
     [(1, "dinov2_3d"), (-1, "random_vit_3d")]),
    ("dinov2", "DINOv2 corregido − DINOv2 anterior", [(1, "dinov2_3d"), (-1, "dinov2_v1_3d")]),
    ("dinov2", "Inicio ViT: DeiT-S inflado − pesos aleatorios",
     [(1, "imagenet_vit_3d"), (-1, "random_vit_3d")]),
    ("byol", "Aporte SSL: BYOL − su inicio (`imagenet_resnet_3d`)",
     [(1, "byol_3d"), (-1, "imagenet_resnet_3d")]),
    ("byol", "BYOL EMA 0.99 − BYOL anterior (EMA 0.996)", [(1, "byol_3d"), (-1, "byol_v1_3d")]),
    ("vicreg", "Aporte SSL: VICReg − su inicio (`imagenet_convnext_3d`)",
     [(1, "vicreg_3d"), (-1, "imagenet_convnext_3d")]),
    ("all", "DINOv2 − VICReg", [(1, "dinov2_3d"), (-1, "vicreg_3d")]),
    ("all", "BYOL − VICReg", [(1, "byol_3d"), (-1, "vicreg_3d")]),
]
# Short DINOv2 runs of 2026-09-30 that led to the final recipe (same campaign
# config except the head's batch norm and the teacher temperature; stopped once
# the trend was clear). Shown in the report as a collapse ablation.
EXPLORATION = [f"dinov2_mask0.5x0.1-0.5_vit_small_3d_2026-09-30_{t}"
               for t in ("13-53-01", "14-25-11", "14-44-25", "14-50-01", "14-53-05",
                         "15-32-36", "15-43-21", "15-45-04")]
DIAGNOSTICS = ("train/dino", "train/ibot", "train/koleo", "train/teacher_entropy",
               "train/teacher_entropy_frac", "train/dino_kl", "train/teacher_unique_frac",
               "train/loss_epoch", "val/loss")


def overrides_for(run: str, seed: int) -> list[str]:
    import run_roi3d
    kind, method, _ = RUNS[run]
    calib = json.loads((run_roi3d.CAMPAIGN / "calibration_s32.json").read_text(encoding="utf-8"))
    aug = run_roi3d.aug3d_overrides(CHAMPION, calib)[1:]  # the group comes from MODEL
    if method == "dinov2":  # champion bias field, but roi3d_dinov2's geometry (none)
        aug = [o for o in aug if not o.startswith("++augmentation.affine_degrees=")]
    out = [*COMMON, *run_roi3d.TRAINER, *MODEL[method], f"seed={seed}", *aug]
    if kind == "random":
        out.append("model.backbone.pretrained=false")
    return out


def effective_rank(x: np.ndarray) -> float:
    """exp(entropy) of the normalized singular values of the centered embeddings."""
    import numpy as np
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    p = s[s > 0] / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def epoch_metrics(exp_dir: Path | str) -> dict[str, list[float]]:
    """Per-epoch series of the epoch-level metrics logged during training."""
    series: dict[str, dict[int, float]] = defaultdict(dict)
    with (Path(exp_dir) / "metrics.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value not in ("", None) and key not in ("epoch", "step", "train/loss_step"):
                    series[key][int(row["epoch"])] = float(value)
    return {key: [values[e] for e in sorted(values)] for key, values in series.items()}


def score(encoder: torch.nn.Module, dm: Any, clean: Any) -> dict[str, Any]:
    import run_roi3d
    import screen_aug_equal as screen2
    from evaluate_frozen_probe import evaluate
    x, y, subjects = screen2.embed(encoder, dm, clean)
    cv = screen2.cv_probe(x, y, {"clean": x})
    test = evaluate(*(run_roi3d.split_embeddings(encoder, dm, clean, s)
                      for s in ("train", "val", "test")))
    return {"subjects": subjects, "labels": y.tolist(), "cv": cv, "test": test,
            "effective_rank": effective_rank(x)}


def run_one(run: str, seed: int) -> None:
    import warnings

    import torch
    from hydra import compose, initialize_config_dir

    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_backbone, build_datamodule, train
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    kind, method, _ = RUNS[run]
    overrides = overrides_for(run, seed)
    start = time.time()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    clean = ROIEval3D(int(cfg.data.roi_size))
    diagnostics, exp_dir = None, None
    if kind == "ssl":
        result = train(cfg)
        model, dm, exp_dir = result["model"], result["datamodule"], result["exp_dir"]
        # DINO/DINOv2 evaluate the EMA teacher; BYOL the online network.
        encoders = ({"main": model.teacher_backbone, "student": model.backbone}
                    if method == "dinov2" else {"main": model.backbone})
        diagnostics = epoch_metrics(exp_dir)
    else:
        torch.manual_seed(seed)
        encoders = {"main": build_backbone(cfg)}
        dm = build_datamodule(cfg, clean, clean)
        dm.setup()
    train_minutes = (time.time() - start) / 60
    scored = {name: score(encoder, dm, clean) for name, encoder in encoders.items()}
    main = scored.pop("main")
    out = {"run": run, "seed": seed, "augmentation": CHAMPION, "overrides": overrides,
           "experiment_dir": str(exp_dir) if exp_dir else None,
           "encoder": "teacher" if "student" in scored else "backbone",
           "train_minutes": train_minutes, "minutes": (time.time() - start) / 60, **main,
           "student": ({k: v for k, v in scored["student"].items()
                        if k not in ("subjects", "labels")} if "student" in scored else None),
           "training_metrics": diagnostics}
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "runs" / f"{run}_s{seed}.json").write_text(json.dumps(out) + "\n")


def import_previous() -> None:
    """Copy the campaign_ssl3d runs whose recipe and evaluation are unchanged."""
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    for run, (_, _, source) in RUNS.items():
        if source is None:
            continue
        for path in sorted((V1 / "runs").glob(f"{source}_s*.json")):
            target = CAMPAIGN / "runs" / f"{run}_s{path.stem.rsplit('_s', 1)[1]}.json"
            if not target.is_file():
                doc = json.loads(path.read_text(encoding="utf-8"))
                doc.update(run=run, imported_from=str(path.relative_to(ROOT)))
                target.write_text(json.dumps(doc) + "\n")


def jobs(runs: list[str]) -> list[tuple[str, int]]:
    todo = []
    for run in runs:
        kind, _, source = RUNS[run]
        if source is not None:
            continue
        seeds = SEEDS[:1] if kind == "imagenet" else SEEDS  # the ImageNet start is fixed
        todo += [(run, s) for s in seeds if not (CAMPAIGN / "runs" / f"{run}_s{s}.json").is_file()]
    # SSL first, one DINOv2 next to one BYOL on the GPU, then the evaluation-only controls.
    return sorted(todo, key=lambda job: (RUNS[job[0]][0] != "ssl", job[1],
                                         RUNS[job[0]][1] != "dinov2"))


def run_all(workers: int, runs: list[str]) -> None:
    CAMPAIGN.mkdir(exist_ok=True)
    import_previous()
    (CAMPAIGN / "logs").mkdir(exist_ok=True)
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
            t, m = doc["test"], doc["cv"]["clean"]["metrics"]
            detail = (f"OK cv={(m['auc'] + m['f1'] + m['aupr']) / 3:.3f} test AUROC="
                      f"{t['test_auc']:.3f} F1={t['test_f1_macro']:.3f} "
                      f"AUPR={t['test_average_precision']:.3f} erank={doc['effective_rank']:.1f}")
        print(f"{time.strftime('%H:%M:%S')} {run} s{seed} {detail} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, jobs(runs)))


# ---------------------------------------------------------------- report
def report() -> None:
    import matplotlib
    import numpy as np
    from report_aug_screen import fast_mean3, mean3
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((CAMPAIGN / "runs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[doc["run"]].append(doc)
    order = [run for run in RUNS if run in docs]
    first = docs[order[0]][0]
    labels = np.array(first["labels"])
    test_first = first["test"]["test_predictions"]
    test_subjects = [p["subject"] for p in test_first]
    test_labels = np.array([p["AD"] for p in test_first])
    for run in order:
        for d in docs[run]:
            if d["subjects"] != first["subjects"] or \
                    [p["subject"] for p in d["test"]["test_predictions"]] != test_subjects:
                raise RuntimeError(f"{run} was scored on different subjects")

    def mean_of(ds: list[dict], get: Any) -> float:
        return float(np.mean([get(d) for d in ds]))

    summary: dict[str, dict[str, Any]] = {}
    for run in order:
        ds = docs[run]
        row: dict[str, Any] = {"n": len(ds), "minutes": mean_of(ds, lambda d: d["minutes"])}
        for key, field in (("auc", "test_auc"), ("f1", "test_f1_macro"),
                           ("aupr", "test_average_precision")):
            row[f"cv_{key}"] = mean_of(ds, lambda d, key=key: d["cv"]["clean"]["metrics"][key])
            row[f"test_{key}"] = mean_of(ds, lambda d, field=field: d["test"][field])
        row["cv_seeds"] = [mean3(d["cv"]["clean"]["metrics"]) for d in ds]
        row["test_seeds"] = [(d["test"]["test_auc"] + d["test"]["test_f1_macro"]
                              + d["test"]["test_average_precision"]) / 3 for d in ds]
        row["cv"], row["test"] = float(np.mean(row["cv_seeds"])), float(np.mean(row["test_seeds"]))
        ranks = [d["effective_rank"] for d in ds if "effective_rank" in d]
        row["effective_rank"] = float(np.mean(ranks)) if ranks else None
        students = [d["student"] for d in ds if d.get("student")]
        if students:
            row["student_cv"] = float(np.mean([mean3(s["cv"]["clean"]["metrics"])
                                               for s in students]))
            row["student_test"] = float(np.mean([(s["test"]["test_auc"] + s["test"]["test_f1_macro"]
                                                  + s["test"]["test_average_precision"]) / 3
                                                 for s in students]))
        summary[run] = row

    rng = np.random.default_rng(42)

    def draws(lab: np.ndarray, n: int = 1000) -> list[np.ndarray]:
        pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
        return [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
                for _ in range(n)]

    boots = {"cv": (draws(labels), labels), "test": (draws(test_labels), test_labels)}
    scores = {"cv": lambda d: np.asarray(d["cv"]["clean"]["oof_mean"]),
              "test": lambda d: np.array([p["score"] for p in d["test"]["test_predictions"]])}

    def contrast(terms: list[tuple[float, str]]) -> dict[str, tuple[float, list[float], float]]:
        """Paired bootstrap over subjects of a signed sum of seed-averaged mean-of-3 scores."""
        out = {}
        for split, (idx_list, lab) in boots.items():
            def value(idx: np.ndarray, split: str = split, lab: np.ndarray = lab) -> float:
                return sum(w * float(np.mean([fast_mean3(lab[idx], scores[split](d)[idx])
                                              for d in docs[run]])) for w, run in terms)
            samples = np.array([value(idx) for idx in idx_list])
            point = sum(w * summary[run][split] for w, run in terms)
            out[split] = (point, np.percentile(samples, [2.5, 97.5]).tolist(),
                          float(np.mean(samples > 0)))
        return out

    comparisons = [(m, text, terms, contrast(terms)) for m, text, terms in COMPARISONS
                   if all(run in docs for _, run in terms)]

    # DINOv2 training diagnostics: last-epoch collapse monitors, new vs previous recipe.
    diag_rows = []
    for run in ("dinov2_3d", "dinov2_v1_3d"):
        for d in docs.get(run, []):
            metrics = d.get("training_metrics")
            if metrics is None and d.get("experiment_dir"):
                path = ROOT / d["experiment_dir"]
                metrics = epoch_metrics(path) if (path / "metrics.csv").is_file() else None
            if metrics:
                n_prototypes = 4096
                if d.get("experiment_dir") and (ROOT / d["experiment_dir"] / "config.yaml").is_file():
                    from omegaconf import OmegaConf
                    saved = OmegaConf.load(ROOT / d["experiment_dir"] / "config.yaml")
                    n_prototypes = int(saved.model.projector.out_dim)
                last = {k: metrics[k][-1] for k in DIAGNOSTICS if k in metrics}
                diag_rows.append({"run": run, "seed": d["seed"], "prototypes": n_prototypes, **last,
                                  "dino_first": metrics.get("train/dino", [None])[0],
                                  "dino_over_lnk": last.get("train/dino", math.nan)
                                  / math.log(n_prototypes)})
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    # Figure: DINOv2 training curves, previous vs corrected recipe (collapse monitors).
    curves = []
    for run, color in (("dinov2_v1_3d", "#9c9a92"), ("dinov2_3d", "#2a78d6")):
        for d in docs.get(run, []):
            metrics = d.get("training_metrics")
            path = ROOT / d["experiment_dir"] if d.get("experiment_dir") else None
            if metrics is None and path is not None and (path / "metrics.csv").is_file():
                metrics = epoch_metrics(path)
            row = next((r for r in diag_rows if r["run"] == run and r["seed"] == d["seed"]), None)
            if metrics and "train/dino" in metrics and row:
                curves.append((run, color, metrics, row["prototypes"]))
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        panels = [("train/dino", "Pérdida DINO (CLS) / ln K", True),
                  ("train/dino_kl", "KL profesor → estudiante", False),
                  ("train/teacher_entropy_frac", "Entropía del profesor / ln K", False)]
        for ax, (key, title, per_lnk) in zip(axes, panels, strict=True):
            for run, color, metrics, prototypes in curves:
                if key not in metrics:
                    continue
                values = np.array(metrics[key]) / (math.log(prototypes) if per_lnk else 1.0)
                ax.plot(values, color=color, linewidth=1.6, alpha=0.9,
                        label="anterior (K=1024)" if run.endswith("v1_3d") else "corregido (K=4096)")
            ax.set_title(title, fontsize=10, color="#0b0b0b")
            ax.set_xlabel("época", fontsize=9)
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0].axhline(1.0, color="#b3261e", linewidth=1, linestyle="--")
        axes[0].text(0.5, 1.005, "ln K (salida uniforme)", color="#b3261e", fontsize=8,
                     va="bottom", transform=axes[0].get_yaxis_transform())
        handles = {}
        for ax in axes:
            for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
                handles.setdefault(name, handle)
        fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=2,
                   frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(CAMPAIGN / "figures" / "dinov2_training.png", dpi=160)
        plt.close(fig)

    # Collapse ablation: the exploration runs next to the final recipe.
    exploration = []
    for name in EXPLORATION:
        path = ROOT / "logs" / name
        if not (path / "metrics.csv").is_file():
            continue
        from omegaconf import OmegaConf
        saved = OmegaConf.load(path / "config.yaml")
        t0 = float(saved.loss.teacher_temp)
        t1 = float(saved.loss.get("teacher_temp_final", t0))
        still = float(saved.augmentation.get("affine_degrees", 10)) == 0 and \
            list(saved.augmentation.get("side_range", [0.84, 1.0])) == [1.0, 1.0]
        views = ("vistas sin geometría" if still else "geometría de la campaña") + \
            f", {int(saved.augmentation.get('n_local_crops', 0))} locales"
        label = (f"{'BN' if saved.model.projector.get('use_bn', False) else 'sin BN'}, "
                 f"τ {t0:g}" + (f"→{t1:g}" if t1 != t0 else "") + f", {views}")
        exploration.append((label, epoch_metrics(path), int(saved.model.projector.out_dim)))
    final = [(f"final (BN, τ 0.01, vistas sin geometría, 0 locales) s{d['seed']}",
              d["training_metrics"], 4096)
             for d in docs.get("dinov2_3d", []) if d.get("training_metrics")]
    if exploration:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        palette = ["#b3261e", "#e0892b", "#c9b037", "#6aa84f", "#2a78d6", "#7e57c2",
                   "#9c9a92", "#00897b"]
        for i, (label, metrics, _) in enumerate(exploration + final):
            color = palette[i] if i < len(exploration) else "#0b0b0b"
            style = "-" if i < len(exploration) else ":"
            for ax, key in zip(axes, ("train/teacher_entropy_frac", "train/dino_kl"), strict=True):
                if key in metrics:
                    ax.plot(metrics[key], style, color=color, linewidth=1.6,
                            label=label if i <= len(exploration) else None)
        axes[0].set_title("Entropía del profesor / ln K (1 = colapso uniforme)", fontsize=10)
        axes[1].set_title("KL profesor → estudiante (0 = nada que aprender)", fontsize=10)
        for ax in axes:
            ax.set_xlabel("época", fontsize=9)
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(CAMPAIGN / "figures" / "dinov2_collapse_ablation.png", dpi=160)
        plt.close(fig)

    (CAMPAIGN / "results.json").write_text(json.dumps(
        {"summary": summary, "diagnostics": diag_rows, "comparisons": [
            {"method": m, "comparison": text, "terms": terms, "result": result}
            for m, text, terms, result in comparisons]}, indent=2) + "\n")

    # Figure: per method, CV and test mean-of-3 for SSL (now / before), start and random.
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    bars = [("ssl", "SSL (esta campaña)", "#2a78d6"), ("v1", "SSL (campaña anterior)", "#9dc3ee"),
            ("imagenet", "ImageNet sin SSL (inicio)", "#9c9a92"),
            ("random", "Pesos aleatorios", "#d6d4cc")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey="row")
    for r, (split, title) in enumerate((("cv", "CV train+val"), ("test", "Test"))):
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            for k, (kind, label, color) in enumerate(bars):
                run = {"ssl": f"{m}_3d", "v1": f"{m}_v1_3d"}.get(kind, f"{kind}_{BACKBONE[m]}_3d")
                if run not in summary:
                    continue
                ax.bar(k, summary[run][split], width=0.75, color=color, label=label,
                       edgecolor="white", linewidth=2)
                ax.scatter([k] * summary[run]["n"], summary[run][f"{split}_seeds"], s=8,
                           color="#0b0b0b", zorder=3)
            ax.set_xticks([])
            ax.set_ylim(0.4, 0.95)
            ax.set_title(f"{LABEL[m]} — {title}", fontsize=10, color="#0b0b0b")
            ax.grid(axis="y", alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[r, 0].set_ylabel("Media AUROC, F1-macro, AUPR", fontsize=9)
    handles = {}
    for ax in axes.flat:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(name, handle)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=4,
               frameon=False, fontsize=9)
    fig.suptitle("3D ROI hipocampal: SSL corregido vs anterior vs mismas redes sin SSL "
                 "(30 épocas; puntos = semillas)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(CAMPAIGN / "figures" / "ssl3d_v2.png", dpi=160)
    plt.close(fig)

    ln_k = math.log(4096)
    lines = [
        "# VICReg, BYOL y DINOv2 en 3D (ROI hipocampal), segunda pasada: DINOv2 y BYOL corregidos",
        "",
        "Mismo protocolo que `campaign_ssl3d` (30 épocas, warmup del LR 6 de 30, warmup de la "
        "temperatura del profesor de DINOv2 9 de 30, split balanceado, augmentation campeona "
        f"`{CHAMPION}` recalibrada en 3D, {len(SEEDS)} semillas, encoder congelado). Solo cambia "
        "lo que señaló el diagnóstico del 2026-09-30:",
        "",
        "- **DINOv2** (`dinov2_3d`): cabeza DINO real (MLP con batch norm → bottleneck de 256 "
        "normalizado L2 → prototipos coseno), temperatura del profesor 0.01 constante (con el "
        "0.04 → 0.07 de DINOv2, y también con 0.04 o 0.02 fijos, el profesor derivó a objetivos "
        "uniformes en este conjunto), centrado Sinkhorn-Knopp, "
        "KoLeo sobre el CLS del backbone, enmascarado por bloques de DINOv2 (la mitad de las "
        "vistas globales, ratio 0.1–0.5, sin parche forzado), prototipos congelados la primera "
        "época, weight decay 0.04 → 0.2, EMA 0.99, parches de 8 vóxeles (64 tokens por vista, "
        "antes 8), ViT-S con pesos DeiT-S ImageNet-1k inflados a 3D (el equivalente ViT de los "
        "pesos ImageNet de VICReg y BYOL) y vistas propias (`roi3d_dinov2`): las dos vistas "
        "globales solo difieren en el bias field y las máscaras, sin recorte, rotación ni "
        "recortes locales, porque cualquier cambio geométrico borraba al sujeto de los rasgos "
        "del ViT (ver la ablación al final). Se evalúa el profesor, como en DINO/DINOv2; el "
        "estudiante va aparte.",
        "- **BYOL** (`byol_3d`): EMA 0.99 en vez de 0.996; nada más.",
        "- Importados de `campaign_ssl3d` sin re-correr: VICReg, los controles ConvNeXt/ResNet y "
        "las corridas anteriores de BYOL y DINOv2 (`*_v1_3d`). Los controles ViT son nuevos "
        "porque cambió el backbone: `imagenet_vit_3d` (el inicio de DINOv2) y `random_vit_3d`.",
        "- Evaluación: CV 5×5 sobre los "
        f"{len(labels)} sujetos AD/HC de train+val, y el probe de las campañas en test "
        f"({len(test_labels)} sujetos, {int(test_labels.sum())} AD). Δ con bootstrap pareado "
        "sobre sujetos, promediando semillas, de la media de AUROC, F1-macro y AUPR. "
        "*Rango efectivo*: exp(entropía) de los valores singulares de los embeddings de "
        "train+val (bajo = colapso parcial).",
        "", "![3D v2](figures/ssl3d_v2.png)", "",
        "## Resultados por corrida", "",
        "| Corrida | Semillas | CV AUROC | CV F1 | CV AUPR | CV media | Test AUROC | Test F1 | "
        "Test AUPR | Test media | Rango ef. | Min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in order:
        s = summary[run]
        rank = f"{s['effective_rank']:.1f}" if s["effective_rank"] is not None else "—"
        lines.append(f"| `{run}` | {s['n']} | {s['cv_auc']:.3f} | {s['cv_f1']:.3f} | "
                     f"{s['cv_aupr']:.3f} | **{s['cv']:.3f}** | {s['test_auc']:.3f} | "
                     f"{s['test_f1']:.3f} | {s['test_aupr']:.3f} | **{s['test']:.3f}** | "
                     f"{rank} | {s['minutes']:.0f} |")
    if "student_cv" in summary.get("dinov2_3d", {}):
        s = summary["dinov2_3d"]
        lines += ["", f"Estudiante de DINOv2 (en vez del profesor): CV media {s['student_cv']:.3f}, "
                  f"test media {s['student_test']:.3f}."]
    lines += ["", "## Comparaciones (media de AUROC, F1-macro y AUPR)", "",
              "| Comparación | Δ CV [IC 95 %] | P(Δ>0) | Δ test [IC 95 %] | P(Δ>0) |",
              "|---|---:|---:|---:|---:|"]
    for _, text, _, result in comparisons:
        cv_d, (cv_lo, cv_hi), cv_p = result["cv"]
        te_d, (te_lo, te_hi), te_p = result["test"]
        lines.append(f"| {text} | {cv_d:+.3f} [{cv_lo:+.3f}, {cv_hi:+.3f}] | {cv_p:.2f} | "
                     f"{te_d:+.3f} [{te_lo:+.3f}, {te_hi:+.3f}] | {te_p:.2f} |")
    if diag_rows:
        lines += ["", "## Diagnóstico de entrenamiento de DINOv2 (última época)", "",
                  "![Curvas DINOv2](figures/dinov2_training.png)", "",
                  f"Pérdida DINO del CLS frente a ln K (K = 4096 ahora: {ln_k:.2f}; K = 1024 antes: "
                  f"{math.log(1024):.2f}). *Entropía rel.*: entropía del profesor / ln K (1 = "
                  "colapso uniforme). *KL*: profesor → estudiante (≈ 0 = el estudiante no aprende "
                  "nada que el profesor no sea ya). *Únicos*: fracción de prototipos argmax "
                  "distintos en el batch (≈ 0 = colapso a un prototipo).", "",
                  "| Corrida | Semilla | K | dino época 0 | dino final | dino final / ln K | iBOT final | "
                  "KoLeo final | Entropía rel. | KL | Únicos |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in diag_rows:
            def fmt(key: str, row: dict = row) -> str:
                return f"{row[key]:.3f}" if row.get(key) is not None else "—"
            lines.append(f"| `{row['run']}` | {row['seed']} | {row['prototypes']} | "
                         f"{fmt('dino_first')} | {fmt('train/dino')} | {fmt('dino_over_lnk')} | "
                         f"{fmt('train/ibot')} | {fmt('train/koleo')} | "
                         f"{fmt('train/teacher_entropy_frac')} | {fmt('train/dino_kl')} | "
                         f"{fmt('train/teacher_unique_frac')} |")
    if exploration:
        lines += ["", "## Cómo se llegó a la receta final de DINOv2 (ablación del colapso)", "",
                  "Corridas cortas con la receta de esta campaña cambiando la batch norm de la "
                  "cabeza, la temperatura del profesor y las vistas, detenidas cuando la tendencia "
                  "era clara. La señal clave es la pérdida DINO del estudiante frente a ln K: en ln K "
                  "el estudiante da una salida uniforme y no aprende nada del profesor (con τ baja la "
                  "entropía y la KL del profesor lo disimulan). Con la geometría de la campaña el "
                  "estudiante cae a la uniforme con cualquier cabeza o temperatura: en el ViT, dos "
                  "vistas recortadas/rotadas del mismo sujeto no se parecen más que dos sujetos "
                  "distintos, así que no hay nada predecible entre vistas. Con vistas que solo "
                  "difieren en bias field y máscaras, la pérdida baja de ln K y sigue bajando.", "",
                  "![Ablación](figures/dinov2_collapse_ablation.png)", "",
                  "| Configuración | Épocas | Pérdida DINO / ln K (ép. 0 → última) | "
                  "Entropía rel. del profesor | iBOT |", "|---|---:|---:|---:|---:|"]
        for label, metrics, prototypes in exploration + final:
            dino, ent = metrics.get("train/dino"), metrics.get("train/teacher_entropy_frac")
            ibot = metrics.get("train/ibot")
            if dino and ent and ibot:
                ln_k = math.log(prototypes)
                lines.append(f"| {label} | {len(dino)} | {dino[0] / ln_k:.3f} → "
                             f"{dino[-1] / ln_k:.3f} | {ent[0]:.3f} → {ent[-1]:.3f} | "
                             f"{ibot[0]:.2f} → {ibot[-1]:.2f} |")
    lines += ["", "Regenerar con `python scripts/run_ssl3d_v2.py report`.", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {CAMPAIGN / 'RESULTADOS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("run", choices=[r for r, (_, _, source) in RUNS.items() if source is None])
    one.add_argument("seed", type=int)
    every = sub.add_parser("all")
    every.add_argument("runs", nargs="*", default=list(RUNS))
    every.add_argument("--workers", type=int, default=2)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "run":
        run_one(args.run, args.seed)
    elif args.command == "all":
        run_all(args.workers, args.runs)
    else:
        report()


if __name__ == "__main__":
    sys.exit(main())
