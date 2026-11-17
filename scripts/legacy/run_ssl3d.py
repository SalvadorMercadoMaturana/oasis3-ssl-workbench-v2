"""VICReg, BYOL and DINOv2 on 3D hippocampal ROI cubes vs 2D slices, under one short protocol.

Does 3D improve AD/HC classification, and does it make self-supervision contribute
more? Every run: 30 epochs on the campaign schedule compressed proportionally (LR
warmup 6/30; DINOv2 teacher-temperature warmup 9/30), the balanced split of
``campaign_full50_pretrained_balanced``, the screen-2 champion augmentation
(``bias``; recalibrated in 3D), 3 seeds.

Backbones follow ``campaign_full50_pretrained``: VICReg ConvNeXt-tiny ImageNet with
projector dropout 0.1, BYOL ResNet-18 ImageNet, DINOv2 ViT-Small from scratch with
a few coarse tokens. The 3D versions inflate the ImageNet weights (I3D) and see the
two-channel hippocampal pair [left, right mirrored] of 32 voxels (64 mm) per side,
which contains the axial, coronal and sagittal planes around each hippocampus.

Controls without SSL for each backbone and dimensionality: ``imagenet_*`` (where
VICReg/BYOL start) and ``random_*`` (the campaigns' control, and DINOv2's start).
Evaluation as ``run_roi3d.py`` without the artifact variants: frozen encoder, 5x5
cross-validated probe on the train+val AD/HC subjects, and the campaign probe
(fit on train, C by validation AUROC) scored once on test.

BYOL seeds 1-2 and the ImageNet ResNet controls are imported from
``campaign_roi3d``, which ran exactly this protocol.

Usage: ``all --workers 2``, then ``report``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import matplotlib
import numpy as np
import run_roi3d
import screen_aug_equal as screen2
import torch
from evaluate_frozen_probe import evaluate
from report_aug_screen import fast_mean3, mean3
from screen_augmentations import MANIFEST, ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMPAIGN = ROOT / "campaign_ssl3d"
PY = str(ROOT / ".venv/Scripts/python.exe")
SEEDS = (1, 2, 3)
CHAMPION = "bias"
METHODS = ("vicreg", "byol", "dinov2")
LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
DIM_LABEL = {"2d": "2D corte completo", "3d": "3D ROI hipocampal"}
BACKBONE = {"vicreg": "convnext", "byol": "resnet", "dinov2": "vit"}
METHOD_OF = {backbone: method for method, backbone in BACKBONE.items()}
MODEL = {
    ("vicreg", "2d"): ["model=convnext", "model.backbone.pretrained=true", "loss=vicreg",
                       "augmentation=ssl_multiview"],
    ("vicreg", "3d"): ["model=vicreg_convnext_3d", "loss=vicreg", "augmentation=roi3d"],
    ("byol", "2d"): ["model=byol_resnet18", "loss=byol", "augmentation=ssl_multiview"],
    ("byol", "3d"): ["model=byol_resnet18_3d", "loss=byol", "augmentation=roi3d"],
    ("dinov2", "2d"): ["model=dinov2_vit", "loss=dinov2", "augmentation=dinov2_multicrop",
                       "loss.teacher_temp_warmup_epochs=9"],
    ("dinov2", "3d"): ["model=dinov2_vit3d", "loss=dinov2", "augmentation=roi3d_multicrop",
                       "loss.teacher_temp_warmup_epochs=9"],
}
RUNTIME = ["trainer.accelerator=gpu", "trainer.precision=32-true",
           "trainer.deterministic=false", "trainer.enable_progress_bar=false",
           "trainer.early_stopping_patience=null", "trainer.enable_checkpointing=false",
           "trainer.enable_health_check=false"]
COMMON_2D = [o for o in screen2.COMMON
             if not o.startswith(("trainer.max_epochs=", "optimizer.warmup_epochs="))]
COMMON_3D = ["data=oasis3_roi3d", "data.roi_size=32", f"data.manifest_path={MANIFEST}",
             "trainer=full50", *RUNTIME]
SSL_RUNS = [f"{m}_{d}" for d in ("3d", "2d") for m in METHODS]
CONTROLS = [f"imagenet_{b}_{d}" for d in ("2d", "3d") for b in ("convnext", "resnet")] + \
           [f"random_{b}_{d}" for d in ("2d", "3d") for b in ("convnext", "resnet", "vit")]
IMPORTED = {"byol_2d": "byol_2d", "byol_3d": "byol_3d32",
            "imagenet_resnet_2d": "init_2d", "imagenet_resnet_3d": "init_3d32"}


def parse(run: str) -> tuple[str, str, str]:
    """Return (kind, method, dim) with kind in {ssl, imagenet, random}."""
    parts = run.split("_")
    if len(parts) == 2:
        return "ssl", parts[0], parts[1]
    return parts[0], METHOD_OF[parts[1]], parts[2]


def overrides_for(run: str, seed: int) -> list[str]:
    kind, method, dim = parse(run)
    if dim == "2d":
        calib = json.loads((screen2.CAMPAIGN / "calibration.json").read_text(encoding="utf-8"))
        aug = screen2.aug_overrides(CHAMPION, calib)
        base = [*COMMON_2D, *run_roi3d.TRAINER]
    else:
        calib = json.loads((run_roi3d.CAMPAIGN / "calibration_s32.json").read_text(encoding="utf-8"))
        aug = run_roi3d.aug3d_overrides(CHAMPION, calib)[1:]  # group comes from MODEL
        base = [*COMMON_3D, *run_roi3d.TRAINER]
    out = [*base, *MODEL[(method, dim)], f"seed={seed}", *aug]
    if kind == "random":
        out.append("model.backbone.pretrained=false")
    return out


def run_one(run: str, seed: int) -> None:
    import warnings

    from hydra import compose, initialize_config_dir

    from t5_ssl.data.augmentations import EvalTransform
    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_backbone, build_datamodule, train
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    kind, _, dim = parse(run)
    overrides = overrides_for(run, seed)
    start = time.time()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    clean = (EvalTransform(tuple(cfg.augmentation.image_size)) if dim == "2d"
             else ROIEval3D(int(cfg.data.roi_size)))
    if kind == "ssl":
        result = train(cfg)
        backbone, dm, exp_dir = result["model"].backbone, result["datamodule"], result["exp_dir"]
    else:
        torch.manual_seed(seed)
        backbone = build_backbone(cfg)
        dm = build_datamodule(cfg, clean, clean)
        dm.setup()
        exp_dir = None
    x, y, subjects = screen2.embed(backbone, dm, clean)
    cv = screen2.cv_probe(x, y, {"clean": x})
    test = evaluate(*(run_roi3d.split_embeddings(backbone, dm, clean, s)
                      for s in ("train", "val", "test")))
    out = {"run": run, "seed": seed, "augmentation": CHAMPION, "overrides": overrides,
           "experiment_dir": str(exp_dir) if exp_dir else None,
           "minutes": (time.time() - start) / 60, "subjects": subjects, "labels": y.tolist(),
           "cv": cv, "test": test}
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "runs" / f"{run}_s{seed}.json").write_text(json.dumps(out) + "\n")


def import_previous() -> None:
    """Copy the campaign_roi3d runs that used exactly this protocol."""
    (CAMPAIGN / "runs").mkdir(parents=True, exist_ok=True)
    for new, old in IMPORTED.items():
        for path in sorted((run_roi3d.CAMPAIGN / "runs").glob(f"{old}_s*.json")):
            target = CAMPAIGN / "runs" / path.name.replace(old, new, 1)
            if not target.is_file():
                doc = json.loads(path.read_text(encoding="utf-8"))
                doc.update(run=new, imported_from=str(path.relative_to(ROOT)))
                target.write_text(json.dumps(doc) + "\n")
    shutil.copy(run_roi3d.CAMPAIGN / "calibration_s32.json", CAMPAIGN / "calibration_s32.json")


def jobs(runs: list[str]) -> list[tuple[str, int]]:
    todo = []
    for run in runs:
        seeds = SEEDS[:1] if run.startswith("imagenet") else SEEDS  # ImageNet init is fixed
        todo += [(run, s) for s in seeds
                 if not (CAMPAIGN / "runs" / f"{run}_s{s}.json").is_file()]
    # Longest first (3D SSL, then 2D SSL, then the evaluation-only controls).
    return sorted(todo, key=lambda job: (parse(job[0])[0] != "ssl", parse(job[0])[2] != "3d",
                                          job[1]))


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
            t = doc["test"]
            detail = (f"OK cv={mean3(doc['cv']['clean']['metrics']):.3f} test AUROC="
                      f"{t['test_auc']:.3f} F1={t['test_f1_macro']:.3f} "
                      f"AUPR={t['test_average_precision']:.3f}")
        print(f"{time.strftime('%H:%M:%S')} {run} s{seed} {detail} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, jobs(runs)))


# ---------------------------------------------------------------- report
def report(only_3d: bool = False) -> None:
    docs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((CAMPAIGN / "runs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not only_3d or doc["run"].endswith("_3d"):
            docs[doc["run"]].append(doc)
    first = next(iter(docs.values()))[0]
    labels = np.array(first["labels"])
    test_first = first["test"]["test_predictions"]
    test_subjects = [p["subject"] for p in test_first]
    test_labels = np.array([p["AD"] for p in test_first])
    for run, ds in docs.items():
        for d in ds:
            if d["subjects"] != first["subjects"] or \
                    [p["subject"] for p in d["test"]["test_predictions"]] != test_subjects:
                raise RuntimeError(f"{run} was scored on different subjects")

    summary: dict[str, dict[str, Any]] = {}
    for run, ds in docs.items():
        row: dict[str, Any] = {"n": len(ds), "minutes": float(np.mean([d["minutes"] for d in ds]))}
        for key, field in (("auc", "test_auc"), ("f1", "test_f1_macro"),
                           ("aupr", "test_average_precision")):
            row[f"cv_{key}"] = float(np.mean([d["cv"]["clean"]["metrics"][key] for d in ds]))
            row[f"test_{key}"] = float(np.mean([d["test"][field] for d in ds]))
        row["cv"] = float(np.mean([mean3(d["cv"]["clean"]["metrics"]) for d in ds]))
        row["test"] = (row["test_auc"] + row["test_f1"] + row["test_aupr"]) / 3
        row["cv_seeds"] = [mean3(d["cv"]["clean"]["metrics"]) for d in ds]
        row["test_seeds"] = [(d["test"]["test_auc"] + d["test"]["test_f1_macro"]
                              + d["test"]["test_average_precision"]) / 3 for d in ds]
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
        """Bootstrap a signed sum of seed-averaged mean-of-3 scores, e.g. [(1, a), (-1, b)]."""
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

    start = {m: ("random" if m == "dinov2" else "imagenet") for m in METHODS}
    comparisons = []
    for m in METHODS:
        b = BACKBONE[m]
        spec = [(f"3D vs 2D ({LABEL[m]} SSL)", [(1, f"{m}_3d"), (-1, f"{m}_2d")])]
        for dim in ("2d", "3d"):
            init = f"{start[m]}_{b}_{dim}"
            spec.append((f"Aporte SSL {dim.upper()} ({LABEL[m]} − {init})",
                         [(1, f"{m}_{dim}"), (-1, init)]))
            if start[m] != "random":
                spec.append((f"{LABEL[m]} {dim.upper()} vs pesos aleatorios",
                             [(1, f"{m}_{dim}"), (-1, f"random_{b}_{dim}")]))
        spec.append((f"¿3D aumenta el aporte SSL? ({LABEL[m]}: Δ3D − Δ2D vs {start[m]})",
                     [(1, f"{m}_3d"), (-1, f"{start[m]}_{b}_3d"),
                      (-1, f"{m}_2d"), (1, f"{start[m]}_{b}_2d")]))
        comparisons += [(m, text, terms, contrast(terms)) for text, terms in spec
                        if all(run in docs for _, run in terms)]
    (CAMPAIGN / "results.json").write_text(json.dumps(
        {"summary": summary, "comparisons": [
            {"method": m, "comparison": text, "terms": terms, "result": result}
            for m, text, terms, result in comparisons]}, indent=2) + "\n")

    # Figure: per method, CV and test mean-of-3 for SSL / ImageNet start / random, 2D vs 3D.
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    kinds = [("ssl", "SSL", "#2a78d6"), ("imagenet", "ImageNet sin SSL", "#9c9a92"),
             ("random", "Pesos aleatorios", "#d6d4cc")]
    dims = ("3d",) if only_3d else ("2d", "3d")
    fig, axes = plt.subplots(2, 3, figsize=(12 if len(dims) == 2 else 9, 6.4), sharey="row")
    for r, (split, title) in enumerate((("cv", "CV train+val"), ("test", "Test"))):
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            for k, (kind, label, color) in enumerate(kinds):
                for x, dim in enumerate(dims):
                    run = f"{m}_{dim}" if kind == "ssl" else f"{kind}_{BACKBONE[m]}_{dim}"
                    if run not in summary:
                        continue
                    pos = x + (k - 1) * 0.22
                    ax.bar(pos, summary[run][split], width=0.2, color=color,
                           label=label if x == 0 else None, edgecolor="white", linewidth=2)
                    ax.scatter([pos] * summary[run]["n"], summary[run][f"{split}_seeds"],
                               s=8, color="#0b0b0b", zorder=3)
            ax.set_xticks(range(len(dims)), [DIM_LABEL[d] for d in dims], fontsize=9)
            ax.set_ylim(0.4, 0.95)
            ax.set_title(f"{LABEL[m]} — {title}", fontsize=10, color="#0b0b0b")
            ax.grid(axis="y", alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[r, 0].set_ylabel("Media AUROC, F1-macro, AUPR", fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(("3D ROI hipocampal: SSL vs mismas redes sin SSL" if only_3d else
                  "2D vs 3D y aporte SSL por método")
                 + " (30 épocas, split balanceado; puntos = semillas)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(CAMPAIGN / "figures" / "ssl3d_vs_2d.png", dpi=160)
    plt.close(fig)

    lines = [
        ("# VICReg, BYOL y DINOv2 en 3D (ROI hipocampal): clasificación y aporte SSL" if only_3d
         else "# VICReg, BYOL y DINOv2 en 3D (ROI hipocampal) vs 2D: clasificación y aporte SSL"), "",
        "Mismo protocolo corto para todo: 30 épocas con el calendario de la campaña comprimido "
        "(warmup del LR 6 de 30; warmup de la temperatura del teacher de DINOv2 9 de 30), split "
        "balanceado de `campaign_full50_pretrained_balanced`, augmentation campeona del screen 2 "
        f"(`{CHAMPION}`, recalibrada en 3D), {len(SEEDS)} semillas por método.", "",
        "- Backbones de `campaign_full50_pretrained`: VICReg ConvNeXt-tiny ImageNet (+ dropout "
        "0,1 en el proyector), BYOL ResNet-18 ImageNet, DINOv2 ViT-Small desde cero con pocos "
        "tokens gruesos (2D: parches de 60 px sobre 120 px = 4 tokens; 3D: parches de 16 vóxeles "
        "sobre el cubo de 32 = 8 tokens, recortes locales de 16 = 1 token).",
        "- **2D**: corte axial completo (la entrada de las campañas). **3D**: par de cubos "
        "hipocampales de 32 vóxeles (64 mm) [izquierdo, derecho espejado] centrados en cada "
        "hipocampo, que contienen los planos axial, coronal y sagital de la región; pesos de "
        "ImageNet inflados a 3D (I3D).",
        "- Controles sin SSL por backbone y dimensión: `imagenet_*` (el punto de partida de "
        "VICReg y BYOL) y `random_*` (el control de las campañas y el punto de partida de "
        "DINOv2). El *aporte SSL* se mide contra el punto de partida de cada método.",
        "- Evaluación con encoder congelado: CV 5×5 sobre los "
        f"{len(labels)} sujetos AD/HC de train+val, y el probe de las campañas en test "
        f"({len(test_labels)} sujetos, {int(test_labels.sum())} AD). Δ con bootstrap pareado "
        "sobre sujetos, promediando semillas, de la media de AUROC, F1-macro y AUPR.",
        "- BYOL semillas 1–2 y los controles ImageNet de ResNet vienen de `campaign_roi3d` "
        "(mismo protocolo).", "",
        "![2D vs 3D](figures/ssl3d_vs_2d.png)", "",
        "## Resultados por corrida", "",
        "| Corrida | Semillas | CV AUROC | CV F1 | CV AUPR | CV media | Test AUROC | Test F1 | "
        "Test AUPR | Test media | Min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = [*SSL_RUNS, *CONTROLS]
    for run in [r for r in order if r in summary]:
        s = summary[run]
        lines.append(f"| `{run}` | {s['n']} | {s['cv_auc']:.3f} | {s['cv_f1']:.3f} | "
                     f"{s['cv_aupr']:.3f} | **{s['cv']:.3f}** | {s['test_auc']:.3f} | "
                     f"{s['test_f1']:.3f} | {s['test_aupr']:.3f} | **{s['test']:.3f}** | "
                     f"{s['minutes']:.0f} |")
    lines += ["", "## Comparaciones (media de AUROC, F1-macro y AUPR)", "",
              "| Comparación | Δ CV [IC 95 %] | P(Δ>0) | Δ test [IC 95 %] | P(Δ>0) |",
              "|---|---:|---:|---:|---:|"]
    for _, text, _, result in comparisons:
        cv_d, (cv_lo, cv_hi), cv_p = result["cv"]
        te_d, (te_lo, te_hi), te_p = result["test"]
        lines.append(f"| {text} | {cv_d:+.3f} [{cv_lo:+.3f}, {cv_hi:+.3f}] | {cv_p:.2f} | "
                     f"{te_d:+.3f} [{te_lo:+.3f}, {te_hi:+.3f}] | {te_p:.2f} |")
    lines += ["", "Regenerar con `python scripts/legacy/run_ssl3d.py report`.", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {CAMPAIGN / 'RESULTADOS.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("run", choices=[*SSL_RUNS, *CONTROLS])
    one.add_argument("seed", type=int)
    every = sub.add_parser("all")
    every.add_argument("runs", nargs="*", default=[*SSL_RUNS, *CONTROLS])
    every.add_argument("--workers", type=int, default=2)
    rep = sub.add_parser("report")
    rep.add_argument("--only-3d", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run_one(args.run, args.seed)
    elif args.command == "all":
        run_all(args.workers, args.runs)
    else:
        report(args.only_3d)


if __name__ == "__main__":
    sys.exit(main())
