"""Tables for the 3D augmentation-recipe screen and the BYOL dropout screen.

    python scripts/report_aug_recipes3d.py campaign_ssl3d_runpod/augs20 [--select 3]
    python scripts/report_aug_recipes3d.py campaign_ssl3d_runpod/dropout20 --dropout

Recipe screen (runs named ``<model>_<variant>__<recipe>_s<seed>``, scored by
train.py with ``--identity``). A recipe is DISCARDED, with criteria fixed before
looking at the results, when its trained encoder

- collapses: effective rank (variance spectrum) of the held-out clean
  embeddings < 1.5, or
- loses identity under its own views: Var_subject / Var_aug < 1 (the
  augmentation moves an embedding more than subjects differ), or centered
  cosine same subject - different subject < 0.10.

Kept recipes are ranked by CV AUROC (seed 1; test is never used to choose).
``--select N`` prints the ``model:recipe`` specs of the N best per model plus
each model's reference recipe (for the extra seeds) and exits. Otherwise it
writes ``TABLA_RECETAS.md`` (every recipe) and ``TABLA_GANADORAS.md`` (winners
over all their seeds, with a paired bootstrap of the AUROC difference against
the model's reference recipe), plus ``recipes.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from t5_ssl import recipes  # noqa: E402

LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
MIN_RANK, MIN_RATIO, MIN_COS_GAP = 1.5, 1.0, 0.10


def load(run_dir: Path) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("*/result.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "cv" not in doc:
            continue
        groups[(doc["model"], doc["variant"], doc.get("aug_recipe") or "")].append(doc)
    return groups


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = int(labels.sum())
    neg = len(labels) - pos
    return float((rankdata(scores)[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def identity(doc: dict, view: str = "own") -> dict:
    return (doc.get("identity") or {}).get(view) or {}


def verdict(doc: dict) -> tuple[bool, str]:
    own = identity(doc)
    if not own:
        return False, "sin métricas de identidad"
    reasons = []
    if own["effective_rank"] < MIN_RANK:
        reasons.append(f"colapso (rango {own['effective_rank']:.2f})")
    if own["between_over_within"] < MIN_RATIO:
        reasons.append(f"Var_suj/Var_aug {own['between_over_within']:.2g} < 1")
    if own["same_subject_cos"] - own["different_subject_cos"] < MIN_COS_GAP:
        reasons.append("cos mismo − distinto < 0.10")
    return (not reasons), "; ".join(reasons) or "pasa"


def val_rise(doc: dict) -> float | None:
    val = (doc.get("training_metrics") or {}).get("val/loss")
    return float(val[-1] - min(val)) if val else None


def mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def std(values: list) -> float:
    values = [v for v in values if v is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def seed1(docs: list[dict]) -> dict:
    return min(docs, key=lambda d: d["seed"])


def rank_recipes(groups: dict) -> dict[str, list[tuple[str, dict, bool, str]]]:
    by_model: dict[str, list[tuple[str, dict, bool, str]]] = defaultdict(list)
    for (model, _, recipe), docs in groups.items():
        if not recipe:
            continue
        doc = seed1(docs)
        ok, why = verdict(doc)
        by_model[model].append((recipe, doc, ok, why))
    for rows in by_model.values():
        rows.sort(key=lambda r: (not r[2], -r[1]["cv_auc"]))
    return by_model


def bootstrap_auc(a: list[dict], b: list[dict], draws: int = 2000) -> dict[str, tuple]:
    """Paired, stratified bootstrap over subjects of the seed-averaged AUROC difference."""
    rng = np.random.default_rng(42)
    out = {}
    for split in ("cv", "test"):
        if split == "cv":
            labels = np.array(a[0]["labels"])
            get = lambda d: np.asarray(d["cv"]["clean"]["oof_mean"])  # noqa: E731
        else:
            labels = np.array([p["AD"] for p in a[0]["test"]["test_predictions"]])
            get = lambda d: np.array([p["score"] for p in d["test"]["test_predictions"]])  # noqa: E731
        sa, sb = [get(d) for d in a], [get(d) for d in b]
        pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        point = mean([auc(labels, s) for s in sa]) - mean([auc(labels, s) for s in sb])
        samples = []
        for _ in range(draws):
            idx = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
            samples.append(np.mean([auc(labels[idx], s[idx]) for s in sa])
                           - np.mean([auc(labels[idx], s[idx]) for s in sb]))
        lo, hi = np.percentile(samples, [2.5, 97.5])
        out[split] = (point, float(lo), float(hi), float(np.mean(np.array(samples) > 0)))
    return out


def recipe_tables(run_dir: Path, groups: dict, n_select: int) -> None:
    ranked = rank_recipes(groups)
    lines = ["# Recetas de augmentation, 20 épocas (semilla 1)", "",
             "Probe congelado (CV 5×5 en 357 sujetos de train+val; test en 56). Identidad en 142 "
             "sujetos no vistos con las vistas de la propia receta (*propia*) y con unas vistas de "
             "referencia comunes (geometría de la campaña con relleno de borde + bias field). "
             f"Descartada si rango efectivo < {MIN_RANK}, Var_sujeto/Var_aug < {MIN_RATIO:g} o "
             f"coseno mismo − distinto < {MIN_COS_GAP:.2f} (criterios fijados antes de correr). "
             "Orden: pasa/descartada, luego AUROC de CV.", ""]
    payload: dict = {}
    for model in ("vicreg", "byol", "dinov2"):
        rows = ranked.get(model, [])
        if not rows:
            continue
        reference = recipes.REFERENCE_RECIPE[model]
        lines += [f"## {LABEL[model]}", "",
                  "| # | Receta | AUROC CV | AUROC test | Media3 CV | Rango ef. | cos mismo | "
                  "cos distinto | Var_suj/Var_aug (propia) | Var_suj/Var_aug (ref.) | Veredicto |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        payload[model] = []
        for i, (recipe, doc, ok, why) in enumerate(rows, 1):
            own, ref = identity(doc), identity(doc, "reference")
            name = f"`{recipe}`" + (" (actual)" if recipe == reference else "")
            lines.append(
                f"| {i} | {name} | {doc['cv_auc']:.3f} | {doc['test_auc']:.3f} | "
                f"{doc['cv_mean3']:.3f} | {own.get('effective_rank', float('nan')):.2f} | "
                f"{own.get('same_subject_cos', float('nan')):.3f} | "
                f"{own.get('different_subject_cos', float('nan')):.3f} | "
                f"{own.get('between_over_within', float('nan')):.3g} | "
                f"{ref.get('between_over_within', float('nan')):.3g} | "
                f"{'✓ ' if ok else '✗ '}{why} |")
            payload[model].append({"recipe": recipe, "ok": ok, "why": why,
                                   "cv_auc": doc["cv_auc"], "test_auc": doc["test_auc"],
                                   "cv_mean3": doc["cv_mean3"], "identity": doc.get("identity")})
        lines.append("")
    (run_dir / "TABLA_RECETAS.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "recipes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    winners = ["# Recetas ganadoras (todas sus semillas)", "",
               f"Las {n_select} mejores recetas por modelo entre las que pasan (AUROC de CV con la "
               "semilla 1), repetidas con 3 semillas, junto a la receta actual del modelo. Media ± "
               "desviación entre semillas; Δ AUROC frente a la receta actual con bootstrap pareado "
               "sobre sujetos (IC 95 %).", "",
               "| Modelo | Receta | n | AUROC CV | AUROC test | Media3 CV | Media3 test | "
               "Rango ef. | Var_suj/Var_aug | Δ AUROC CV vs actual | Δ AUROC test vs actual |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("vicreg", "byol", "dinov2"):
        rows = [r for r in ranked.get(model, []) if r[2]][:n_select]
        reference = recipes.REFERENCE_RECIPE[model]
        ref_docs = next((d for (m, _, r), d in groups.items() if m == model and r == reference),
                        None)
        names = [r[0] for r in rows] + ([reference] if reference not in [r[0] for r in rows]
                                        else [])
        for recipe in names:
            docs = next(d for (m, _, r), d in groups.items() if m == model and r == recipe)
            own = [identity(d) for d in docs]
            delta = ""
            if ref_docs and recipe != reference and len(docs) > 1:
                b = bootstrap_auc(docs, ref_docs)
                delta = " | ".join(f"{b[s][0]:+.3f} [{b[s][1]:+.3f}, {b[s][2]:+.3f}]"
                                   for s in ("cv", "test"))
            else:
                delta = "— | —"
            tag = " (actual)" if recipe == reference else ""
            winners.append(
                f"| {LABEL[model]} | `{recipe}`{tag} | {len(docs)} | "
                f"{mean([d['cv_auc'] for d in docs]):.3f} ± {std([d['cv_auc'] for d in docs]):.3f} | "
                f"{mean([d['test_auc'] for d in docs]):.3f} ± {std([d['test_auc'] for d in docs]):.3f} | "
                f"{mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f} | "
                f"{mean([o.get('effective_rank') for o in own]):.2f} | "
                f"{mean([o.get('between_over_within') for o in own]):.3g} | {delta} |")
    winners.append("")
    (run_dir / "TABLA_GANADORAS.md").write_text("\n".join(winners), encoding="utf-8")
    print(f"Wrote {run_dir / 'TABLA_RECETAS.md'} and {run_dir / 'TABLA_GANADORAS.md'}")


def dropout_table(run_dir: Path, groups: dict) -> None:
    lines = ["# BYOL `border_wide` con dropout, 20 épocas (3 semillas)", "",
             "*Subida val* = pérdida de validación final − su mínimo (sobreajuste del objetivo); "
             "*brecha* = validación − entrenamiento en la última época.", "",
             "| Variante | n | AUROC CV | AUROC test | Media3 CV | Media3 test | Subida val | "
             "Brecha val−train | Var_suj/Var_aug | Rango ef. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for (_, variant, _), docs in groups.items():
        gap = [(d["training_metrics"]["val/loss"][-1] - d["training_metrics"]["train/loss_epoch"][-1])
               for d in docs if d.get("training_metrics")]
        rows.append((mean([d["cv_auc"] for d in docs]), variant, docs, gap))
    for cv, variant, docs, gap in sorted(rows, reverse=True):
        own = [identity(d) for d in docs]
        lines.append(
            f"| `{variant}` | {len(docs)} | {cv:.3f} | {mean([d['test_auc'] for d in docs]):.3f} | "
            f"{mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f} | "
            f"{mean([val_rise(d) for d in docs]):.3f} | {mean(gap):.3f} | "
            f"{mean([o.get('between_over_within') for o in own]):.3g} | "
            f"{mean([o.get('effective_rank') for o in own]):.2f} |")
    (run_dir / "TABLA_DROPOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--select", type=int, default=0)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--dropout", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    groups = load(run_dir)
    if args.dropout:
        dropout_table(run_dir, groups)
        return
    if args.select:
        specs = []
        for model, rows in rank_recipes(groups).items():
            chosen = [r[0] for r in rows if r[2]][:args.select]
            reference = recipes.REFERENCE_RECIPE[model]
            specs += [f"{model}:{r}" for r in dict.fromkeys([*chosen, reference])]
        print(" ".join(specs))
        return
    recipe_tables(run_dir, groups, args.top)


if __name__ == "__main__":
    main()
