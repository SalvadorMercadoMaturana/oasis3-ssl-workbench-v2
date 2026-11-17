"""Rank the augmentation screen (``screen_augmentations.py``) and render its report.

Sets are ordered by the mean of validation AUROC, F1-macro and AUPR of the
frozen BYOL probe. Every run shares the same validation subjects, so each set
is compared with ``base`` by a paired bootstrap over validation subjects
(resampled within class). Test metrics are listed for reference only.
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from screen_augmentations import CAMPAIGN, ROOT

# Moved to the package (2026-11-16); re-exported for the legacy campaign scripts.
from t5_ssl.evaluation.metrics import fast_mean3, mean3, scores

REFERENCE = "base"
METRICS = (("auc", "AUROC"), ("f1", "F1-macro"), ("aupr", "AUPR"))
VAL_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def load_runs() -> list[dict]:
    with (CAMPAIGN / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "ok"]
    runs = []
    for row in rows:
        doc = json.loads((ROOT / row["experiment_dir"] / "ad_hc_subject_probe_best.json")
                         .read_text(encoding="utf-8"))["results"]
        run = {"name": row["name"], "minutes": float(row["minutes"]),
               "specs": json.loads(row["specs"]), "random": doc["random"]}
        for split in ("val", "test"):
            pred = doc["trained"][f"{split}_predictions"]
            run[f"{split}_subjects"] = [p["subject"] for p in pred]
            run[f"{split}_labels"] = np.array([p["AD"] for p in pred])
            run[f"{split}_probs"] = np.array([p["score"] for p in pred])
            run[split] = scores(run[f"{split}_labels"], run[f"{split}_probs"])
        runs.append(run)
    return runs


def paired_delta(run: dict, ref: dict, repetitions: int = 2000) -> dict:
    """Bootstrap the val mean-of-three difference run - ref over shared subjects."""
    if run["val_subjects"] != ref["val_subjects"]:
        raise RuntimeError(f"{run['name']} was validated on different subjects")
    labels = run["val_labels"]
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(repetitions):
        idx = np.concatenate((rng.choice(positive, len(positive), replace=True),
                              rng.choice(negative, len(negative), replace=True)))
        deltas.append(fast_mean3(labels[idx], run["val_probs"][idx])
                      - fast_mean3(labels[idx], ref["val_probs"][idx]))
    deltas = np.array(deltas)
    return {"delta": mean3(run["val"]) - mean3(ref["val"]),
            "ci": [float(v) for v in np.percentile(deltas, [2.5, 97.5])],
            "p_positive": float(np.mean(deltas > 0))}


def describe(specs: list[dict] | None) -> str:
    if specs is None:
        return "ssl_multiview (flip V+H, rot 15°, blur, ruido, escala)"
    if not specs:
        return "solo base"

    def one(spec: dict) -> str:
        name = spec["name"].removeprefix("Random")
        if name == "OneOf":
            return "OneOf(" + ", ".join(one(child) for child in spec["transforms"]) + f") p={spec['p']}"
        return f"{name} p={spec['p']}" if "p" in spec else name
    return " + ".join(one(spec) for spec in specs)


def figure(runs: list[dict], ref: dict) -> None:
    order = runs[::-1]  # best at the top
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.34 * len(order) + 1.6), sharey=True)
    for ax, (key, title) in zip(axes, METRICS, strict=True):
        ax.axvline(ref["val"][key], color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.scatter([r["val"][key] for r in order], y, s=42, color=VAL_COLOR, zorder=3,
                   label="Validación (criterio)")
        ax.scatter([r["test"][key] for r in order], y, s=42, facecolors="none",
                   edgecolors=TEST_COLOR, linewidths=2, zorder=3, label="Test (referencia)")
        ax.set_title(title, fontsize=11, color=INK)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(y, [r["name"] for r in order], color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Screen de augmentations (BYOL, 30 épocas): probe AD/HC por set, "
                 "ordenado por la media de validación. Línea punteada = base.", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(CAMPAIGN / "figures" / "screen_ranking.png", dpi=160)
    plt.close(fig)


def main() -> None:
    runs = load_runs()
    by_name = {run["name"]: run for run in runs}
    if REFERENCE not in by_name:
        raise RuntimeError(f"Reference run '{REFERENCE}' has not finished")
    ref = by_name[REFERENCE]
    for run in runs:
        run["vs_ref"] = paired_delta(run, ref)
    runs.sort(key=lambda r: mean3(r["val"]), reverse=True)
    (CAMPAIGN / "figures").mkdir(exist_ok=True)
    figure(runs, ref)

    random = ref["random"]
    n_val, n_test = len(ref["val_labels"]), len(ref["test_labels"])
    lines = [
        "# Screen de augmentations tipo hospital (BYOL, 30 épocas)", "",
        "BYOL con ResNet-18 ImageNet1K (el método más rápido de la campaña), 30 épocas en GPU "
        "sobre el split balanceado de `campaign_full50_pretrained_balanced`, una corrida por "
        "set y una sola semilla. Todos los sets salvo `current` parten de "
        "`configs/augmentation/hospital_base.yaml` (crop SSL + flip izquierda-derecha + "
        "rotación ±10°) y agregan artefactos de TorchIO sobre el corte completo antes del crop "
        "([ejemplos](figures/augmentation_examples.png)). `current` es el `ssl_multiview` "
        "usado en las campañas, como referencia.", "",
        f"**Criterio:** media de AUROC, F1-macro y AUPR del probe en validación ({n_val} sujetos "
        f"AD/HC). El test ({n_test} sujetos) se muestra solo como referencia: elegir con él "
        "ajustaría las augmentations al test. Δ y P(Δ>0) comparan cada set con `base` por "
        "bootstrap pareado sobre los mismos sujetos de validación; con una sola corrida por set, "
        "un P(Δ>0) entre ~0,1 y ~0,9 no distingue al set de `base`.", "",
        f"Control aleatorio (idéntico en todas las corridas): AUROC val {random['val_auc']:.3f}, "
        f"test {random['test_auc']:.3f}.", "",
        "![Ranking](figures/screen_ranking.png)", "",
        "| # | Set | AUROC val | F1 val | AUPR val | Media val | Δ media vs base [IC 95 %] | "
        "P(Δ>0) | AUROC test | F1 test | AUPR test | Min |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, run in enumerate(runs, start=1):
        v, t, d = run["val"], run["test"], run["vs_ref"]
        delta = ("—" if run is ref else
                 f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
        p_positive = "—" if run is ref else f"{d['p_positive']:.2f}"
        lines.append(
            f"| {rank} | `{run['name']}` | {v['auc']:.3f} | {v['f1']:.3f} | {v['aupr']:.3f} | "
            f"**{mean3(v):.3f}** | {delta} | {p_positive} | "
            f"{t['auc']:.3f} | {t['f1']:.3f} | {t['aupr']:.3f} | {run['minutes']:.0f} |")
    lines += ["", "## Composición de cada set", "",
              "| Set | Augmentations TorchIO sobre la base |", "|---|---|"]
    lines += [f"| `{run['name']}` | {describe(run['specs'])} |" for run in runs]
    lines += ["", "Parámetros exactos en `scripts/legacy/screen_augmentations.py`; resultados crudos en "
              "`results.csv`. Regenerar con `python scripts/legacy/report_aug_screen.py`.", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    summary = [{"name": r["name"], "val": r["val"], "test": r["test"],
                "val_mean": mean3(r["val"]), "vs_base": r["vs_ref"]} for r in runs]
    (CAMPAIGN / "ranking.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {CAMPAIGN / 'RESULTADOS.md'} ({len(runs)} sets)")


if __name__ == "__main__":
    main()
