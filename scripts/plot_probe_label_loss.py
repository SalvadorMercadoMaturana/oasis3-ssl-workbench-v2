"""Compare OASIS3 AD/HC probe log-loss against image and prevalence baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from omegaconf import OmegaConf
from sklearn.metrics import log_loss

from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tuning_pretext"
METHODS = ("vicreg", "byol", "dinov2")
DISPLAY = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}


def predictions(document: dict, kind: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = document["results"][kind]["test_predictions"]
    subjects = [row["subject"] for row in rows]
    if len(subjects) != len(set(subjects)):
        raise RuntimeError("Duplicate test subjects")
    labels = np.asarray([row["AD"] for row in rows], dtype=int)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    if not set(labels) <= {0, 1} or not np.isfinite(scores).all():
        raise RuntimeError("Invalid AD/HC predictions")
    return subjects, labels, scores


def train_prior(experiment: Path) -> tuple[float, int, int]:
    train_ids: set[str] = set()
    with (experiment / "subject_splits.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "train":
                train_ids.add(row["subject"])
    cfg = OmegaConf.load(experiment / "config.yaml")
    all_labels: dict[str, set[str]] = {}
    for row in clinical_rows(cfg.data.csv_path):
        all_labels.setdefault(row["subject"], set()).add(row["group"])
    eligible = {subject: next(iter(values)) for subject, values in all_labels.items()
                if subject in train_ids and len(values) == 1 and
                next(iter(values)) in ("AD", "HC")}
    n_ad = sum(label == "AD" for label in eligible.values())
    n = len(eligible)
    if not 0 < n_ad < n:
        raise RuntimeError("Train prior requires both AD and HC")
    return n_ad / n, n_ad, n


def per_subject_loss(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, 1e-15, 1 - 1e-15)
    return -(labels * np.log(scores) + (1 - labels) * np.log1p(-scores))


def paired_interval(labels: np.ndarray, first: np.ndarray, second: np.ndarray
                    ) -> tuple[float, list[float]]:
    differences = per_subject_loss(labels, first) - per_subject_loss(labels, second)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(42)
    samples = np.empty(2000, dtype=float)
    for index in range(len(samples)):
        chosen = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        samples[index] = differences[chosen].mean()
    return float(differences.mean()), [float(value) for value in
                                        np.quantile(samples, [0.025, 0.975])]


def collect() -> dict:
    rows = []
    common_subjects: list[str] | None = None
    common_labels: np.ndarray | None = None
    common_prior: float | None = None
    for method in METHODS:
        refit = json.loads((OUT / f"{method}_refit.json").read_text(encoding="utf-8"))
        original_dir = Path(refit["baseline_experiment"])
        tuned_dir = Path(refit["tuned_experiment"])
        paths = {
            "original_last": original_dir / "ad_hc_subject_probe.json",
            "tuned_best": tuned_dir / "ad_hc_subject_probe_best.json",
            "tuned_last": tuned_dir / "ad_hc_subject_probe.json",
        }
        documents = {name: json.loads(path.read_text(encoding="utf-8"))
                     for name, path in paths.items()}
        if len({doc["test_manifest"] for doc in documents.values()}) != 1:
            raise RuntimeError(f"Different test manifests for {method}")
        if documents["tuned_best"]["trained_checkpoint"] == documents["tuned_last"]["trained_checkpoint"]:
            raise RuntimeError(f"Best and last checkpoints were not distinguished for {method}")
        variants = {name: predictions(doc, "trained") for name, doc in documents.items()}
        random_variants = {name: predictions(doc, "random") for name, doc in documents.items()}
        subjects, labels, best_scores = variants["tuned_best"]
        if common_subjects is not None and (subjects != common_subjects or
                                            not np.array_equal(labels, common_labels)):
            raise RuntimeError("Methods evaluated different test subjects")
        common_subjects, common_labels = subjects, labels
        for name, (ids, y, _) in {**variants, **{f"random_{k}": v for k, v in random_variants.items()}}.items():
            if ids != subjects or not np.array_equal(y, labels):
                raise RuntimeError(f"Misaligned labels: {method}/{name}")
        random_scores = random_variants["tuned_best"][2]
        if any(not np.array_equal(random_scores, variant[2])
               for variant in random_variants.values()):
            raise RuntimeError(f"Random control changed for {method}")
        prior, n_train_ad, n_train = train_prior(tuned_dir)
        if common_prior is not None and prior != common_prior:
            raise RuntimeError("Methods used different labelled train subjects")
        common_prior = prior
        losses = {name: float(log_loss(labels, variant[2], labels=[0, 1]))
                  for name, variant in variants.items()}
        losses["random_encoder"] = float(log_loss(labels, random_scores, labels=[0, 1]))
        losses["train_prior"] = float(log_loss(labels, np.full(len(labels), prior), labels=[0, 1]))
        delta, interval = paired_interval(labels, best_scores, random_scores)
        rows.append({"method": method, "losses": losses,
                     "delta_best_minus_random": delta,
                     "delta_95ci_subject_bootstrap": interval,
                     "auroc_best": documents["tuned_best"]["results"]["trained"]["test_auc"],
                     "auroc_random": documents["tuned_best"]["results"]["random"]["test_auc"],
                     "n_train_ad": n_train_ad, "n_train_ad_hc": n_train,
                     "n_val_ad_hc": documents["tuned_best"]["results"]["trained"]["val_subjects"],
                     "source_files": {name: str(path) for name, path in paths.items()}})
    assert common_labels is not None and common_prior is not None
    return {"metric": "subject-level binary cross-entropy / log-loss; lower is better",
            "protocol": "OASIS3 only; frozen encoder; logistic regression trained on labelled train subjects; "
                        "L2 C chosen by validation AUROC; fixed expanded held-out test; "
                        "SSL checkpoints chosen without AD/HC labels",
            "test_subjects": len(common_labels), "test_ad": int(common_labels.sum()),
            "test_hc": int(len(common_labels) - common_labels.sum()),
            "train_ad_prior": common_prior,
            "bootstrap_note": "Paired stratified subject bootstrap; intervals condition on one fitted "
                              "probe, one split, and one model seed",
            "results": rows}


def draw_table(report: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 7.5), dpi=170)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#f4f7fb")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.035, 0.95, "OASIS3  |  Pérdida de clasificación AD vs HC",
            fontsize=23, weight="bold", color="#15324b", va="center")
    ax.text(0.035, 0.893, "Test independiente: 806 sujetos (186 AD · 620 HC)  ·  log-loss binaria; menor es mejor",
            fontsize=12.5, color="#42576b", va="center")
    widths = [0.105, 0.125, 0.135, 0.125, 0.125, 0.125, 0.19]
    starts = [0.035]
    for width in widths[:-1]:
        starts.append(starts[-1] + width)

    def cell(x: float, y: float, w: float, h: float, color: str, *, edge: str = "#dce4ed") -> None:
        ax.add_patch(FancyBboxPatch((x + 0.002, y + 0.004), w - 0.004, h - 0.008,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    facecolor=color, edgecolor=edge, linewidth=0.8))

    # Group headers make the reference definitions visible at a glance.
    groups = [(starts[1], sum(widths[1:4]), "PROBES CON ENCODER CONGELADO", "#1f5f86"),
              (starts[4], sum(widths[4:6]), "BASELINES", "#607487"),
              (starts[6], widths[6], "COMPARACIÓN", "#1c7769")]
    for x, w, label, color in groups:
        cell(x, 0.785, w, 0.065, color, edge=color)
        ax.text(x + w / 2, 0.817, label, ha="center", va="center", fontsize=10.5,
                color="white", weight="bold")
    headers = ("MODELO", "Original\n100 ep", "Ajustado\nmejor SSL", "Ajustado\népoca 100",
               "Encoder\naleatorio", "Prior AD\nsin imagen", "Δ mejor − aleatorio\nIC 95 %")
    for x, w, label in zip(starts, widths, headers, strict=True):
        cell(x, 0.67, w, 0.105, "#e3eaf2")
        ax.text(x + w / 2, 0.722, label, ha="center", va="center", fontsize=11.2,
                weight="bold", color="#223c53", linespacing=1.2)

    for index, row in enumerate(report["results"]):
        y = 0.535 - index * 0.145
        losses = row["losses"]
        values = (DISPLAY[row["method"]], f"{losses['original_last']:.4f}",
                  f"{losses['tuned_best']:.4f}", f"{losses['tuned_last']:.4f}",
                  f"{losses['random_encoder']:.4f}", f"{losses['train_prior']:.4f}")
        for col, (x, w, value) in enumerate(zip(starts, widths, values, strict=True)):
            color = "#163f5b" if col == 0 else ("#ebf4fa" if col in (1, 2, 3)
                                                   else "#eff2f5")
            cell(x, y, w, 0.13, color)
            ax.text(x + w / 2, y + 0.065, value, ha="center", va="center",
                    fontsize=14 if col else 14.5, color="white" if col == 0 else "#203b50",
                    weight="bold" if col in (0, 2) else "medium")
        delta = row["delta_best_minus_random"]
        low, high = row["delta_95ci_subject_bootstrap"]
        color = "#d9f0e6" if high < 0 else "#fff1d7" if low <= 0 <= high else "#f9dfdf"
        cell(starts[6], y, widths[6], 0.13, color)
        ax.text(starts[6] + widths[6] / 2, y + 0.083, f"{delta:+.4f}",
                ha="center", va="center", fontsize=14.5, weight="bold", color="#1b604d" if high < 0 else "#79551d")
        ax.text(starts[6] + widths[6] / 2, y + 0.037, f"[{low:+.4f}, {high:+.4f}]",
                ha="center", va="center", fontsize=9.7, color="#485d6e")

    ax.text(0.035, 0.17, "Δ < 0 favorece al probe SSL. El prior usa 14 AD / 61 sujetos de train; no ve imágenes.",
            fontsize=11, color="#30485c")
    ax.text(0.035, 0.122, "Probe: regresión logística sobre embeddings; C elegido en 13 sujetos de validación.",
            fontsize=10.5, color="#526679")
    ax.text(0.035, 0.08, "IC: bootstrap emparejado por sujeto; no incluye variación entre splits o semillas. AUROC DINOv2: 0.745 vs 0.701 aleatorio.",
            fontsize=10.5, color="#526679")
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    report = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_label_loss_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    draw_table(report, figures / "probe_baseline_label_loss.png")
    lines = ["# Pérdida del probe AD/HC frente a baselines", "",
             "Log-loss binaria media por sujeto en 806 sujetos de test OASIS3 "
             "(186 AD, 620 HC). Menor es mejor. El probe usa el encoder congelado, "
             "regresión logística entrenada en 61 sujetos AD/HC y regularización "
             "elegida por AUROC en 13 sujetos de validación. Los checkpoints SSL "
             "se eligieron sin mirar AD/HC.", "",
             "| Método | Original 100 ep | Ajustado mejor SSL | Ajustado ep 100 | "
             "Encoder aleatorio | Prior AD | Δ mejor − aleatorio [IC 95 %] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        loss = row["losses"]
        lo, hi = row["delta_95ci_subject_bootstrap"]
        lines.append(f"| {DISPLAY[row['method']]} | {loss['original_last']:.4f} | "
                     f"{loss['tuned_best']:.4f} | {loss['tuned_last']:.4f} | "
                     f"{loss['random_encoder']:.4f} | {loss['train_prior']:.4f} | "
                     f"{row['delta_best_minus_random']:+.4f} "
                     f"[{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "El baseline de encoder aleatorio también entrena la regresión "
              "logística con labels; no equivale a adivinar. El baseline de prior "
              "predice siempre la prevalencia AD de train (14/61). Los IC son "
              "bootstrap emparejados sobre estos sujetos de test y no capturan "
              "variación entre modelos, semillas o splits. Un AUROC alto puede "
              "coexistir con peor log-loss si las probabilidades están mal calibradas.", "",
              "Figura: `figures/probe_baseline_label_loss.png`. Métricas y fuentes: "
              "`probe_label_loss_results.json`.", ""]
    (OUT / "PROBE_LABEL_LOSS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"figure": str(figures / "probe_baseline_label_loss.png"),
                      "results": report["results"]}, indent=2))


if __name__ == "__main__":
    main()
