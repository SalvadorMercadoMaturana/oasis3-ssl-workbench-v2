"""Deployment split report: supervised vs SSL vs sin SSL, ADNI, the ADNI-SSL variant, the model.

    python scripts/report_deploy.py [--runs campaign_ssl3d_runpod/deploy/runs]

Reads the pod tags ``dep50_ssl`` (each method on its best backbone, 3 seeds),
``dep50a_ssl`` (the same with 174 ADNI SSL-only subjects, seed 1), ``dep50_init`` (the
encoders before SSL) and ``dep50_sup`` (end-to-end supervised), all on split_deploy, and
``deploy/ad_mri_v1/validation.json`` (scripts/build_deploy_model.py) when present.

Writes ``<runs>/REPORT.md`` and figures ``campaign_ssl3d_runpod/figures/dep50_*.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_bb50 as rb  # noqa: E402

base, plt = rb.base, rb.plt
SPLIT = base.CAMPAIGN / "split_deploy"
CELLS = [("vicreg", "convnext_tiny"), ("byol", "convnext_tiny"), ("dinov2", "vit_small")]
TICK = {"vicreg": "VICReg\nConvNeXt-T", "byol": "BYOL\nConvNeXt-T", "dinov2": "DINOv2\nViT-S"}


def ages() -> tuple[dict[str, float], dict[str, set[str]]]:
    """Age at the evaluated session of every split_deploy and ADNI subject, and the labels."""
    rows = list(csv.DictReader((base.ROOT / "data/dementia/mni-dataset.csv").open(encoding="utf-8")))
    by_idx = {r["idx"]: r for r in rows}
    out, split = {}, {"train": set(), "val": set(), "test": set(), "adni": set()}
    for row in csv.DictReader((SPLIT / "manifest.csv").open(encoding="utf-8")):
        if row["labeled"] == "1":
            out.setdefault(row["subject"], float(by_idx[row["idx"]]["age"]))
            split[row["split"]].add(row["subject"])
    for row in csv.DictReader((SPLIT / "external_adni.csv").open(encoding="utf-8")):
        out[row["subject"]] = float(row["age"])
        split["adni"].add(row["subject"])
    return out, split


def age_reference(any_doc: dict) -> dict[str, dict[str, float]]:
    """Age-only class-balanced logistic regression (train -> each evaluation set)."""
    age, split = ages()
    labels = {s: a for s, (a, _) in rb.predictions(any_doc, "test").items()}
    labels |= {s: a for s, (a, _) in rb.predictions(any_doc, "adni").items()}
    train = sorted(split["train"] & set(any_doc["subjects"]))
    y_train = [any_doc["labels"][any_doc["subjects"].index(s)] for s in train]
    model = LogisticRegression(class_weight="balanced").fit([[age[s]] for s in train], y_train)
    out = {}
    for where in ("test", "agem", "adni"):
        subjects = sorted(rb.predictions(any_doc, where))
        y = np.array([labels[s] for s in subjects])
        p = model.predict_proba([[age[s]] for s in subjects])[:, 1]
        out[where] = {"auc": rb.fast_auc(y, p), "aupr": rb.fast_ap(y, p),
                      "f1": float(rb.f1_score(y, p >= 0.5, average="macro"))}
    return out


def adni_figure(g: dict, variant: dict, age: dict) -> None:
    """ADNI and representative-test AUROC per method: supervised, SSL, SSL + ADNI-SSL, sin SSL."""
    kinds = (("Supervisado", lambda m, bb: g["sup"].get((m, bb)), rb.SUP_COLOR, "///"),
             ("SSL (OASIS3)", lambda m, bb: g["ssl"].get((m, bb)), None, None),
             ("SSL + 174 ADNI sin etiqueta (1 semilla)", lambda m, bb: variant["ssl"].get((m, bb)),
              None, ".."),
             ("Sin SSL", lambda m, bb: g["init"].get((m, bb)), base.BASE, None))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, where, title in ((axes[0], "test", "Test OASIS3 representativo (41 AD + 137 HC)"),
                             (axes[1], "adni", "ADNI, otro centro (118 AD + 248 HC)")):
        for i, (m, bb) in enumerate(CELLS):
            for k, (label, get, color, hatch) in enumerate(kinds):
                docs = get(m, bb)
                if not docs:
                    continue
                v = [rb.value(d, where, "auc") for d in docs]
                x = i + (k - 1.5) * 0.2
                ax.bar(x, np.mean(v), width=0.2, color=color or base.COLOR[m], hatch=hatch,
                       edgecolor="white", alpha=0.85 if hatch else 1.0,
                       label=label if i == 0 else None)
                if len(v) > 1:
                    ax.scatter([x] * len(v), v, s=9, color=base.INK, zorder=3, linewidths=0)
                ax.text(x, 0.32, f"{np.mean(v):.2f}", ha="center", fontsize=6.5, rotation=90)
        ax.axhline(0.5, color=base.INK2, linewidth=1, linestyle=":")
        ax.axhline(age[where]["auc"], color=rb.AGE_COLOR, linewidth=1.1, linestyle="--",
                   label="Solo edad" if where == "test" else None)
        ax.set_xticks(range(len(CELLS)), [TICK[m] for m, _ in CELLS], fontsize=8)
        ax.set_ylim(0.3, 0.95)
        ax.set_title(f"AUROC — {title}", fontsize=9.2)
        base.style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               fontsize=8)
    fig.suptitle("¿Generaliza a otro centro? Mismo probe entrenado en OASIS3 (línea punteada = "
                 "azar)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(base.FIGURES / "dep50_adni.png", dpi=160)
    plt.close(fig)
    print("wrote dep50_adni.png")


def model_figure(validation: dict) -> None:
    """ROC curves of the deployed ensemble on each validation set, with its operating points."""
    from sklearn.metrics import roc_curve
    sets = (("oof", "CV fuera de fold (train+val)", "#52514e"),
            ("oasis3_test", "Test OASIS3 representativo", "#2a78d6"),
            ("oasis3_test_age_matched", "Test OASIS3 emparejado por edad", "#1baf7a"),
            ("adni_raw", "ADNI sin calibrar", "#eb6834"),
            ("adni_site_calibrated", "ADNI calibrado por sitio", "#b3261e"))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for key, label, color in sets:
        if key not in validation:
            continue
        d = validation[key]
        fpr, tpr, thr = roc_curve(d["y"], d["p"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label}: AUROC {d['auc']:.2f} [{d['auc_ci'][0]:.2f}, {d['auc_ci'][1]:.2f}]")
        at = d["at"]["youden"]
        ax.plot(1 - at["specificity"], at["sensitivity"], "o", color=color, markersize=6)
    ax.plot([0, 1], [0, 1], ":", color=base.INK2)
    ax.set_xlabel("1 − especificidad")
    ax.set_ylabel("sensibilidad")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title(f"Modelo desplegable: {validation['candidate']}\n(punto = umbral Youden "
                 f"{validation['thresholds']['youden']:.2f}, elegido en CV)", fontsize=9.5)
    base.style(ax)
    fig.tight_layout()
    fig.savefig(base.FIGURES / "dep50_modelo_roc.png", dpi=160)
    plt.close(fig)
    print("wrote dep50_modelo_roc.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "deploy" / "runs")
    parser.add_argument("--model", type=Path, default=base.ROOT / "deploy" / "ad_mri_v1")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    main_runs = rb.load(args.runs, "dep50")
    variant_runs = rb.load(args.runs, "dep50a")
    for name in list(main_runs):  # dep50a_* also matches the dep50 prefix
        main_runs[name] = [d for d in main_runs[name] if "/dep50a_ssl/" not in d["experiment_dir"]]
    g, variant = rb.groups_of(main_runs), rb.groups_of(variant_runs)
    matched = rb.age_matched_test(SPLIT / "manifest.csv")
    rb.AGE_MATCHED.update(matched["subjects"])
    rb.TICK.update(TICK)
    rb.WHERE.update({"cv": "CV (train+val emparejados por edad, 294)",
                     "test": "Test representativo (41 AD + 137 HC)",
                     "agem": f"Test emparejado por edad ({matched['ad']} AD + {matched['hc']} HC)",
                     "adni": "ADNI externo (118 AD + 248 HC)"})
    first = g["ssl"][CELLS[0]][0]
    age = age_reference(first)
    lines = ["# Split de despliegue, 50 épocas", "",
             "Train/val con etiqueta emparejados por edad 1:1 (117+117 / 30+30); 743 sujetos para "
             "SSL; test representativo de OASIS3 (prevalencia y edades reales); ADNI como test "
             "externo de otro centro. Probe congelado como siempre; supervisado end-to-end; sin "
             "SSL = encoder de partida. Δ = bootstrap pareado (IC 95 %, * = excluye 0).", "",
             "Solo edad: " + ", ".join(f"{w} AUROC {age[w]['auc']:.3f}" for w in age), "",
             "| Método | Modelo | n | AUROC CV | AUROC test | AUPR test | AUROC emp. edad | "
             "AUROC ADNI | AUPR ADNI |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    sets = [("Supervisado", g["sup"]), ("SSL", g["ssl"]), ("SSL + ADNI-SSL", variant["ssl"]),
            ("Sin SSL", g["init"])]
    for m, bb in CELLS:
        for name, group in sets:
            if docs := group.get((m, bb)):
                lines.append(f"| {base.LABEL[m]} {rb.BB[bb]} | {name} | {len(docs)} | "
                             f"{rb.mean_sd(docs, 'cv', 'auc')} | {rb.mean_sd(docs, 'test', 'auc')} | "
                             f"{rb.mean_sd(docs, 'test', 'aupr')} | {rb.mean_sd(docs, 'agem', 'auc')} | "
                             f"{rb.mean_sd(docs, 'adni', 'auc')} | {rb.mean_sd(docs, 'adni', 'aupr')} |")
    lines += ["", "| Δ AUROC | CV | Test | Test emp. edad | ADNI |", "|---|---:|---:|---:|---:|"]
    for m, bb in CELLS:
        ssl = g["ssl"].get((m, bb))
        refs = [("sin SSL", g["init"].get((m, bb))), ("supervisado", g["sup"].get((m, bb)))]
        same_seed = [d for d in (ssl or []) if d["seed"] == 1]
        refs_v = [("SSL + ADNI-SSL − SSL (semilla 1)", variant["ssl"].get((m, bb)), same_seed)]
        for name, ref in refs:
            if ssl and ref:
                cv = rb.fmt(rb.bootstrap(ssl, ref, "cv")) if ssl[0]["subjects"] == ref[0]["subjects"] else "—"
                lines.append(f"| {base.LABEL[m]} SSL − {name} | {cv} | "
                             f"{rb.fmt(rb.bootstrap(ssl, ref, 'test'))} | "
                             f"{rb.fmt(rb.bootstrap(ssl, ref, 'agem'))} | "
                             f"{rb.fmt(rb.bootstrap(ssl, ref, 'adni'))} |")
        for name, a, b in refs_v:
            if a and b:
                lines.append(f"| {base.LABEL[m]} {name} | — | {rb.fmt(rb.bootstrap(a, b, 'test'))} | "
                             f"{rb.fmt(rb.bootstrap(a, b, 'agem'))} | "
                             f"{rb.fmt(rb.bootstrap(a, b, 'adni'))} |")
    rb.metrics_figure(g, CELLS, "Split de despliegue (edad neutralizada en train/val), 50 épocas",
                      "dep50_metricas.png", age, wheres=("cv", "test", "agem", "adni"))
    adni_figure(g, variant, age)
    validation_path = args.model / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        model_figure(validation)
        lines += ["", (args.model / "MODEL_CARD.md").read_text(encoding="utf-8")]
    (args.runs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
