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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_deploy_model as bd  # noqa: E402
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
    """Score each evaluation set with age itself (older = AD): the age shortcut.

    A logistic fit on the labeled train set cannot serve: train is age-matched 1:1, so the
    fitted slope is noise and its sign arbitrary (it came out negative). No F1: a threshold
    would have to be fitted somewhere.
    """
    age, _ = ages()
    labels = {s: a for s, (a, _) in rb.predictions(any_doc, "test").items()}
    labels |= {s: a for s, (a, _) in rb.predictions(any_doc, "adni").items()}
    out = {}
    for where in ("test", "agem", "adni"):
        subjects = sorted(rb.predictions(any_doc, where))
        y = np.array([labels[s] for s in subjects])
        p = np.array([age[s] for s in subjects])
        out[where] = {"auc": rb.fast_auc(y, p), "aupr": rb.fast_ap(y, p)}
    return out


def apply_input_qc(*runs: dict) -> dict[str, int]:
    """Score ADNI as the deployed model does: without the scans its input QC rejects.

    One scan-level rule for every model (the hippocampal cube, build_deploy_model.qc_range):
    45 ADNI AD scans carry another intensity normalization. The unfiltered entry stays as
    ``adni_all``. Returns the AD/HC counts kept and rejected.
    """
    qc = bd.qc_range("hippo_s32")
    ext = list(csv.DictReader((SPLIT / "external_adni.csv").open(encoding="utf-8")))
    value = bd.cache_intensity("adni_hippo_s32", [r["idx"] for r in ext])
    keep = {r["subject"] for r, v in zip(ext, value, strict=True) if qc["lo"] <= v <= qc["hi"]}
    for group in runs:
        for docs in group.values():
            for doc in docs:
                extras = doc["test"]["extra_tests"]
                extras["adni_all"] = extras["adni"]
                rows = [r for r in extras["adni_all"]["predictions"] if r["subject"] in keep]
                y = np.array([r["AD"] for r in rows])
                p = np.array([r["score"] for r in rows])
                extras["adni"] = {"predictions": rows, "auc": rb.fast_auc(y, p),
                                  "average_precision": rb.fast_ap(y, p),
                                  "f1_macro": float(rb.f1_score(y, p >= 0.5, average="macro"))}
    groups = {r["subject"]: r["group"] for r in ext}
    count = lambda subjects, g: sum(groups[s] == g for s in subjects)  # noqa: E731
    rejected = set(groups) - keep
    return {"ad": count(keep, "AD"), "hc": count(keep, "HC"),
            "rejected_ad": count(rejected, "AD"), "rejected_hc": count(rejected, "HC")}


def adni_figure(g: dict, variant: dict, age: dict) -> None:
    """ADNI and representative-test AUROC per method: supervised, SSL, SSL + ADNI-SSL, sin SSL."""
    kinds = (("Supervisado", lambda m, bb: g["sup"].get((m, bb)), rb.SUP_COLOR, "///"),
             ("SSL (OASIS3)", lambda m, bb: g["ssl"].get((m, bb)), None, None),
             ("SSL + 174 ADNI sin etiqueta (1 semilla)", lambda m, bb: variant["ssl"].get((m, bb)),
              None, ".."),
             ("Sin SSL", lambda m, bb: g["init"].get((m, bb)), base.BASE, None))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, where, title in ((axes[0], "test", "Test OASIS3 representativo (41 AD + 137 HC)"),
                             (axes[1], "adni", rb.WHERE["adni"])):
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


def metrics_figure(g: dict) -> None:
    """CV and representative-test metrics: supervised, SSL and sin SSL per method, values on top."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    def pale(color: str) -> tuple[float, float, float]:
        return tuple(0.3 * c + 0.7 for c in to_rgb(color))

    kinds = (("sup", "Supervisado (end-to-end)"), ("ssl", "SSL 50 épocas, congelado + probe"),
             ("init", "Sin SSL: mismo encoder antes del SSL + probe"))
    rows = (("cv", "CV (train + val emparejados por edad, 294)"),
            ("test", "Test representativo (41 AD + 137 HC)"))
    cols = (("auc", "AUROC"), ("f1", "F1-macro"), ("aupr", "AUPR"))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(15, 8.6), sharey=True)
    width = 0.28
    for r, (where, row_title) in enumerate(rows):
        for c, (key, metric) in enumerate(cols):
            ax = axes[r, c]
            for i, (m, bb) in enumerate(CELLS):
                for k, (kind, _) in enumerate(kinds):
                    docs = g[kind].get((m, bb))
                    if not docs:
                        continue
                    v = [rb.value(d, where, key) for d in docs]
                    x, mean = i + (k - 1) * (width + 0.02), float(np.mean(v))
                    sd = float(np.std(v)) if len(v) > 1 else 0.0
                    style = {"sup": {"color": rb.SUP_COLOR, "hatch": "///", "edgecolor": "white"},
                             "ssl": {"color": base.COLOR[m], "edgecolor": "white"},
                             "init": {"color": pale(base.COLOR[m]), "edgecolor": base.COLOR[m],
                                      "linewidth": 1.0}}[kind]
                    ax.bar(x, mean, width=width, **style)
                    if sd:
                        ax.errorbar(x, mean, yerr=sd, color=base.INK, capsize=2.5, linewidth=1)
                    ax.text(x, mean + sd + 0.015, f"{mean:.3f}", ha="center", va="bottom",
                            fontsize=9.5, fontweight="bold" if kind == "ssl" else "normal",
                            color=base.INK)
            ax.axhline(0.5, color=base.INK2, linewidth=1.1, linestyle=":", zorder=0)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(len(CELLS)), [TICK[m] for m, _ in CELLS], fontsize=9.5)
            ax.set_title(f"{metric} — {row_title}", fontsize=10)
            base.style(ax)
    handles = [Patch(facecolor=rb.SUP_COLOR, hatch="///", edgecolor="white", label=kinds[0][1]),
               Patch(facecolor=base.COLOR["vicreg"], label=kinds[1][1] + " (color del método)"),
               Patch(facecolor=pale(base.COLOR["vicreg"]), edgecolor=base.COLOR["vicreg"],
                     label=kinds[2][1]),
               plt.Line2D([], [], color=base.INK2, linestyle=":", label="0.5")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Split de despliegue, 50 épocas: CV y test representativo (media de 3 "
                 "semillas ± DE; sin SSL = 1 encoder)", fontsize=11.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(base.FIGURES / "dep50_metricas.png", dpi=160)
    plt.close(fig)
    print("wrote dep50_metricas.png")


def split_figure() -> None:
    """Split_deploy at a glance: subjects per partition and class, SSL-only pool, ages."""
    clinical = {r["idx"]: r for r in csv.DictReader(
        (base.ROOT / "data/dementia/mni-dataset.csv").open(encoding="utf-8"))}
    rows = list(csv.DictReader((SPLIT / "manifest.csv").open(encoding="utf-8")))
    groups: dict[str, set[str]] = {}
    for r in rows:
        groups.setdefault(r["subject"], set()).add(r["group"])
    labeled = {k: [r for r in rows if r["split"] == k and r["labeled"] == "1"]
               for k in ("train", "val", "test")}
    ssl_rows = [r for r in rows if r["labeled"] == "0"]
    ssl_subjects = sorted({r["subject"] for r in ssl_rows})

    def kind(s: str) -> str:
        only = next(iter(groups[s]))
        return only if len(groups[s]) == 1 and only in ("AD", "HC") else "otro"

    ssl_count = {k: sum(kind(s) == k for s in ssl_subjects) for k in ("HC", "AD", "otro")}
    count = {k: {g: sum(r["group"] == g for r in v) for g in ("AD", "HC")}
             for k, v in labeled.items()}
    ad_c, hc_c, other_c = "#c4314b", "#2a78d6", "#b9b8b3"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    bars = [(f"Solo SSL, sin etiqueta\n{len(ssl_subjects)} sujetos · {len(ssl_rows)} sesiones\n"
             f"({ssl_count['HC']} HC · {ssl_count['AD']} AD · {ssl_count['otro']} otro dx o "
             "desconocido)",
             [(ssl_count["HC"], hc_c, "HC"), (ssl_count["AD"], ad_c, "AD"),
              (ssl_count["otro"], other_c, "otro dx / desconocido")], 0.45),
            (f"Train con etiqueta\n{len(labeled['train'])} sujetos · 1 sesión c/u",
             [(count["train"]["AD"], ad_c, "AD"), (count["train"]["HC"], hc_c, "HC")], 1.0),
            (f"Validación\n{len(labeled['val'])} sujetos",
             [(count["val"]["AD"], ad_c, "AD"), (count["val"]["HC"], hc_c, "HC")], 1.0),
            (f"Test representativo\n{len(labeled['test'])} sujetos",
             [(count["test"]["AD"], ad_c, "AD"), (count["test"]["HC"], hc_c, "HC")], 1.0)]
    for y, (_, parts, alpha) in enumerate(bars):
        left = 0
        for n, color, name in parts:
            ax.barh(y, n, left=left, color=color, alpha=alpha, edgecolor="white", height=0.62)
            if n >= 25:
                ax.text(left + n / 2, y, f"{n} {name}" if n >= 60 and len(name) <= 3 else str(n),
                        ha="center",
                        va="center", fontsize=8, color="white" if alpha == 1.0 else base.INK)
            left += n
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("sujetos")
    n_ssl = len(ssl_subjects) + len(labeled["train"])
    ax.text(max(len(ssl_subjects), 1) + 15, 0.5,
            f"Preentrenamiento SSL: {n_ssl} sujetos\n({len(ssl_subjects)} solo-SSL + "
            f"{len(labeled['train'])} de train;\nlas etiquetas no se usan)", fontsize=8.2,
            va="center", color=base.INK)
    cv_ad = count["train"]["AD"] + count["val"]["AD"]
    cv_hc = count["train"]["HC"] + count["val"]["HC"]
    ax.text(max(len(ssl_subjects), 1) + 15, 1.6,
            f"CV del probe (5×5): train + val =\n{cv_ad + cv_hc} con etiqueta "
            f"({cv_ad} AD + {cv_hc} HC, 50/50)", fontsize=8.2, va="center", color=base.INK)
    ax.text(max(len(ssl_subjects), 1) + 15, 3,
            "Prevalencia real de OASIS3 (23 % AD),\nnunca visto en SSL ni en el probe",
            fontsize=8.2, va="center", color=base.INK)
    ax.set_xlim(0, len(ssl_subjects) + 260)
    ax.set_title("Particiones (OASIS3, un sujeto en una sola partición)", fontsize=9.5)
    base.style(ax)
    # ages: matched in train/val, real in test
    data, ticks, colors = [], [], []
    for name, subset in (("Train + val", labeled["train"] + labeled["val"]),
                         ("Test", labeled["test"])):
        for g, color in (("AD", ad_c), ("HC", hc_c)):
            ages = [float(clinical[r["idx"]]["age"]) for r in subset if r["group"] == g]
            data.append(ages)
            ticks.append(f"{name}\n{g} (n={len(ages)})\n{np.mean(ages):.1f} años")
            colors.append(color)
    box = bx.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": base.INK})
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    bx.set_xticks(range(1, len(data) + 1), ticks, fontsize=8)
    bx.set_ylabel("edad en la sesión evaluada")
    bx.set_title("Edad: emparejada 1:1 en train/val, real en test", fontsize=9.5)
    base.style(bx)
    fig.suptitle("Split de despliegue: test representativo, entrenamiento sin atajo de edad",
                 fontsize=11, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(base.FIGURES / "dep50_split.png", dpi=160)
    plt.close(fig)
    print("wrote dep50_split.png")


def model_figure(validation: dict) -> None:
    """ROC curves of the deployed ensemble on each validation set, with its operating points."""
    from sklearn.metrics import roc_curve
    sets = (("oof", "CV fuera de fold (train+val)", "#52514e"),
            ("oasis3_test", "Test representativo (41 AD + 137 HC)", "#2a78d6"))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for key, label, color in sets:
        if key not in validation:
            continue
        d = validation[key]
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
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
    qc = apply_input_qc(main_runs, variant_runs)
    g, variant = rb.groups_of(main_runs), rb.groups_of(variant_runs)
    matched = rb.age_matched_test(SPLIT / "manifest.csv")
    rb.AGE_MATCHED.update(matched["subjects"])
    rb.TICK.update(TICK)
    rb.WHERE.update({"cv": "CV (train+val emparejados por edad, 294)",
                     "test": "Test representativo (41 AD + 137 HC)",
                     "agem": f"Test emparejado por edad ({matched['ad']} AD + {matched['hc']} HC)",
                     "adni": f"ADNI externo con QC ({qc['ad']} AD + {qc['hc']} HC)"})
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
    metrics_figure(g)
    adni_figure(g, variant, age)
    split_figure()
    validation_path = args.model / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        model_figure(validation)
        lines += ["", (args.model / "MODEL_CARD.md").read_text(encoding="utf-8")]
    (args.runs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
