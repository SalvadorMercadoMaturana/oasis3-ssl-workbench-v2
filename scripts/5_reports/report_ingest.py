#!/usr/bin/env python
"""Report of the ingestion / backbone tuning of 2026-11-22.

    python scripts/5_reports/report_ingest.py [--final cnx2d_sag_x2]

Reads the downloaded runs in ``campaign_ssl3d_runpod/ingest/runs`` (scripts/3_train/
jobs_ingest.py: tags ing_init_f<k>, ing_f<k>, ing_<variant>_f<k>, ing_sup_f<k>) and the outer-fold references
of the 2026-11-16 campaign (scripts/5_reports/report_outer5.py); writes
``campaign_ssl3d_runpod/ingest/REPORT.md`` and ``summary.json``.

1. Frozen starts (sin SSL) of every ingestion on fold 0 (split_deploy).
2. SSL screen on fold 0: the four methods (VICReg, BYOL, DINOv2, rotation pretext) on
   each ingestion, against the current one (convnext_tiny, same seed, same run); the
   variants ``ing_<v>_f0`` (learning rates x f: ``lr<f>``; frozen backbone BatchNorm:
   ``fbn``) are rows of their own (``<backbone> <v>``).
3. ``--final``: the chosen ingestion on the five outer folds (3-seed ensembles) against
   the outer-fold results of the current ingestion: the test numbers of the report.

Every number is scored with protocol v2 (representative CV of 702 subjects on fold 0;
mean3 = (AUROC + AUPR + F1-macro) / 3, F1 threshold chosen out of fold).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_outer5 as r5  # noqa: E402
from report_v2fix import boot_delta, fmt_delta  # noqa: E402

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402

CAMPAIGN = r5.CAMPAIGN
RUNS = CAMPAIGN / "ingest/runs"
OUT = CAMPAIGN / "ingest"
REF_BB = "convnext_tiny"
# key: (label, run name without backbone/roi/seed, outer-fold reference key of report_outer5)
METHODS = {
    "vicreg": ("VICReg", "vicreg_campaign_augval__geo", "vicreg"),
    "byol": ("BYOL", "byol_wide_dproj0.1__noise", "byol"),
    "dino": ("DINOv2", "dinov2_b64_k256__elastic_tio_gpu", "dino_cnx"),
    "pretext": ("Rotación", "pretext_rot24_h2048__geo", "pretext"),
}
PRETEXT_REF = ("pretext_tune", "pretext_rot24_h2048__geo__bbconvnext_tiny__roi32")


def run_name(base: str, bb: str) -> str:
    return f"{base}__bb{bb}__roi32"


def backbone_of(dirname: str) -> str:
    """``<method>__bb<backbone>__roi<roi>_s<seed>`` -> backbone (+ ``@<roi>`` if not 32)."""
    bb, roi = dirname.split("__bb", 1)[1].rsplit("__roi", 1)
    roi = roi.rsplit("_s", 1)[0]
    return bb if roi == "32" else f"{bb}@{roi}"


def variant_tags(k: int = 0) -> list[tuple[str, str]]:
    """(tag, row suffix) of the SSL runs of fold k.

    ``ing_f<k>`` and its variants ``ing_<v>_f<k>`` (``lr0.1``: learning rates x 0.1;
    ``fbn``: backbone BatchNorm frozen after a calibration on the training views).
    """
    tags = [(f"ing_f{k}", "")]
    for p in sorted(RUNS.glob(f"ing_*_f{k}")):
        v = p.name[len("ing_"):].rsplit("_f", 1)[0]
        if v not in ("init", "sup"):
            tags.append((p.name, f" {v}"))
    return tags


def ens(tag: str, name: str, k: int = 0, root: Path = RUNS) -> dict | None:
    docs = pp.load_runs(root, tag, name)
    if not docs:
        return None
    out = pp.ensemble(docs, r5.AGES[k])
    out["n_seeds"] = len(docs)
    out["minutes"] = [json.loads((d["dir"] / "result.json").read_text().splitlines()[0])
                      .get("minutes") for d in docs]
    return out


def m3(block: dict) -> float:
    return block["mean3"]


def cell(e: dict, where: str) -> str:
    b = e[where]
    return f"{b['auc']:.3f} | {b['aupr']:.3f} | {b['f1_macro']:.3f} | **{b['mean3']:.3f}**"


def cv_delta(a: dict, b: dict) -> dict:
    y, sa, _ = a["_cv"]
    _, sb, _ = b["_cv"]
    return boot_delta(y, sa, sb, a["threshold_rep"], b["threshold_rep"], draws=1000)


def test_delta(a: dict, b: dict) -> dict:
    y, sa = a["_test"]
    _, sb = b["_test"]
    return boot_delta(y, sa, sb, a["threshold_rep"], b["threshold_rep"], draws=1000)


# ---------------------------------------------------------------- 1. frozen starts
def frozen_table() -> tuple[list[str], dict]:
    tag = "ing_init_f0"
    names = sorted({p.name.rsplit("_s", 1)[0] for p in (RUNS / tag).glob("vicreg_init__bb*_s*")})
    rows = {backbone_of(n + "_s1"): ens(tag, n) for n in names}
    rows = {k: v for k, v in rows.items() if v is not None}
    ref = rows.get(REF_BB)
    lines = ["| Ingesta / backbone | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
             "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. vs actual [IC 95 %] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for bb, e in sorted(rows.items(), key=lambda kv: -m3(kv[1]["cv_rep"])):
        delta = "—" if bb == REF_BB or ref is None else fmt_delta(cv_delta(e, ref)["mean3"])
        lines.append(f"| `{bb}` | {cell(e, 'cv_rep')} | {cell(e, 'test')} | {delta} |")
    summary = {bb: {"cv": e["cv_rep"], "test": e["test"]} for bb, e in rows.items()}
    return lines, summary


# ---------------------------------------------------------------- 2. SSL screen
def screen_table() -> tuple[list[str], dict]:
    found: dict[str, dict[str, dict]] = {}
    for tag, suffix in variant_tags(0):
        for key, (_, base, _) in METHODS.items():
            for p in (RUNS / tag).glob(f"{base}__bb*__roi32_s*"):
                bb = backbone_of(p.name)
                e = ens(tag, run_name(base, bb))
                if e is not None:
                    found.setdefault(bb + suffix, {})[key] = e
    ref = found.get(REF_BB, {})
    head = " | ".join(f"{METHODS[k][0]} CV / test" for k in METHODS)
    lines = [f"| Ingesta / backbone | {head} | media CV comp. | media test comp. | "
             "Δ CV comp. vs actual (media de métodos) |",
             "|---|" + "---|" * len(METHODS) + "---:|---:|---|"]
    summary = {}
    order = sorted(found, key=lambda bb: -np.mean([m3(e["cv_rep"]) for e in found[bb].values()]))
    for bb in order:
        cells, deltas = [], []
        for key in METHODS:
            e = found[bb].get(key)
            if e is None:
                cells.append("—")
                continue
            cells.append(f"{m3(e['cv_rep']):.3f} / {m3(e['test']):.3f}")
            if bb != REF_BB and key in ref:
                deltas.append(cv_delta(e, ref[key])["mean3"][0])
        cv_mean = np.mean([m3(e["cv_rep"]) for e in found[bb].values()])
        te_mean = np.mean([m3(e["test"]) for e in found[bb].values()])
        d = f"{np.mean(deltas):+.3f} ({len(deltas)} métodos)" if deltas else "—"
        lines.append(f"| `{bb}` | " + " | ".join(cells) + f" | {cv_mean:.3f} | {te_mean:.3f} | {d} |")
        summary[bb] = {k: {"cv": e["cv_rep"], "test": e["test"], "minutes": e["minutes"]}
                       for k, e in found[bb].items()}
    # per-method detail with bootstrap intervals
    detail = ["", "| Método | Ingesta | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
              "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. [IC 95 %] | Δ test comp. [IC 95 %] |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for key in METHODS:
        for bb in order:
            e = found[bb].get(key)
            if e is None:
                continue
            if bb == REF_BB or key not in ref:
                dc = dt = "—"
            else:
                dc = fmt_delta(cv_delta(e, ref[key])["mean3"])
                dt = fmt_delta(test_delta(e, ref[key])["mean3"])
            detail.append(f"| {METHODS[key][0]} | `{bb}` | {cell(e, 'cv_rep')} | {cell(e, 'test')} "
                          f"| {dc} | {dt} |")
    return lines + detail, summary


# ---------------------------------------------------------------- 3. outer folds
def fold_ens(root: Path, tag: str, name: str, k: int, seeds: set[int] | None) -> dict | None:
    """Seed ensemble of one fold (only ``seeds`` when given), as report_outer5 scores it."""
    docs = pp.load_runs(root, tag, name)
    if seeds:
        docs = [d for d in docs if d.get("seed") in seeds]
    if not docs:
        return None
    out = pp.ensemble(docs, r5.AGES[k])
    out["n_seeds"] = len(docs)
    return out


def outer_folds(source, seeds: set[int] | None) -> dict | None:
    """``{"folds": {k: ensemble}}`` over the five folds; ``source(k)`` = (root, tag, name)."""
    folds = {}
    for k in r5.FOLDS:
        e = fold_ens(*source(k), k, seeds)
        if e is None:
            return None
        folds[k] = e
    return {"folds": folds}


def pooled(folds: dict) -> dict:
    y = np.concatenate([folds[k]["_test"][0] for k in r5.FOLDS])
    p = np.concatenate([folds[k]["_test"][1] for k in r5.FOLDS])
    shift = np.concatenate([folds[k]["_test"][1] - folds[k]["threshold_rep"] for k in r5.FOLDS])
    out = {"n": len(y), "ad": int(y.sum()), "auc": pp.auc(y, p), "aupr": pp.aupr(y, p),
           "f1_macro": pp.at_threshold(y, shift, 0.0)["f1_macro"]}
    out["mean3"] = (out["auc"] + out["aupr"] + out["f1_macro"]) / 3
    return out


def fold_mean(folds: dict, where: str, metric: str) -> tuple[float, float]:
    v = [folds[k][where][metric] for k in r5.FOLDS]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def outer_table(bb: str, prefix: str = "ing", seeds: set[int] | None = None
                ) -> tuple[list[str], dict]:
    """Compare the chosen ingestion ``bb`` with the current one on the five outer folds.

    New runs: ``<prefix>_f<k>`` (SSL) and ``ing_init_f<k>`` (sin SSL) of this campaign;
    current ingestion: the outer-fold runs of report_outer5 (fold 0 of the earlier
    campaigns). ``seeds`` keeps the same seeds on both sides (e.g. {1}).
    """
    lines = ["| Modelo | Ingesta | Semillas | CV comp. (media ± DE) | Test AUROC | Test AUPR | "
             "Test F1 | **Test comp.** | Test agrupado AUROC / AUPR / F1 / comp. | "
             "Δ test comp. vs actual [IC 95 %] |",
             "|---|---|---:|---|---|---|---|---|---|---|"]
    summary = {}
    rows = [(key, label, lambda k, base=base: (RUNS, f"{prefix}_f{k}", run_name(base, bb)),
             lambda k, ref_key=ref_key: r5.fold_source(
                 ref_key, k, PRETEXT_REF if ref_key == "pretext" else None))
            for key, (label, base, ref_key) in METHODS.items()]
    rows.append(("init", "sin SSL", lambda k: (RUNS, f"ing_init_f{k}", f"vicreg_init__bb{bb}__roi32"),
                 lambda k: r5.fold_source("init_cnx", k)))
    for key, label, new_source, ref_source in rows:
        new = outer_folds(new_source, seeds)
        ref = outer_folds(ref_source, seeds)
        for name, side in ((bb, new), (REF_BB, ref)):
            if side is None:
                continue
            folds = side["folds"]
            pool = pooled(folds)
            fm = {m: fold_mean(folds, "test", m) for m in ("auc", "aupr", "f1_macro", "mean3")}
            cv = fold_mean(folds, "cv_rep", "mean3")
            n_seeds = "/".join(str(folds[k]["n_seeds"]) for k in r5.FOLDS)
            delta = "—"
            if name == bb and ref is not None:
                d = r5.fold_delta(new, ref)
                delta = fmt_delta(d["mean3"])
                summary.setdefault(key, {})["delta"] = d
            lines.append(f"| {label} | `{name}` | {n_seeds} | {cv[0]:.3f} ± {cv[1]:.3f} | "
                         + " | ".join(f"{fm[m][0]:.3f} ± {fm[m][1]:.3f}"
                                      for m in ("auc", "aupr", "f1_macro"))
                         + f" | **{fm['mean3'][0]:.3f} ± {fm['mean3'][1]:.3f}** | "
                         f"{pool['auc']:.3f} / {pool['aupr']:.3f} / {pool['f1_macro']:.3f} / "
                         f"{pool['mean3']:.3f} | {delta} |")
            summary.setdefault(key, {})[name] = {"test": fm, "cv_mean3": cv, "pooled": pool}
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None, help="ingestion run on the five outer folds")
    ap.add_argument("--final-tag", default="ing",
                    help="tag prefix of its runs (ing, or ing_lr0.1 for scaled learning rates)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="outer folds: only these seeds on both sides (default all)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the tables carry Greek deltas
    md = ["# Tuning de ingesta y backbone (2026-11-22)", "",
          "Protocolo v2. CV = CV representativa del pliegue 0 (702 sujetos AD/HC, 23 % AD); "
          "test = test representativo (178 sujetos del pliegue 0; 880 en los 5 pliegues). "
          "comp. = (AUROC + AUPR + F1-macro)/3. Δ = bootstrap pareado estratificado; "
          "* = IC sin el 0.", ""]
    summary: dict = {}
    if (RUNS / "ing_init_f0").is_dir():
        lines, summary["frozen"] = frozen_table()
        md += ["## 1. Inicio congelado (sin SSL) de cada ingesta, pliegue 0", "", *lines, ""]
    if (RUNS / "ing_f0").is_dir():
        lines, summary["screen"] = screen_table()
        md += ["## 2. Los cuatro métodos SSL sobre cada ingesta, pliegue 0 (semilla 1)", "",
               *lines, ""]
    if args.final:
        seeds = set(args.seeds) if args.seeds else None
        lines, summary["outer"] = outer_table(args.final, args.final_tag, seeds)
        which = ("semillas " + ", ".join(map(str, sorted(seeds)))) if seeds else "todas las semillas"
        md += [f"## 3. `{args.final}` en los 5 pliegues externos ({which}, ambos lados)", "",
               "Test agrupado = los 880 sujetos de test de los 5 pliegues juntos; Δ = media "
               "sobre pliegues de la diferencia pareada en cada test.", "", *lines, ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
