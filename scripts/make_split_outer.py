"""Outer 5-fold version of the deployment split: every OASIS3 AD/HC subject is test once.

    python scripts/make_split_outer.py [--out campaign_ssl3d_runpod/split_outer5]

Why (2026-11-16, scripts/analyze_cv_vs_test.py): on split_deploy the representative CV and
the representative test are the same population (age-only AUROC 0.752 vs 0.758), the
pooled-fold estimator and the smaller CV probes cost < 0.005 AUROC, and the test of every
model falls inside the 95 % band of pseudo-tests of the same size drawn from the CV
population (+-0.05 AUROC with 41 AD). The "test better than CV" was the luck of one fixed
draw of 41 AD, shared by every model (so it looked systematic). A larger fixed test would
take subjects from training and still be one draw; the remedy that keeps the real-data
goal is the outer cross-validation used for small clinical cohorts: the AD/HC subjects
are dealt into 5 representative folds (cohort prevalence, age tertile within class), each
fold is the test once, and SSL + probe are refit without it. The test then covers all
203 AD + 677 HC subjects (band +-0.025) and CV and test are estimated the same way.

- fold 0 = split_deploy unchanged (its 178 test subjects), so every model already run
  there is fold 0 of this design;
- folds 1-4 deal the other 162 AD + 540 HC subjects (seed 42), i.e. 40-41 AD + 135 HC each;
- inside each outer fold the recipe of split_deploy is repeated on the remaining subjects:
  labeled train/val AD and HC age-matched 1:1 inside 2.5-year bins, val 30 + 30, every
  other subject SSL-only with all its sessions; the evaluated session of every subject is
  the one split_deploy uses.

Writes ``<out>/fold{k}/manifest.csv`` (+ SPLIT.md, split.json) and ``<out>/folds.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from make_split_361_63_84 import allocate
from make_split_imbalanced import SUBJECT, day, write

ROOT = Path(__file__).resolve().parent.parent
CLINICAL = ROOT / "data/dementia/mni-dataset.csv"
DEPLOY = ROOT / "campaign_ssl3d_runpod/split_deploy/manifest.csv"
CACHE_META = ROOT / "data/roi3d/hippo_s32.json"
FOLDS = 5
VAL_PER_CLASS = 30
BIN_YEARS = 2.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="campaign_ssl3d_runpod/split_outer5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [r for r in csv.DictReader(CLINICAL.open(encoding="utf-8"))
            if r["study_name"].strip().upper() == "OASIS3"]
    by_idx = {r["idx"]: r for r in rows}
    sessions: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        s = SUBJECT.match(r["original_code"]).group(1)
        sessions[s].append(r)
        groups[s].add(r["group"].strip().upper())
    deploy = list(csv.DictReader(DEPLOY.open(encoding="utf-8")))
    primary = {r["subject"]: by_idx[r["idx"]] for r in deploy if r["primary"] == "1"}
    test0 = sorted({r["subject"] for r in deploy if r["split"] == "test"})

    def label(s: str) -> str:
        only = next(iter(groups[s]))
        return only if len(groups[s]) == 1 and only in ("AD", "HC") else "OTHER"

    def age(s: str) -> float:
        return float(primary[s]["age"])

    def row_of(split: str, r: dict, is_primary: str, labeled: str = "1") -> dict:
        return {"split": split, "subject": SUBJECT.match(r["original_code"]).group(1),
                "idx": r["idx"], "session": r["original_code"],
                "group": r["group"].strip().upper(), "primary": is_primary, "labeled": labeled}

    subjects = sorted(sessions)
    assert set(subjects) == set(primary), "split_deploy must list every OASIS3 subject"
    rng = random.Random(args.seed)
    # ---- outer folds: fold 0 = split_deploy's test; the rest dealt by class x age tertile
    fold_of = {s: 0 for s in test0}
    for diagnosis in ("AD", "HC"):
        members = [s for s in subjects if label(s) == diagnosis and s not in fold_of]
        cuts = np.quantile([age(s) for s in members], [1 / 3, 2 / 3])
        strata: dict[int, list[str]] = defaultdict(list)
        for s in members:
            strata[int(np.searchsorted(cuts, age(s), side="right"))].append(s)
        offset = 0
        for key in sorted(strata):
            group = sorted(strata[key])
            rng.shuffle(group)
            for i, s in enumerate(group):
                fold_of[s] = 1 + (offset + i) % (FOLDS - 1)
            offset += len(group)
    cached = set(json.loads(CACHE_META.read_text(encoding="utf-8"))["idx"])
    summary = {}
    for k in range(FOLDS):
        out = ROOT / args.out / f"fold{k}"
        if k == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.csv").write_text(DEPLOY.read_text(encoding="utf-8"), encoding="utf-8")
            (out / "SPLIT.md").write_text("# Pliegue externo 0 = split_deploy\n\nCopia exacta de "
                                          "`campaign_ssl3d_runpod/split_deploy/manifest.csv`.\n",
                                          encoding="utf-8")
            summary[k] = {"test_ad": sum(label(s) == "AD" for s in test0),
                          "test_hc": sum(label(s) == "HC" for s in test0)}
            continue
        test = {s for s, f in fold_of.items() if f == k}
        bins = {d: defaultdict(list) for d in ("AD", "HC")}
        for s in subjects:
            if label(s) in bins and s not in test:
                bins[label(s)][math.floor(age(s) / BIN_YEARS)].append(s)
        labeled: dict[str, list[str]] = {"AD": [], "HC": []}
        for b in sorted(set(bins["AD"]) | set(bins["HC"])):
            n = min(len(bins["AD"][b]), len(bins["HC"][b]))
            for d in ("AD", "HC"):
                labeled[d] += rng.sample(sorted(bins[d][b]), n)
        val: set[str] = set()
        for members in labeled.values():
            strata = defaultdict(list)
            for s in members:
                strata[math.floor(age(s) / BIN_YEARS)].append(s)
            for b, n in allocate(VAL_PER_CLASS, strata).items():
                val.update(rng.sample(sorted(strata[b]), n))
        is_labeled = set(labeled["AD"]) | set(labeled["HC"])
        out_rows: list[dict] = []
        for s in subjects:
            first = primary[s]
            if s in test:
                out_rows.append(row_of("test", first, "1"))
            elif s in val:
                out_rows.append(row_of("val", first, "1"))
            elif s in is_labeled:
                out_rows.append(row_of("train", first, "1"))
            else:
                out_rows += [row_of("train", r, "1" if r["idx"] == first["idx"] else "0", "0")
                             for r in sorted(sessions[s], key=lambda r: day(r["original_code"]))]
        write(out, out_rows, label, age, f"Pliegue externo {k} de 5 (split_outer5)", [
            f"Test: pliegue {k} de los AD/HC de OASIS3 fuera del test de split_deploy, repartidos "
            "por clase y tercil de edad (prevalencia de la cohorte, edades reales).",
            "Train/val con etiqueta: AD y HC restantes (incluido el test de split_deploy) "
            f"emparejados 1:1 por edad en tramos de {BIN_YEARS} años; validación "
            f"{VAL_PER_CLASS} + {VAL_PER_CLASS}. El resto, solo-SSL con todas sus sesiones.",
            "Generado con `scripts/make_split_outer.py`."])
        missing = sorted({r["idx"] for r in out_rows} - cached)
        summary[k] = {"test_ad": sum(label(s) == "AD" for s in test),
                      "test_hc": sum(label(s) == "HC" for s in test),
                      "labeled_per_class": len(labeled["AD"]), "missing_cache": len(missing)}
        print(k, summary[k])
    (ROOT / args.out / "folds.json").write_text(json.dumps(
        {"fold_of": dict(sorted(fold_of.items())), "summary": summary}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
