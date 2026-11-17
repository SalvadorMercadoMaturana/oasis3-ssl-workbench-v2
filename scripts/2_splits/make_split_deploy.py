"""Deployment-oriented OASIS3 split + an external ADNI test.

    python scripts/2_splits/make_split_deploy.py [--out campaign_ssl3d_runpod/split_deploy]

Goal: a model that finds AD from the MRI alone, selected where age cannot help, with
as much unlabeled SSL data as OASIS3 has, and judged on data as it would arrive.

- ``test``: a representative sample of OASIS3 AD/HC subjects at the cohort's own
  prevalence (203 AD : 677 HC) and ages: 41 AD + 137 HC, allocated proportionally
  over age tertiles within each class (seed 42). One session (the evaluated one).
- labeled ``train`` / ``val``: the remaining AD and HC age-matched 1:1 (50/50) inside
  2.5-year bins (age at the evaluated session), drawn at random; validation takes 30 AD
  + 30 HC proportionally over the bins. One session per labeled subject (both classes
  alike).
- SSL-only ``train`` subjects (``labeled=0``): every other OASIS3 subject (the HC and
  AD left over by the matching and all subjects with another or mixed diagnosis),
  with every session.
- ``external_adni.csv``: every ADNI subject whose sessions are all AD or all HC, its
  earliest session (118 AD + 248 HC): another site, scanner park and protocol, with HC
  older than AD, so an age shortcut fails there.

Writes ``<out>/manifest.csv``, ``SPLIT.md``, ``split.json``, ``external_adni.csv`` and
``new_sessions.txt`` (OASIS3 keys missing from the ROI caches) and
``adni_sessions.txt`` (for ``scripts/1_data/extend_roi_caches.py --into adni_<cache>``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from make_split_361_63_84 import allocate
from make_split_imbalanced import FIELDS, ORIGINAL, SUBJECT, day, write

ROOT = Path(__file__).resolve().parents[2]
CLINICAL = ROOT / "data/dementia/mni-dataset.csv"
CACHE_META = ROOT / "data/roi3d/hippo_s32.json"
TEST_AD = 41
VAL_PER_CLASS = 30
BIN_YEARS = 2.5
ADNI_SUBJECT = re.compile(r"^(sub-\d+S\d+)ses(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="campaign_ssl3d_runpod/split_deploy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = list(csv.DictReader(CLINICAL.open(encoding="utf-8")))
    oasis = [r for r in rows if r["study_name"].strip().upper() == "OASIS3"]
    sessions: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in oasis:
        subject = SUBJECT.match(row["original_code"]).group(1)
        sessions[subject].append(row)
        groups[subject].add(row["group"].strip().upper())
    by_idx = {r["idx"]: r for r in oasis}
    original: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(ORIGINAL.open(encoding="utf-8")):
        original[row["subject"]].append(row)

    def label(subject: str) -> str:
        values = groups[subject]
        only = next(iter(values))
        return only if len(values) == 1 and only in ("AD", "HC") else "OTHER"

    def eval_session(subject: str) -> dict:
        """Return the original manifest's primary session, else the earliest one."""
        primary = [by_idx[r["idx"]] for r in original.get(subject, []) if r["primary"] == "1"]
        return min(primary or sessions[subject], key=lambda r: day(r["original_code"]))

    def age(subject: str) -> float:
        return float(eval_session(subject)["age"])

    def row_of(split: str, clinical_row: dict, primary: str, labeled: str = "1") -> dict:
        return {"split": split, "subject": SUBJECT.match(clinical_row["original_code"]).group(1),
                "idx": clinical_row["idx"], "session": clinical_row["original_code"],
                "group": clinical_row["group"].strip().upper(), "primary": primary,
                "labeled": labeled}

    rng = random.Random(args.seed)
    subjects = sorted(sessions)
    pool = {d: [s for s in subjects if label(s) == d] for d in ("AD", "HC")}
    # ---- representative test: cohort prevalence, age tertiles within class
    test: set[str] = set()
    test_n = {"AD": TEST_AD, "HC": round(TEST_AD * len(pool["HC"]) / len(pool["AD"]))}
    for diagnosis, members in pool.items():
        cuts = np.quantile([age(s) for s in members], [1 / 3, 2 / 3])
        strata: dict[int, list[str]] = defaultdict(list)
        for s in members:
            strata[int(np.searchsorted(cuts, age(s), side="right"))].append(s)
        for key, n in allocate(test_n[diagnosis], strata).items():
            test.update(rng.sample(sorted(strata[key]), n))
    # ---- labeled train/val: 1:1 age matching inside 2.5-year bins
    bins = {d: defaultdict(list) for d in pool}
    for diagnosis, members in pool.items():
        for s in members:
            if s not in test:
                bins[diagnosis][math.floor(age(s) / BIN_YEARS)].append(s)
    labeled: dict[str, list[str]] = {"AD": [], "HC": []}
    for b in sorted(set(bins["AD"]) | set(bins["HC"])):
        n = min(len(bins["AD"][b]), len(bins["HC"][b]))
        for diagnosis in ("AD", "HC"):
            labeled[diagnosis] += rng.sample(sorted(bins[diagnosis][b]), n)
    val: set[str] = set()
    for members in labeled.values():
        strata = defaultdict(list)
        for s in members:
            strata[math.floor(age(s) / BIN_YEARS)].append(s)
        for b, k in allocate(VAL_PER_CLASS, strata).items():
            val.update(rng.sample(sorted(strata[b]), k))
    is_labeled = set(labeled["AD"]) | set(labeled["HC"])
    # ---- manifest
    out_rows: list[dict] = []
    for subject in subjects:
        first = eval_session(subject)
        if subject in test:
            out_rows.append(row_of("test", first, "1"))
        elif subject in val:
            out_rows.append(row_of("val", first, "1"))
        elif subject in is_labeled:
            out_rows.append(row_of("train", first, "1"))
        else:  # SSL-only: every session
            out_rows += [row_of("train", r, "1" if r["idx"] == first["idx"] else "0", "0")
                         for r in sorted(sessions[subject], key=lambda r: day(r["original_code"]))]
    out = ROOT / args.out
    write(out, out_rows, label, age, "Split de despliegue: test representativo + entrenamiento "
          "sin atajo de edad", [
              f"Test representativo de OASIS3: {test_n['AD']} AD + {test_n['HC']} HC (la "
              f"prevalencia de la cohorte, {len(pool['AD'])}:{len(pool['HC'])}), proporcional por "
              "tercil de edad dentro de cada clase: las edades son las reales (AD mayores).",
              "Train/val con etiqueta: AD y HC restantes emparejados 1:1 por edad en tramos de "
              f"{BIN_YEARS} años (50/50); validación {VAL_PER_CLASS} + {VAL_PER_CLASS}. Una "
              "sesión por sujeto con etiqueta. Todos los demás sujetos OASIS3 (HC/AD sobrantes, "
              "otros diagnósticos) quedan en train como solo-SSL con todas sus sesiones.",
              "Test externo: ADNI (otro centro), ver external_adni.csv.",
              "Generado con `scripts/2_splits/make_split_deploy.py`."])
    # ---- external ADNI test
    adni = [r for r in rows if r["study_name"].strip().upper() == "ADNI"]
    adni_sessions: dict[str, list[dict]] = defaultdict(list)
    for r in adni:
        adni_sessions[ADNI_SUBJECT.match(r["original_code"]).group(1)].append(r)
    external = []
    for subject, rs in sorted(adni_sessions.items()):
        values = {r["group"].strip().upper() for r in rs}
        if len(values) == 1 and next(iter(values)) in ("AD", "HC"):
            first = min(rs, key=lambda r: ADNI_SUBJECT.match(r["original_code"]).group(2))
            external.append({"subject": subject, "idx": first["idx"],
                             "session": first["original_code"], "group": next(iter(values)),
                             "age": first["age"]})
    with (out / "external_adni.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(external[0]))
        writer.writeheader()
        writer.writerows(external)
    for diagnosis in ("AD", "HC"):
        ages = [float(e["age"]) for e in external if e["group"] == diagnosis]
        print(f"ADNI {diagnosis}: {len(ages)} subjects, age {np.mean(ages):.1f} ± {np.std(ages):.1f}")
    (out / "adni_sessions.txt").write_text("\n".join(e["idx"] for e in external) + "\n",
                                           encoding="utf-8")
    # ---- variant: the ADNI subjects that are neither AD nor HC (SCC, mixed) as SSL-only
    # train subjects with every session, so the encoder sees a second site's scanners
    # without touching the 366 external test subjects or any label.
    test_subjects = {e["subject"] for e in external}
    adni_rows = [{"split": "train", "subject": subject, "idx": r["idx"],
                  "session": r["original_code"], "group": r["group"].strip().upper(),
                  "primary": "0", "labeled": "0"}
                 for subject, rs in sorted(adni_sessions.items()) if subject not in test_subjects
                 for r in rs]
    variant = ROOT / f"{args.out}_adni"
    variant.mkdir(parents=True, exist_ok=True)
    with (variant / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows + adni_rows)
    n_adni = len({r["subject"] for r in adni_rows})
    (variant / "SPLIT.md").write_text(
        f"# Split de despliegue + ADNI solo-SSL\n\nIgual que `{Path(args.out).name}` más {n_adni} "
        f"sujetos ADNI que no son AD ni HC ({len(adni_rows)} sesiones: SCC y diagnóstico mixto) "
        "en train como solo-SSL. Los 366 sujetos AD/HC de ADNI (external_adni.csv) no aparecen: "
        "siguen siendo el test externo.\n", encoding="utf-8")
    print(f"variant {variant.name}: + {n_adni} ADNI SSL-only subjects, {len(adni_rows)} sessions")
    cached = set(json.loads(CACHE_META.read_text(encoding="utf-8"))["idx"])
    (variant / "new_sessions.txt").write_text(
        "\n".join(sorted({r["idx"] for r in adni_rows} - cached)) + "\n", encoding="utf-8")
    missing = sorted({r["idx"] for r in out_rows} - cached)
    (out / "new_sessions.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"{len(missing)} OASIS3 sessions missing from the ROI caches; "
          f"{len(external)} ADNI sessions for the external caches")
    assert list(out_rows[0]) == FIELDS


if __name__ == "__main__":
    main()
