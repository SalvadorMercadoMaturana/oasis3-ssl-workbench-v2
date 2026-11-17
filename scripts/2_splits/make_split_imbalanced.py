"""Class-imbalanced test set and an age-matched 55/45 split, both on top of split 361/63/84.

    python scripts/2_splits/make_split_imbalanced.py

Two manifests (same columns as campaign_ssl3d_runpod/split_361_63_84/manifest.csv
plus ``labeled``), both with the same held-out subjects:

- ``test``: the 43 HC + 41 AD of split 361 (unchanged);
- ``test_imb``: 203 OASIS3 HC never used before, so ``test`` + ``test_imb`` =
  41 AD + 246 HC (1:6, 14 % AD). They are drawn one per target, cycling over the
  ages of the 43 test HC in random order (nearest age without replacement, seed
  42), so the imbalanced test has the age distribution of the balanced one: only
  the prevalence changes. One session per subject (the earliest).

``split_361_imb``: split 361 unchanged (train 361 / val 63) + ``test_imb``. The
SSL and probes train exactly as in split 361; only the test scoring grows.

``split_agematch``: train/val re-drawn from every other OASIS3 subject (162 AD,
431 HC, 101 others):

- labeled AD and HC are age-matched at HC:AD = 1.25 (55/45) inside 2.5-year bins
  (age at the evaluated session): per bin ``min(AD, floor(HC / 1.25))`` AD and
  1.25 times as many HC, drawn at random (seed 42); the AD and HC left over
  become SSL-only train subjects (``labeled=0``), as the others;
- validation takes 24 AD + 30 HC of the labeled ones, proportionally over the
  age bins (largest remainder), so train and validation keep the same ages and ratio;
- sessions: subjects of the original balanced manifest keep their cached
  sessions there (about 1.13 per HC/AD subject); new labeled HC get one session
  (the earliest), so the classes keep a similar number of sessions per subject;
  new SSL-only subjects keep every session; validation keeps one session.

Writes ``<out>/split_361_imb/`` and ``<out>/split_agematch/`` (``manifest.csv``
+ ``SPLIT.md``) and ``<out>/split_agematch/new_sessions.txt`` (LMDB keys
missing from the ROI caches: scripts/1_data/extend_roi_caches.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from make_split_361_63_84 import allocate

ROOT = Path(__file__).resolve().parents[2]
SPLIT361 = ROOT / "campaign_ssl3d_runpod/split_361_63_84/manifest.csv"
ORIGINAL = ROOT / "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
CLINICAL = ROOT / "data/dementia/mni-dataset.csv"
CACHE_META = ROOT / "data/roi3d/hippo_s32.json"
IMBALANCE = 6          # HC per AD in test + test_imb
RATIO = 1.25           # labeled HC per AD in split_agematch train/val
BIN_YEARS = 2.5
VAL_AD = 24
SUBJECT = re.compile(r"^(sub-OAS3\d+)")
DAY = re.compile(r"d(\d+)$")
FIELDS = ["split", "subject", "idx", "session", "group", "primary", "labeled"]


def day(session: str) -> int:
    return int(DAY.search(session).group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="campaign_ssl3d_runpod")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    clinical = [r for r in csv.DictReader(CLINICAL.open(encoding="utf-8"))
                if r["study_name"].strip().upper() == "OASIS3"]
    by_idx = {r["idx"]: r for r in clinical}
    csv_sessions: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in clinical:
        subject = SUBJECT.match(row["original_code"]).group(1)
        csv_sessions[subject].append(row)
        groups[subject].add(row["group"].strip().upper())

    def label(subject: str) -> str:
        values = groups[subject]
        only = next(iter(values))
        return only if len(values) == 1 and only in ("AD", "HC") else "OTHER"

    def row_of(split: str, clinical_row: dict, primary: str, labeled: str = "1") -> dict:
        return {"split": split, "subject": SUBJECT.match(clinical_row["original_code"]).group(1),
                "idx": clinical_row["idx"], "session": clinical_row["original_code"],
                "group": clinical_row["group"].strip().upper(), "primary": primary,
                "labeled": labeled}

    split361 = list(csv.DictReader(SPLIT361.open(encoding="utf-8")))
    original: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(ORIGINAL.open(encoding="utf-8")):
        original[row["subject"]].append(row)
    assignment = {row["subject"]: row["split"] for row in split361}
    eval_row = {row["subject"]: row for row in split361 if row["split"] != "train"}

    def eval_session(subject: str) -> dict:
        """Return the evaluated session: split 361's, else the original primary, else the earliest."""
        if subject in eval_row:
            return by_idx[eval_row[subject]["idx"]]
        primary = [by_idx[r["idx"]] for r in original.get(subject, []) if r["primary"] == "1"]
        rows = primary or [by_idx[r["idx"]] for r in original.get(subject, [])] or csv_sessions[subject]
        return min(rows, key=lambda r: day(r["original_code"]))

    def age(subject: str) -> float:
        return float(eval_session(subject)["age"])

    rng = random.Random(args.seed)
    # ---- test_imb: unused HC, age-matched to the balanced test HC.
    test_hc = sorted(s for s, sp in assignment.items() if sp == "test" and label(s) == "HC")
    test_ad = sorted(s for s, sp in assignment.items() if sp == "test" and label(s) == "AD")
    unused = sorted(s for s in csv_sessions if s not in assignment)
    available = [s for s in unused if label(s) == "HC"]
    extra_n = IMBALANCE * len(test_ad) - len(test_hc)
    imb: list[str] = []
    targets = [age(s) for s in test_hc]
    while len(imb) < extra_n:
        for target in rng.sample(targets, len(targets)):
            if len(imb) == extra_n:
                break
            best = min(available, key=lambda s: (abs(age(s) - target), s))
            available.remove(best)
            imb.append(best)
    imb_rows = [row_of("test_imb", eval_session(s), "1") for s in sorted(imb)]

    # ---- split_361_imb
    base = [{**row, "labeled": "1"} for row in split361]
    write(ROOT / args.out / "split_361_imb", base + imb_rows, label, age,
          "Split 361/63/84 + test desbalanceado", [
              "Train y validación idénticos al split 361/63/84; `test` igual (43 HC + 41 AD). "
              f"`test_imb` agrega {len(imb)} HC de OASIS3 no usados antes, emparejados por edad con "
              "los HC de test (vecino más cercano sin reemplazo, semilla "
              f"{args.seed}): test + test_imb = {len(test_ad)} AD + {len(test_hc) + len(imb)} HC "
              f"(1:{IMBALANCE}).",
              "Generado con `scripts/2_splits/make_split_imbalanced.py`."])

    # ---- split_agematch: everything that is not test re-drawn.
    held = set(test_hc) | set(test_ad) | set(imb)
    pool = sorted(s for s in csv_sessions if s not in held)
    ad_pool = [s for s in pool if label(s) == "AD"]
    hc_pool = [s for s in pool if label(s) == "HC"]

    def bin_of(subject: str) -> int:
        return math.floor(age(subject) / BIN_YEARS)

    ad_bins: dict[int, list[str]] = defaultdict(list)
    hc_bins: dict[int, list[str]] = defaultdict(list)
    for s in ad_pool:
        ad_bins[bin_of(s)].append(s)
    for s in hc_pool:
        hc_bins[bin_of(s)].append(s)
    labeled_ad, labeled_hc = [], []
    for b in sorted(ad_bins):
        n_ad = min(len(ad_bins[b]), math.floor(len(hc_bins[b]) / RATIO))
        n_hc = min(len(hc_bins[b]), round(RATIO * n_ad))
        labeled_ad += rng.sample(sorted(ad_bins[b]), n_ad)
        labeled_hc += rng.sample(sorted(hc_bins[b]), n_hc)
    val: set[str] = set()
    for members, n in ((labeled_ad, VAL_AD), (labeled_hc, round(RATIO * VAL_AD))):
        strata: dict[int, list[str]] = defaultdict(list)
        for s in members:
            strata[bin_of(s)].append(s)
        for b, k in allocate(n, strata).items():
            val.update(rng.sample(sorted(strata[b]), k))
    labeled = set(labeled_ad) | set(labeled_hc)
    rows: list[dict] = []
    new_sessions: list[str] = []
    for subject in pool:
        if subject in val:
            rows.append(row_of("val", eval_session(subject), "1"))
            continue
        flag = "1" if subject in labeled else "0"
        if subject in original:  # cached sessions of the balanced manifest
            first = eval_session(subject)["idx"]
            rows += [row_of("train", by_idx[r["idx"]], "1" if r["idx"] == first else "0", flag)
                     for r in original[subject]]
        elif flag == "1":  # new labeled HC: one session, like the HC/AD of the manifest
            rows.append(row_of("train", eval_session(subject), "1", flag))
        else:  # new SSL-only subject: every session
            first = eval_session(subject)["idx"]
            rows += [row_of("train", r, "1" if r["idx"] == first else "0", flag)
                     for r in sorted(csv_sessions[subject], key=lambda r: day(r["original_code"]))]
    rows += [{**row, "labeled": "1"} for row in split361 if row["split"] == "test"] + imb_rows
    out = ROOT / args.out / "split_agematch"
    write(out, rows, label, age, "Split emparejado por edad 55/45 + test desbalanceado", [
        "Test y test_imb idénticos a `split_361_imb`. Train/val re-sorteados con todos los demás "
        f"sujetos OASIS3 ({len(ad_pool)} AD, {len(hc_pool)} HC, "
        f"{len(pool) - len(ad_pool) - len(hc_pool)} otros): AD y HC con etiqueta emparejados por "
        f"edad en tramos de {BIN_YEARS} años a HC:AD = {RATIO} (55/45); los AD/HC sobrantes y los "
        "otros quedan en train como sujetos solo-SSL (`labeled=0`: ningún probe ni modelo "
        "supervisado ve su diagnóstico).",
        f"Validación: {VAL_AD} AD + {round(RATIO * VAL_AD)} HC con etiqueta, proporcional por tramo "
        "de edad. Sesiones: los sujetos del manifiesto balanceado original conservan sus sesiones "
        "cacheadas; los HC nuevos con etiqueta, una (la primera); los nuevos solo-SSL, todas.",
        "Generado con `scripts/2_splits/make_split_imbalanced.py`."])
    cached = set(json.loads(CACHE_META.read_text(encoding="utf-8"))["idx"])
    new_sessions = sorted({r["idx"] for r in rows} - cached)
    (out / "new_sessions.txt").write_text("\n".join(new_sessions) + "\n", encoding="utf-8")
    print(f"{len(new_sessions)} sessions missing from the ROI caches -> {out / 'new_sessions.txt'}")


def write(out: Path, rows: list[dict], label, age, title: str, notes: list[str]) -> None:
    """Manifest, leakage checks and the balance table (SPLIT.md + split.json)."""
    from scipy.stats import ks_2samp
    subjects: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subjects[row["split"]].add(row["subject"])
    names = list(subjects)
    assert not any(subjects[a] & subjects[b] for a in names for b in names if a < b), "leakage"
    assert len({r["idx"] for r in rows}) == len(rows), "duplicated session"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    labeled = {r["subject"]: r["labeled"] for r in rows}
    lines = [f"# {title}", "", *[n + "\n" for n in notes],
             "| Split | Sujetos | HC etiq. | AD etiq. | Solo SSL (HC/AD/otros) | Sesiones | "
             "Edad HC (media ± DE) | Edad AD | KS edad HC vs AD (p) |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    summary = {}
    for split in ("train", "val", "test", "test_imb"):
        subs = sorted(subjects.get(split, ()))
        if not subs:
            continue
        on = [s for s in subs if labeled[s] == "1"]
        off = Counter(label(s) for s in subs if labeled[s] == "0" or label(s) == "OTHER")
        ages = {d: [age(s) for s in on if label(s) == d] for d in ("HC", "AD")}
        cell = {d: f"{np.mean(a):.1f} ± {np.std(a):.1f}" if a else "—" for d, a in ages.items()}
        p = ks_2samp(ages["HC"], ages["AD"]).pvalue if ages["HC"] and ages["AD"] else float("nan")
        lines.append(f"| {split} | {len(subs)} | {len(ages['HC'])} | {len(ages['AD'])} | "
                     f"{off['HC']}/{off['AD']}/{off['OTHER']} | "
                     f"{sum(r['split'] == split for r in rows)} | {cell['HC']} | {cell['AD']} | "
                     f"{p:.3f} |")
        summary[split] = {"subjects": len(subs), "hc": len(ages["HC"]), "ad": len(ages["AD"]),
                          "ssl_only": dict(off), "age_hc": ages["HC"], "age_ad": ages["AD"]}
    lines += ["", "Ningún sujeto ni sesión aparece en dos splits; edad de la sesión evaluada.", ""]
    (out / "SPLIT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "split.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
