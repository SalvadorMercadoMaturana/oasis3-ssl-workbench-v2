"""Subject-level OASIS3 split 361 / 63 / 84, stratified by diagnosis and age.

    python scripts/make_split_361_63_84.py [--out campaign_ssl3d_runpod/split_361_63_84]

Re-partitions the same 508 subjects and 594 cached sessions of the balanced
manifest (campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv)
so that only the split changes:

- labels as everywhere in the pipeline: a subject is AD or HC when every one of
  its OASIS3 sessions carries that group; the rest (mixed or unknown diagnosis,
  LBD, FTD, VD) are "OTHER" and all go to train;
- test: 43 HC + 41 AD; validation: 32 HC + 31 AD; train: everything else;
- within each diagnosis, subjects are allocated to test, then validation,
  proportionally over age-tertile strata (largest remainder, seed 42); sex would
  join the strata, but no OASIS3 row of the clinical CSV records it;
- train keeps every manifest session of its subjects; validation and test keep
  one session per subject (the manifest's primary one, else the earliest), as
  the original "eval primary" design. No subject (hence no session) is in two splits.

Writes ``manifest.csv`` (same columns as the original) and ``SPLIT.md`` with the
balance per split.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
CLINICAL = ROOT / "data/dementia/mni-dataset.csv"
SIZES = {"test": {"HC": 43, "AD": 41}, "val": {"HC": 32, "AD": 31}}
SUBJECT = re.compile(r"^(sub-OAS3\d+)")
DAY = re.compile(r"d(\d+)$")


def allocate(total: int, strata: dict[str, list[str]]) -> dict[str, int]:
    """Largest-remainder proportional allocation of ``total`` over the strata."""
    n = sum(len(v) for v in strata.values())
    quotas = {k: total * len(v) / n for k, v in strata.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[:total - sum(counts.values())]:
        counts[k] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="campaign_ssl3d_runpod/split_361_63_84")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = list(csv.DictReader(ORIGINAL.open(encoding="utf-8")))
    clinical = [r for r in csv.DictReader(CLINICAL.open(encoding="utf-8"))
                if r["study_name"].strip().upper() == "OASIS3"]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in clinical:
        groups[SUBJECT.match(row["original_code"]).group(1)].add(row["group"].strip().upper())
    by_idx = {row["idx"]: row for row in clinical}
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        sessions[row["subject"]].append(row)

    def label(subject: str) -> str:
        values = groups[subject]
        only = next(iter(values))
        return only if len(values) == 1 and only in ("AD", "HC") else "OTHER"

    def eval_session(subject: str) -> dict:
        rows = sessions[subject]
        primary = [r for r in rows if r["primary"] == "1"]
        return min(primary or rows, key=lambda r: int(DAY.search(r["session"]).group(1)))

    def age_sex(subject: str) -> tuple[float, str]:
        row = by_idx[eval_session(subject)["idx"]]
        return float(row["age"]), row["sex"].strip().lower()

    rng = random.Random(args.seed)
    assignment: dict[str, str] = {}
    for diagnosis in ("HC", "AD"):
        pool = sorted(s for s in sessions if label(s) == diagnosis)
        ages = np.array([age_sex(s)[0] for s in pool])
        cuts = np.quantile(ages, [1 / 3, 2 / 3])
        for split in ("test", "val"):
            remaining = [s for s in pool if s not in assignment]
            strata: dict[str, list[str]] = defaultdict(list)
            for s in remaining:
                age, sex = age_sex(s)
                strata[f"{sex}_{int(np.searchsorted(cuts, age, side='right'))}"].append(s)
            for key, n in allocate(SIZES[split][diagnosis], strata).items():
                members = sorted(strata[key])
                rng.shuffle(members)
                assignment.update({s: split for s in members[:n]})
    for subject in sessions:
        assignment.setdefault(subject, "train")

    out_rows = []
    for subject in sorted(sessions):
        split = assignment[subject]
        rows = sessions[subject] if split == "train" else [eval_session(subject)]
        for row in rows:
            out_rows.append({**row, "split": split})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Checks and balance report.
    subjects = {sp: {r["subject"] for r in out_rows if r["split"] == sp} for sp in ("train", "val", "test")}
    assert not (subjects["train"] & subjects["val"] or subjects["train"] & subjects["test"]
                or subjects["val"] & subjects["test"]), "subject in two splits"
    assert len({r["idx"] for r in out_rows}) == len(out_rows), "duplicated session"
    lines = ["# Split OASIS3 361 / 63 / 84 (por sujeto)", "",
             f"Generado con `scripts/make_split_361_63_84.py` (semilla {args.seed}). Mismos 508 "
             "sujetos y sesiones cacheadas que el manifiesto balanceado original; edad y sexo de la "
             "sesión evaluada.", "",
             "| Split | Sujetos | HC | AD | Otros | Sesiones | Edad HC (media ± DE) | Edad AD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        subs = sorted(subjects[split])
        counts = Counter(label(s) for s in subs)
        cells = []
        for diagnosis in ("HC", "AD"):
            ages = [age_sex(s)[0] for s in subs if label(s) == diagnosis]
            cells.append(f"{np.mean(ages):.1f} ± {np.std(ages):.1f}")
        lines.append(f"| {split} | {len(subs)} | {counts['HC']} | {counts['AD']} | "
                     f"{counts['OTHER']} | {sum(r['split'] == split for r in out_rows)} | "
                     f"{cells[0]} | {cells[1]} |")
    lines += ["", "Otros en train: " + ", ".join(
        f"{'/'.join(sorted(k))} {v}" for k, v in Counter(
            tuple(sorted(groups[s])) for s in subjects["train"] if label(s) == "OTHER").most_common()),
        "", "Estratificado por diagnóstico y tercil de edad: el sexo no está registrado para "
        "ninguna sesión OASIS3 del CSV clínico.",
        "", "Ningún sujeto ni sesión aparece en dos splits; val y test solo contienen AD/HC.", ""]
    (out / "SPLIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
