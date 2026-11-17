"""Select a reproducible OASIS3 subject cohort without copying image data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from t5_ssl.data.paths import DEFAULT_CSV

SUBJECT_RE = re.compile(r"^(sub-OAS\d+)ses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("representative_oasis3.csv"))
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    subjects: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["study_name"].strip().upper() != "OASIS3":
                continue
            match = SUBJECT_RE.match(row["original_code"])
            if match is None:
                raise ValueError(f"Invalid OASIS3 code: {row['original_code']}")
            subjects[match.group(1)].append(row)
    strata: dict[str, list[str]] = defaultdict(list)
    for subject, rows in subjects.items():
        groups = {r["group"].strip().upper() for r in rows}
        stratum = next(iter(groups)) if len(groups) == 1 and next(iter(groups)) in ("AD", "HC") else "OTHER"
        strata[stratum].append(subject)
    rng = random.Random(args.seed)
    selected: dict[str, str] = {}
    counts: dict[str, Counter] = {split: Counter() for split in ("train", "val", "test")}
    for stratum, ids in sorted(strata.items()):
        ids.sort()
        rng.shuffle(ids)
        size = min(len(ids), max(3, round(len(ids) * args.fraction)))
        chosen = ids[:size]
        n_train = max(1, int(size * 0.70))
        n_val = max(1, int(size * 0.15))
        if n_train + n_val >= size:
            n_train, n_val = size - 2, 1
        for split, cohort in (("train", chosen[:n_train]),
                              ("val", chosen[n_train:n_train + n_val]),
                              ("test", chosen[n_train + n_val:])):
            for subject in cohort:
                selected[subject] = split
                counts[split][stratum] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "subject", "idx", "session", "group", "primary"])
        for subject in sorted(selected):
            rows = sorted(subjects[subject], key=lambda row: (row["original_code"], row["idx"]))
            for index, row in enumerate(rows):
                writer.writerow([selected[subject], subject, row["idx"],
                                 row["original_code"], row["group"], int(index == 0)])
    summary = {"source_csv": str(args.csv),
               "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               "fraction": args.fraction, "seed": args.seed,
               "subject_counts": {split: dict(value) for split, value in counts.items()},
               "selected_subjects": len(selected),
               "selected_sessions": sum(len(subjects[s]) for s in selected),
               "protocol": "OASIS3 only; group-stratified subject sample; all sessions assigned "
                           "to one split; primary marks one deterministic session per subject"}
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "summary": str(summary_path),
                      **summary}, indent=2))


if __name__ == "__main__":
    main()
