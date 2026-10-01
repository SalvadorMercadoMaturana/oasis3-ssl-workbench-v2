"""Predeclare a larger OASIS3 test set without changing SSL train/validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
from t5_ssl.data.paths import DEFAULT_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("representative_oasis3.csv"))
    parser.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV))
    parser.add_argument("--output", type=Path, default=Path("expanded_test_oasis3.csv"))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    rows = clinical_rows(args.csv)
    indexed = {row["idx"]: row for row in rows}
    subject_rows = defaultdict(list)
    for row in rows:
        subject_rows[row["subject"]].append(row)
    assigned = {}
    for item in original:
        row = indexed.get(item["idx"])
        if (row is None or row["subject"] != item["subject"] or
                row["session"] != item["session"] or row["group"] != item["group"]):
            raise ValueError(f"Original manifest differs from clinical CSV: {item['idx']}")
        previous = assigned.setdefault(item["subject"], item["split"])
        if previous != item["split"]:
            raise ValueError(f"Subject in multiple original splits: {item['subject']}")
    extras = []
    for subject in sorted(set(subject_rows) - set(assigned)):
        for index, row in enumerate(sorted(subject_rows[subject],
                                           key=lambda r: (r["session"], r["idx"]))):
            extras.append({"split": "test", "subject": subject, "idx": row["idx"],
                           "session": row["session"], "group": row["group"],
                           "primary": str(int(index == 0))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "subject", "idx", "session",
                                                    "group", "primary"))
        writer.writeheader()
        writer.writerows(original + extras)
    subject_counts = Counter({"train": 0, "val": 0, "test": 0})
    for split in list(assigned.values()) + ["test"] * len(set(subject_rows) - set(assigned)):
        subject_counts[split] += 1
    summary = {"source_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               "original_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "extended_manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "subjects_by_split": dict(subject_counts),
               "additional_test_subjects": len(set(subject_rows) - set(assigned)),
               "protocol": "Original SSL train and val unchanged; every remaining OASIS3 "
                           "subject assigned to held-out test; one primary MNI session per subject"}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
