"""Audit session-level label availability and demographic imbalance in OASIS3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
from t5_ssl.data.paths import DEFAULT_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV))
    parser.add_argument("--manifest", type=Path,
                        default=Path("campaign_full50/all_sessions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("campaign_anatomy/label_audit.json"))
    args = parser.parse_args()
    rows = clinical_rows(args.csv)
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    labels = {subject: next(iter(values)) for subject, subject_rows in by_subject.items()
              if len(values := {row["group"] for row in subject_rows}) == 1 and
              next(iter(values)) in ("AD", "HC")}
    source_by_idx = {row["idx"]: row for row in rows}
    manifest_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    splits: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in manifest_rows:
        if item["primary"] == "1" and item["subject"] in labels:
            splits[item["split"]].append(source_by_idx[item["idx"]])
    summary = {}
    for split, entries in splits.items():
        summary[split] = {
            "subjects": len(entries),
            "AD": sum(labels[row["subject"]] == "AD" for row in entries),
            "HC": sum(labels[row["subject"]] == "HC" for row in entries),
            "mean_primary_age_AD": float(np.mean(
                [float(row["age"]) for row in entries if labels[row["subject"]] == "AD"])),
            "mean_primary_age_HC": float(np.mean(
                [float(row["age"]) for row in entries if labels[row["subject"]] == "HC"])),
        }
    output = {
        "source_csv": str(args.csv),
        "source_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "split_manifest": str(args.manifest),
        "split_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "oasis3_sessions": len(rows),
        "oasis3_subjects": len(by_subject),
        "group_counts_per_session": dict(Counter(row["group"] for row in rows)),
        "subjects_with_mixed_groups": sum(
            len({row["group"] for row in subject_rows}) > 1
            for subject_rows in by_subject.values()),
        "subjects_eligible_for_AD_HC_probe": len(labels),
        "missing_diagnosis_sessions": sum(row["diagnosis"] in ("", "-") for row in rows),
        "missing_sex_sessions": sum(not row["sex"].strip() for row in rows),
        "split_primary_session_summary": summary,
        "interpretation": "The CSV has group per MRI session, but its diagnosis field is '-' "
                          "throughout OASIS3. Mixed-group subjects are excluded from the probe. "
                          "For constant-group subjects, the CSV alone cannot verify whether the "
                          "group reflects clinical status at each MRI date.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
