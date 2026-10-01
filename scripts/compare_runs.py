"""Paired-bootstrap comparisons between groups of scored runs (train.py / campaigns).

    python scripts/compare_runs.py runs/screen30 [more dirs...] \
        --pair byol_border byol_campaign --pair byol_border imagenet_resnet_3d ... \
        --out runs/screen30/COMPARISONS.md

Run documents are ``<dir>/<group>_s<seed>/result.json`` (train.py) or
``<dir>/<group>_s<seed>.json`` (campaign ``runs/`` folders, e.g.
``campaign_ssl3d_v2/runs``); the group is the name without ``_s<seed>``.
Each group's score is the seed-averaged mean of AUROC, F1-macro and AUPR, on
the 5x5 CV over train+val and on the test probe; a comparison is the paired
bootstrap over subjects (stratified, 2000 draws) of the difference, as
``scripts/run_ssl3d_v2.py report``. ``--spatial`` scores the spatial probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_aug_screen import fast_mean3


def load(dirs: list[str], spatial: bool) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for root in map(Path, dirs):
        paths = sorted(root.glob("*/result.json")) + sorted(root.glob("*_s*.json"))
        for path in paths:
            doc = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name if path.name == "result.json" else path.stem
            group = name.rsplit("_s", 1)[0]
            if "cv" not in doc:
                continue
            if spatial:
                if not doc.get("spatial"):
                    continue
                doc = {**doc, "cv": doc["spatial"]["cv"], "test": doc["spatial"]["test"]}
            groups[group].append(doc)
    return groups


def mean3(m: dict) -> float:
    return (m["auc"] + m["f1"] + m["aupr"]) / 3


def test3(t: dict) -> float:
    return (t["test_auc"] + t["test_f1_macro"] + t["test_average_precision"]) / 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    groups = load(args.dirs, args.spatial)
    first = next(iter(groups.values()))[0]
    labels = np.array(first["labels"])
    test_subjects = [p["subject"] for p in first["test"]["test_predictions"]]
    test_labels = np.array([p["AD"] for p in first["test"]["test_predictions"]])
    for name, docs in groups.items():
        for d in docs:
            if d["subjects"] != first["subjects"] or \
                    [p["subject"] for p in d["test"]["test_predictions"]] != test_subjects:
                raise RuntimeError(f"{name} was scored on different subjects")
    rng = np.random.default_rng(42)

    def draws(lab: np.ndarray) -> list[np.ndarray]:
        pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
        return [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
                for _ in range(args.draws)]

    boots = {"cv": (draws(labels), labels), "test": (draws(test_labels), test_labels)}
    scores = {"cv": lambda d: np.asarray(d["cv"]["clean"]["oof_mean"]),
              "test": lambda d: np.array([p["score"] for p in d["test"]["test_predictions"]])}
    summary = {name: {"n": len(docs),
                      "cv": float(np.mean([mean3(d["cv"]["clean"]["metrics"]) for d in docs])),
                      "test": float(np.mean([test3(d["test"]) for d in docs])),
                      "rank": float(np.mean([d.get("effective_rank", np.nan) for d in docs]))}
               for name, docs in sorted(groups.items())}
    lines = ["| Grupo | n | CV media3 | Test media3 | Rango ef. |", "|---|---:|---:|---:|---:|"]
    lines += [f"| `{k}` | {v['n']} | {v['cv']:.3f} | {v['test']:.3f} | {v['rank']:.1f} |"
              for k, v in summary.items()]
    results = []
    if args.pair:
        lines += ["", "| Comparación | Δ CV [IC 95 %] | P(Δ>0) | Δ test [IC 95 %] | P(Δ>0) |",
                  "|---|---:|---:|---:|---:|"]
    for a, b in args.pair:
        if a not in groups or b not in groups:
            print(f"skip {a} vs {b}: missing group", flush=True)
            continue
        row = {"a": a, "b": b}
        cells = []
        for split, (idx_list, lab) in boots.items():
            def value(idx: np.ndarray, split: str = split, lab: np.ndarray = lab,
                      a: str = a, b: str = b) -> float:
                return sum(w * float(np.mean([fast_mean3(lab[idx], scores[split](d)[idx])
                                              for d in groups[g]]))
                           for w, g in ((1, a), (-1, b)))
            samples = np.array([value(idx) for idx in idx_list])
            point = summary[a][split] - summary[b][split]
            lo, hi = np.percentile(samples, [2.5, 97.5])
            row[split] = {"delta": point, "ci": [float(lo), float(hi)],
                          "p_positive": float(np.mean(samples > 0))}
            cells.append(f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.mean(samples > 0):.2f}")
        results.append(row)
        lines.append(f"| `{a}` − `{b}` | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps({"summary": summary, "comparisons": results}, indent=2) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
