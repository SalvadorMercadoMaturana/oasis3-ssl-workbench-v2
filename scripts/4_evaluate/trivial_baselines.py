#!/usr/bin/env python
"""Trivial baselines scored exactly like the SSL encoders, on the five outer folds.

    python scripts/4_evaluate/trivial_baselines.py [--sets stats aseg hippo_vol imagenet random_s1 random_s2]
                                                   [--out campaign_ssl3d_runpod/outer5/baselines]

Question (2026-11-21): the frozen ImageNet ConvNeXt-T ("sin SSL") and the fully supervised
model score close to the SSL encoders and rise and fall with them from fold to fold. Is the
AD/HC signal of the hippocampal cubes something any feature extractor picks up?

Per-session features of every cached session (``data/roi3d/hippo_s32``); no label is used
to compute any of them:

- ``imagenet``: the frozen ConvNeXt-T 3D with inflated ImageNet weights, i.e. "sin SSL"
  (``--check-init`` compares it with the embeddings saved by the sin SSL runs);
- ``random_s1``, ``random_s2``: the same architecture with random weights (two seeds);
- ``stats``: 19 intensity statistics per channel of the evaluated 32^3 cube (mean, std,
  7 quantiles, 10-bin histogram of the [0, 1] normalized intensities);
- ``aseg``: FreeSurfer aparc+aseg volumes stored in the LMDB (``roi``), in voxels of the
  2 mm MNI grid: hippocampus, amygdala, inferior lateral and lateral ventricles,
  entorhinal and parahippocampal cortex (left and right), white matter, cortex, brain;
  ``hippo_vol`` keeps the two hippocampal volumes only. Sessions without a parcellation
  (OASIS3 ``has_roi`` False) get the mean of the labeled subjects of the fold.

Each set goes through ``t5_ssl.evaluation.protocol_v2.evaluate`` with the subjects of each
outer fold (labeled train+val, SSL-only pool, test; one session per subject, the manifest
primary, as in ``train.py``): the same probe, C grid, folds, thresholds and metrics as the
SSL runs. Writes ``features.npz`` and ``protocol_v2.json`` (``"<set>|<fold>"`` -> document,
resumable). CPU only, ~10 min.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402

CAMPAIGN = ROOT / "campaign_ssl3d_runpod"
CACHE = ROOT / "data/roi3d/hippo_s32"
LMDB = ROOT / "data/dementia/mni-dataset.lmdb"
FOLDS = range(5)
SETS = ("stats", "aseg", "hippo_vol", "imagenet", "random_s1", "random_s2")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BINS = np.linspace(0, 1, 11)
ASEG = {"hippocampus_L": [17], "hippocampus_R": [53], "amygdala_L": [18], "amygdala_R": [54],
        "inf_lat_vent_L": [5], "inf_lat_vent_R": [44], "lat_vent_L": [4], "lat_vent_R": [43],
        "entorhinal_L": [1006], "entorhinal_R": [2006],
        "parahippocampal_L": [1016], "parahippocampal_R": [2016], "white_matter": [2, 41]}
ASEG_NAMES = (*ASEG, "cortex", "brain")
# sin SSL runs of report_outer5.REFERENCE["init_cnx"] (only read by --check-init)
INIT_RUNS = {0: CAMPAIGN / "deploy/runs/dep50_init"} | {
    k: CAMPAIGN / f"outer5/runs/outer_f{k}_init" for k in FOLDS if k}
INIT_NAME = "vicreg_init__bbconvnext_tiny__roi32_s1"


def manifest(k: int) -> Path:
    """Fold 0 is split_deploy, folds 1-4 come from scripts/2_splits/make_split_outer.py."""
    return (CAMPAIGN / "split_deploy/manifest.csv" if k == 0
            else CAMPAIGN / f"split_outer5/fold{k}/manifest.csv")


# ---------------------------------------------------------------- per-session features
def cube_batches(cubes: np.ndarray, size: int = 64):
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    clean = ROIEval3D(32)
    for start in range(0, len(cubes), size):
        yield torch.stack([clean(torch.from_numpy(np.asarray(cubes[i], dtype=np.float32)))
                           for i in range(start, min(start + size, len(cubes)))])


def cube_statistics(cubes: np.ndarray) -> np.ndarray:
    rows = []
    for batch in cube_batches(cubes):
        for cube in batch.numpy():
            row = []
            for channel in cube:
                v = channel.ravel()
                row += [v.mean(), v.std(), *np.quantile(v, QUANTILES)]
                row += list(np.histogram(v, BINS)[0] / v.size)
            rows.append(row)
    return np.asarray(rows, np.float32)


def stats_names() -> list[str]:
    names = []
    for side in ("L", "R"):
        names += [f"{side}_mean", f"{side}_std", *(f"{side}_q{round(q * 100)}" for q in QUANTILES)]
        names += [f"{side}_hist{b:.1f}" for b in BINS[:-1]]
    return names


def frozen_embeddings(cubes: np.ndarray, pretrained: bool, seed: int) -> np.ndarray:
    import torch

    from t5_ssl.models.backbones.resnet3d import ConvNeXt3DBackbone
    torch.manual_seed(seed)
    encoder = ConvNeXt3DBackbone("convnext_tiny_3d", in_channels=2, pretrained=pretrained)
    encoder = encoder.float().eval()
    with torch.inference_mode():
        return np.concatenate([encoder(b).float().numpy() for b in cube_batches(cubes)])


def aseg_volumes(idx: list[str]) -> np.ndarray:
    import lmdb
    out = np.full((len(idx), len(ASEG_NAMES)), np.nan, np.float32)
    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for n, key in enumerate(idx):
            payload = pickle.loads(txn.get(key.encode("ascii")))
            out[n, -1] = float((np.asarray(payload["image"]) != 0).sum())
            if payload.get("roi") is not None and payload.get("has_roi", True):
                counts = np.bincount(np.asarray(payload["roi"]).ravel().clip(0), minlength=3000)
                for j, labels in enumerate(ASEG.values()):
                    out[n, j] = counts[labels].sum()
                out[n, -2] = counts[1000:3000].sum()  # cortical parcels of aparc
    return out


def compute_features(path: Path) -> None:
    t0 = time.time()
    idx = json.loads(CACHE.with_suffix(".json").read_text(encoding="utf-8"))["idx"]
    cubes = np.load(CACHE.with_suffix(".npy"), mmap_mode="r")
    feats = {"stats": cube_statistics(cubes)}
    feats["imagenet"] = frozen_embeddings(cubes, True, 1)
    for seed in (1, 2):
        feats[f"random_s{seed}"] = frozen_embeddings(cubes, False, seed)
    print(f"[baselines] encoders done {time.time() - t0:.0f}s", flush=True)
    feats["aseg"] = aseg_volumes(idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, idx=np.array(idx), stats_names=np.array(stats_names()),
             aseg_names=np.array(ASEG_NAMES), **feats)
    print(f"[baselines] features {time.time() - t0:.0f}s -> {path}", flush=True)


# ---------------------------------------------------------------- fold subjects
def fold_subjects(k: int) -> dict:
    """Labeled CV, pool and test subjects of fold k in the order train.py scores them."""
    labels = pp.subject_labels()
    with manifest(k).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    primary = {r["subject"]: r["idx"] for r in rows if r["primary"] == "1"}
    lab = sorted({r["subject"] for r in rows if r["subject"] in labels and (
        r["split"] == "val" or (r["split"] == "train" and r["labeled"] == "1"))})
    pool = [r["subject"] for r in pp.pool_records(manifest(k))]
    test = sorted({r["subject"] for r in rows if r["split"] == "test" and r["subject"] in labels})
    return {"lab": lab, "pool": pool, "test": test, "primary": primary, "labels": labels}


def matrix(x: np.ndarray, row_of: dict[str, int], subjects: list[str], primary: dict) -> np.ndarray:
    return np.stack([x[row_of[primary[s]]] for s in subjects]).astype(np.float64)


def check_init(feats: np.lib.npyio.NpzFile) -> None:
    """Stop unless the recomputed ImageNet embeddings are the saved sin SSL embeddings."""
    row_of = {k: i for i, k in enumerate(feats["idx"])}
    for k in FOLDS:
        saved = np.load(INIT_RUNS[k] / INIT_NAME / "embeddings.npz")
        subjects = fold_subjects(k)
        for part in ("cv", "test"):
            mine = matrix(feats["imagenet"], row_of, list(saved[f"{part}_subjects"]),
                          subjects["primary"])
            rel = np.abs(mine - saved[f"{part}_x"]).max() / np.abs(saved[f"{part}_x"]).max()
            print(f"[baselines] check fold {k} {part}: max relative difference {rel:.1e}")
            if rel > 1e-4:
                raise SystemExit("recomputed ImageNet embeddings differ from the sin SSL run")


def evaluate_sets(feats: np.lib.npyio.NpzFile, sets: list[str], out: Path) -> None:
    row_of = {k: i for i, k in enumerate(feats["idx"])}
    names = list(feats["aseg_names"])
    arrays = {name: feats[name] for name in ("stats", "aseg", "imagenet", "random_s1", "random_s2")}
    arrays["hippo_vol"] = feats["aseg"][:, [names.index("hippocampus_L"), names.index("hippocampus_R")]]
    results = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    t0 = time.time()
    for k in FOLDS:
        sub = fold_subjects(k)
        ages = pp.session_ages(manifest(k))
        y = {part: np.array([sub["labels"][s] for s in sub[part]]) for part in ("lab", "pool", "test")}
        for name in sets:
            key = f"{name}|{k}"
            if key in results:
                continue
            x = {part: matrix(arrays[name], row_of, sub[part], sub["primary"])
                 for part in ("lab", "pool", "test")}
            missing = {part: int(np.isnan(m).any(1).sum()) for part, m in x.items()}
            fill = np.nanmean(x["lab"], axis=0)
            for m in x.values():
                r, c = np.nonzero(np.isnan(m))
                m[r, c] = fill[c]
            doc = pp.evaluate(*((x[p], y[p], sub[p]) for p in ("lab", "pool", "test")), ages)
            doc["missing"] = missing
            results[key] = doc
            out.write_text(json.dumps(results), encoding="utf-8")
            r, t = doc["cv_rep"], doc["test"]
            print(f"[baselines] fold {k} {name:9s} CV AUROC {r['auc']:.3f} test AUROC {t['auc']:.3f} "
                  f"auc_age {t['auc_age']:.3f} mean3 {t['mean3']:.3f} missing {missing} "
                  f"{time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--out", type=Path, default=CAMPAIGN / "outer5/baselines")
    ap.add_argument("--check-init", action="store_true",
                    help="compare the ImageNet features with the saved sin SSL embeddings")
    args = ap.parse_args(argv)
    features = args.out / "features.npz"
    if not features.is_file():
        compute_features(features)
    feats = np.load(features)
    if args.check_init:
        check_init(feats)
    evaluate_sets(feats, args.sets, args.out / "protocol_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
