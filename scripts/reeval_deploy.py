#!/usr/bin/env python
"""Re-score split_deploy runs with the v2 probe protocol (scripts/probe_protocol.py).

    python scripts/reeval_deploy.py --tags dep50_ssl dep50_init [--runs campaign_ssl3d_runpod/deploy/runs]

For every run with ``embeddings.npz`` and ``encoder.pt``: embeds the SSL-only AD/HC
subjects of train (primary session, clean central cube) with the run's own encoder on
the CPU unless ``embeddings.npz`` already holds ``pool_x`` (runs from the fixed
train.py) or ``embeddings_pool.npz`` exists, then writes ``protocol_v2.json`` next to
``result.json``. Supervised runs (no encoder) carry their v2 scores in ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import probe_protocol as pp  # noqa: E402

SPLIT = ROOT / "campaign_ssl3d_runpod/split_deploy"


def load_encoder(path: Path):
    import torch
    import train as entry

    from t5_ssl.train import build_backbone
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = entry.compose([o for o in saved["overrides"]
                         if not o.startswith(("++log_root", "++run_name"))])
    cfg.model.backbone.pretrained = False
    encoder = build_backbone(cfg)
    encoder.load_state_dict({k: v.float() for k, v in saved["state"].items()})
    return encoder.eval(), cfg


def embed_records(encoder, cache_path: str, records: list[dict], roi_size: int) -> np.ndarray:
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    meta = json.loads(Path(cache_path).with_suffix(".json").read_text(encoding="utf-8"))
    row = {k: i for i, k in enumerate(meta["idx"])}
    cubes = np.load(cache_path, mmap_mode="r")
    clean = ROIEval3D(roi_size)
    out = []
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            chunk = records[start:start + 32]
            batch = torch.stack([clean(torch.from_numpy(np.asarray(cubes[row[r["idx"]]],
                                                                   dtype=np.float32)))
                                 for r in chunk])
            out.append(encoder(batch).float().numpy())
    return np.concatenate(out)


def pool_embeddings(run: Path, records: list[dict]) -> np.ndarray:
    z = np.load(run / "embeddings.npz")
    if "pool_x" in z.files:
        return z["pool_x"]
    side = run / "embeddings_pool.npz"
    if side.is_file():
        saved = np.load(side)
        if list(saved["subjects"]) == [r["subject"] for r in records]:
            return saved["x"]
    encoder, cfg = load_encoder(run / "encoder.pt")
    x = embed_records(encoder, str(ROOT / cfg.data.cache_path), records, int(cfg.data.roi_size))
    np.savez_compressed(side, x=x.astype(np.float32),
                        subjects=np.array([r["subject"] for r in records]))
    return x


def adni_qc_mask(subjects: list[str]) -> np.ndarray:
    """ADNI input QC of the deployed model (build_deploy_model.qc_range on the hippocampal cube)."""
    import csv

    import build_deploy_model as bd
    qc = bd.qc_range("hippo_s32")
    idx = {r["subject"]: r["idx"] for r in
           csv.DictReader((SPLIT / "external_adni.csv").open(encoding="utf-8"))}
    value = bd.cache_intensity("adni_hippo_s32", [idx[s] for s in subjects])
    return (value >= qc["lo"]) & (value <= qc["hi"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "campaign_ssl3d_runpod/deploy/runs")
    parser.add_argument("--tags", nargs="+", default=["dep50_ssl", "dep50_init"])
    parser.add_argument("--manifest", type=Path, default=SPLIT / "manifest.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    records = pp.pool_records(args.manifest)
    ages = pp.session_ages(args.manifest, external=SPLIT / "external_adni.csv")
    y_pool = np.array([r["label"] for r in records])
    print(f"pool: {len(records)} SSL-only subjects ({int(y_pool.sum())} AD)", flush=True)
    for tag in args.tags:
        for run in sorted((args.runs / tag).glob("*/embeddings.npz")):
            run = run.parent
            out = run / "protocol_v2.json"
            if out.is_file() and not args.force:
                continue
            start = time.time()
            z = np.load(run / "embeddings.npz")
            x_pool = pool_embeddings(run, records)
            extra = {}
            if "adni_x" in z.files:
                ok = adni_qc_mask(list(z["adni_subjects"]))
                extra["adni_qc"] = (z["adni_x"][ok], z["adni_y"][ok], list(z["adni_subjects"][ok]))
            doc = pp.evaluate((z["cv_x"], z["cv_y"], list(z["cv_subjects"])),
                              (x_pool, y_pool, [r["subject"] for r in records]),
                              (z["test_x"], z["test_y"], list(z["test_subjects"])), ages, extra)
            out.write_text(json.dumps(doc) + "\n", encoding="utf-8")
            r, m, t = doc["cv_rep"], doc["cv_matched"], doc["test"]
            print(f"{tag}/{run.name}: rep AUROC {r['auc']:.3f} AUPR {r['aupr']:.3f} "
                  f"F1 {r['f1_macro']:.3f} mean3 {r['mean3']:.3f} | matched {m['auc']:.3f} | "
                  f"test {t['auc']:.3f}/{t['aupr']:.3f}/{t['f1_macro']:.3f} | "
                  f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
