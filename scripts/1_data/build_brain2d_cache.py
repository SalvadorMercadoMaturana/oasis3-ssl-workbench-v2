#!/usr/bin/env python
"""Build the whole-brain canvas cache of the multiplanar 2D ingestion (t5_ssl.data.brain2d).

    python scripts/1_data/build_brain2d_cache.py [--like data/roi3d/hippo_s32.json]
                                                 [--out data/brain2d/brain_mni2.npy]

Sessions: the ``idx`` list of an existing ROI cache (``--like``), so every manifest the
ROI campaigns ran on (split_deploy, split_outer5 folds) is covered, in the same order.
Each volume is cropped to its minimal tissue box (voxels > 0), z-scored over its tissue
voxels and stored at the origin of a shared canvas filled with its background value
(float16, memory-mapped .npy + .json with the per-session boxes and tissue statistics).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from t5_ssl.data.brain2d import build_cache  # noqa: E402
from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_LMDB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--like", default=str(ROOT / "data/roi3d/hippo_s32.json"),
                    help="ROI cache whose sessions (idx) to preprocess")
    ap.add_argument("--out", default=str(ROOT / "data/brain2d/brain_mni2.npy"))
    ap.add_argument("--lmdb", default=DEFAULT_LMDB)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke test)")
    args = ap.parse_args()
    idx = json.loads(Path(args.like).read_text(encoding="utf-8"))["idx"]
    if args.limit:
        idx = idx[:args.limit]
    with Path(args.csv).open(encoding="utf-8") as stream:
        codes = {row["idx"]: row["original_code"] for row in csv.DictReader(stream)}
    start = time.time()
    meta = build_cache(args.out, lmdb_path=args.lmdb, idx=idx,
                       sessions=[codes[i] for i in idx])
    print(f"{meta['sessions']} sessions -> {args.out}: canvas {meta['canvas']}, boxes "
          f"{meta['box_shape_min']}..{meta['box_shape_max']} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
