"""Append sessions to existing ROI cube caches, cropped at the caches' own cube positions.

    python scripts/1_data/extend_roi_caches.py campaign_ssl3d_runpod/split_agematch/new_sessions.txt \
        hippo_s32 multi3_s32

The cube starts (and so every cached cube) stay those of the cache: no centroid
or brain box is recomputed, so the existing rows are byte-identical and every
manifest that worked keeps working. Before writing, the crop is checked against
``--verify`` cached sessions (exact float16 equality). Each cache is rewritten
as ``<name>.npy`` (old rows first, then the new ones) and its ``.json`` gets the
extended ``idx`` list and an ``extended`` record; the old pair is kept as
``<name>.pre_extend.{npy,json}`` unless ``--no-backup``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import lmdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from t5_ssl.data.oasis3_datamodule import normalize_volume  # noqa: E402
from t5_ssl.data.roi3d import crop_cubes  # noqa: E402

LMDB = ROOT / "data/dementia/mni-dataset.lmdb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", type=Path, help="text file, one LMDB key per line")
    parser.add_argument("caches", nargs="+", help="cache names in data/roi3d (e.g. hippo_s32)")
    parser.add_argument("--verify", type=int, default=8)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--into", default=None, metavar="PREFIX",
                        help="write only the listed sessions to a new cache <PREFIX><name> "
                             "(same cube positions; e.g. another cohort), leave <name> as is")
    args = parser.parse_args()
    wanted = [line.strip() for line in args.sessions.read_text().splitlines() if line.strip()]
    env = lmdb.open(str(LMDB), readonly=True, lock=False, readahead=False)
    for name in args.caches:
        start = time.time()
        npy, meta_path = ROOT / "data/roi3d" / f"{name}.npy", ROOT / "data/roi3d" / f"{name}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old = np.load(npy, mmap_mode="r")
        starts, extract = meta["cube_starts"], int(meta["extract"])
        if old.shape[1:] != (len(starts), extract, extract, extract):
            raise RuntimeError(f"{name}: cache shape {old.shape} does not match its metadata")
        present = set() if args.into else set(meta["idx"])
        new = [idx for idx in wanted if idx not in present]
        with env.begin() as txn:
            def volume(idx: str) -> np.ndarray:
                return normalize_volume(np.asarray(pickle.loads(txn.get(idx.encode()))["image"]))

            step = max(1, len(meta["idx"]) // max(1, args.verify))
            for row in range(0, len(meta["idx"]), step)[:args.verify]:
                if not np.array_equal(crop_cubes(volume(meta["idx"][row]), starts, extract), old[row]):
                    raise RuntimeError(f"{name}: re-crop of cached row {row} differs")
            print(f"{name}: crop reproduces {args.verify} cached sessions exactly", flush=True)
            kept = 0 if args.into else len(old)
            target = (npy.with_name(f"{args.into}{name}.npy") if args.into
                      else npy.with_suffix(".tmp.npy"))
            cubes = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                              shape=(kept + len(new), *old.shape[1:]))
            cubes[:kept] = old[:kept]
            for i, idx in enumerate(new):
                cubes[kept + i] = crop_cubes(volume(idx), starts, extract)
            cubes.flush()
            del cubes
        del old
        source = args.sessions.resolve()
        if args.into:
            meta.update(idx=new, sessions=len(new), source_cache=name, extended=[])
            target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n",
                                                   encoding="utf-8")
            print(f"{args.into}{name}: {len(new)} sessions ({time.time() - start:.0f} s)",
                  flush=True)
            continue
        meta.setdefault("extended", []).append(
            {"added": len(new), "date": time.strftime("%Y-%m-%d"),
             "from": (source.relative_to(ROOT) if source.is_relative_to(ROOT) else source).as_posix()})
        meta["idx"] = meta["idx"] + new
        meta["sessions"] = len(meta["idx"])
        meta_tmp = meta_path.with_suffix(".tmp.json")
        meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if not args.no_backup:
            shutil.copy2(npy, npy.with_suffix(".pre_extend.npy"))
            shutil.copy2(meta_path, meta_path.with_suffix(".pre_extend.json"))
        target.replace(npy)  # the cubes and their index list change together
        meta_tmp.replace(meta_path)
        print(f"{name}: +{len(new)} sessions -> {meta['sessions']} ({time.time() - start:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
