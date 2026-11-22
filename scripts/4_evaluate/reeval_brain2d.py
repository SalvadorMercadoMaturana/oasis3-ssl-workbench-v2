#!/usr/bin/env python
"""Re-score multiplanar runs (``--roi brain2d``) with another multi-slice probe input.

    python scripts/4_evaluate/reeval_brain2d.py RUN_DIR [RUN_DIR ...] \
        --offsets=-10,-8,-6,-4,-2,0,2,4,6,8,10 --name ms10

Every run needs the ``encoder.pt`` of ``train.py --export-encoder`` (the evaluated
encoder: the teacher for DINOv2). The encoder embeds the axial, coronal and sagittal
planes at each offset from the centre of mass and averages the embeddings
(t5_ssl.evaluation.run_scoring.score_multislice). Protocol v2 is written to
``multislice_<name>.json`` next to ``result.json``, with the same layout as
``result.json["multislice"]``. Training is not repeated, so the probe input can be
swept after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _plain(value: object) -> object:
    return value.tolist() if hasattr(value, "tolist") else float(value)


def rescore(run: Path, offsets: list[int], name: str) -> dict:
    import torch
    import train as entry

    from t5_ssl.data.roi3d import probe_transform
    from t5_ssl.evaluation.run_scoring import score_multislice
    from t5_ssl.train import build_backbone, build_datamodule
    saved = torch.load(run / "encoder.pt", map_location="cpu", weights_only=True)
    cfg = entry.compose([o for o in saved["overrides"]
                         if not o.startswith(("++log_root", "++run_name"))])
    if str(cfg.data.get("source", "")) != "brain2d":
        raise ValueError(f"{run}: not a brain2d run")
    clean = probe_transform(cfg)
    dm = build_datamodule(cfg, clean, clean)
    dm.setup()
    encoder = build_backbone(cfg)
    encoder.load_state_dict({k: v.float() if v.is_floating_point() else v
                             for k, v in saved["state"].items()})
    start = time.time()
    doc = score_multislice(encoder.float().eval(), dm, cfg, offsets)
    doc["name"], doc["minutes"] = name, (time.time() - start) / 60
    (run / f"multislice_{name}.json").write_text(json.dumps(doc, default=_plain) + "\n",
                                                encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--offsets", required=True, help="comma-separated slice offsets")
    ap.add_argument("--name", required=True, help="suffix of multislice_<name>.json")
    args = ap.parse_args()
    offsets = [int(o) for o in args.offsets.split(",")]
    runs = [r.resolve() for r in args.runs]
    os.environ.setdefault("OASIS3_CSV", str(ROOT / "data/dementia/mni-dataset.csv"))
    os.chdir(ROOT)  # relative data/manifest paths in the configs
    for run in runs:
        doc = rescore(run, offsets, args.name)
        cv, te = doc["protocol_v2"]["cv_rep"], doc["protocol_v2"]["test"]
        print(f"{run.name} {args.name}: CV {cv['mean3']:.3f} test {te['mean3']:.3f} "
              f"({doc['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
