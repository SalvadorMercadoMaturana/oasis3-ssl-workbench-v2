"""P(AD) for new MRIs with a model built by scripts/6_deploy/build_deploy_model.py.

    python scripts/6_deploy/predict_ad.py --model deploy/ad_mri_v1 --lmdb-keys 001303 002151
    python scripts/6_deploy/predict_ad.py --model deploy/ad_mri_v1 --volumes scan1.nii.gz scan2.npy \
        [--site site.json] [--threshold f1|youden|sens80] [--out predictions.csv]
    python scripts/6_deploy/predict_ad.py --model deploy/ad_mri_v1 --volumes new_site/*.nii.gz \
        --calibrate-site site.json            # unlabeled batch of a new site, no prediction

Each volume must be a T1 MRI on the dataset's MNI 2 mm grid (97, 115, 97), preprocessed
as data/dementia/mni-dataset.lmdb; it is normalized with ``normalize_volume``, the ROI
cubes are cut at the model's fixed positions (left/right mirrored pairs) and the central
32-voxel cube goes through every seed's encoder and probe. Output per scan: P(AD) (mean
over seeds; 50/50 prior), the decision at the chosen threshold and an out-of-distribution
flag (Mahalanobis distance above the training 99th percentile in a majority of seeds:
treat the prediction as unreliable). A scan whose cube intensity falls outside the
model's QC range (OASIS3's, see ``qc_intensity``) is rejected: decision ``QC``, no P(AD).

``--calibrate-site`` stores the embedding mean and SD of an unlabeled batch of a new site
(tens of scans, any mix of patients); ``--site`` then standardizes that site's embeddings
with them instead of OASIS3's (the remedy measured on ADNI in the model card).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from t5_ssl.evaluation.input_qc import qc_intensity  # noqa: E402


def load_volumes(args: argparse.Namespace) -> tuple[list[str], list[np.ndarray]]:
    names, volumes = [], []
    if args.lmdb_keys:
        import lmdb
        env = lmdb.open(str(ROOT / "data/dementia/mni-dataset.lmdb"), readonly=True, lock=False,
                        readahead=False)
        with env.begin() as txn:
            for key in args.lmdb_keys:
                payload = pickle.loads(txn.get(key.encode()))
                names.append(key)
                volumes.append(np.asarray(payload["image"]))
    for path in args.volumes or []:
        p = Path(path)
        if p.suffix == ".npy":
            volume = np.load(p)
        else:
            import nibabel as nib
            volume = np.asarray(nib.load(str(p)).get_fdata())
        names.append(p.name)
        volumes.append(volume)
    for name, volume in zip(names, volumes, strict=True):
        if volume.shape != (97, 115, 97):
            raise SystemExit(f"{name}: shape {volume.shape}, expected the MNI 2 mm grid (97, 115, 97)")
    return names, volumes


def encoders(model: dict, model_dir: Path) -> list:
    import torch
    import train as entry

    from t5_ssl.train import build_backbone
    out = []
    for seed in model["seeds"]:
        saved = torch.load(model_dir / seed["encoder"], map_location="cpu", weights_only=False)
        cfg = entry.compose([o for o in saved["overrides"]
                             if not o.startswith(("++log_root", "++run_name"))])
        cfg.model.backbone.pretrained = False  # weights come from the file
        encoder = build_backbone(cfg)
        encoder.load_state_dict({k: v.float() for k, v in saved["state"].items()})
        out.append(encoder.eval())
    return out


def embed(model: dict, nets: list, volumes: list[np.ndarray]
          ) -> tuple[list[np.ndarray], np.ndarray]:
    """Return every seed's embeddings and the QC intensity of each scan."""
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from t5_ssl.data.oasis3_datamodule import normalize_volume
    from t5_ssl.data.roi3d import ROIEval3D, crop_cubes
    cube = model["cube"]
    clean = ROIEval3D(model["roi_size"])
    cubes = np.stack([crop_cubes(normalize_volume(v.astype(np.float32)), cube["cube_starts"],
                           cube["extract"]).astype(np.float32) for v in volumes])
    batch = torch.stack([clean(torch.from_numpy(c)) for c in cubes])
    with torch.inference_mode():
        return ([net(batch).float().numpy() for net in nets],
                qc_intensity(cubes, model["roi_size"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lmdb-keys", nargs="*", default=[])
    parser.add_argument("--volumes", nargs="*", default=[])
    parser.add_argument("--threshold", default=None,
                        help="f1 | youden | sens80 (default: the model's decision_threshold, else youden)")
    parser.add_argument("--site", type=Path, default=None, help="site.json from --calibrate-site")
    parser.add_argument("--calibrate-site", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model = json.loads((args.model / "model.json").read_text(encoding="utf-8"))
    names, volumes = load_volumes(args)
    if not volumes:
        raise SystemExit("no input volumes")
    embeddings, intensity = embed(model, encoders(model, args.model), volumes)
    qc = model.get("qc")
    passed = (np.ones(len(volumes), bool) if qc is None
              else (intensity >= qc["lo"]) & (intensity <= qc["hi"]))
    if args.calibrate_site:
        if len(volumes) < 20:
            print(f"warning: {len(volumes)} scans is a small calibration batch (use tens)")
        stats = {"scans": len(volumes), "seeds": [{"mean": e.mean(0).tolist(),
                                                  "sd": (e.std(0) + 1e-6).tolist()}
                                                 for e in embeddings]}
        args.calibrate_site.write_text(json.dumps(stats) + "\n", encoding="utf-8")
        print(f"site statistics of {len(volumes)} scans -> {args.calibrate_site}")
        return
    site = json.loads(args.site.read_text(encoding="utf-8")) if args.site else None
    probs, flags = [], []
    for k, (seed, e) in enumerate(zip(model["seeds"], embeddings, strict=True)):
        probe = seed["probe"]
        mean = np.array(site["seeds"][k]["mean"] if site else probe["mean"])
        sd = np.array(site["seeds"][k]["sd"] if site else probe["sd"])
        z = (e - mean) / sd
        probs.append(expit(z @ np.array(probe["coef"]) + probe["intercept"]))
        ood = seed["ood"]
        p = (z - np.array(ood["center"])) @ np.array(ood["components"]).T
        flags.append(np.sqrt((p**2 / np.array(ood["var"])).sum(1)) > ood["threshold"])
    prob = np.mean(probs, axis=0)
    flag = np.mean(flags, axis=0) >= 0.5
    threshold = model["thresholds"][args.threshold or model.get("decision_threshold", "youden")]
    rows = [{"scan": n, "p_ad": f"{p:.4f}" if ok else "",
             "decision": ("AD" if p >= threshold else "HC") if ok else "QC",
             "threshold": f"{threshold:.4f}", "qc_intensity": f"{v:.3f}",
             "out_of_distribution": bool(f)}
            for n, p, f, v, ok in zip(names, prob, flag, intensity, passed, strict=True)]
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for r in rows:
        if r["decision"] == "QC":
            print(f"{r['scan']}: qc_rechazada (intensidad {r['qc_intensity']} fuera de "
                  f"[{qc['lo']:.3f}, {qc['hi']:.3f}]: revisar el preprocesamiento; no se puntúa)")
            continue
        print(f"{r['scan']}: P(AD)={r['p_ad']} -> {r['decision']}"
              + ("  [fuera de distribución: no confiar]" if r["out_of_distribution"] else ""))


if __name__ == "__main__":
    main()
