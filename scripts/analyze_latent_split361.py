"""Latent space of the split 361/63/84 champions: what the embeddings encode.

For each champion (seed 1, ``campaign_ssl3d_runpod/split361_30``) and its starting
encoder without SSL, on the probe input (clean central cube of every session):

- ``compute``: session embeddings, per-session variables (age, education,
  hippocampal / lateral-ventricle / brain volume in MNI space from the LMDB
  segmentation, cube intensity and an image-noise estimate), the probe's
  out-of-fold AD score, t-SNE, robustness of the test AUROC to added Gaussian
  noise, cross-session retrieval (same subject scanned on another day) and an
  occlusion map of the probe's score on the test subjects. Cached in
  ``campaign_ssl3d_runpod/latent_split361/``.
- ``plot``: ``latent.png``, ``latent_corr.png`` and ``noise_checks.png`` in
  ``docs/figures_split361/``.

Usage: ``python scripts/analyze_latent_split361.py [compute|noise|plot|all]`` (default all).
Another run directory: ``--runs split361_100 --out latent_split361_100 --suffix _100
--run byol=byol_gn_dp0.3_dq0_wd0.01__noise__roi32`` (directories under
campaign_ssl3d_runpod; ``--suffix`` goes before ``.png`` in every figure name).
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "src")]

RUNS = ROOT / "campaign_ssl3d_runpod" / "split361_30"
OUT = ROOT / "campaign_ssl3d_runpod" / "latent_split361"
RUN = {"vicreg": "vicreg_campaign__geo__roi32", "byol": "byol_wide_dproj0.1__noise__roi32",
       "dinov2": "dinov2_campaign__elastic__roimulti3"}
PREFIX = {"vicreg": "backbone.", "byol": "backbone.", "dinov2": "teacher_backbone."}
SEED = 1
NOISE_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.3)
BLOCK, STRIDE = 8, 4
OCCLUSION = "--occlusion" in sys.argv
HIPPO, VENTRICLES = (17, 53), (4, 43, 5, 44)  # aseg: hippocampus; lateral + inferior lateral ventricles
DROP = ("+perf", "perf.", "data.pin_memory", "++data.prefetch_factor", "data.num_workers",
        "trainer.precision", "++log_root", "++run_name")


# ---------------------------------------------------------------- models and data
def compose(model: str):
    import train as entry
    doc = json.loads((RUNS / f"{RUN[model]}_s{SEED}" / "result.json").read_text(encoding="utf-8"))
    overrides = [o for o in doc["overrides"] if not o.startswith(DROP)]
    return entry.compose(overrides + ["data.num_workers=0", "trainer.precision=32-true"])


def encoders(model: str, cfg) -> dict:
    import torch

    from t5_ssl.train import build_backbone
    torch.manual_seed(SEED)
    start = build_backbone(cfg)
    trained = build_backbone(cfg)
    ckpt = RUNS / f"{RUN[model]}_s{SEED}" / "checkpoints" / "last.ckpt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    prefix = PREFIX[model]
    trained.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return {"ssl": trained.eval(), "init": start.eval()}


def sessions(cfg) -> tuple[list[dict], dict]:
    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_datamodule
    clean = ROIEval3D(int(cfg.data.roi_size))
    dm = build_datamodule(cfg, clean, clean)
    dm.setup()
    records = []
    for split in ("train", "val", "test"):
        for r in getattr(dm, f"{split}_dataset").records:
            records.append({"idx": r["idx"], "subject": r["subject"], "split": split,
                            "label": {"AD": 1, "HC": 0}.get(dm.subject_labels.get(r["subject"], ""), -1)})
    return records, dm.cubes


def clean_cubes(records: list[dict], cubes: dict):
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    clean = ROIEval3D(32)
    return torch.stack([clean(torch.from_numpy(np.asarray(cubes[r["idx"]], dtype=np.float32)))
                        for r in records])


def embed(encoder, x, batch: int = 32) -> np.ndarray:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    with torch.inference_mode():
        out = [encoder(x[i:i + batch].to(device)).float().cpu() for i in range(0, len(x), batch)]
    encoder.cpu()
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- variables
def session_variables(records: list[dict], x_hippo) -> dict[str, np.ndarray]:
    """Clinical columns, MNI-space volumes from the segmentation and image statistics."""
    import lmdb
    from scipy.ndimage import gaussian_filter

    from t5_ssl.data.oasis3_datamodule import normalize_volume
    from t5_ssl.data.paths import DEFAULT_LMDB
    rows = {r["idx"]: r for r in csv.DictReader((ROOT / "data/dementia/mni-dataset.csv").open(encoding="utf-8"))}
    num = lambda v: float(v) if v not in ("", "nan", None) else np.nan  # noqa: E731
    out = defaultdict(list)
    env = lmdb.open(str(DEFAULT_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for i, r in enumerate(records):
            row = rows[r["idx"]]
            out["edad"].append(num(row["age"]))
            out["educación"].append(num(row["education"]))
            payload = pickle.loads(txn.get(r["idx"].encode("ascii")))
            volume = normalize_volume(np.asarray(payload["image"]))
            out["vol. cerebral"].append(float((volume > 0.1).sum()) * 8e-3)  # cm^3 (2 mm voxels)
            if "roi" in payload:
                roi = np.asarray(payload["roi"])
                out["vol. hipocampo"].append(float(np.isin(roi, HIPPO).sum()) * 8e-3)
                out["vol. ventrículos"].append(float(np.isin(roi, VENTRICLES).sum()) * 8e-3)
            else:
                out["vol. hipocampo"].append(np.nan)
                out["vol. ventrículos"].append(np.nan)
            cube = x_hippo[i].numpy()
            tissue = cube > 0.1
            residual = cube - gaussian_filter(cube, sigma=(0, 1, 1, 1))
            res = residual[tissue]
            out["intensidad media"].append(float(cube[tissue].mean()))
            out["ruido de imagen"].append(float(1.4826 * np.median(np.abs(res - np.median(res)))))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def by_subject(records: list[dict], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["subject"]].append(i)
    subjects = sorted(groups)
    return subjects, np.stack([np.nanmean(values[groups[s]], axis=0) for s in subjects])


# ---------------------------------------------------------------- probes
def probe():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1, 10], cv=5, scoring="roc_auc", class_weight="balanced",
        max_iter=5000))


def oof_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        out[test] = probe().fit(x[train], y[train]).decision_function(x[test])
    return out


def oof_ridge(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = np.isfinite(t)
    out = np.full(len(t), np.nan)
    xs, ts, idx = x[ok], t[ok], np.flatnonzero(ok)
    for train, test in KFold(5, shuffle=True, random_state=0).split(xs):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        out[idx[test]] = model.fit(xs[train], ts[train]).predict(xs[test])
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------- compute
def compute() -> None:
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results: dict = (json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                     else {"seed": SEED, "models": {}})
    variables = None
    for model in RUN:
        if model in results["models"]:  # resume: models already embedded are kept
            continue
        cfg = compose(model)
        records, cubes = sessions(cfg)
        x = clean_cubes(records, cubes)
        if variables is None and (OUT / "variables.npz").is_file():
            v = np.load(OUT / "variables.npz")
            variables, subjects = dict(zip(v["names"], v["values"].T, strict=True)), list(v["subjects"])
            var_subj = v["values"]
            label = {r["subject"]: r["label"] for r in records}
            split = {r["subject"]: r["split"] for r in records}
        if variables is None:  # the hippocampal cube is the first two channels of every cache
            hippo_cfg = compose("vicreg")
            hrec, hcubes = sessions(hippo_cfg)
            assert [r["idx"] for r in hrec] == [r["idx"] for r in records]
            variables = session_variables(records, clean_cubes(hrec, hcubes))
            subjects, var_subj = by_subject(records, np.stack(list(variables.values()), axis=1))
            label = {r["subject"]: r["label"] for r in records}
            split = {r["subject"]: r["split"] for r in records}
            np.savez(OUT / "variables.npz", subjects=np.array(subjects), names=np.array(list(variables)),
                     values=var_subj, labels=np.array([label[s] for s in subjects]),
                     splits=np.array([split[s] for s in subjects]))
        enc = encoders(model, cfg)
        res: dict = {}
        for role, encoder in enc.items():
            print(f"[{model}/{role}] embedding {len(x)} sessions", flush=True)
            z = embed(encoder, x)
            subj, zs = by_subject(records, z)
            assert subj == subjects
            y = np.array([label[s] for s in subjects])
            adhc = y >= 0
            za, ya = zs[adhc], y[adhc]
            scores = oof_scores(za, ya)
            r: dict = {"oof_auroc": auroc(ya, scores)}
            std = StandardScaler().fit_transform(za)
            pcs = PCA(5, random_state=0).fit(std)
            r["pca_var"] = pcs.explained_variance_ratio_.tolist()
            np.savez(OUT / f"{model}_{role}.npz", sessions=z, subject_emb=zs, scores=scores,
                     pcs=pcs.transform(std),
                     tsne=TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(std),
                     ridge=np.stack([oof_ridge(za, var_subj[adhc, k]) for k in range(var_subj.shape[1])], 1))
            # robustness: probe fit on clean train+val subjects, test sessions with added noise
            tr = np.array([split[s] in ("train", "val") for s in subjects])[adhc]
            clf = probe().fit(za[tr], ya[tr])
            test_idx = [i for i, rec in enumerate(records) if rec["split"] == "test" and rec["label"] >= 0]
            yt = np.array([records[i]["label"] for i in test_idx])
            xt = x[test_idx]
            r["noise_auroc"] = {}
            gen = torch.Generator().manual_seed(0)
            for sigma in NOISE_SIGMAS:
                noisy = (xt + sigma * torch.randn(xt.shape, generator=gen)).clamp(0, 1) if sigma else xt
                r["noise_auroc"][str(sigma)] = auroc(yt, clf.decision_function(embed(encoder, noisy)))
            # cross-session retrieval: subjects with >= 2 sessions, nearest other session
            groups = defaultdict(list)
            for i, rec in enumerate(records):
                groups[rec["subject"]].append(i)
            multi = [i for s, ids in groups.items() if len(ids) >= 2 for i in ids]
            zm = StandardScaler().fit(z).transform(z[multi])
            zm /= np.linalg.norm(zm, axis=1, keepdims=True)
            sim = zm @ zm.T
            np.fill_diagonal(sim, -np.inf)
            owner = [records[i]["subject"] for i in multi]
            r["retrieval"] = {"top1": float(np.mean([owner[j] == owner[i] for i, j in enumerate(sim.argmax(1))])),
                              "sessions": len(multi), "subjects": len({*owner}),
                              "chance": float(np.mean([(owner.count(o) - 1) / (len(owner) - 1) for o in owner]))}
            # occlusion (slow: 343 blocks x 84 subjects) only on request
            if role == "ssl" and OCCLUSION:
                r["occlusion_file"] = f"{model}_occlusion.npy"
                np.save(OUT / r["occlusion_file"], occlusion(encoder, clf, xt))
            if role == "ssl" and model == "dinov2":
                r["region_ablation"] = region_ablation(encoder, clf, xt, yt)
            res[role] = r
            print(f"[{model}/{role}] oof AUROC {r['oof_auroc']:.3f} | noise {r['noise_auroc']} | "
                  f"retrieval {r['retrieval']['top1']:.3f}", flush=True)
        results["models"][model] = res
        (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def occlusion(encoder, clf, xt) -> np.ndarray:
    base = clf.decision_function(embed(encoder, xt))
    heat, count = np.zeros((32, 32, 32)), np.zeros((32, 32, 32))
    fill = xt.mean(dim=(2, 3, 4), keepdim=True)
    for a in range(0, 32 - BLOCK + 1, STRIDE):
        for b in range(0, 32 - BLOCK + 1, STRIDE):
            for c in range(0, 32 - BLOCK + 1, STRIDE):
                occ = xt.clone()
                occ[:, :, a:a + BLOCK, b:b + BLOCK, c:c + BLOCK] = fill.expand(-1, -1, BLOCK, BLOCK, BLOCK)
                delta = np.abs(clf.decision_function(embed(encoder, occ)) - base).mean()
                heat[a:a + BLOCK, b:b + BLOCK, c:c + BLOCK] += delta
                count[a:a + BLOCK, b:b + BLOCK, c:c + BLOCK] += 1
    return heat / np.maximum(count, 1)


def region_ablation(encoder, clf, xt, yt) -> dict[str, float]:
    out = {"ninguna": auroc(yt, clf.decision_function(embed(encoder, xt)))}
    for k, region in enumerate(("hipocampo", "ventrículo", "posteromedial")):
        occ = xt.clone()
        ch = slice(2 * k, 2 * k + 2)
        occ[:, ch] = occ[:, ch].mean(dim=(2, 3, 4), keepdim=True).expand_as(occ[:, ch])
        out[region] = auroc(yt, clf.decision_function(embed(encoder, occ)))
    return out


WHITE_MATTER = (2, 41)  # aseg: left / right cerebral white matter
OLD_NOISE, NOISE = "ruido de imagen", "ruido (sust. blanca)"


def scanner_noise() -> None:
    """Replace the cube's high-frequency estimate (it mixes noise with atrophy edges) with a
    scan-level noise estimate: robust spread of the high-pass residual inside the eroded
    cerebral white matter, a homogeneous tissue. Re-fits only the ridge column of that variable
    from the saved subject embeddings."""
    import lmdb
    from scipy.ndimage import binary_erosion, gaussian_filter

    from t5_ssl.data.oasis3_datamodule import normalize_volume
    from t5_ssl.data.paths import DEFAULT_LMDB
    v = dict(np.load(OUT / "variables.npz"))
    names = list(v["names"])
    col = names.index(OLD_NOISE) if OLD_NOISE in names else names.index(NOISE)
    manifest = list(csv.DictReader((RUNS.parent / "split_361_63_84" / "manifest.csv").open(encoding="utf-8")))
    per_subject: dict[str, list[float]] = defaultdict(list)
    env = lmdb.open(str(DEFAULT_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for row in manifest:
            payload = pickle.loads(txn.get(row["idx"].encode("ascii")))
            if "roi" not in payload:
                continue
            volume = normalize_volume(np.asarray(payload["image"]))
            wm = binary_erosion(np.isin(np.asarray(payload["roi"]), WHITE_MATTER), iterations=2)
            res = (volume - gaussian_filter(volume, 1.0))[wm]
            per_subject[row["subject"]].append(float(1.4826 * np.median(np.abs(res - np.median(res)))))
    v["values"][:, col] = [np.mean(per_subject[s]) if per_subject.get(s) else np.nan for s in v["subjects"]]
    names[col] = NOISE
    v["names"] = np.array(names)
    np.savez(OUT / "variables.npz", **v)
    adhc = v["labels"] >= 0
    for model in RUN:
        for role in ("ssl", "init"):
            e = dict(np.load(OUT / f"{model}_{role}.npz"))
            e["ridge"][:, col] = oof_ridge(e["subject_emb"][adhc], v["values"][adhc, col])
            np.savez(OUT / f"{model}_{role}.npz", **e)
    print("scanner noise: done", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", nargs="?", default="all", choices=("compute", "noise", "plot", "all"))
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--runs", help="run directory under campaign_ssl3d_runpod")
    parser.add_argument("--out", help="cache directory under campaign_ssl3d_runpod")
    parser.add_argument("--run", action="append", default=[], metavar="MODEL=NAME")
    parser.add_argument("--suffix", default="", help="figure name suffix, e.g. _100")
    args = parser.parse_args()
    if args.runs:
        RUNS = ROOT / "campaign_ssl3d_runpod" / args.runs
    if args.out:
        OUT = ROOT / "campaign_ssl3d_runpod" / args.out
    RUN.update(item.split("=", 1) for item in args.run)
    stage = args.stage
    if stage in ("compute", "all"):
        compute()
    if stage in ("noise", "compute", "all"):
        scanner_noise()
    if stage in ("plot", "all"):
        import plot_latent_split361
        plot_latent_split361.OUT, plot_latent_split361.SUFFIX = OUT, args.suffix
        plot_latent_split361.main()
