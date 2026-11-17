"""Select, fit and validate a deployable AD-vs-HC MRI model from the split_deploy runs.

    python scripts/4_evaluate/reeval_deploy.py --tags dep50_init dep50_ssl v2_ssl
    python scripts/6_deploy/build_deploy_model.py --runs campaign_ssl3d_runpod/deploy/runs \
        --out deploy/ad_mri_v2

Candidates are the frozen encoders of the deployment campaigns (``CANDIDATES``: each SSL
method and the encoders without SSL), 3 seeds each, read from ``embeddings.npz``,
``encoder.pt`` and ``protocol_v2.json`` (src/t5_ssl/evaluation/protocol_v2.py, protocol v2).

1. Selection (fixed before looking at any test): the seed ensemble with the highest
   ``mean3 = (AUROC + AUPR + F1-macro) / 3`` on the representative CV: probes fit on the
   age-matched labeled train+val folds score, out of fold, those subjects and the SSL-only
   AD/HC subjects of train (702 subjects, cohort prevalence 23 % AD and real ages).
   Until 2026-11-07 (ad_mri_v1) it was the AUROC on the age-matched 50/50 subjects only.
2. Final probe per seed: standardization + class-balanced logistic regression on every
   labeled train+val subject, C by inner 3-fold CV log-loss over probe_protocol.CS. The
   model is the mean of the seeds' probabilities. Operating thresholds from the
   representative out-of-fold ensemble probabilities: max F1-macro (the default
   decision at the cohort prevalence), Youden and sensitivity >= 0.80.
3. Site calibration (no labels): a new site's embeddings are standardized with that
   site's own mean and SD instead of OASIS3's; measured on ADNI with all 366 subjects and
   with random unlabeled batches of 30/60/120 scans at other AD shares.
4. Out-of-distribution flag: Mahalanobis distance in the 95 % PCA subspace of the
   training embeddings above their 99th percentile.
5. Validation: representative CV, representative OASIS3 test (cohort prevalence and
   ages, 41 AD + 137 HC), its age-matched subset, and ADNI (another site) raw and site-
   calibrated; class-stratified bootstrap 95 % CIs; age itself as the reference score.

Writes ``<out>/model.json`` (probe, thresholds, OOD and site statistics, cube geometry,
overrides), ``<out>/encoder_s<seed>.pt``, ``<out>/MODEL_CARD.md`` and ``<out>/validation.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402
from t5_ssl.evaluation.input_qc import QC_MARGIN, cache_intensity, qc_range  # noqa: E402
from t5_ssl.evaluation.metrics import fast_ap, fast_auc  # noqa: E402

SPLIT = ROOT / "campaign_ssl3d_runpod/split_deploy"
# label -> (tag, run name without _s<seed>)
CANDIDATES = {
    "VICReg · ConvNeXt-T": ("v2_ssl", "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
    "BYOL · ConvNeXt-T · cabezas BN": ("v2_ssl",
                                       "byol_wide_dproj0.1__noise__bbconvnext_tiny__roi32"),
    "BYOL · ResNet-18": ("v2_ssl", "byol_gn_dp0.3_dq0_wd0.01__noise__bbresnet18__roi32"),
    "BYOL · ConvNeXt-T · cabezas GN (v1, colapsado)": (
        "dep50_ssl", "byol_gn_dp0.3_dq0_wd0.01__noise__bbconvnext_tiny__roi32"),
    "DINOv2 · ViT-S (3 regiones)": ("v2_ssl",
                                    "dinov2_campaign__elastic_tio_gpu__bbvit_small__roimulti3"),
    "Sin SSL · ConvNeXt-T": ("dep50_init", "vicreg_init__bbconvnext_tiny__roi32"),
    "Sin SSL · ResNet-18": ("dep50_init", "vicreg_init__bbresnet18__roi32"),
    "Sin SSL · ViT-S (3 regiones)": ("dep50_init", "vicreg_init__bbvit_small__roimulti3"),
}


# ---------------------------------------------------------------- probe
class Probe:
    """Standardization + class-balanced logistic regression (numpy parameters)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
        del seed  # the inner folds are fixed (probe_protocol.make_probe)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = pp.make_probe().fit(x, y)
        scaler, model = pipe[0], pipe[-1]
        self.mean, self.sd = scaler.mean_, scaler.scale_
        self.c = pp.chosen_c(pipe)
        self.coef, self.intercept = model.coef_[0], float(model.intercept_[0])

    def proba(self, x: np.ndarray, mean: np.ndarray | None = None,
              sd: np.ndarray | None = None) -> np.ndarray:
        z = (x - (self.mean if mean is None else mean)) / (self.sd if sd is None else sd)
        return expit(z @ self.coef + self.intercept)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "sd": self.sd.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "C": self.c}


class OOD:
    """Mahalanobis distance in the 95 % PCA subspace of the standardized training embeddings."""

    def __init__(self, z: np.ndarray) -> None:
        self.center = z.mean(0)
        _, s, vt = np.linalg.svd(z - self.center, full_matrices=False)
        var = s**2 / (len(z) - 1)
        k = int(np.searchsorted(np.cumsum(var) / var.sum(), 0.95) + 1)
        self.components, self.var = vt[:k], var[:k]
        self.threshold = float(np.percentile(self.distance(z), 99))

    def distance(self, z: np.ndarray) -> np.ndarray:
        p = (z - self.center) @ self.components.T
        return np.sqrt((p**2 / self.var).sum(1))


# ---------------------------------------------------------------- metrics
def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    f1_pos, f1_neg = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "f1_macro": (f1_pos + f1_neg) / 2,
            "balanced_accuracy": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def with_ci(y: np.ndarray, p: np.ndarray, thresholds: dict[str, float],
            draws: int = 2000) -> dict:
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = {"auc": fast_auc(y, p), "aupr": fast_ap(y, p), "prevalence": float(y.mean()),
             "n": len(y), "ad": int(y.sum())}
    samples = defaultdict(list)
    for _ in range(draws):
        idx = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
        samples["auc"].append(fast_auc(y[idx], p[idx]))
        samples["aupr"].append(fast_ap(y[idx], p[idx]))
    for k in ("auc", "aupr"):
        point[f"{k}_ci"] = [float(v) for v in np.percentile(samples[k], [2.5, 97.5])]
    point["at"] = {name: threshold_metrics(y, p, t) for name, t in thresholds.items()}
    point["y"], point["p"] = y.astype(int).tolist(), [float(v) for v in p]  # ROC curves
    return point


# ---------------------------------------------------------------- data
def load_candidate(runs: Path, tag: str, name: str) -> list[dict]:
    seeds = []
    for result in sorted(runs.glob(f"{tag}/{name}_s*/result.json")):
        run = result.parent
        if not (run / "embeddings.npz").is_file() or not (run / "protocol_v2.json").is_file():
            continue
        doc = json.loads(result.read_text(encoding="utf-8"))
        z = np.load(run / "embeddings.npz")
        seeds.append({"dir": run, "doc": doc, "seed": doc["seed"],
                      **{k: z[k] for k in z.files}})
    return seeds


def fit_set(seeds: list[dict]) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Return every seed's labeled train+val embeddings, their labels and subjects."""
    x_fit = [np.concatenate([s["train_x"], s["val_x"]]) for s in seeds]
    y_fit = np.concatenate([seeds[0]["train_y"], seeds[0]["val_y"]])
    subjects = list(seeds[0]["train_subjects"]) + list(seeds[0]["val_subjects"])
    for s in seeds[1:]:
        assert list(s["train_subjects"]) + list(s["val_subjects"]) == subjects
    return x_fit, y_fit, subjects


def ages_by_subject() -> dict[str, float]:
    """Age at the evaluated (primary) session, as probe_protocol, plus ADNI."""
    return pp.session_ages(SPLIT / "manifest.csv", external=SPLIT / "external_adni.csv")


def age_matched(subjects: list[str], y: np.ndarray, ages: dict[str, float],
                per_ad: int = 2, caliper: float = 2.5) -> np.ndarray:
    """Return the indices of an age-matched subset: each AD with up to ``per_ad`` HC near in age."""
    ad = sorted((i for i in range(len(y)) if y[i] == 1), key=lambda i: -ages[subjects[i]])
    free = [i for i in range(len(y)) if y[i] == 0]
    keep = []
    for _ in range(per_ad):
        for i in ad:
            best = min(free, key=lambda j: abs(ages[subjects[j]] - ages[subjects[i]]), default=None)
            if best is not None and abs(ages[subjects[best]] - ages[subjects[i]]) <= caliper:
                keep += [i, best]
                free.remove(best)
    return np.array(sorted(set(keep)))


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "campaign_ssl3d_runpod/deploy/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "deploy/ad_mri_v2")
    parser.add_argument("--candidate", default=None, help="force a candidate (default: by CV)")
    parser.add_argument("--reason", default=None,
                        help="why --candidate was forced (Markdown, written into the model card)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ages = ages_by_subject()
    protocol = {label: pp.load_runs(args.runs, tag, name)
                for label, (tag, name) in CANDIDATES.items()}
    protocol = {k: v for k, v in protocol.items() if v}
    ens = {k: pp.ensemble(v, ages) for k, v in protocol.items()}
    score = {k: e["cv_rep"]["mean3"] for k, e in ens.items()}
    lines = ["Selección (fijada antes de ver el test): ensamble de las semillas con mayor "
             "**compuesto = (AUROC + AUPR + F1-macro) / 3** en la CV representativa: los probes "
             "de cada fold (entrenados con AD/HC emparejados por edad, 50/50) puntúan fuera de "
             "fold a esos sujetos y a los AD/HC de train que solo vio el SSL; 702 sujetos con la "
             "prevalencia (23 % AD) y las edades reales de OASIS3, como el test. F1-macro con "
             "umbral elegido fuera de fold. Compuesto (AUROC / AUPR / F1-macro):", ""]
    lines += [f"- {k}: **{score[k]:.3f}** ({ens[k]['cv_rep']['auc']:.3f} / "
              f"{ens[k]['cv_rep']['aupr']:.3f} / {ens[k]['cv_rep']['f1_macro']:.3f}; "
              f"{len(protocol[k])} semillas)"
              for k in sorted(score, key=score.get, reverse=True)]
    chosen = args.candidate or max(score, key=score.get)
    tag, name = CANDIDATES[chosen]
    seeds = load_candidate(args.runs, tag, name)
    lines += ["", f"Elegido: **{chosen}**", ""]
    if args.reason:
        lines += [args.reason, ""]
    # ---- final probes, thresholds (representative out-of-fold ensemble), OOD
    x_fit, y_fit, _ = fit_set(seeds)
    probes = [Probe(x, y_fit, s["seed"]) for x, s in zip(x_fit, seeds, strict=True)]
    y_rep, oof, n_lab = ens[chosen]["_cv"]
    grid = np.unique(oof)
    youden = float(max(grid, key=lambda t: sum(threshold_metrics(y_rep, oof, t)[k]
                                               for k in ("sensitivity", "specificity"))))
    sens80 = float(max(t for t in grid if threshold_metrics(y_rep, oof, t)["sensitivity"] >= 0.8))
    thresholds = {"f1": pp.best_f1_threshold(y_rep, oof), "youden": youden, "sens80": sens80}
    oods = [OOD((x - p.mean) / p.sd) for x, p in zip(x_fit, probes, strict=True)]

    def ensemble(key: str, site: bool = False, rows: np.ndarray | None = None,
                 calib: np.ndarray | None = None) -> np.ndarray:
        out = []
        for s, p in zip(seeds, probes, strict=True):
            x = s[f"{key}_x"] if rows is None else s[f"{key}_x"][rows]
            if site:
                ref = s[f"{key}_x"] if calib is None else s[f"{key}_x"][calib]
                out.append(p.proba(x, ref.mean(0), ref.std(0) + 1e-6))
            else:
                out.append(p.proba(x))
        return np.mean(out, axis=0)

    def ood_rate(key: str, site: bool = False) -> float:
        flags = []
        for s, p, o in zip(seeds, probes, oods, strict=True):
            x = s[f"{key}_x"]
            mean, sd = (x.mean(0), x.std(0) + 1e-6) if site else (p.mean, p.sd)
            flags.append(o.distance((x - mean) / sd) > o.threshold)
        return float(np.mean(np.mean(flags, axis=0) >= 0.5))

    roi = seeds[0]["doc"]["roi"]
    cache = "multi3_s32" if roi == "multi3" else "hippo_s32"
    qc = qc_range(cache)
    validation: dict = {"candidate": chosen, "selection_mean3": score,
                        "cv_rep_out_of_fold": {k: ens[chosen]["cv_rep"][k]
                                               for k in ("auc", "aupr", "f1_macro", "mean3")},
                        "thresholds": thresholds, "qc": qc,
                        "oof": with_ci(y_rep, oof, thresholds),
                        "oof_matched": with_ci(y_rep[:n_lab], oof[:n_lab], thresholds)}
    cv_subjects = protocol[chosen][0]["scores"]["cv_subjects"]
    validation["age_only_oof"] = with_ci(y_rep, np.array([ages[s] for s in cv_subjects]), {})
    test_y, test_subjects = seeds[0]["test_y"], list(seeds[0]["test_subjects"])
    p_test = ensemble("test")
    validation["oasis3_test"] = with_ci(test_y, p_test, thresholds)
    matched = age_matched(test_subjects, test_y, ages)
    validation["oasis3_test_age_matched"] = with_ci(test_y[matched], p_test[matched], thresholds)
    age_p = np.array([ages[s] for s in test_subjects])  # age itself: older = AD
    validation["age_only_test"] = with_ci(test_y, age_p, {})
    validation["age_only_test_age_matched"] = with_ci(test_y[matched], age_p[matched], {})
    validation["ood_rate"] = {"oasis3_test": ood_rate("test")}
    if "adni_x" in seeds[0]:
        adni_y, adni_subjects = seeds[0]["adni_y"], list(seeds[0]["adni_subjects"])
        validation["adni_raw"] = with_ci(adni_y, ensemble("adni"), thresholds)
        validation["adni_site_calibrated"] = with_ci(adni_y, ensemble("adni", site=True),
                                                     thresholds)
        validation["ood_rate"].update(adni_raw=ood_rate("adni"),
                                      adni_site_calibrated=ood_rate("adni", site=True))
        validation["age_only_adni"] = with_ci(
            adni_y, np.array([ages[s] for s in adni_subjects]), {})
        # Input QC: scans outside the OASIS3 intensity range are rejected, not scored.
        adni_idx = {r["subject"]: r["idx"] for r in
                    csv.DictReader((SPLIT / "external_adni.csv").open(encoding="utf-8"))}
        value = cache_intensity(f"adni_{cache}", [adni_idx[s] for s in adni_subjects])
        ok = (value >= qc["lo"]) & (value <= qc["hi"])
        qc["adni_rejected"] = {"AD": int((~ok & (adni_y == 1)).sum()),
                               "HC": int((~ok & (adni_y == 0)).sum())}
        validation["adni_qc"] = with_ci(adni_y[ok], ensemble("adni", rows=ok), thresholds)
        validation["adni_qc_site_calibrated"] = with_ci(
            adni_y[ok], ensemble("adni", site=True, rows=ok, calib=ok), thresholds)
        validation["age_only_adni_qc"] = with_ci(
            adni_y[ok], np.array([ages[s] for s in np.array(adni_subjects)[ok]]), {})
        # Calibration from small unlabeled batches with other AD shares (no labels used).
        rng = np.random.default_rng(0)
        pos, neg = np.flatnonzero(adni_y == 1), np.flatnonzero(adni_y == 0)
        robust = {}
        for n in (30, 60, 120):
            for share in (0.0, 0.3, 0.5):
                aucs = []
                for _ in range(50):
                    k = round(n * share)
                    calib = np.concatenate((rng.choice(pos, k, replace=False),
                                            rng.choice(neg, n - k, replace=False)))
                    aucs.append(fast_auc(adni_y, ensemble("adni", True, calib=calib)))
                robust[f"n{n}_ad{int(share * 100)}"] = [float(np.mean(aucs)),
                                                        float(np.percentile(aucs, 5))]
        validation["adni_calibration_batches"] = robust
    # ---- write the model
    args.out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        shutil.copy2(s["dir"] / "encoder.pt", args.out / f"encoder_s{s['seed']}.pt")
    geometry = json.loads((ROOT / "data/roi3d" / f"{cache}.json").read_text(encoding="utf-8"))
    model = {"name": args.out.name, "candidate": chosen, "roi": roi, "roi_size": 32,
             "cube": {k: geometry[k] for k in ("extract", "cube_starts", "channels")},
             "input": "MNI 2 mm grid (97, 115, 97) as data/dementia/mni-dataset.lmdb; "
                      "t5_ssl.data.oasis3_datamodule.normalize_volume",
             "seeds": [{"seed": s["seed"], "encoder": f"encoder_s{s['seed']}.pt",
                        "probe": p.to_json(),
                        "ood": {"center": o.center.tolist(), "components": o.components.tolist(),
                                "var": o.var.tolist(), "threshold": o.threshold}}
                       for s, p, o in zip(seeds, probes, oods, strict=True)],
             "thresholds": thresholds, "qc": {k: qc[k] for k in ("statistic", "lo", "hi")},
             "decision_threshold": "f1", "training_prevalence": 0.5,
             "protocol": "v2 (src/t5_ssl/evaluation/protocol_v2.py)",
             "oasis3_prevalence": 203 / 880}
    (args.out / "model.json").write_text(json.dumps(model) + "\n", encoding="utf-8")
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1) + "\n",
                                              encoding="utf-8")
    card = model_card(chosen, validation, lines)
    (args.out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(card)


def model_card(chosen: str, v: dict, selection: list[str]) -> str:
    def row(name: str, d: dict) -> str:
        at = d["at"]
        f, s = at.get("f1"), at.get("sens80")
        aupr = (f"{d['aupr']:.3f} [{d['aupr_ci'][0]:.3f}, {d['aupr_ci'][1]:.3f}] "
                f"(azar {d['prevalence']:.2f})")
        return (f"| {name} | {d['n']} ({d['ad']} AD) | {d['auc']:.3f} [{d['auc_ci'][0]:.3f}, "
                f"{d['auc_ci'][1]:.3f}] | {aupr} | "
                + (f"{f['f1_macro']:.3f} | {f['sensitivity']:.2f} / {f['specificity']:.2f} / "
                   f"{f['ppv']:.2f} / {f['npv']:.2f} | " if f else "— | — | ")
                + (f"{s['sensitivity']:.2f} / {s['specificity']:.2f} / {s['ppv']:.2f} |"
                   if s else "— |"))

    t = v["thresholds"]
    lines = [f"# Ficha del modelo: AD vs HC desde MRI ({chosen})", "", *selection,
             "Probe: estandarización + regresión logística balanceada por semilla (C por "
             "log-loss en CV interna), ajustada con todos los sujetos con etiqueta de train+val "
             "(emparejados por edad, 50/50); salida = media de las semillas. Umbrales elegidos "
             f"en la CV representativa fuera de fold: F1-macro máximo {t['f1']:.3f} (decisión por "
             f"defecto), Youden {t['youden']:.3f}, sensibilidad ≥ 0.80 {t['sens80']:.3f}.",
             "", "La fila de la CV representativa usa los umbrales elegidos en esos mismos "
             "datos (F1 optimista); con el umbral elegido fuera de fold, como en la selección: "
             f"AUROC {v['cv_rep_out_of_fold']['auc']:.3f}, AUPR "
             f"{v['cv_rep_out_of_fold']['aupr']:.3f}, F1-macro "
             f"{v['cv_rep_out_of_fold']['f1_macro']:.3f}.",
             "", "| Conjunto | Sujetos | AUROC [IC 95 %] | AUPR [IC 95 %] | F1-macro | "
             "Umbral F1: sens / espec / PPV / NPV | Sens ≥ 0.80: sens / espec / PPV |",
             "|---|---|---|---|---|---|---|",
             row("CV representativa fuera de fold (prevalencia y edades reales)", v["oof"]),
             row("Solo edad, CV representativa", v["age_only_oof"]),
             row("CV emparejada por edad (50/50)", v["oof_matched"]),
             row("Test OASIS3 representativo", v["oasis3_test"]),
             row("Test OASIS3 emparejado por edad", v["oasis3_test_age_matched"]),
             row("Solo edad, test OASIS3", v["age_only_test"]),
             row("Solo edad, test emparejado", v["age_only_test_age_matched"])]
    if "adni_raw" in v:
        rejected = v["qc"]["adni_rejected"]
        lines += [row("ADNI (otro centro), con QC de entrada", v["adni_qc"]),
                  row("ADNI con QC, calibrado por sitio (sin etiquetas)",
                      v["adni_qc_site_calibrated"]),
                  row("Solo edad, ADNI con QC", v["age_only_adni_qc"]),
                  row("ADNI sin QC (todas las MRIs)", v["adni_raw"]),
                  row("ADNI sin QC, calibrado por sitio", v["adni_site_calibrated"]),
                  row("Solo edad, ADNI sin QC", v["age_only_adni"]), "",
                  f"QC de entrada: {v['qc']['statistic']} dentro de [{v['qc']['lo']:.3f}, "
                  f"{v['qc']['hi']:.3f}] (rango de las {v['qc']['oasis3_sessions']} sesiones "
                  f"OASIS3 ± {QC_MARGIN}). En ADNI rechaza {rejected['AD']} AD y "
                  f"{rejected['HC']} HC: MRIs con otra normalización de intensidad (0.85–0.95 "
                  "contra ≤ 0.72 en OASIS3), todas AD; sin QC el modelo las puntúa como "
                  "anatomía y el AUROC de ADNI se hunde.", "",
                  "Calibración de sitio con lotes chicos sin etiquetas (AUROC medio / percentil 5, "
                  "50 lotes): " + ", ".join(f"{k}: {m:.3f} / {p:.3f}"
                                             for k, (m, p) in v["adni_calibration_batches"].items())]
    lines += ["", "Tasa de alertas fuera de distribución (Mahalanobis > p99 de entrenamiento): "
              + ", ".join(f"{k} {r:.1%}" for k, r in v["ood_rate"].items()), "",
              "## Uso y límites", "",
              "- Entrada: una MRI T1 en la grilla MNI de 2 mm del dataset (mismo preprocesamiento "
              "que mni-dataset.lmdb); `scripts/6_deploy/predict_ad.py` recorta los cubos (hipocampo, o 3 "
              "regiones) en posiciones fijas y devuelve P(AD), la decisión en el umbral elegido "
              "y una alerta fuera de distribución.",
              "- QC de entrada obligatorio: una MRI fuera del rango de intensidad de OASIS3 se "
              "rechaza (`qc_rechazada`) y no se puntúa; hay que revisar su preprocesamiento.",
              *(["- Centro nuevo: con el QC, ADNI funciona sin calibrar; la calibración por "
                 "sitio sin etiquetas (`predict_ad.py --calibrate-site`) es opcional y apenas "
                 "cambia el AUROC (ver tabla)."] if "adni_raw" in v else
                ["- Centro nuevo: no validado con este modelo (ADNI no se evaluó en v2; ver "
                 "deploy/ad_mri_v1 para el efecto del QC y la calibración por sitio)."]),
              "- Entrenado solo con OASIS3 (AD y HC 'puros'); no distingue AD de otras demencias "
              "(LBD, FTD, vascular), que nunca vio con etiqueta.",
              "- Las probabilidades suponen prevalencia 50/50 (probe balanceado); el umbral F1 "
              "está elegido para la prevalencia de OASIS3 (23 % AD); el PPV real depende de la "
              "prevalencia del lugar donde se use (ver tabla).",
              "- Validación retrospectiva en dos cohortes de investigación; no es un dispositivo "
              "médico validado ni reemplaza el diagnóstico clínico.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
