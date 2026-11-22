#!/usr/bin/env python
"""Report of the whole-brain multiplanar 2D ingestion (2026-11-22, t5_ssl.data.brain2d).

    python scripts/5_reports/report_brain2d.py

What is ingested:

- Input: the axial, coronal and sagittal plane through the centre of mass of each brain.
- Preprocessing: tissue box cropped and z-scored over the tissue voxels; each plane is
  ``224 x 224``, repeated on the RGB inputs of a 2D ImageNet ConvNeXt-T.
- SSL views: the planes are shifted by up to +-5 slices.

Runs come from ``campaign_ssl3d_runpod/brain2d/runs``:

- tags ``tri_init*_f0`` (frozen start, sin SSL);
- tags ``tri_f0`` / ``tri_<v>_f0`` (the four methods with their tuned recipes: VICReg
  campaign_augval geo, BYOL wide_dproj0.1 noise, DINOv2 b64_k256 elastic_tio_gpu,
  rotation pretext with the 4 in-plane turns, rot4_h2048 geo; ``tri_k3``: three plane
  triples per SSL view).

Each run is scored with two probe inputs:

- the centre-of-mass planes (``centro``);
- the multi-slice probe of ``++eval_slice_offsets`` or ``reeval_brain2d.py``
  (``k cortes ±m``: embeddings averaged over k slices up to m from the centre).

These are compared on fold 0 (split_deploy, seed 1) with the ingestions of
campaign_ssl3d_runpod/ingest:

- the current 3D hippocampal cubes (``convnext_tiny``);
- the hippocampal 2D slices (``cnx2d_cor_x3_s2``).

Writes ``campaign_ssl3d_runpod/brain2d/REPORT.md`` and ``summary.json``. Protocol v2:
representative CV of 702 subjects; mean3 = (AUROC + AUPR + F1-macro) / 3; Δ = paired
bootstrap (1000 draws) against the current 3D ingestion, the same subjects and thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_ingest as ri  # noqa: E402
import report_outer5 as r5  # noqa: E402
from report_v2fix import fmt_delta  # noqa: E402

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402

CAMPAIGN = ri.CAMPAIGN
RUNS = CAMPAIGN / "brain2d/runs"
OUT = CAMPAIGN / "brain2d"
# key: (label, run name of the brain2d runs without backbone/roi/seed)
METHODS = {
    "vicreg": ("VICReg", "vicreg_campaign_augval__geo"),
    "byol": ("BYOL", "byol_wide_dproj0.1__noise"),
    "dino": ("DINOv2", "dinov2_b64_k256__elastic_tio_gpu"),
    "pretext": ("Rotación", "pretext_rot4_h2048__geo"),
}
REFERENCES = {"convnext_tiny": "3D cubos hipocampo (actual)",
              "cnx2d_cor_x3_s2": "2D cortes coronales hipocampo"}
REF_NAME = f"convnext_tiny ({REFERENCES['convnext_tiny']})"


def probe_label(offsets: list[int]) -> str:
    return f"{len(offsets)} cortes ±{max(abs(o) for o in offsets)}"


def probes(run: Path) -> dict[str, dict]:
    """Protocol-v2 documents of one run by probe input: ``centro`` and multi-slice ones."""
    result = json.loads((run / "result.json").read_text(encoding="utf-8").splitlines()[0])
    out = {"centro": result["protocol_v2"]}
    extra = [result["multislice"]] if result.get("multislice") else []
    extra += [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(run.glob("multislice_*.json"))]
    for doc in extra:
        out[probe_label(doc["offsets"])] = doc["protocol_v2"]
    for doc in out.values():
        doc["seed"], doc["minutes"] = result.get("seed"), result.get("minutes")
    return out


def ens_runs(root: Path, tag: str, name: str) -> dict[str, dict]:
    """Seed ensembles of ``<root>/<tag>/<name>_s*`` for each probe input they share."""
    runs = [r for r in sorted((root / tag).glob(f"{name}_s*")) if (r / "result.json").is_file()]
    if not runs:
        return {}
    per_run = [probes(r) for r in runs]
    shared = set.intersection(*(set(p) for p in per_run))
    out = {}
    for label in sorted(shared, key=lambda s: (s != "centro", s)):
        docs = [p[label] for p in per_run]
        e = pp.ensemble(docs, r5.AGES[0])
        e["n_seeds"], e["minutes"] = len(docs), [d["minutes"] for d in docs]
        out[label] = e
    return out


def backbone_of(dirname: str) -> str:
    return dirname.split("__bb", 1)[1].split("__roi", 1)[0]


def frozen_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for tag in sorted(p.name for p in RUNS.glob("tri_init*_f0")):
        for p in sorted((RUNS / tag).glob("vicreg_init__bb*__roibrain2d_s*")):
            bb = backbone_of(p.name)
            for label, e in ens_runs(RUNS, tag, f"vicreg_init__bb{bb}__roibrain2d").items():
                rows.setdefault(f"{bb} multiplano, {label}", e)
    for bb, label in REFERENCES.items():
        e = ri.ens("ing_init_f0", f"vicreg_init__bb{bb}__roi32")
        if e is not None:
            rows[f"{bb} ({label})"] = e
    return rows


def ssl_rows() -> dict[str, dict[str, dict]]:
    """``{ingestion + probe: {method: ensemble}}`` of the brain2d runs and the references."""
    found: dict[str, dict[str, dict]] = {}
    for tag in sorted(p.name for p in RUNS.glob("tri_*f0") if not p.name.startswith("tri_init")):
        variant = tag[len("tri_"):-len("_f0")] if tag != "tri_f0" else ""
        for key, (_, base) in METHODS.items():
            for p in (RUNS / tag).glob(f"{base}__bb*__roibrain2d_s*"):
                bb = backbone_of(p.name)
                for label, e in ens_runs(RUNS, tag, f"{base}__bb{bb}__roibrain2d").items():
                    name = f"{bb}{' ' + variant if variant else ''} multiplano, {label}"
                    found.setdefault(name, {})[key] = e
    for bb, label in REFERENCES.items():
        for key, (_, base, _) in ri.METHODS.items():
            e = ri.ens("ing_f0", ri.run_name(base, bb))
            if e is not None:
                found.setdefault(f"{bb} ({label})", {})[key] = e
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lines = ["# Ingesta 2D multiplano (cerebro completo) — pliegue 0, semilla 1", "",
             "Protocolo v2. CV representativa (702 sujetos) y test del split_deploy. "
             "comp. = (AUROC + AUPR + F1-macro) / 3. Probe `centro`: los 3 planos por el "
             "centro de masa; `k cortes ±m`: embeddings promediados sobre k cortes por plano "
             "hasta m del centro. Δ = bootstrap pareado contra la ingesta 3D actual "
             "(IC 95 %; * = excluye 0).", "",
             "## 1. Encoder congelado (sin SSL)", "",
             "| Ingesta, probe | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
             "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. | Δ test comp. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    frozen = frozen_rows()
    ref = frozen.get(REF_NAME)
    summary: dict = {"frozen": {}, "ssl": {}}
    for name, e in sorted(frozen.items(), key=lambda kv: -kv[1]["cv_rep"]["mean3"]):
        dc = dt = "—"
        if ref is not None and name != REF_NAME:
            dc = fmt_delta(ri.cv_delta(e, ref)["mean3"])
            dt = fmt_delta(ri.test_delta(e, ref)["mean3"])
        lines.append(f"| `{name}` | {ri.cell(e, 'cv_rep')} | {ri.cell(e, 'test')} | {dc} | {dt} |")
        summary["frozen"][name] = {"cv": e["cv_rep"], "test": e["test"]}
    found = ssl_rows()
    refs = found.get(REF_NAME, {})
    lines += ["", "## 2. Los cuatro métodos SSL", "",
              "| Ingesta, probe | " + " | ".join(f"{METHODS[k][0]} CV / test" for k in METHODS)
              + " | media CV comp. | **media test comp.** |",
              "|---|" + "---|" * len(METHODS) + "---:|---:|"]
    order = sorted(found, key=lambda n: -np.mean([e["cv_rep"]["mean3"]
                                                  for e in found[n].values()]))
    for name in order:
        cells = [f"{found[name][k]['cv_rep']['mean3']:.3f} / {found[name][k]['test']['mean3']:.3f}"
                 if k in found[name] else "—" for k in METHODS]
        cv = np.mean([e["cv_rep"]["mean3"] for e in found[name].values()])
        te = np.mean([e["test"]["mean3"] for e in found[name].values()])
        n = len(found[name])
        lines.append(f"| `{name}` | " + " | ".join(cells)
                     + f" | {cv:.3f}{'' if n == 4 else f' ({n})'} | **{te:.3f}** |")
        summary["ssl"][name] = {k: {"cv": e["cv_rep"], "test": e["test"], "minutes": e["minutes"]}
                                for k, e in found[name].items()}
    lines += ["", "### Detalle por método", "",
              "| Método | Ingesta, probe | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
              "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. | Δ test comp. | min |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for key in METHODS:
        for name in order:
            e = found[name].get(key)
            if e is None:
                continue
            dc = dt = "—"
            if name != REF_NAME and key in refs:
                dc = fmt_delta(ri.cv_delta(e, refs[key])["mean3"])
                dt = fmt_delta(ri.test_delta(e, refs[key])["mean3"])
            minutes = "/".join(f"{m:.0f}" for m in e["minutes"] if m is not None) or "—"
            lines.append(f"| {METHODS[key][0]} | `{name}` | {ri.cell(e, 'cv_rep')} | "
                         f"{ri.cell(e, 'test')} | {dc} | {dt} | {minutes} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n",
                                      encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
