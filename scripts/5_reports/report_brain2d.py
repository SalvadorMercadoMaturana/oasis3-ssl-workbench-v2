#!/usr/bin/env python
"""Report of the whole-brain multiplanar 2D ingestion (2026-11-22, t5_ssl.data.brain2d).

    python scripts/5_reports/report_brain2d.py

What is ingested:

- Input: the axial, coronal and sagittal plane through the centre of mass of each brain.
- Preprocessing: tissue box cropped and z-scored over the tissue voxels; each plane is
  ``224 x 224``, repeated on the RGB inputs of a 2D ImageNet ConvNeXt-T.
- SSL views: the planes are shifted by up to +-5 slices.

Runs come from ``campaign_ssl3d_runpod/brain2d/runs``:

- tags ``tri_init_f<k>`` (frozen start, sin SSL);
- tags ``tri_f<k>`` and variants ``tri_<v>_f<k>`` (the four methods with their tuned
  recipes: VICReg campaign_augval geo, BYOL wide_dproj0.1 noise, DINOv2 b64_k256
  elastic_tio_gpu, rotation pretext with the 4 in-plane turns, rot4_h2048 geo).

These are compared on fold 0 (split_deploy, seed 1) with the ingestions of
campaign_ssl3d_runpod/ingest:

- the current 3D hippocampal cubes (``convnext_tiny``);
- the best hippocampal 2D slices (``cnx2d_cor_x3_s2``).

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
from report_v2fix import fmt_delta  # noqa: E402

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


def ens(root: Path, tag: str, name: str) -> dict | None:
    return ri.ens(tag, name, 0, root)


def frozen_rows() -> dict[str, dict]:
    rows = {}
    for p in sorted((RUNS / "tri_init_f0").glob("vicreg_init__bb*__roibrain2d_s*")):
        bb = p.name.split("__bb", 1)[1].split("__roi", 1)[0]
        e = ens(RUNS, "tri_init_f0", f"vicreg_init__bb{bb}__roibrain2d")
        if e is not None:
            rows[f"{bb} (multiplano)"] = e
    for bb, label in REFERENCES.items():
        e = ens(ri.RUNS, "ing_init_f0", f"vicreg_init__bb{bb}__roi32")
        if e is not None:
            rows[f"{bb} ({label})"] = e
    return rows


def ssl_rows() -> dict[str, dict[str, dict]]:
    """``{ingestion: {method: ensemble}}`` of the brain2d runs and the references."""
    found: dict[str, dict[str, dict]] = {}
    tags = [("tri_f0", "")] + [(p.name, " " + p.name[4:].rsplit("_f", 1)[0])
                               for p in sorted(RUNS.glob("tri_*_f0"))
                               if p.name not in ("tri_init_f0",)]
    for tag, suffix in tags:
        for key, (_, base) in METHODS.items():
            for p in (RUNS / tag).glob(f"{base}__bb*__roibrain2d_s*"):
                bb = p.name.split("__bb", 1)[1].split("__roi", 1)[0]
                e = ens(RUNS, tag, f"{base}__bb{bb}__roibrain2d")
                if e is not None:
                    found.setdefault(f"{bb}{suffix} (multiplano)", {})[key] = e
    for bb, label in REFERENCES.items():
        for key, (_, base, _) in ri.METHODS.items():
            e = ens(ri.RUNS, "ing_f0", ri.run_name(base, bb))
            if e is not None:
                found.setdefault(f"{bb} ({label})", {})[key] = e
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ref_name = f"convnext_tiny ({REFERENCES['convnext_tiny']})"
    lines = ["# Ingesta 2D multiplano (cerebro completo) — pliegue 0, semilla 1", "",
             "Protocolo v2. CV representativa (702 sujetos) y test del split_deploy. "
             "comp. = (AUROC + AUPR + F1-macro) / 3. Δ = bootstrap pareado contra la "
             "ingesta 3D actual (IC 95 %; * = excluye 0).", "",
             "## 1. Encoder congelado (sin SSL)", "",
             "| Ingesta | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | Test AUPR | "
             "Test F1 | **Test comp.** | Δ CV comp. | Δ test comp. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    frozen = frozen_rows()
    ref = frozen.get(ref_name)
    summary: dict = {"frozen": {}, "ssl": {}}
    for name, e in sorted(frozen.items(), key=lambda kv: -kv[1]["test"]["mean3"]):
        dc = dt = "—"
        if ref is not None and name != ref_name:
            dc = fmt_delta(ri.cv_delta(e, ref)["mean3"])
            dt = fmt_delta(ri.test_delta(e, ref)["mean3"])
        lines.append(f"| `{name}` | {ri.cell(e, 'cv_rep')} | {ri.cell(e, 'test')} | {dc} | {dt} |")
        summary["frozen"][name] = {"cv": e["cv_rep"], "test": e["test"]}
    found = ssl_rows()
    refs = found.get(ref_name, {})
    lines += ["", "## 2. Los cuatro métodos SSL", "",
              "| Ingesta | " + " | ".join(f"{METHODS[k][0]} CV / test" for k in METHODS)
              + " | media CV comp. | **media test comp.** |",
              "|---|" + "---|" * len(METHODS) + "---:|---:|"]
    order = sorted(found, key=lambda n: -np.mean([e["test"]["mean3"] for e in found[n].values()]))
    for name in order:
        cells = [f"{found[name][k]['cv_rep']['mean3']:.3f} / {found[name][k]['test']['mean3']:.3f}"
                 if k in found[name] else "—" for k in METHODS]
        cv = np.mean([e["cv_rep"]["mean3"] for e in found[name].values()])
        te = np.mean([e["test"]["mean3"] for e in found[name].values()])
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {cv:.3f} | **{te:.3f}** |")
        summary["ssl"][name] = {k: {"cv": e["cv_rep"], "test": e["test"], "minutes": e["minutes"]}
                                for k, e in found[name].items()}
    lines += ["", "### Detalle por método", "",
              "| Método | Ingesta | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
              "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. | Δ test comp. | min |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for key in METHODS:
        for name in order:
            e = found[name].get(key)
            if e is None:
                continue
            dc = dt = "—"
            if name != ref_name and key in refs:
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
