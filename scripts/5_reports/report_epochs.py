#!/usr/bin/env python
"""Report of the 200-epoch runs (2026-11-23): the four methods trained four times longer.

    python scripts/5_reports/report_epochs.py

Runs (``campaign_ssl3d_runpod/epochs200/runs/ep200_f0``, from
``scripts/3_train/jobs_ingest.py ssl --backbones convnext_tiny --tag ep200 --epochs 200
--extra "--probe-every 50"``): the four methods with their tuned recipes on the current 3D
ingestion (ConvNeXt-T ImageNet-1k inflated, hippocampal 32-voxel cubes), fold 0 of
split_deploy, seed 1. Each run is scored at epochs 50, 100 and 150
(``probe_epoch<NNN>.json``, t5_ssl.callbacks.periodic_probe) and 200 (``result.json``).

Reference: the 50-epoch runs of the same recipes and seed (``ingest/runs/ing_f0``,
``convnext_tiny``), whose schedule (warmup = epochs / 5, cosine decay) ends at epoch 50;
at epoch 50 of a 200-epoch run the learning rate is still near its peak.

Protocol v2: representative CV (702 subjects) and the representative test; mean3 =
(AUROC + AUPR + F1-macro) / 3; Δ = paired bootstrap (1000 draws) against the reference,
same subjects. Writes ``campaign_ssl3d_runpod/epochs200/REPORT.md`` and ``summary.json``.
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
RUNS = CAMPAIGN / "epochs200/runs/ep200_f0"
OUT = CAMPAIGN / "epochs200"
BACKBONE = "convnext_tiny"


def by_epoch(run: Path) -> dict[int, dict]:
    """Protocol-v2 documents of one run by training epoch (periodic probes + the end)."""
    docs = {}
    for path in sorted(run.glob("probe_epoch*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[int(doc["epoch"])] = doc["protocol_v2"]
    result = run / "result.json"
    if result.is_file():
        doc = json.loads(result.read_text(encoding="utf-8").splitlines()[0])
        docs[int(doc["epochs"])] = doc["protocol_v2"]
    return {epoch: pp.ensemble([doc], r5.AGES[0]) for epoch, doc in docs.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    found: dict[str, dict[int, dict]] = {}
    refs: dict[str, dict] = {}
    for key, (_, base, _) in ri.METHODS.items():
        runs = sorted(RUNS.glob(f"{ri.run_name(base, BACKBONE)}_s*"))
        if runs:
            found[key] = by_epoch(runs[0])
        ref = ri.ens("ing_f0", ri.run_name(base, BACKBONE))
        if ref is not None:
            refs[key] = ref
    epochs = sorted({e for docs in found.values() for e in docs})
    lines = ["# 200 épocas: los cuatro métodos entrenados más tiempo (pliegue 0, semilla 1)", "",
             "Protocolo v2. Referencia: la corrida de 50 épocas de la misma receta (su "
             "calendario de lr termina en la época 50). comp. = (AUROC + AUPR + F1-macro) / 3. "
             "Δ = bootstrap pareado contra la referencia (IC 95 %; * = excluye 0).", "",
             "## Resumen: comp. CV / test por época", "",
             "| Método | 50 ép. (ref.) | " + " | ".join(f"ép. {e} de 200" for e in epochs) + " |",
             "|---|---|" + "---|" * len(epochs)]
    summary: dict = {"reference": {}, "runs": {}}
    for key, docs in found.items():
        ref = refs.get(key)
        cells = [f"{ref['cv_rep']['mean3']:.3f} / {ref['test']['mean3']:.3f}" if ref else "—"]
        cells += [f"{docs[e]['cv_rep']['mean3']:.3f} / {docs[e]['test']['mean3']:.3f}"
                  if e in docs else "—" for e in epochs]
        lines.append(f"| {ri.METHODS[key][0]} | " + " | ".join(cells) + " |")
        summary["runs"][key] = {e: {"cv": d["cv_rep"], "test": d["test"]} for e, d in docs.items()}
        if ref:
            summary["reference"][key] = {"cv": ref["cv_rep"], "test": ref["test"]}
    means = []
    for e in epochs:
        if all(e in docs for docs in found.values()) and len(found) == len(ri.METHODS):
            means.append(f"{np.mean([d[e]['cv_rep']['mean3'] for d in found.values()]):.3f} / "
                         f"{np.mean([d[e]['test']['mean3'] for d in found.values()]):.3f}")
        else:
            means.append("—")
    ref_mean = (f"{np.mean([r['cv_rep']['mean3'] for r in refs.values()]):.3f} / "
                f"{np.mean([r['test']['mean3'] for r in refs.values()]):.3f}"
                if len(refs) == len(ri.METHODS) else "—")
    lines.append(f"| **media de los 4** | **{ref_mean}** | "
                 + " | ".join(f"**{m}**" for m in means) + " |")
    lines += ["", "## Detalle por método", "",
              "| Método | Épocas | CV AUROC | CV AUPR | CV F1 | **CV comp.** | Test AUROC | "
              "Test AUPR | Test F1 | **Test comp.** | Δ CV comp. | Δ test comp. |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for key, docs in found.items():
        ref = refs.get(key)
        if ref:
            lines.append(f"| {ri.METHODS[key][0]} | 50 (ref.) | {ri.cell(ref, 'cv_rep')} | "
                         f"{ri.cell(ref, 'test')} | — | — |")
        for e in sorted(docs):
            dc = fmt_delta(ri.cv_delta(docs[e], ref)["mean3"]) if ref else "—"
            dt = fmt_delta(ri.test_delta(docs[e], ref)["mean3"]) if ref else "—"
            lines.append(f"| {ri.METHODS[key][0]} | {e} de 200 | {ri.cell(docs[e], 'cv_rep')} | "
                         f"{ri.cell(docs[e], 'test')} | {dc} | {dt} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n",
                                      encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
