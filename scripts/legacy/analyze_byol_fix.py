"""Identity and augmentation-sensitivity analyses on the BYOL-fix screen (experiment 2).

Repeats the diagnosis of ``analyze_ssl3d_encoders.py`` and
``analyze_byol_sensitivity.py`` (docs/informe_ssl3d_v2.pdf, sec. 8.2-8.3) on the
encoders that ``cloud/run_all.py --preset byol-screen`` trained with
``--export-encoders``: for the 142 held-out subjects (val + test), per run and
encoder (online backbone, and BYOL's EMA target),

- identity under the campaign views with zero-padded and with border-padded
  rotation: centered cosine same vs different subject, top-1 view matching,
  between-subject / within-subject (augmentation) variance;
- effective rank of the clean embeddings and linear CKA to the ImageNet start;
- sensitivity to each augmentation component alone (within / between variance).

Usage (on the GPU host): ``python scripts/legacy/analyze_byol_fix.py runs/screen30``.
Writes ``<dir>/analysis.json`` and ``<dir>/ANALYSIS.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "src")]
import analyze_byol_sensitivity as sens  # noqa: E402
import analyze_ssl3d_encoders as enc  # noqa: E402

from t5_ssl import recipes  # noqa: E402

ROTATION_ZERO = "rotación ±10° (relleno con ceros)"
ROTATION_BORDER = "rotación ±10° (relleno con borde)"
FULL_ZERO = "vistas completas de la campaña"
FULL_BORDER = "vistas completas (relleno con borde)"


def compose(model: str, variant: str = "campaign"):
    import train as entry
    return entry.compose(recipes.overrides(model, variant, fast=False))


def main() -> None:
    from omegaconf import OmegaConf

    from t5_ssl.data.roi3d import ROIEval3D, ROIView3DTransform
    from t5_ssl.train import build_backbone
    run_dir = Path(sys.argv[1])
    base = compose("byol")
    cubes, _ = enc.held_out_cubes(base)
    clean_inputs = [ROIEval3D(32)(c) for c in cubes]
    bias = [{**spec, "p": 1.0} for spec in OmegaConf.to_container(base.augmentation.torchio,
                                                                  resolve=True)]
    components = sens.components(bias)
    components[FULL_BORDER] = ROIView3DTransform(size=32, side_range=(0.84, 1.0), degrees=10.0,
                                                 lr_swap_p=0.5, torchio_transforms=bias,
                                                 rotation_padding="border")
    draws = {name: enc.views(t, cubes, seed=11) for name, t in components.items()}
    campaign_views = {"ceros": draws[FULL_ZERO], "borde": draws[FULL_BORDER]}
    results: dict = {"subjects": len(cubes), "chance": 1 / len(cubes), "runs": {}}

    initial = {}
    for model in ("byol", "vicreg"):
        torch.manual_seed(0)
        encoder = build_backbone(compose(model))
        clean = enc.embed(encoder, clean_inputs)
        initial[model] = clean
        results["runs"][f"{model}_init"] = {"encoder": {
            "identity": {k: enc.scores(clean, *(enc.embed(encoder, v) for v in pair))
                         for k, pair in campaign_views.items()},
            "sensitivity": sens.measure(encoder, clean_inputs, draws)}}
        print(model, "init done", flush=True)

    for path in sorted(run_dir.glob("*/encoders.pt")):
        name = path.parent.name
        model = name.split("_", 1)[0]
        doc = torch.load(path, map_location="cpu", weights_only=False)
        cfg = compose(model, name.split("_", 1)[1].rsplit("_s", 1)[0])
        entry = {}
        for role in ("backbone", "target_backbone"):
            if role not in doc["state"]:
                continue
            encoder = build_backbone(cfg)
            encoder.load_state_dict(doc["state"][role])
            clean = enc.embed(encoder, clean_inputs)
            entry["online" if role == "backbone" else "target"] = {
                "identity": {k: enc.scores(clean, *(enc.embed(encoder, v) for v in pair))
                             for k, pair in campaign_views.items()},
                "cka_to_initial": enc.linear_cka(initial[model], clean),
                "sensitivity": sens.measure(encoder, clean_inputs, draws)}
        results["runs"][name] = entry
        print(name, {r: (round(e["identity"]["ceros"]["between_over_within"], 3),
                         round(e["sensitivity"][FULL_ZERO]["sensitivity"], 2))
                     for r, e in entry.items()}, flush=True)
        (run_dir / "analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)
                                               + "\n", encoding="utf-8")
    (run_dir / "analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)
                                           + "\n", encoding="utf-8")
    write_markdown(run_dir, results)


def write_markdown(run_dir: Path, results: dict) -> None:
    lines = ["# Identidad y sensibilidad tras el arreglo de BYOL", "",
             f"{results['subjects']} sujetos no vistos (val + test); azar top-1 = "
             f"{results['chance']:.3f}. Vistas de la campaña con relleno de ceros / de borde.", "",
             "| Corrida | Encoder | cos dif. (ceros) | top-1 (ceros) | entre/dentro (ceros) | "
             "entre/dentro (borde) | rango ef. | CKA | sens. rot. ceros | sens. rot. borde | "
             "sens. vistas ceros | sens. vistas borde |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in results["runs"].items():
        for role, e in entry.items():
            z, b, s = e["identity"]["ceros"], e["identity"]["borde"], e["sensitivity"]
            cka = e.get("cka_to_initial")
            lines.append(
                f"| {name} | {role} | {z['different_subject_cos']:.3f} | "
                f"{z['view_matching_top1']:.3f} | {z['between_over_within']:.3g} | "
                f"{b['between_over_within']:.3g} | {z['effective_rank']:.1f} | "
                + (f"{cka:.2f}" if cka is not None else "—")
                + f" | {s[ROTATION_ZERO]['sensitivity']:.3g} | {s[ROTATION_BORDER]['sensitivity']:.3g}"
                f" | {s[FULL_ZERO]['sensitivity']:.3g} | {s[FULL_BORDER]['sensitivity']:.3g} |")
    lines += ["", "*sens.* = varianza dentro del sujeto entre dos sorteos del componente / varianza "
              "entre sujetos del cubo limpio (BYOL de la campaña: ~6 700 en las vistas completas). "
              "*entre/dentro* = varianza entre sujetos del cubo limpio / varianza que mete la "
              "augmentation (VICReg de la campaña: 3.83).", ""]
    (run_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_dir / 'ANALYSIS.md'}", flush=True)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
