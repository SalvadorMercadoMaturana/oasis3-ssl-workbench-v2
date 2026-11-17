#!/usr/bin/env python
"""Measure real training throughput: 100 optimizer steps per model and runtime config.

    python cloud/bench.py [--models byol vicreg dinov2] [--steps 100] [--warmup 15]
                    [--configs baseline fast-eager fast-compile]

Each (model, config) runs ``train.py --bench-steps`` in a fresh ``spawn``
process, on the real data pipeline (CPU augmentations included), one at a
time so the numbers do not interfere. After ``--warmup`` steps (compilation,
cuDNN autotuning, worker start-up) it times ``--steps`` steps between two CUDA
synchronizations and reports images/s (batch x views), ms/step, the host-side
data-wait share and peak VRAM. Configs:

- ``baseline``: the campaigns' runtime (fp32, no DataLoader workers, no perf);
- ``fast-eager``: bf16 autocast, TF32, fused AdamW, channels_last_3d, workers;
- ``fast-compile``: fast-eager + torch.compile (with fallback);
- ``fast-nocl``: fast-compile without channels_last_3d.

Results: ``<out>/bench.json`` and ``<out>/BENCH.md`` with the fastest config
per model.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repository (this file is in cloud/)
sys.path.insert(0, str(ROOT / "src"))
from t5_ssl import recipes  # noqa: E402

CONFIGS = {
    "baseline": ["--no-fast"],
    "fast-eager": ["--compile", "off"],
    "fast-compile": ["--compile", "on"],
    "fast-nocl": ["--compile", "on", "--channels-last", "off"],
}


def _child(argv: list[str], log_path: str) -> None:
    log = open(log_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sys.path.insert(0, str(ROOT))
    import train as entry
    sys.exit(entry.main(argv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(recipes.MODELS), choices=recipes.MODELS)
    parser.add_argument("--configs", nargs="+", default=["baseline", "fast-eager", "fast-compile"],
                        choices=sorted(CONFIGS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--byol-variant", default="campaign")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    out = Path(args.out or f"bench_results/{time.strftime('%Y%m%d_%H%M%S')}")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    results = []
    for model in args.models:
        variant = args.byol_variant if model == "byol" else "campaign"
        for config in args.configs:
            name = f"{model}_{config}"
            target = out / f"{name}.json"
            argv_ = ["--model", model, "--variant", variant, "--seed", "1",
                     "--bench-steps", str(args.steps), "--bench-warmup", str(args.warmup),
                     "--bench-out", str(target), "--batch-size", str(args.batch_size),
                     "--workers", str(args.workers), "--out-root", str(out / "runs"),
                     "--no-eval", *CONFIGS[config]]
            start = time.time()
            process = ctx.Process(target=_child, args=(argv_, str(out / "logs" / f"{name}.log")))
            process.start()
            process.join()
            row = {"model": model, "config": config, "exit_code": process.exitcode,
                   "wall_seconds": time.time() - start}
            if target.is_file():
                row.update(json.loads(target.read_text(encoding="utf-8")))
            results.append(row)
            print(f"{name:28s} exit={process.exitcode} "
                  + (f"{row['images_per_second']:7.0f} img/s {row['ms_per_step']:6.1f} ms/step "
                     f"data-wait {row['data_wait_fraction']:.0%} "
                     f"VRAM {row.get('peak_vram_reserved_gb', 0):.2f} GB "
                     f"(wall {row['wall_seconds']:.0f} s)" if "images_per_second" in row
                     else "no result (see log)"), flush=True)
    (out / "bench.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [f"# bench: {args.steps} pasos tras {args.warmup} de calentamiento, batch "
             f"{args.batch_size}", "",
             "| Modelo | Config | img/s | ms/paso | Espera de datos | VRAM pico (GB) | Reloj (s) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in results:
        if "images_per_second" in row:
            lines.append(f"| {row['model']} | {row['config']} | {row['images_per_second']:.0f} | "
                         f"{row['ms_per_step']:.1f} | {row['data_wait_fraction']:.0%} | "
                         f"{row.get('peak_vram_reserved_gb', 0):.2f} | {row['wall_seconds']:.0f} |")
        else:
            lines.append(f"| {row['model']} | {row['config']} | falló (exit {row['exit_code']}) "
                         "| | | | |")
    lines += ["", "## Más rápida por modelo", ""]
    for model in args.models:
        ok = [r for r in results if r["model"] == model and "images_per_second" in r]
        if ok:
            best = max(ok, key=lambda r: r["images_per_second"])
            base = next((r for r in ok if r["config"] == "baseline"), None)
            gain = (f" ({best['images_per_second'] / base['images_per_second']:.1f}x baseline)"
                    if base else "")
            lines.append(f"- {model}: `{best['config']}` {best['images_per_second']:.0f} img/s{gain}")
    (out / "BENCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'BENCH.md'}")
    return 0 if all("images_per_second" in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
