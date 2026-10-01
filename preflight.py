#!/usr/bin/env python
"""Check the GPU host before training: GPU, VRAM, versions, data and a tiny compile.

    python preflight.py [--min-vram-gb 20] [--require-data] [--skip-compile] [--json out.json]

Exit code 0 when every critical check passes; 1 otherwise (each failure is
printed). Warnings (no MPS binary, few CPUs, no Triton) do not fail.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILES = ["data/roi3d/hippo_s32.npy", "data/roi3d/hippo_s32.json",
              "data/dementia/mni-dataset.csv",
              "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"]


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("+")[0].split(".")[:2] if part.isdigit())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-vram-gb", type=float, default=20.0)
    parser.add_argument("--min-torch", default="2.4")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    report: dict[str, object] = {}
    failures: list[str] = []
    warnings: list[str] = []

    def check(name: str, ok: bool, detail: object, critical: bool = True) -> None:
        report[name] = detail
        mark = "OK  " if ok else ("FAIL" if critical else "WARN")
        print(f"[{mark}] {name}: {detail}", flush=True)
        if not ok:
            (failures if critical else warnings).append(name)

    check("python", sys.version_info >= (3, 10), platform.python_version())
    try:
        import torch
    except ImportError as exc:
        check("torch", False, str(exc))
        return 1
    check("torch", version_tuple(torch.__version__) >= version_tuple(args.min_torch),
          torch.__version__)
    cuda_version = torch.version.cuda or "none"
    check("cuda_runtime", cuda_version.startswith("12"), cuda_version)
    check("cuda_available", torch.cuda.is_available(), torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 2**30
        check("gpu", True, f"{props.name} (sm_{props.major}{props.minor}, "
              f"{props.multi_processor_count} SMs)")
        check("vram_gb", vram >= args.min_vram_gb, round(vram, 1))
        check("bf16_native", props.major >= 8, props.major >= 8)
        check("cudnn", torch.backends.cudnn.is_available(), torch.backends.cudnn.version())
        free, total = torch.cuda.mem_get_info()
        check("vram_free_gb", free / total > 0.8, round(free / 2**30, 1), critical=False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        a = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):  # cuBLAS initialization and heuristics, outside the timing
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a @ a
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a @ a
        torch.cuda.synchronize()
        tflops = 10 * 2 * 4096**3 / (time.perf_counter() - start) / 1e12
        check("bf16_matmul_tflops", tflops > 20, round(tflops, 1), critical=False)
        x = torch.randn(16, 2, 32, 32, 32, device="cuda").contiguous(
            memory_format=torch.channels_last_3d)
        conv = torch.nn.Conv3d(2, 64, 3, padding=1).cuda().to(memory_format=torch.channels_last_3d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = conv(x)
        torch.cuda.synchronize()
        check("conv3d_channels_last_bf16", y.dtype == torch.bfloat16, str(tuple(y.shape)))
        del a, x, y, conv
        torch.cuda.empty_cache()
    if not args.skip_compile:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            fn = torch.compile(lambda t: torch.nn.functional.gelu(t) * 2 + 1)
            out = fn(torch.randn(64, 64, device=device))
            check("torch_compile", bool(torch.isfinite(out).all()), "inductor ok", critical=False)
        except Exception as exc:
            check("torch_compile", False, f"{type(exc).__name__}: {str(exc)[:200]} "
                  "(train.py falls back to eager)", critical=False)
    mps = shutil.which("nvidia-cuda-mps-control")
    check("cuda_mps_control", mps is not None, mps or "missing (run_all falls back to "
          "all-at-once)", critical=False)
    for module in ("lightning", "hydra", "omegaconf", "torchio", "sklearn", "scipy",
                   "matplotlib", "t5_ssl"):
        try:
            mod = importlib.import_module(module)
            check(f"import_{module}", True, getattr(mod, "__version__", "ok"))
        except Exception as exc:
            check(f"import_{module}", False, f"{type(exc).__name__}: {exc}")
    from run_all import effective_cpus
    cpus = effective_cpus()
    check("cpus", cpus >= 8, cpus, critical=False)
    with open("/proc/meminfo", encoding="utf-8") if Path("/proc/meminfo").exists() else \
            open(os.devnull, encoding="utf-8") as stream:
        meminfo = {line.split(":")[0]: line.split()[1] for line in stream if ":" in line}
    if "MemAvailable" in meminfo:
        ram = int(meminfo["MemAvailable"]) / 2**20
        check("ram_available_gb", ram >= 16, round(ram, 1), critical=False)
    disk = shutil.disk_usage(ROOT)
    check("disk_free_gb", disk.free / 2**30 >= 10, round(disk.free / 2**30, 1))
    for name in DATA_FILES:
        present = (ROOT / name).is_file()
        check(f"data:{name}", present or not args.require_data,
              f"{(ROOT / name).stat().st_size / 2**20:.1f} MB" if present else "missing",
              critical=args.require_data)
    env = {key: os.environ.get(key) for key in (
        "PYTORCH_CUDA_ALLOC_CONF", "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "OMP_NUM_THREADS", "OASIS3_CSV", "TORCH_HOME")}
    check("env", True, env)
    report["failures"], report["warnings"] = failures, warnings
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n",
                                   encoding="utf-8")
    print(f"preflight: {len(failures)} failures, {len(warnings)} warnings", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
