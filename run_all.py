#!/usr/bin/env python
"""Run several SSL trainings on one GPU and report throughput, VRAM and probes.

    python run_all.py --parallel mps|all-at-once|sequential [--models byol vicreg dinov2]
                      [--seeds 1 2 3] [--epochs 50] [--byol-variant border_h3] [--tag final50]
    python run_all.py --preset byol-screen --parallel mps --max-procs 4

Each job is ``train.py`` run in its own process (``multiprocessing`` with the
``spawn`` start method, so no CUDA state is inherited), with stdout/stderr in
``<out-root>/<tag>/logs/<job>.log``. Parallel modes:

- ``mps``: start the CUDA MPS control daemon so the processes share the GPU
  through one MPS server (concurrent kernels, no context switching); if it
  cannot start (no binary, no permission) fall back to ``all-at-once``;
- ``all-at-once``: up to ``--max-procs`` processes time-slice the GPU;
- ``sequential``: one job at a time.

A job that exits with code 75 (CUDA OOM after train.py's own retries) is
relaunched once with half the batch and twice the gradient accumulation.
``nvidia-smi`` is sampled every few seconds for whole-GPU utilization and memory.
The report (``REPORT.md`` + ``report.json``) lists images/s, s/epoch and peak
VRAM per job, and the frozen-probe scores when the jobs were evaluated.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from t5_ssl import recipes  # noqa: E402  (light: no torch)

EXIT_OOM = 75
PRESETS: dict[str, list[tuple[str, str]]] = {
    # The BYOL experiments of docs/informe_ssl3d_v2.pdf sec. 8.5 (without the
    # VICReg-terms one), re-running the campaign BYOL and VICReg as same-hardware
    # baselines; 30 epochs as campaign_ssl3d(_v2).
    "byol-screen": [("byol", v) for v in ("campaign", "border", "border_wide", "border_gn",
                                          "border_predlr", "border_h3")]
                   + [("vicreg", "campaign"), ("vicreg", "border")],
    # Dropout against BYOL's overfitting (validation loss rising at 50 epochs).
    "byol-dropout": [("byol", v) for v in ("border_wide", "wide_dproj0.1", "wide_dproj0.3",
                                          "wide_dpred0.1", "wide_dpred0.3", "wide_dboth0.2")],
}
MPS_DIRS = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
            "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log"}


def effective_cpus() -> int:
    """CPUs this container may use: the cgroup quota (containers report the host's count)."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for path, parse in (("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
                        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None)):
        try:
            if parse is not None:
                quota, period = parse(Path(path).read_text())
            else:
                quota = Path(path).read_text().strip()
                period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
            if quota not in ("max", "-1"):
                return max(1, min(count, int(int(quota) / int(period))))
        except (OSError, ValueError):
            continue
    return count


# ---------------------------------------------------------------- child process
def _child(argv: list[str], log_path: str) -> None:
    """Spawned process: run train.py's main with stdout/stderr redirected to the log."""
    log = open(log_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - lives with the process
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sys.path.insert(0, str(ROOT))
    import train as entry
    code = entry.main(argv)
    log.flush()
    sys.exit(code)


# ---------------------------------------------------------------- MPS / GPU monitor
def start_mps() -> bool:
    binary = shutil.which("nvidia-cuda-mps-control")
    if binary is None:
        print("[mps] nvidia-cuda-mps-control not found", flush=True)
        return False
    for key, value in MPS_DIRS.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    started = subprocess.run([binary, "-d"], capture_output=True, text=True)
    if started.returncode != 0:
        print(f"[mps] daemon failed: {started.stderr.strip() or started.stdout.strip()}",
              flush=True)
        return False
    probe = subprocess.run([binary], input="get_server_list\n", capture_output=True, text=True,
                           timeout=10)
    if probe.returncode != 0:
        print(f"[mps] control not answering: {probe.stderr.strip()}", flush=True)
        stop_mps()
        return False
    print("[mps] daemon started", flush=True)
    return True


def stop_mps() -> None:
    binary = shutil.which("nvidia-cuda-mps-control")
    if binary:
        with contextlib.suppress(Exception):
            subprocess.run([binary], input="quit\n", capture_output=True, text=True, timeout=20)
    for key in MPS_DIRS:
        os.environ.pop(key, None)


class GPUSampler(threading.Thread):
    """Sample whole-GPU utilization and memory with nvidia-smi into a CSV."""

    def __init__(self, path: Path, every: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.path, self.every = path, every
        self.stop_event = threading.Event()
        self.rows: list[tuple[float, float, float, float]] = []

    def run(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        with self.path.open("w", encoding="utf-8") as stream:
            stream.write("time,util_percent,memory_used_mib,memory_total_mib\n")
            while not self.stop_event.is_set():
                with contextlib.suppress(Exception):
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        timeout=10).stdout.strip().splitlines()[0]
                    util, used, total = (float(v) for v in out.split(","))
                    row = (time.time(), util, used, total)
                    self.rows.append(row)
                    stream.write(",".join(f"{v:.1f}" for v in row) + "\n")
                    stream.flush()
                self.stop_event.wait(self.every)

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {}
        utils = [r[1] for r in self.rows]
        return {"samples": len(self.rows), "mean_util_percent": statistics.mean(utils),
                "busy_util_percent": statistics.mean([u for u in utils if u > 0] or [0]),
                "peak_memory_used_gb": max(r[2] for r in self.rows) / 1024,
                "memory_total_gb": self.rows[0][3] / 1024}


# ---------------------------------------------------------------- jobs
def make_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    def variant_of(model: str) -> str:
        return {"byol": args.byol_variant, "vicreg": args.vicreg_variant}.get(model, "campaign")

    if args.preset:
        triples = [(m, v, None) for m, v in PRESETS[args.preset]]
    elif args.specs:  # explicit model:recipe[:roi[:variant]] jobs (e.g. a screen's winners)
        parts = [[*spec.split(":"), "", ""][:4] for spec in args.specs]
        triples = [(m, v or variant_of(m), r if r not in ("", "-") else None)
                   for m, r, _, v in parts]
        spec_roi = {i: roi for i, (_, _, roi, _) in enumerate(parts) if roi}
    else:  # recipe-major, models interleaved so concurrent jobs mix heavy and light models
        triples = [(m, variant_of(m), r) for r in (args.aug_recipes or [None])
                   for m in args.models]
    pairs = [(m, v) for m, v, _ in triples]
    jobs = []
    rois = args.rois or [None]
    spec_roi = locals().get("spec_roi", {})
    for seed in args.seeds:  # seed-major: every model of seed 1 starts first
        for backbone, roi, (model, variant, recipe) in (
                (b, spec_roi.get(i, r), t) for b in (args.backbones or [None]) for r in rois
                for i, t in enumerate(triples)):
            name = recipes.run_name(model, variant, seed, recipe, roi, backbone)
            argv = ["--model", model, "--variant", variant, "--seed", str(seed),
                    "--epochs", str(args.epochs), "--tag", args.tag,
                    "--out-root", args.out_root, "--workers", str(args.workers),
                    "--batch-size", str(args.batch_size), "--accumulate", str(args.accumulate),
                    "--compile", args.compile, "--precision", args.precision,
                    *(["--aug-recipe", recipe] if recipe else []),
                    *(["--roi", roi] if roi else []),
                    *(["--backbone", backbone] if backbone else []), *args.train_args]
            jobs.append({"name": name, "model": model, "variant": variant, "recipe": recipe,
                         "roi": roi, "backbone": backbone,
                         "seed": seed, "argv": argv, "batch_size": args.batch_size,
                         "accumulate": args.accumulate})
    if args.init_controls:
        for model in sorted({m for m, _ in pairs}):
            argv = ["--model", model, "--seed", "1", "--init-only", "--tag", args.tag,
                    "--out-root", args.out_root, *[a for a in args.train_args
                                                   if a == "--spatial-probe"]]
            jobs.append({"name": f"{model}_init_s1", "model": model, "variant": "init",
                         "recipe": None, "seed": 1, "argv": argv, "batch_size": args.batch_size,
                         "accumulate": 1})
    return jobs


def run_jobs(jobs: list[dict[str, Any]], max_procs: int, logs: Path) -> None:
    ctx = mp.get_context("spawn")
    pending = list(jobs)
    running: dict[str, tuple[Any, dict, float]] = {}
    while pending or running:
        while pending and len(running) < max_procs:
            job = pending.pop(0)
            log_path = logs / f"{job['name']}.log"
            process = ctx.Process(target=_child, args=(job["argv"], str(log_path)),
                                  name=job["name"])
            process.start()
            running[job["name"]] = (process, job, time.time())
            print(f"{time.strftime('%H:%M:%S')} start {job['name']} (pid {process.pid})",
                  flush=True)
        time.sleep(2)
        for name, (process, job, start) in list(running.items()):
            if process.is_alive():
                continue
            process.join()
            del running[name]
            minutes = (time.time() - start) / 60
            code = process.exitcode
            job.setdefault("attempts", []).append({"exit_code": code, "minutes": minutes,
                                                   "batch_size": job["batch_size"]})
            if code == EXIT_OOM and job["batch_size"] > 1 and len(job["attempts"]) < 2:
                job["batch_size"] //= 2
                job["accumulate"] *= 2
                job["argv"] = _replace(job["argv"], "--batch-size", str(job["batch_size"]))
                job["argv"] = _replace(job["argv"], "--accumulate", str(job["accumulate"]))
                print(f"{time.strftime('%H:%M:%S')} OOM {name}: relaunch with batch "
                      f"{job['batch_size']} x accumulate {job['accumulate']}", flush=True)
                pending.insert(0, job)
                continue
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            print(f"{time.strftime('%H:%M:%S')} {status} {name} ({minutes:.1f} min)"
                  f"{_score_line(job)}", flush=True)


def _replace(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def _result_path(job: dict[str, Any], args: argparse.Namespace) -> Path:
    return Path(args.out_root) / args.tag / job["name"] / "result.json"


def _score_line(job: dict[str, Any]) -> str:
    path = Path(job["argv"][job["argv"].index("--out-root") + 1]) / job["argv"][
        job["argv"].index("--tag") + 1] / job["name"] / "result.json"
    if not path.is_file():
        return ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    perf = doc.get("perf") or {}
    parts = []
    if "cv_mean3" in doc:
        parts.append(f"cv={doc['cv_mean3']:.3f} test={doc['test_mean3']:.3f}")
    if perf.get("images_per_second"):
        parts.append(f"{perf['images_per_second']:.0f} img/s "
                     f"{perf['seconds_per_epoch']:.1f} s/ep "
                     f"VRAM {perf.get('peak_vram_reserved_gb', 0):.1f} GB")
    return (" | " + " | ".join(parts)) if parts else ""


# ---------------------------------------------------------------- report
def report(args: argparse.Namespace, jobs: list[dict[str, Any]], wall_seconds: float,
           mode: str, gpu: dict[str, float]) -> dict[str, Any]:
    rows = []
    for job in jobs:
        path = _result_path(job, args)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        perf = doc.get("perf") or {}
        student = doc.get("student") or {}
        spatial = doc.get("spatial") or {}
        own = (doc.get("identity") or {}).get("own") or {}
        ref = (doc.get("identity") or {}).get("reference") or {}
        rows.append({
            "job": job["name"], "model": job["model"], "variant": job["variant"],
            "recipe": ((job.get("recipe") or "") + (f" @roi{job['roi']}" if job.get("roi") else "")
                       + (f" [{job['backbone']}]" if job.get("backbone") else "") or None),
            "seed": job["seed"], "status": "ok" if doc else "failed",
            "cv_auc": doc.get("cv_auc"), "test_auc": doc.get("test_auc"),
            "var_subject_over_aug": own.get("between_over_within"),
            "var_subject_over_aug_reference": ref.get("between_over_within"),
            "same_subject_cos": own.get("same_subject_cos"),
            "different_subject_cos": own.get("different_subject_cos"),
            "view_matching_top1": own.get("view_matching_top1"),
            "heldout_effective_rank": own.get("effective_rank"),
            "batch_size": (perf.get("batch_size") or job["batch_size"]),
            "accumulate": perf.get("accumulate_grad_batches", job["accumulate"]),
            "images_per_second": perf.get("images_per_second"),
            "seconds_per_epoch": perf.get("seconds_per_epoch"),
            "first_epoch_seconds": perf.get("first_epoch_seconds"),
            "fit_minutes": (perf.get("fit_seconds") or 0) / 60 or None,
            "peak_vram_allocated_gb": perf.get("peak_vram_allocated_gb"),
            "peak_vram_reserved_gb": perf.get("peak_vram_reserved_gb"),
            "compile_mode": perf.get("compile_mode"), "precision": perf.get("precision"),
            "channels_last": perf.get("channels_last"),
            "minutes": doc.get("minutes"), "cv_mean3": doc.get("cv_mean3"),
            "test_mean3": doc.get("test_mean3"), "effective_rank": doc.get("effective_rank"),
            "student_cv_mean3": (_mean3(student["cv"]["clean"]["metrics"])
                                 if student.get("cv") else None),
            "spatial_cv_mean3": (_mean3(spatial["cv"]["clean"]["metrics"])
                                 if spatial.get("cv") else None),
            "spatial_test_mean3": (_mean3_test(spatial["test"]) if spatial.get("test") else None),
            "attempts": job.get("attempts", []),
        })
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(row["model"], row["variant"], row["recipe"] or "")].append(row)

    def avg(values: list[Any]) -> float | None:
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    summary = [{"model": m, "variant": v + (f" / {r}" if r else ""), "n": len(rs),
                **{k: avg([row[k] for row in rs]) for k in (
                    "cv_mean3", "test_mean3", "effective_rank", "images_per_second",
                    "seconds_per_epoch", "peak_vram_reserved_gb", "minutes",
                    "spatial_cv_mean3", "spatial_test_mean3", "student_cv_mean3", "cv_auc",
                    "test_auc", "var_subject_over_aug", "heldout_effective_rank")}}
               for (m, v, r), rs in sorted(groups.items())]
    doc = {"tag": args.tag, "mode": mode, "epochs": args.epochs, "wall_minutes": wall_seconds / 60,
           "gpu": gpu, "jobs": rows, "summary": summary}
    out = Path(args.out_root) / args.tag
    (out / "report.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def fmt(value: Any, spec: str = ".3f") -> str:
        return "—" if value is None else format(value, spec)

    lines = [f"# run_all `{args.tag}`", "",
             f"Modo `{mode}`, {args.epochs} épocas, {len(jobs)} trabajos, "
             f"reloj total {wall_seconds / 60:.1f} min."]
    if gpu:
        lines.append(f"GPU completa (nvidia-smi): utilización media {gpu['mean_util_percent']:.0f} % "
                     f"({gpu['busy_util_percent']:.0f} % mientras trabajaba), memoria pico "
                     f"{gpu['peak_memory_used_gb']:.1f} de {gpu['memory_total_gb']:.0f} GB.")
    lines += ["", "## Por modelo y variante (media de semillas)", "",
              "| Modelo | Variante | n | CV media3 | Test media3 | Rango ef. | img/s | s/época | "
              "VRAM pico (GB) | min/corrida |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['variant']} | {s['n']} | {fmt(s['cv_mean3'])} | "
                     f"{fmt(s['test_mean3'])} | {fmt(s['effective_rank'], '.1f')} | "
                     f"{fmt(s['images_per_second'], '.0f')} | {fmt(s['seconds_per_epoch'], '.1f')} | "
                     f"{fmt(s['peak_vram_reserved_gb'], '.2f')} | {fmt(s['minutes'], '.1f')} |")
    if any(s["spatial_cv_mean3"] is not None for s in summary):
        lines += ["", "Probe espacial (mapa previo al GAP en 2³ celdas):", ""]
        lines += [f"- {s['model']} {s['variant']}: CV {fmt(s['spatial_cv_mean3'])}, test "
                  f"{fmt(s['spatial_test_mean3'])}" for s in summary
                  if s["spatial_cv_mean3"] is not None]
    lines += ["", "## Por trabajo", "",
              "| Trabajo | Estado | Batch×acum | Compile | img/s | s/época | 1ª época (s) | "
              "VRAM pico asign./reserv. (GB) | CV | Test |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['job']} | {r['status']} | {r['batch_size']}×{r['accumulate']} | "
                     f"{r['compile_mode'] or 'eager'} | {fmt(r['images_per_second'], '.0f')} | "
                     f"{fmt(r['seconds_per_epoch'], '.1f')} | {fmt(r['first_epoch_seconds'], '.0f')} | "
                     f"{fmt(r['peak_vram_allocated_gb'], '.2f')}/{fmt(r['peak_vram_reserved_gb'], '.2f')} | "
                     f"{fmt(r['cv_mean3'])} | {fmt(r['test_mean3'])} |")
    lines += ["", "img/s = volúmenes de entrada por segundo en el encoder online (batch × vistas), "
              "sin la primera época (compilación, arranque de workers). VRAM por proceso "
              "(torch.cuda.max_memory_*). CV/Test = media de AUROC, F1-macro y AUPR del probe "
              "congelado (protocolo de campaign_ssl3d_v2).", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'REPORT.md'}", flush=True)
    return doc


def _mean3(m: dict[str, float]) -> float:
    return (m["auc"] + m["f1"] + m["aupr"]) / 3


def _mean3_test(t: dict[str, float]) -> float:
    return (t["test_auc"] + t["test_f1_macro"] + t["test_average_precision"]) / 3


# ---------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parallel", choices=("mps", "all-at-once", "sequential"), default="mps")
    parser.add_argument("--models", nargs="+", default=list(recipes.MODELS),
                        choices=recipes.MODELS)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--aug-recipes", nargs="+", default=None,
                        choices=sorted(recipes.AUG_RECIPES),
                        help="one job per model x recipe (t5_ssl.recipes.AUG_RECIPES)")
    parser.add_argument("--rois", nargs="+", default=None, choices=sorted(recipes.ROIS),
                        help="repeat every job for these regions of interest (listed order)")
    parser.add_argument("--specs", nargs="+", default=None, metavar="MODEL:RECIPE",
                        help="explicit model:recipe jobs (recipe '-' = the model's own views)")
    parser.add_argument("--backbones", nargs="+", default=None, choices=sorted(recipes.BACKBONES),
                        help="repeat every job with each encoder (t5_ssl.recipes.BACKBONES)")
    parser.add_argument("--byol-variant", default="campaign")
    parser.add_argument("--vicreg-variant", default="campaign")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--compile", choices=("on", "off"), default="off")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader workers per job (default: CPUs / concurrent jobs)")
    parser.add_argument("--max-procs", type=int, default=None,
                        help="concurrent jobs (default: number of models; 1 if sequential)")
    parser.add_argument("--tag", default=time.strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--init-controls", action="store_true",
                        help="also score each model's untrained (ImageNet) encoder")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="after '--': extra train.py flags, e.g. -- --spatial-probe")
    args = parser.parse_args(argv)
    args.train_args = [a for a in args.train_args if a != "--"]
    n_models = (len(PRESETS[args.preset]) if args.preset else len(args.specs) if args.specs
                else len(args.models))
    if args.max_procs is None:
        args.max_procs = 1 if args.parallel == "sequential" else min(n_models, 4)
    if args.parallel == "sequential":
        args.max_procs = 1
    if args.workers is None:
        args.workers = max(2, min(8, effective_cpus() // max(1, args.max_procs) - 1))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(ROOT)
    out = Path(args.out_root) / args.tag
    (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    if args.report_only:
        report(args, jobs, 0.0, args.parallel, {})
        return 0
    mode = args.parallel
    if mode == "mps" and not start_mps():
        print("[mps] falling back to all-at-once", flush=True)
        mode = "all-at-once"
    sampler = GPUSampler(out / "gpu_samples.csv")
    sampler.start()
    start = time.time()
    try:
        run_jobs(jobs, args.max_procs, out / "logs")
    finally:
        sampler.stop_event.set()
        sampler.join(timeout=15)
        if mode == "mps":
            stop_mps()
    doc = report(args, jobs, time.time() - start, mode, sampler.summary())
    failed = [r["job"] for r in doc["jobs"] if r["status"] != "ok"]
    if failed:
        print(f"FAILED: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

