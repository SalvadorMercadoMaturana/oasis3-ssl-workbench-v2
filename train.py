#!/usr/bin/env python
"""Train and evaluate one SSL model on 3D ROI cubes (entry point of every campaign since 2026-10).

    python train.py --model byol|vicreg|dinov2|pretext [--variant V] [--aug-recipe R]
                    [--roi 32|48|64|multi3|atrophy5] [--backbone B] [--epochs 50] [--seed 1]
    python train.py --model byol --dry-run      # print the composed configuration only

What one call does (``run``):

1. Configuration. ``build_overrides`` turns the flags into Hydra overrides through
   ``t5_ssl.recipes`` (split manifest, views, schedule, encoder, run name) and ``compose``
   builds the configuration from ``configs/``.
2. Training. ``t5_ssl.train.train`` (data module, encoder, SSL module, Lightning trainer).
   CUDA out-of-memory errors are retried with half the batch and twice the gradient
   accumulation (same effective batch); exit code 75 means the retries ran out.
   ``--init-only`` skips training and scores the encoder SSL would start from (the
   "sin SSL" control).
3. Evaluation. ``t5_ssl.evaluation.run_scoring.evaluate_encoders`` freezes the encoder
   (DINOv2: the EMA teacher, plus the student) and fits the linear AD-vs-HC probes of
   protocol v1 and v2; ``--external NAME=CSV`` adds other cohorts (ADNI), ``--identity``
   identity/collapse scores on held-out subjects, ``--spatial-probe`` a second probe on the
   feature map before global pooling (pooled to 2x2x2).
4. Outputs. ``save_artifacts`` and ``write_result`` write, in
   ``<out-root>/<tag>/<run name>/`` (``recipes.run_name``, e.g.
   ``byol_wide_dproj0.1__noise__roi32_s1``), next to the Lightning logs and rotated
   checkpoints: ``result.json`` (probe metrics, training curves, runtime), ``perf.json``
   (s/epoch, images/s, peak VRAM) and, on request, ``embeddings.npz``, ``encoder.pt``
   (fp16, ``--export-encoder``) and ``encoders.pt`` (every encoder, ``--export-encoders``).

Unless ``--no-fast``, the ``perf=rtx4090`` runtime is on: bf16 autocast, TF32, fused AdamW,
channels_last_3d, optional ``torch.compile`` with fallback and persistent pinned DataLoader
workers. Everything degrades gracefully without a GPU (``--cpu`` forces it).
``--bench-steps N`` times N optimizer steps instead of training (``cloud/bench.py``).

Many runs on one GPU: ``cloud/run_all.py`` and ``cloud/queue_runner.py``. The plain Hydra
entry point (``python -m t5_ssl.train`` / ``t5-ssl-train``) trains without evaluating.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:  # also without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from t5_ssl import recipes  # noqa: E402  (light: no torch)

EXIT_OOM = 75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=recipes.MODELS)
    parser.add_argument("--variant", default="campaign",
                        help="recipe variant (t5_ssl.recipes.VARIANTS), e.g. border_h3 for BYOL")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--precision", default="bf16-mixed",
                        help="Lightning precision (degrades to 16-mixed / 32-true when unsupported)")
    parser.add_argument("--compile", choices=("on", "off"), default="off",
                        help="torch.compile the encoders (cloud/bench.py: slower at batch 16 of 32^3)")
    parser.add_argument("--compile-modes", default="default,reduce-overhead",
                        help="torch.compile modes tried in order before falling back to eager")
    parser.add_argument("--channels-last", choices=("on", "off"), default="on")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--prefetch", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--no-fast", action="store_true",
                        help="campaign runtime instead (fp32, no workers, no perf profile)")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--tag", default="adhoc", help="sub-directory of --out-root")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--keep-ckpts", type=int, default=2,
                        help="periodic checkpoints kept on disk (rotation); last.ckpt always")
    parser.add_argument("--ckpt-every", type=int, default=5, help="epochs between checkpoints")
    parser.add_argument("--no-eval", action="store_true", help="skip the frozen probes")
    parser.add_argument("--spatial-probe", action="store_true",
                        help="also probe the pre-GAP feature map pooled to 2^3 (H4)")
    parser.add_argument("--external", action="append", default=[], metavar="NAME=CSV",
                        help="external labeled test set (subject, idx, group) whose cubes are in "
                             "data/roi3d/<NAME>_<cache>; scored with the same probe (repeatable)")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="embeddings.npz: probe features of every split and external set")
    parser.add_argument("--export-encoder", action="store_true",
                        help="encoder.pt: the evaluated encoder only (fp16) + overrides")
    parser.add_argument("--export-encoders", action="store_true",
                        help="save encoder state_dicts to encoders.pt (for analyses)")
    parser.add_argument("--aug-recipe", default=None, choices=sorted(recipes.AUG_RECIPES),
                        help="replace the views by a recipe of t5_ssl.recipes.AUG_RECIPES")
    parser.add_argument("--roi", default=None, choices=sorted(recipes.ROIS),
                        help="region of interest: 32 (default), 48, 64 or multi3")
    parser.add_argument("--backbone", default=None, choices=sorted(recipes.BACKBONES),
                        help="swap the method's encoder (t5_ssl.recipes.BACKBONES)")
    parser.add_argument("--collapse-every", type=int, default=0,
                        help="collapse metrics on held-out subjects every N epochs (0 = off)")
    parser.add_argument("--identity", action="store_true",
                        help="identity/collapse metrics on the 142 held-out subjects")
    parser.add_argument("--init-only", action="store_true",
                        help="no SSL: score the ImageNet-initialized encoder (control)")
    parser.add_argument("--oom-retries", type=int, default=2)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra Hydra override (repeatable)")
    parser.add_argument("--bench-steps", type=int, default=0)
    parser.add_argument("--bench-warmup", type=int, default=15)
    parser.add_argument("--bench-out", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the overrides and exit")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace, batch_size: int, accumulate: int,
                    compile_on: bool) -> list[str]:
    fast = not args.no_fast
    bench = args.bench_steps > 0
    extra = [f"data.batch_size={batch_size}", f"++trainer.accumulate_grad_batches={accumulate}",
             f"++trainer.checkpoint_keep_last={args.keep_ckpts}",
             f"trainer.checkpoint_every_n_epochs={args.ckpt_every}"]
    if fast:
        extra += [f"trainer.precision={args.precision}", f"data.num_workers={args.workers}",
                  f"++data.prefetch_factor={args.prefetch}",
                  f"perf.compile={str(compile_on).lower()}",
                  f"perf.compile_modes=[{args.compile_modes}]",
                  f"perf.channels_last={str(args.channels_last == 'on').lower()}"]
    if bench:
        extra += [f"trainer.max_steps={args.bench_warmup + args.bench_steps}",
                  "trainer.max_epochs=1000", "trainer.limit_val_batches=0",
                  "trainer.log_every_n_steps=1000"]
        if fast:
            extra.append("perf.monitor=false")
    tag_root = Path(args.out_root) / ("bench" if bench else args.tag)
    name = recipes.run_name(args.model, args.variant, args.seed, args.aug_recipe, args.roi,
                            args.backbone)
    if args.init_only:
        name = (f"{args.model}_init" + (f"__bb{args.backbone}" if args.backbone else "")
                + (f"__roi{args.roi}" if args.roi else "") + f"_s{args.seed}")
    return recipes.overrides(
        args.model, args.variant, epochs=args.epochs, seed=args.seed, fast=fast,
        checkpoints=not (args.no_checkpoints or bench or args.init_only),
        accelerator="cpu" if args.cpu else "auto", log_root=tag_root.as_posix(),
        run_name=name, aug_recipe=args.aug_recipe, roi=args.roi or "32",
        backbone=args.backbone, extra=extra + list(args.set))


def compose(overrides: list[str]) -> Any:
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return hydra_compose(config_name="train", overrides=overrides)


def is_oom(exc: BaseException) -> bool:
    import torch
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def is_compile_failure(exc: BaseException) -> bool:
    names = {type(e).__module__ + "." + type(e).__name__ for e in _chain(exc)}
    return any(part in name for name in names for part in ("_dynamo", "_inductor", "triton"))


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def free_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- main
def run(args: argparse.Namespace) -> int:
    """Dry run, "sin SSL" scoring, or training (with OOM/compile retries) + evaluation."""
    batch_size, accumulate = args.batch_size, args.accumulate
    compile_on = args.compile == "on"
    if args.dry_run:
        return dry_run(build_overrides(args, batch_size, accumulate, compile_on))
    started = time.time()
    if args.init_only:
        return score_init(args, build_overrides(args, batch_size, accumulate, compile_on),
                          started)
    import torch

    from t5_ssl.evaluation.identity import identity_scores
    from t5_ssl.evaluation.run_scoring import evaluate_encoders
    from t5_ssl.train import train
    from t5_ssl.utils.runtime import StepTimer
    attempts: list[dict[str, Any]] = []
    for attempt in range(args.oom_retries + 2):
        overrides = build_overrides(args, batch_size, accumulate, compile_on)
        cfg = compose(overrides)
        callbacks = []
        if args.collapse_every:
            from t5_ssl.callbacks.collapse_monitor import CollapseMonitor
            callbacks.append(CollapseMonitor(cfg, args.collapse_every))
        timer = None
        if args.bench_steps:
            out = args.bench_out or str(Path(args.out_root) / "bench" / "bench_step.json")
            timer = StepTimer(args.bench_warmup, args.bench_steps, out,
                              extra={"model": args.model, "variant": args.variant,
                                     "batch_size": batch_size, "precision": args.precision,
                                     "compile": compile_on, "channels_last": args.channels_last,
                                     "workers": args.workers, "fast": not args.no_fast})
            callbacks.append(timer)
        try:
            result = train(cfg, extra_callbacks=callbacks)
        except Exception as exc:
            oom, compile_failure = is_oom(exc), compile_on and is_compile_failure(exc)
            attempts.append({"batch_size": batch_size, "accumulate": accumulate,
                             "compile": compile_on, "error": f"{type(exc).__name__}: "
                             f"{str(exc).splitlines()[0][:300] if str(exc) else ''}"})
            traceback.print_exc()
            del exc
            free_cuda()
            if compile_failure and not oom:
                print("[retry] torch.compile failed during training: retrying eagerly", flush=True)
                compile_on = False
                continue
            if oom and attempt < args.oom_retries and batch_size > 1:
                batch_size, accumulate = max(1, batch_size // 2), accumulate * 2
                print(f"[retry] CUDA OOM: batch {batch_size} x accumulate {accumulate}",
                      flush=True)
                continue
            return EXIT_OOM if oom else 1
        break
    else:
        return EXIT_OOM

    exp_dir, model, dm = result["exp_dir"], result["model"], result["datamodule"]
    if timer is not None:
        print(f"[bench] {json.dumps(timer.result)}", flush=True)
        return 0 if timer.result else 1
    encoders = {"main": model.backbone}
    if args.model == "dinov2":  # DINO/DINOv2 evaluate the EMA teacher
        encoders = {"main": model.teacher_backbone, "student": model.backbone}
    if args.export_encoders:
        from t5_ssl.utils.runtime import uncompile
        uncompile(model)
        state = {"backbone": model.backbone.state_dict()}
        for name in ("target_backbone", "teacher_backbone"):
            if hasattr(model, name):
                state[name] = getattr(model, name).state_dict()
        torch.save({"overrides": overrides, "state": state}, exp_dir / "encoders.pt")
    scored = None
    attach_externals(dm, args, cfg)
    if not args.no_eval:
        scored = evaluate_encoders(args.model, encoders, dm, cfg, args.spatial_probe,
                                   args.save_embeddings)
    save_artifacts(exp_dir, args, overrides, scored, encoders["main"],
                   "teacher_backbone" if args.model == "dinov2" else "backbone")
    identity = identity_scores(encoders["main"], dm, cfg) if args.identity else None
    write_result(exp_dir, args, overrides, scored, result.get("runtime"), started, attempts,
                 identity=identity)
    return 0


def dry_run(overrides: list[str]) -> int:
    """Print the overrides and the composed nodes (validates them); no torch import."""
    from omegaconf import OmegaConf
    cfg = compose(overrides)
    print("\n".join(overrides))
    print(OmegaConf.to_yaml({k: cfg[k] for k in ("model", "augmentation", "optimizer",
                                                 "loss") if k in cfg}))
    print(OmegaConf.to_yaml({"perf": cfg.get("perf"), "precision": cfg.trainer.precision,
                             "num_workers": cfg.data.num_workers,
                             "log_root": cfg.get("log_root"),
                             "run_name": cfg.get("run_name")}))
    return 0


def score_init(args: argparse.Namespace, overrides: list[str], started: float) -> int:
    """``--init-only``: score the encoder SSL would start from ("sin SSL"), no training."""
    import torch

    from t5_ssl.data.roi3d import probe_transform
    from t5_ssl.evaluation.run_scoring import evaluate_encoders
    from t5_ssl.train import build_backbone, build_datamodule
    cfg = compose(overrides)
    torch.manual_seed(args.seed)
    clean = probe_transform(cfg)
    dm = build_datamodule(cfg, clean, clean)
    dm.setup()
    attach_externals(dm, args, cfg)
    encoder = build_backbone(cfg)
    scored = evaluate_encoders(args.model, {"main": encoder}, dm, cfg,
                               args.spatial_probe, args.save_embeddings)
    exp_dir = Path(cfg.log_root) / cfg.run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts(exp_dir, args, overrides, scored, encoder, "backbone")
    write_result(exp_dir, args, overrides, scored, None, started, [], init=True)
    return 0


def attach_externals(dm: Any, args: argparse.Namespace, cfg: Any) -> None:
    """``--external NAME=CSV`` sets as ``dm.external_sets`` (cubes from <NAME>_<cache>)."""
    from t5_ssl.data.roi3d import load_external
    dm.external_sets = {}
    for spec in args.external:
        name, csv_path = spec.split("=", 1)
        dm.external_sets[name] = load_external(name, csv_path, cfg.data.cache_path)


def save_artifacts(exp_dir: Path, args: argparse.Namespace, overrides: list[str],
                   scored: dict | None, encoder: Any, role: str) -> None:
    """``embeddings.npz`` (probe features per split) and ``encoder.pt`` (fp16 weights)."""
    import numpy as np
    import torch
    embeddings = (scored or {}).get("main", {}).pop("_embeddings", None)
    if embeddings is not None:
        arrays = {}
        for name, (x, y, subjects) in embeddings.items():
            arrays.update({f"{name}_x": x.astype(np.float32), f"{name}_y": y,
                           f"{name}_subjects": np.array(subjects)})
        np.savez_compressed(exp_dir / "embeddings.npz", **arrays)
    if args.export_encoder:
        state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                 for k, v in encoder.state_dict().items()}
        torch.save({"overrides": overrides, "role": role, "state": state}, exp_dir / "encoder.pt")


def write_result(exp_dir: Path, args: argparse.Namespace, overrides: list[str],
                 scored: dict | None, runtime: dict | None, started: float,
                 attempts: list[dict], init: bool = False,
                 identity: dict | None = None) -> None:
    from t5_ssl.evaluation.run_scoring import mean3_cv, mean3_test

    perf_path = exp_dir / "perf.json"
    perf = json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.is_file() else None
    doc: dict[str, Any] = {"model": args.model, "variant": "init" if init else args.variant,
                           "seed": args.seed, "epochs": 0 if init else args.epochs,
                           "overrides": overrides, "runtime": runtime, "perf": perf,
                           "retries": attempts, "minutes": (time.time() - started) / 60,
                           "experiment_dir": exp_dir.as_posix(),
                           "aug_recipe": getattr(args, "aug_recipe", None),
                           "roi": getattr(args, "roi", None) or "32",
                           "backbone": (getattr(args, "backbone", None)
                                        or recipes.DEFAULT_BACKBONE[args.model]),
                           "identity": identity}
    if scored:
        main = scored.pop("main")
        doc.update(main)
        doc["encoder"] = "teacher" if "student" in scored else "backbone"
        doc["student"] = ({k: v for k, v in scored["student"].items()
                           if k not in ("subjects", "labels")} if "student" in scored else None)
        doc["cv_mean3"], doc["test_mean3"] = mean3_cv(main), mean3_test(main)
        doc["cv_auc"], doc["test_auc"] = main["cv"]["clean"]["metrics"]["auc"], main["test"]["test_auc"]
    if (exp_dir / "collapse.json").is_file():
        doc["collapse"] = json.loads((exp_dir / "collapse.json").read_text(encoding="utf-8"))
    if (exp_dir / "metrics.csv").is_file():
        from t5_ssl.evaluation.curves import epoch_metrics
        doc["training_metrics"] = epoch_metrics(exp_dir)
    (exp_dir / "result.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(f"[done] {exp_dir.as_posix()} in {doc['minutes']:.1f} min"
          + (f" cv={doc['cv_mean3']:.3f} test={doc['test_mean3']:.3f}" if scored else ""),
          flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("OASIS3_CSV", str(ROOT / "data/dementia/mni-dataset.csv"))
    os.chdir(ROOT)  # relative data/manifest paths in the configs
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
