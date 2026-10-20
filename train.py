#!/usr/bin/env python
"""Train one SSL model on the 3D hippocampal ROI pair, fast (RTX 4090 / RunPod).

    python train.py --model byol|vicreg|dinov2 [--variant border_h3] [--epochs 50] [--seed 1]

Composes the campaign protocol of ``scripts/run_ssl3d_v2.py`` through
``t5_ssl.recipes`` (same split, views, schedules and probes) and, unless
``--no-fast``, the ``perf=rtx4090`` runtime: bf16 autocast, TF32, fused AdamW,
channels_last_3d, ``torch.compile`` with fallback and persistent pinned
DataLoader workers. Everything degrades gracefully without a GPU.

Outputs go to ``<out-root>/<tag>/<model>_<variant>_s<seed>/``: the Lightning
logs and rotated checkpoints, ``perf.json`` (s/epoch, images/s, peak VRAM),
``result.json`` (probe metrics as in campaign_ssl3d_v2 plus the spatial probe),
and ``encoders.pt`` with ``--export-encoders``.

CUDA out-of-memory errors are retried with half the batch and twice the
gradient accumulation (same effective batch); exit code 75 means the retries
ran out. ``--bench-steps N`` times N optimizer steps instead (bench.py).
The original Hydra entry point (``python -m t5_ssl.train`` / ``t5-ssl-train``)
is untouched.
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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
                        help="torch.compile the encoders (bench.py: slower at batch 16 of 32^3)")
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


# ---------------------------------------------------------------- evaluation
def spatial_encoder(backbone: Any) -> Any:
    """Pre-GAP feature map average-pooled to 2^3 cells and flattened (probe H4).

    ResNet-18 3D: layer4 (512 x 2^3 = 4096); ConvNeXt-T 3D: stage 3, the last
    map wider than one voxel at 32^3 (384 x 2^3 = 3072); ViT-S 3D: the 4^3 patch
    tokens pooled to 2^3 (384 x 8 = 3072).
    """
    import torch.nn.functional as F
    from torch import nn

    class SpatialEncoder(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            b = self.inner
            if hasattr(b, "forward_tokens"):
                _, patches = b.forward_tokens(x)
                grid = round(patches.shape[1] ** (1 / 3))
                fmap = patches.transpose(1, 2).reshape(x.shape[0], -1, grid, grid, grid)
            elif hasattr(b, "layer4"):
                h = b.relu(b.bn1(b.conv1(x)))
                fmap = b.layer4(b.layer3(b.layer2(b.layer1(h))))
            else:
                fmap = b.features[:6](x)
            return F.adaptive_avg_pool3d(fmap, 2).flatten(1)

    return SpatialEncoder(backbone)


def evaluate(model_name: str, encoders: dict[str, Any], dm: Any, cfg: Any,
             spatial: bool) -> dict[str, Any]:
    """Score each encoder as campaign_ssl3d_v2 (CV 5x5 on train+val, test probe)."""
    import warnings

    import run_ssl3d_v2 as v2
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    # Probe features in full fp32, as the campaigns computed them.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clean = ROIEval3D(int(cfg.data.roi_size))
    out: dict[str, Any] = {}
    for role, encoder in encoders.items():
        for attr in ("forward", "forward_tokens"):  # compiled entry points -> eager again
            if attr in encoder.__dict__:
                delattr(encoder, attr)
        encoder = encoder.float().to(memory_format=torch.contiguous_format)
        start = time.time()
        out[role] = v2.score(encoder, dm, clean)
        if spatial:
            spatial_scores = v2.score(spatial_encoder(encoder), dm, clean)
            out[role]["spatial"] = {k: v for k, v in spatial_scores.items()
                                    if k not in ("subjects", "labels")}
        out[role]["eval_minutes"] = (time.time() - start) / 60
        print(f"[eval] {model_name}/{role}: cv={mean3_cv(out[role]):.3f} "
              f"test={mean3_test(out[role]):.3f} erank={out[role]['effective_rank']:.1f}"
              + (f" | spatial cv={mean3_cv(out[role]['spatial']):.3f} "
                 f"test={mean3_test(out[role]['spatial']):.3f}" if spatial else ""), flush=True)
    return out


def identity_scores(encoder: Any, dm: Any, cfg: Any) -> dict[str, Any]:
    """Identity retention and collapse on the val + test subjects (one session each).

    Under the run's own training views and under a fixed reference set (the
    campaign geometry with border padding + bias field, the same for every
    recipe): centered cosine same vs different subject, top-1 view matching,
    between-subject variance of the clean cube over within-subject variance of
    the views (``between_over_within`` = Var_subject / Var_aug) and the
    effective rank (variance spectrum) of the clean embeddings.
    """
    import analyze_ssl3d_encoders as enc
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from t5_ssl.data.roi3d import ROIEval3D, analysis_views, build_roi3d_transforms
    for attr in ("forward", "forward_tokens"):
        if attr in encoder.__dict__:
            delattr(encoder, attr)
    encoder = encoder.float().to(memory_format=torch.contiguous_format)
    records = dm.val_dataset.records + dm.test_dataset.records
    cubes = [torch.from_numpy(dm.cubes[r["idx"]].astype(np.float32)) for r in records]
    size = int(cfg.data.roi_size)
    clean = enc.embed(encoder, [ROIEval3D(size)(c) for c in cubes])
    reference = OmegaConf.create({"data": {"roi_size": size},
                                  "augmentation": recipes.aug_spec("campaign_border", size)})
    out: dict[str, Any] = {"subjects": len(cubes)}
    own_cfg = cfg
    if cfg.augmentation.get("gpu_elastic"):  # same deformation, CPU (TorchIO) version
        own_cfg = OmegaConf.create({"data": {"roi_size": size},
                                    "augmentation": recipes.aug_spec("elastic", size)})
    own = (analysis_views(cfg) if cfg.augmentation.get("tio_gpu_elastic")
           else build_roi3d_transforms(own_cfg)[0])
    for name, transform in (("own", own),
                            ("reference", build_roi3d_transforms(reference)[0])):
        first, second = enc.views(transform, cubes, seed=0)
        out[name] = enc.scores(clean, enc.embed(encoder, first), enc.embed(encoder, second))
    print(f"[identity] own: Var_subject/Var_aug={out['own']['between_over_within']:.3g} "
          f"same={out['own']['same_subject_cos']:.3f} diff={out['own']['different_subject_cos']:.3f} "
          f"erank={out['own']['effective_rank']:.2f}", flush=True)
    return out


def mean3_cv(doc: dict) -> float:
    m = doc["cv"]["clean"]["metrics"]
    return (m["auc"] + m["f1"] + m["aupr"]) / 3


def mean3_test(doc: dict) -> float:
    t = doc["test"]
    return (t["test_auc"] + t["test_f1_macro"] + t["test_average_precision"]) / 3


# ---------------------------------------------------------------- main
def run(args: argparse.Namespace) -> int:
    batch_size, accumulate = args.batch_size, args.accumulate
    compile_on = args.compile == "on"
    if args.dry_run:  # compose only (validates the overrides), no torch import
        from omegaconf import OmegaConf
        overrides = build_overrides(args, batch_size, accumulate, compile_on)
        cfg = compose(overrides)
        print("\n".join(overrides))
        print(OmegaConf.to_yaml({k: cfg[k] for k in ("model", "augmentation", "optimizer",
                                                     "loss") if k in cfg}))
        print(OmegaConf.to_yaml({"perf": cfg.get("perf"), "precision": cfg.trainer.precision,
                                 "num_workers": cfg.data.num_workers,
                                 "log_root": cfg.get("log_root"),
                                 "run_name": cfg.get("run_name")}))
        return 0
    import torch

    from t5_ssl.train import build_backbone, build_datamodule, train
    from t5_ssl.utils.runtime import StepTimer
    started = time.time()
    attempts: list[dict[str, Any]] = []
    for attempt in range(args.oom_retries + 2):
        overrides = build_overrides(args, batch_size, accumulate, compile_on)
        cfg = compose(overrides)
        if args.init_only:
            from t5_ssl.data.roi3d import ROIEval3D
            torch.manual_seed(args.seed)
            clean = ROIEval3D(int(cfg.data.roi_size))
            dm = build_datamodule(cfg, clean, clean)
            dm.setup()
            scored = evaluate(args.model, {"main": build_backbone(cfg)}, dm, cfg,
                              args.spatial_probe)
            exp_dir = Path(cfg.log_root) / cfg.run_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            write_result(exp_dir, args, overrides, scored, None, started, attempts, init=True)
            return 0
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
    if not args.no_eval:
        scored = evaluate(args.model, encoders, dm, cfg, args.spatial_probe)
    identity = identity_scores(encoders["main"], dm, cfg) if args.identity else None
    write_result(exp_dir, args, overrides, scored, result.get("runtime"), started, attempts,
                 identity=identity)
    return 0


def write_result(exp_dir: Path, args: argparse.Namespace, overrides: list[str],
                 scored: dict | None, runtime: dict | None, started: float,
                 attempts: list[dict], init: bool = False,
                 identity: dict | None = None) -> None:
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
        import run_ssl3d_v2 as v2
        doc["training_metrics"] = v2.epoch_metrics(exp_dir)
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
