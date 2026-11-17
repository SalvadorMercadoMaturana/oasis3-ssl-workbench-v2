"""Optional GPU runtime knobs for fast training (the ``perf`` config group).

Nothing here runs unless a config enables it (``+perf=rtx4090``; the default
configs carry no ``perf`` node), and every knob degrades to a no-op without a
GPU:

- TF32 matmul/cuDNN and ``torch.set_float32_matmul_precision("high")``;
- mixed precision resolved to what the device supports (bf16 -> fp16 -> fp32);
- ``torch.compile`` of the encoders, trying a chain of modes (default, then
  ``reduce-overhead``) on a throw-away copy before compiling the real modules,
  and falling back to eager when every mode fails (no Triton, Windows, CPU);
- ``channels_last(_3d)`` weights and inputs for convolutional encoders;
- fused AdamW;
- :class:`PerfMonitor`: per-epoch wall time, images/s and peak VRAM, written to
  ``perf.json`` with one CUDA synchronize per epoch boundary (none per step).
"""

from __future__ import annotations

import contextlib
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import nn

from t5_ssl.utils.logging import get_logger

logger = get_logger(__name__)

PROFILE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "tf32": False,
    "matmul_precision": None,
    "cudnn_benchmark": False,
    "compile": False,
    "compile_modes": ["default", "reduce-overhead"],
    "channels_last": False,
    "fused_optimizer": False,
    "monitor": False,
}


def perf_settings(cfg: Any) -> dict[str, Any]:
    """Return the ``perf`` node merged over :data:`PROFILE_DEFAULTS` (all off)."""
    node = cfg.get("perf") if hasattr(cfg, "get") else None
    settings = dict(PROFILE_DEFAULTS)
    if node:
        from omegaconf import OmegaConf
        settings.update(OmegaConf.to_container(node, resolve=True))
    return settings


def configure_backends(tf32: bool = True, matmul_precision: str | None = "high",
                       cudnn_benchmark: bool = True) -> dict[str, Any]:
    """Enable TF32 and cuDNN autotuning (fixed input shapes) when CUDA is present."""
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return {"tf32": tf32 and torch.cuda.is_available(), "matmul_precision": matmul_precision,
            "cudnn_benchmark": cudnn_benchmark and torch.cuda.is_available()}


def resolve_precision(requested: str | int | None, accelerator: str | None = "auto") -> str:
    """Return a Lightning precision the device can run, degrading gracefully.

    ``bf16-mixed`` needs CUDA with native bf16 (Ampere+); otherwise it becomes
    ``16-mixed`` on older GPUs and ``32-true`` on CPU. Other values pass through.
    """
    text = str(requested or "32-true")
    on_cpu = accelerator == "cpu" or not torch.cuda.is_available()
    if text in ("bf16-mixed", "bf16"):
        if on_cpu:
            logger.warning("bf16-mixed requested without CUDA: training in 32-true.")
            return "32-true"
        major, _ = torch.cuda.get_device_capability()
        if major < 8:  # no bf16 tensor cores before Ampere: emulated, slow
            logger.warning("GPU capability %d.x has no native bf16: using 16-mixed.", major)
            return "16-mixed"
        return "bf16-mixed"
    if text in ("16-mixed", "16") and on_cpu:
        logger.warning("16-mixed requested without CUDA: training in 32-true.")
        return "32-true"
    return text


def autocast_dtype(precision: str) -> torch.dtype | None:
    return {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}.get(precision)


def is_convolutional(module: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv3d | nn.Conv2d) for m in module.modules()) and not hasattr(
        module, "forward_tokens")


# ---------------------------------------------------------------- compile
def _encoders(model: nn.Module) -> list[tuple[str, nn.Module, bool]]:
    """(attribute, module, trains with gradients) for every encoder of an SSL module."""
    found = []
    for name, trains in (("backbone", True), ("target_backbone", False),
                         ("teacher_backbone", False)):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            found.append((name, module, trains))
    return found


def _entry_point(module: nn.Module) -> str:
    """ViTs are driven through ``forward_tokens`` (DINOv2 masks); CNNs through ``forward``."""
    return "forward_tokens" if hasattr(module, "forward_tokens") else "forward"


def _probe_compile(module: nn.Module, method: str, mode: str, example: torch.Tensor,
                   trains: bool, precision: str) -> None:
    """Compile and run forward (+ backward) on a copy, so failures never touch the run."""
    probe = copy.deepcopy(module).to(example.device)
    probe.train(True)
    if not trains:
        probe.requires_grad_(False)
    fn = torch.compile(getattr(probe, method), mode=mode, dynamic=False)
    dtype = autocast_dtype(precision)
    autocast = (torch.autocast(device_type=example.device.type, dtype=dtype) if dtype
                else contextlib.nullcontext())
    grad = torch.enable_grad() if trains else torch.no_grad()
    with autocast, grad:
        out = fn(example)
    tensors = [t for t in (out if isinstance(out, tuple | list) else (out,))
               if isinstance(t, torch.Tensor)]
    if trains:
        sum(t.float().sum() for t in tensors).backward()
    if example.is_cuda:
        torch.cuda.synchronize(example.device)
    del probe, fn, out, tensors


def compile_encoders(model: nn.Module, modes: list[str], example: torch.Tensor | None,
                     precision: str = "32-true") -> str | None:
    """Compile every encoder of ``model`` with the first mode of ``modes`` that works.

    Each mode is probed on a deep copy (forward + backward on ``example``); the
    real modules are compiled only with a mode that passed, so a failure costs
    time but never corrupts BatchNorm statistics or weights. Returns the mode
    used, or ``None`` (eager) when no mode works or compilation is unavailable.
    Compiled entry points are instance attributes: ``state_dict`` keys are
    unchanged and :func:`uncompile` restores eager execution (for evaluation).
    """
    if not modes or not hasattr(torch, "compile"):
        return None
    encoders = _encoders(model)
    if not encoders:
        return None
    for mode in modes:
        if mode in ("none", "eager", None):
            return None
        try:
            if example is not None:
                for _, module, trains in encoders:
                    _probe_compile(module, _entry_point(module), mode, example, trains, precision)
            for _, module, _ in encoders:
                method = _entry_point(module)
                setattr(module, method, torch.compile(getattr(module, method), mode=mode,
                                                      dynamic=False))
            logger.info("torch.compile mode=%s on %s", mode, [n for n, _, _ in encoders])
            return mode
        except Exception as exc:
            logger.warning("torch.compile mode=%s failed (%s: %s); trying the next mode.",
                           mode, type(exc).__name__, str(exc).splitlines()[0][:200])
            with contextlib.suppress(Exception):
                torch._dynamo.reset()
            uncompile(model)
    logger.warning("No torch.compile mode worked; training eagerly.")
    return None


def uncompile(model: nn.Module) -> None:
    """Drop compiled entry points set by :func:`compile_encoders` (back to eager)."""
    for _, module, _ in _encoders(model):
        for method in ("forward", "forward_tokens"):
            if method in module.__dict__:
                delattr(module, method)


# ---------------------------------------------------------------- apply
@torch.no_grad()
def to_channels_last_3d(module: nn.Module) -> None:
    """Store 5-D (Conv3d) weights/buffers channels-last; leave every other tensor as is.

    ``Module.to(memory_format=...)`` also touches 4-D tensors (e.g. the inflated
    ConvNeXt layer scale), which cannot take the 3-D channels-last format.
    """
    for tensor in [*module.parameters(), *module.buffers()]:
        if tensor.dim() == 5:
            tensor.data = tensor.data.contiguous(memory_format=torch.channels_last_3d)


def apply_runtime(cfg: Any, model: nn.Module, precision: str,
                  example: torch.Tensor | None = None) -> dict[str, Any]:
    """Apply the ``perf`` settings of ``cfg`` to ``model``; return what was applied."""
    settings = perf_settings(cfg)
    report: dict[str, Any] = {"profile": settings.get("name"), "precision": precision,
                              "device": (torch.cuda.get_device_name(0)
                                         if torch.cuda.is_available() else "cpu")}
    if not settings["enabled"]:
        return report
    cuda = torch.cuda.is_available() and cfg.trainer.get("accelerator", "auto") != "cpu"
    report.update(configure_backends(settings["tf32"], settings["matmul_precision"],
                                     settings["cudnn_benchmark"]))
    if settings["fused_optimizer"] and cuda:
        model.fused_optimizer = True
    report["fused_optimizer"] = bool(getattr(model, "fused_optimizer", False))
    channels_last = bool(settings["channels_last"]) and cuda and is_convolutional(model.backbone)
    if channels_last:
        to_channels_last_3d(model)
        model.input_memory_format = torch.channels_last_3d
        if example is not None:
            example = example.contiguous(memory_format=torch.channels_last_3d)
    report["channels_last"] = channels_last
    report["compile_mode"] = None
    if settings["compile"] and cuda:
        if example is not None:
            example = example.cuda()
        report["compile_mode"] = compile_encoders(model, list(settings["compile_modes"]),
                                                  example, precision)
    return report


# ---------------------------------------------------------------- monitor
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PerfMonitor(L.Callback):
    """Epoch wall time, training images/s and peak VRAM, written to ``perf.json``.

    Synchronizes CUDA only at epoch/validation boundaries. Images are input
    volumes through the online encoder: batch size x number of views. The
    first epoch (compilation, worker start-up, cuDNN autotuning) is reported
    apart from the steady-state figures.
    """

    def __init__(self, out_path: str | Path, extra: dict[str, Any] | None = None) -> None:
        self.out_path = Path(out_path)
        self.extra = dict(extra or {})
        self.epochs: list[dict[str, float]] = []
        self._epoch_start = self._train_end = 0.0
        self._images = self._samples = 0
        self._fit_start = 0.0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._fit_start = time.perf_counter()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _sync()
        self._epoch_start = time.perf_counter()
        self._train_end = 0.0
        self._images = self._samples = 0

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        images = batch.get("image") if isinstance(batch, dict) else None
        views = images if isinstance(images, list | tuple) else [images]
        size = len(batch["subject"]) if isinstance(batch, dict) and "subject" in batch else (
            int(views[0].shape[0]) if isinstance(views[0], torch.Tensor) else 0)
        self._samples += size
        self._images += size * len(views)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        _sync()
        self._train_end = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _sync()
        end = time.perf_counter()
        train_seconds = (self._train_end or end) - self._epoch_start
        self.epochs.append({"epoch": trainer.current_epoch, "seconds": end - self._epoch_start,
                            "train_seconds": train_seconds, "samples": self._samples,
                            "images": self._images,
                            "images_per_second": self._images / max(train_seconds, 1e-9)})
        self.write(trainer)

    def summary(self) -> dict[str, Any]:
        steady = self.epochs[1:] or self.epochs
        out: dict[str, Any] = {**self.extra, "epochs_completed": len(self.epochs)}
        if self.epochs:
            out.update(
                first_epoch_seconds=self.epochs[0]["seconds"],
                seconds_per_epoch=statistics.median(e["seconds"] for e in steady),
                train_seconds_per_epoch=statistics.median(e["train_seconds"] for e in steady),
                images_per_second=(sum(e["images"] for e in steady)
                                   / max(sum(e["train_seconds"] for e in steady), 1e-9)),
                samples_per_second=(sum(e["samples"] for e in steady)
                                    / max(sum(e["train_seconds"] for e in steady), 1e-9)),
                fit_seconds=time.perf_counter() - self._fit_start)
        if torch.cuda.is_available():
            out.update(peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                       peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                       gpu=torch.cuda.get_device_name(0))
        out["per_epoch"] = self.epochs
        return out

    def write(self, trainer: L.Trainer | None = None) -> None:
        del trainer
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.write(trainer)


class StepTimer(L.Callback):
    """Steady-state throughput over ``steps`` optimizer steps after ``warmup`` (cloud/bench.py).

    One CUDA synchronize when the window opens and one when it closes; the gap
    between a batch's end and the next batch's start (host side) is reported as
    the data-wait share, a hint that CPU augmentation is the bottleneck.
    """

    def __init__(self, warmup: int, steps: int, out_path: str | Path,
                 extra: dict[str, Any] | None = None) -> None:
        self.warmup, self.steps = warmup, steps
        self.out_path, self.extra = Path(out_path), dict(extra or {})
        self._batches = 0
        self._start = self._last_end = 0.0
        self._wait = 0.0
        self._images = 0
        self.result: dict[str, Any] | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule,
                             batch: Any, batch_idx: int) -> None:
        now = time.perf_counter()
        if self._batches == self.warmup:
            _sync()
            self._start = self._last_end = time.perf_counter()
        elif self._batches > self.warmup and self._last_end:
            self._wait += now - self._last_end

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        self._batches += 1
        if self._batches <= self.warmup:
            return
        images = batch["image"]
        views = images if isinstance(images, list | tuple) else [images]
        self._images += int(views[0].shape[0]) * len(views)
        self._last_end = time.perf_counter()
        if self._batches == self.warmup + self.steps:
            _sync()
            seconds = time.perf_counter() - self._start
            self.result = {**self.extra, "steps": self.steps, "warmup_steps": self.warmup,
                           "seconds": seconds, "ms_per_step": 1000 * seconds / self.steps,
                           "images_per_second": self._images / seconds,
                           "data_wait_fraction": self._wait / seconds}
            if torch.cuda.is_available():
                self.result.update(peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                                   peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                                   gpu=torch.cuda.get_device_name(0))
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(json.dumps(self.result, indent=2) + "\n", encoding="utf-8")
            trainer.should_stop = True
