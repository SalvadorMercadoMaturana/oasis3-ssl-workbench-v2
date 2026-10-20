"""Track representation collapse during SSL training on the held-out ROI subjects.

Every ``every_n_epochs`` (and before the first epoch) the encoders of the SSL
module — the online backbone, and BYOL's EMA target or DINOv2's teacher — embed
the val + test sessions (one per subject, 142 here): the clean central cube and
two fixed augmented views per subject drawn once at the start with the run's
own recipe (TorchIO's elastic stands in for the GPU elastic). Written to
``<exp_dir>/collapse.json`` after every measurement:

- ``effective_rank``: exp(entropy) of the variance spectrum of the clean
  embeddings (1 = everything on one axis);
- ``mean_feature_std``: average per-dimension standard deviation (→ 0 = constant);
- ``between_over_within``: Var_subject (clean) / Var_aug (between the two views);
- ``same_subject_cos`` / ``different_subject_cos``: centered cosine between the
  two views of the same / different subjects;
- ``view_matching_top1``: fraction of first views whose nearest second view is the
  same subject (chance = 1 / subjects).

For BYOL it also records the objective itself (symmetric ``2 - 2 cos`` between
the online prediction of one view and the EMA target of the other, every module
in eval mode) on the held-out subjects and on as many training subjects with the
same kind of fixed views (``objective`` rows): a held-out loss that rises while
the training one falls is the objective overfitting the training subjects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

ROLES = (("backbone", "online"), ("target_backbone", "target"), ("teacher_backbone", "teacher"))


def view_transforms(cfg: Any) -> Any:
    """Return the run's own training views (GPU-elastic recipes → TorchIO's elastic)."""
    from omegaconf import OmegaConf

    from t5_ssl import recipes
    from t5_ssl.data.roi3d import analysis_views, build_roi3d_transforms
    if cfg.augmentation.get("tio_gpu_elastic"):
        return analysis_views(cfg)
    if cfg.augmentation.get("gpu_elastic"):
        size = int(cfg.data.roi_size)
        cfg = OmegaConf.create({"data": {"roi_size": size},
                                "augmentation": recipes.aug_spec("elastic", size)})
    return build_roi3d_transforms(cfg)[0]


def collapse_scores(clean: np.ndarray, view_a: np.ndarray, view_b: np.ndarray) -> dict:
    centered = clean - clean.mean(0)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    p = spectrum[spectrum > 0] / spectrum.sum()
    mu = np.concatenate([view_a, view_b]).mean(0)
    a = F.normalize(torch.from_numpy(view_a - mu), dim=1)
    b = F.normalize(torch.from_numpy(view_b - mu), dim=1)
    sim = (a @ b.T).numpy()
    n = len(sim)
    within = float(((view_a - view_b) ** 2).sum(1).mean() / 2)
    between = float((centered ** 2).sum(1).mean())
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "mean_feature_std": float(centered.std(0).mean()),
            "between_over_within": between / max(within, 1e-12),
            "same_subject_cos": float(np.diag(sim).mean()),
            "different_subject_cos": float(sim[~np.eye(n, dtype=bool)].mean()),
            "view_matching_top1": float((sim.argmax(1) == np.arange(n)).mean()),
            "chance": 1 / n}


class CollapseMonitor(L.Callback):
    """Collapse metrics of every encoder on fixed held-out views, every few epochs."""

    def __init__(self, cfg: Any, every_n_epochs: int = 2, seed: int = 0) -> None:
        self.cfg, self.every, self.seed = cfg, max(1, every_n_epochs), seed
        self.history: list[dict] = []
        self.inputs: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        from t5_ssl.data.roi3d import ROIEval3D
        dm = trainer.datamodule
        records = dm.val_dataset.records + dm.test_dataset.records
        seen, cubes = set(), []
        for record in records:  # one session per subject
            if record["subject"] not in seen:
                seen.add(record["subject"])
                cubes.append(torch.from_numpy(np.asarray(dm.cubes[record["idx"]],
                                                         dtype=np.float32)))
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        random.seed(self.seed), np.random.seed(self.seed), torch.manual_seed(self.seed)
        transform = view_transforms(self.cfg)
        pairs = [transform(cube)[:2] for cube in cubes]
        random.setstate(state[0]), np.random.set_state(state[1]), torch.set_rng_state(state[2])
        clean = ROIEval3D(int(self.cfg.data.roi_size))
        self.inputs = {"clean": torch.stack([clean(c) for c in cubes]),
                       "a": torch.stack([p[0] for p in pairs]),
                       "b": torch.stack([p[1] for p in pairs])}
        self.train_views: dict[str, torch.Tensor] = {}
        if hasattr(pl_module, "predictor") and hasattr(pl_module, "target_backbone"):
            by_subject: dict[str, str] = {}
            for record in dm.train_dataset.records:
                by_subject.setdefault(record["subject"], record["idx"])
            subjects = sorted(by_subject)
            random.Random(self.seed).shuffle(subjects)
            train_cubes = [torch.from_numpy(np.asarray(dm.cubes[by_subject[s]], dtype=np.float32))
                           for s in subjects[:len(cubes)]]
            random.seed(self.seed + 1), np.random.seed(self.seed + 1), torch.manual_seed(self.seed + 1)
            train_pairs = [transform(cube)[:2] for cube in train_cubes]
            random.setstate(state[0]), np.random.set_state(state[1]), torch.set_rng_state(state[2])
            self.train_views = {"a": torch.stack([p[0] for p in train_pairs]),
                                "b": torch.stack([p[1] for p in train_pairs])}
        self.out_path = Path(trainer.log_dir or ".") / "collapse.json"

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._measure(trainer, pl_module, epoch=0)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        done = trainer.current_epoch + 1
        if done % self.every == 0 or done == trainer.max_epochs:
            self._measure(trainer, pl_module, epoch=done)

    @torch.no_grad()
    def _embed(self, encoder: torch.nn.Module, x: torch.Tensor, device: torch.device,
               memory_format: torch.memory_format | None) -> np.ndarray:
        out = []
        for i in range(0, len(x), 32):
            batch = x[i:i + 32].to(device, non_blocking=True)
            if memory_format is not None:
                batch = batch.contiguous(memory_format=memory_format)
            out.append(encoder(batch).float().cpu())
        return torch.cat(out).numpy()

    def _measure(self, trainer: L.Trainer, pl_module: L.LightningModule, epoch: int) -> None:
        if not self.inputs:
            return
        device = pl_module.device
        memory_format = getattr(pl_module, "input_memory_format", None)
        for attr, role in ROLES:
            encoder = getattr(pl_module, attr, None)
            if encoder is None:
                continue
            was_training = encoder.training
            encoder.eval()
            with torch.autocast(device_type=device.type, enabled=False):
                emb = {k: self._embed(encoder, v, device, memory_format)
                       for k, v in self.inputs.items()}
            encoder.train(was_training)
            self.history.append({"epoch": epoch, "role": role,
                                 **collapse_scores(emb["clean"], emb["a"], emb["b"])})
        if self.train_views:
            self.history.append({"epoch": epoch, "role": "objective",
                                 "byol_loss_heldout": self._byol_loss(pl_module, self.inputs,
                                                                      device, memory_format),
                                 "byol_loss_train": self._byol_loss(pl_module, self.train_views,
                                                                    device, memory_format)})
        self.out_path.write_text(json.dumps(self.history, indent=1) + "\n", encoding="utf-8")
        if device.type == "cuda":  # fp32 cuDNN autotuning here reserves GBs; hand them back
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _byol_loss(self, pl_module: L.LightningModule, views: dict[str, torch.Tensor],
                   device: torch.device, memory_format: torch.memory_format | None) -> float:
        modules = [pl_module.backbone, pl_module.projector, pl_module.predictor,
                   pl_module.target_backbone, pl_module.target_projector]
        modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        total, count = 0.0, 0
        with torch.autocast(device_type=device.type, enabled=False):
            for i in range(0, len(views["a"]), 32):
                a = views["a"][i:i + 32].to(device)
                b = views["b"][i:i + 32].to(device)
                if memory_format is not None:
                    a, b = (t.contiguous(memory_format=memory_format) for t in (a, b))
                pa = F.normalize(pl_module.predictor(pl_module.projector(pl_module.backbone(a))), dim=-1)
                pb = F.normalize(pl_module.predictor(pl_module.projector(pl_module.backbone(b))), dim=-1)
                ta = F.normalize(pl_module.target_projector(pl_module.target_backbone(a)), dim=-1)
                tb = F.normalize(pl_module.target_projector(pl_module.target_backbone(b)), dim=-1)
                loss = (2 - 2 * (pa * tb).sum(-1) + 2 - 2 * (pb * ta).sum(-1)) / 2
                total += float(loss.sum())
                count += len(loss)
        for m, mode in zip(modules, modes, strict=True):
            m.train(mode)
        return total / max(count, 1)
