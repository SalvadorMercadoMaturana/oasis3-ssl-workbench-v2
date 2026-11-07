"""Main training script for Self-Supervised Learning with Hydra configuration."""

from __future__ import annotations

import contextlib
import csv
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from t5_ssl.callbacks.representation_health import RepresentationHealthCallback
from t5_ssl.data.augmentations import (
    DeterministicSSLViews,
    DINOv2ViewTransform,
    EvalTransform,
    SSLViewTransform,
)
from t5_ssl.data.oasis3_datamodule import OASIS3DataModule
from t5_ssl.data.oasis3_lmdb_datamodule import (
    OASIS3LMDBDataModule,
    OASIS3LMDBSliceDataset,
    clinical_rows,
)
from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_IMAGE_ROOT, DEFAULT_LMDB
from t5_ssl.losses.byol import BYOLLoss
from t5_ssl.losses.dinov2 import DINOv2Loss
from t5_ssl.losses.mae import SimMIMLoss
from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.backbones.convnext import ConvNeXtBackbone
from t5_ssl.models.backbones.resnet import ResNetBackbone
from t5_ssl.models.backbones.vit import ViTBackbone
from t5_ssl.models.byol_module import BYOLLightningModule
from t5_ssl.models.dinov2_module import DINOv2LightningModule
from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.heads.projector import MLPProjector
from t5_ssl.models.mae_module import MAELightningModule, MAEVICRegLightningModule
from t5_ssl.models.ssl_module import SSLLightningModule
from t5_ssl.models.vicreg_module import VICRegLightningModule
from t5_ssl.utils.logging import get_logger
from t5_ssl.utils.runtime import PerfMonitor, apply_runtime, perf_settings, resolve_precision

logger = get_logger(__name__)


def get_experiment_dir_name(cfg: DictConfig) -> str:
    """Generate a descriptive, concise experiment directory name.

    Format: [loss-name]_[short-params]_[backbone]_[timestamp]
    Example: vicreg_s25-v25-c1_convnext_tiny_2026-08-24_09-45-00
    """
    loss_name = str(cfg.loss.get("name", "ssl")).lower()

    params_parts: list[str] = []
    if loss_name == "vicreg":
        s = cfg.loss.get("sim_coeff", 25.0)
        v = cfg.loss.get("var_coeff", 25.0)
        c = cfg.loss.get("cov_coeff", 1.0)
        params_parts.append(f"s{s:g}-v{v:g}-c{c:g}")
    elif loss_name == "byol":
        params_parts.append(f"ema{cfg.loss.get('ema_base', 0.996):g}")
    elif loss_name == "dinov2":
        low, high = cfg.loss.get("mask_ratio_min_max", (0.1, 0.5))
        params_parts.append(f"mask{cfg.loss.get('mask_probability', 0.5):g}x{low:g}-{high:g}")
    elif loss_name in ("mae", "mae_vicreg"):
        params_parts.append(f"mask{cfg.loss.get('mask_ratio', 0.6):g}")

    # Model backbone variant
    backbone_variant = str(cfg.model.backbone.get("variant", "convnext_tiny"))
    params_parts.append(backbone_variant)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_params = "_".join(params_parts)
    return f"{loss_name}_{short_params}_{timestamp}"


def _sanitize_environment(accelerator: str | None = None) -> None:
    """Configure warnings, suppress third-party noise, and safely handle GPU drivers."""
    # Suppress verbose HTTP logs from Hugging Face and httpx
    for noisy_logger in ["httpx", "urllib3", "huggingface_hub", "datasets", "lightning"]:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    # Filter third-party deprecation / benign system warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning\..*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*Total length of `DataLoader`.*")
    warnings.filterwarnings("ignore", message=".*tensorboardX.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
    warnings.filterwarnings("ignore", message=".*litlogger.*")

    if accelerator == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # PyTorch's Windows CPU wheel sizes its native intra-op thread pool from
        # hardware_concurrency() and does not reliably pick up OMP_NUM_THREADS
        # from the environment; without this, N parallel processes (e.g. three
        # simultaneous campaign runs) each grab one thread per physical core and
        # thrash each other instead of sharing them, stalling all of them.
        try:
            import torch

            requested = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
            if requested:
                torch.set_num_threads(max(1, int(requested)))
                torch.set_num_interop_threads(1)
        except Exception as exc:
            logger.warning("Could not pin CPU thread count (%s).", exc)
        return

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception as exc:
        logger.warning("CUDA driver initialization failed (%s). Falling back to CPU.", exc)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def build_backbone(cfg: DictConfig) -> ConvNeXtBackbone | ResNetBackbone | ViTBackbone:
    """Instantiate the backbone encoder based on configuration."""
    variant = cfg.model.backbone.get("variant", "convnext_tiny")
    in_channels = cfg.model.backbone.get("in_channels", 1)
    pretrained = cfg.model.backbone.get("pretrained", False)

    logger.info("Initializing backbone: %s (in_channels=%d)", variant, in_channels)
    if variant == "vit_small_3d":
        from t5_ssl.models.backbones.vit3d import ViT3DBackbone
        return ViT3DBackbone(
            in_channels=in_channels, image_size=cfg.model.backbone.get("image_size", 32),
            patch_size=cfg.model.backbone.get("patch_size", 8),
            embed_dim=cfg.model.backbone.get("embed_dim", 384),
            depth=cfg.model.backbone.get("depth", 12),
            num_heads=cfg.model.backbone.get("num_heads", 6),
            mlp_ratio=cfg.model.backbone.get("mlp_ratio", 4.0),
            pool=cfg.model.backbone.get("pool", "cls"),
            pretrained=pretrained or None,
            drop_path=cfg.model.backbone.get("drop_path", 0.0),
            layer_scale=cfg.model.backbone.get("layer_scale", None),
        )
    if variant.endswith("_3d"):
        from t5_ssl.models.backbones.resnet3d import ConvNeXt3DBackbone, ResNet3DBackbone
        backbone_3d = ConvNeXt3DBackbone if variant.startswith("convnext") else ResNet3DBackbone
        return backbone_3d(variant=variant, in_channels=in_channels, pretrained=pretrained)
    if variant == "vit_small":
        return ViTBackbone(
            in_channels=in_channels,
            image_size=cfg.model.backbone.get("image_size", 64),
            patch_size=cfg.model.backbone.get("patch_size", 8),
            embed_dim=cfg.model.backbone.get("embed_dim", 192),
            depth=cfg.model.backbone.get("depth", 4),
            num_heads=cfg.model.backbone.get("num_heads", 3),
            pool=cfg.model.backbone.get("pool", "cls"),
        )
    if variant.startswith("resnet"):
        return ResNetBackbone(
            variant=variant,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    return ConvNeXtBackbone(
        variant=variant,
        in_channels=in_channels,
        pretrained=pretrained,
    )


def build_loss(cfg: DictConfig) -> VICRegLoss | BYOLLoss | DINOv2Loss | SimMIMLoss:
    """Instantiate the SSL loss criterion based on configuration.

    ``mae_vicreg`` combines two losses and is constructed directly in
    :func:`train` instead, since this function returns a single criterion.
    """
    loss_name = cfg.loss.get("name", "vicreg").lower()
    if loss_name == "vicreg":
        return VICRegLoss(
            sim_coeff=cfg.loss.get("sim_coeff", 25.0),
            var_coeff=cfg.loss.get("var_coeff", 25.0),
            cov_coeff=cfg.loss.get("cov_coeff", 1.0),
            gamma=cfg.loss.get("gamma", 1.0),
            eps=cfg.loss.get("eps", 1e-4),
        )
    if loss_name == "byol":
        return BYOLLoss()
    if loss_name == "dinov2":
        return DINOv2Loss(
            out_dim=cfg.model.projector.get("out_dim", 4096),
            student_temp=cfg.loss.get("student_temp", 0.1),
            teacher_temp=cfg.loss.get("teacher_temp", 0.04),
            center_momentum=cfg.loss.get("center_momentum", 0.9),
            ibot_weight=cfg.loss.get("ibot_weight", 1.0),
            koleo_weight=cfg.loss.get("koleo_weight", 0.1),
            centering=cfg.loss.get("centering", "sinkhorn_knopp"),
            sinkhorn_iterations=cfg.loss.get("sinkhorn_iterations", 3),
        )
    if loss_name == "mae":
        return SimMIMLoss()
    msg = f"Unsupported SSL loss '{loss_name}'."
    raise ValueError(msg)


def build_datamodule(cfg: DictConfig, train_transform: Any,
                     eval_transform: Any) -> OASIS3DataModule | OASIS3LMDBDataModule:
    """Select the local OASIS3 data source while preserving one batch contract."""
    data_cfg = cfg.data
    if int(cfg.model.backbone.in_channels) != int(data_cfg.get("channels", 1)):
        raise ValueError("model.backbone.in_channels must match data.channels.")
    if str(data_cfg.get("source", "")) == "roi3d":
        from t5_ssl.data.roi3d import ROI3DDataModule
        # augmentation.val_views=train: the SSL validation loss sees the training
        # augmentations (fixed per session), not the clean cube and its mirror;
        # downstream evaluation keeps its own deterministic transform.
        val_views = str(cfg.augmentation.get("val_views", "fixed"))
        if val_views not in ("fixed", "train"):
            raise ValueError(f"augmentation.val_views must be 'fixed' or 'train', got {val_views}")
        augmented_val = val_views == "train"
        return ROI3DDataModule(
            csv_path=data_cfg.get("csv_path", DEFAULT_CSV), cache_path=data_cfg.cache_path,
            manifest_path=data_cfg.manifest_path, batch_size=data_cfg.get("batch_size", 16),
            num_workers=data_cfg.get("num_workers", 0), pin_memory=data_cfg.get("pin_memory", False),
            one_session_per_subject_train=data_cfg.get("one_session_per_subject_train", True),
            seed=data_cfg.get("seed", 42), train_transform=train_transform,
            eval_transform=eval_transform, prefetch_factor=data_cfg.get("prefetch_factor", None),
            val_transform=train_transform if augmented_val else None,
            val_view_seed=int(cfg.augmentation.get("val_view_seed", 12345)) if augmented_val else None,
            train_monitor_subjects=int(data_cfg.get("train_monitor_subjects", 0)) if augmented_val else 0)
    common = dict(
        csv_path=data_cfg.get("csv_path", DEFAULT_CSV),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        channels=data_cfg.get("channels", 1),
        slices_per_volume=data_cfg.get("slices_per_volume", 1),
        slice_range=tuple(data_cfg.get("slice_range", (0.45, 0.55))),
        train_ratio=data_cfg.get("train_ratio", 0.70),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        seed=data_cfg.get("seed", 42),
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    source = str(data_cfg.get("source", "nifti"))
    if source == "lmdb":
        return OASIS3LMDBDataModule(
            lmdb_path=data_cfg.get("lmdb_path", DEFAULT_LMDB),
            manifest_path=data_cfg.get("manifest_path", None),
            primary_only=data_cfg.get("primary_only", False),
            plane=data_cfg.get("plane", "axial"),
            one_session_per_subject_train=data_cfg.get("one_session_per_subject_train", False),
            views_per_subject_epoch=data_cfg.get("views_per_subject_epoch", 1),
            train_cache_capacity=data_cfg.get("train_cache_capacity", 128),
            eval_cache_capacity=data_cfg.get("eval_cache_capacity", 32),
            balanced_sampler=data_cfg.get("balanced_sampler", False), **common)
    if source == "nifti":
        return OASIS3DataModule(
            image_root=data_cfg.get("image_root", DEFAULT_IMAGE_ROOT),
            max_volumes=data_cfg.get("max_volumes", None), **common)
    raise ValueError(f"Unknown OASIS3 data source: {source}")


def _build_health_callback(
    cfg: DictConfig, dm: OASIS3LMDBDataModule, exp_dir: Path,
) -> RepresentationHealthCallback:
    """Assemble the representation-health callback from the val split already in `dm`."""
    assert dm.val_dataset is not None
    records = dm.val_dataset.records
    ages: dict[str, float] = {}
    sexes: dict[str, str] = {}
    groups: dict[str, str] = {}
    for record in records:
        with contextlib.suppress(KeyError, ValueError, TypeError):
            ages.setdefault(record["subject"], float(record["age"]))
        sex = record.get("sex", "")
        if sex:
            sexes.setdefault(record["subject"], sex)
        groups.setdefault(record["subject"], record["group"])

    image_size = tuple(cfg.augmentation.get("image_size", (96, 96)))
    slice_range = tuple(cfg.data.get("slice_range", (0.45, 0.55)))
    center = (slice_range[0] + slice_range[1]) / 2
    neighbor_dataset, neighbor_group_ids = None, None
    if dm.slices_per_volume == 1:
        band = max(0.02, 1.5 / 97)
        neighbor_dataset = OASIS3LMDBSliceDataset(
            records, dm.lmdb_path, channels=dm.channels, plane=dm.plane,
            slices_per_volume=2, slice_range=(max(0.0, center - band), min(1.0, center + band)),
            transform=EvalTransform(image_size), cache_capacity=dm.eval_cache_capacity,
        )
        neighbor_group_ids = [str(i // 2) for i in range(len(neighbor_dataset))]

    # The manifest's val split usually keeps one (primary) session per subject,
    # so `dm.val_dataset` alone cannot exercise same-subject session retrieval.
    # Pull in every other real session those same val subjects have, without
    # touching train/test: this only adds more of the val subjects' own images.
    val_subjects = set(dm.split_subject_ids.get("val", []))
    session_dataset = None
    if val_subjects:
        session_rows = [row for row in clinical_rows(dm.csv_path) if row["subject"] in val_subjects]
        if len({row["session"] for row in session_rows}) > len(val_subjects):
            session_dataset = OASIS3LMDBSliceDataset(
                session_rows, dm.lmdb_path, channels=dm.channels, plane=dm.plane,
                slice_range=slice_range, transform=EvalTransform(image_size),
                cache_capacity=dm.eval_cache_capacity,
            )

    return RepresentationHealthCallback(
        dm.val_dataset, exp_dir, every_n_epochs=cfg.trainer.get("health_check_every_n_epochs", 1),
        batch_size=cfg.data.get("batch_size", 16),
        neighbor_dataset=neighbor_dataset, neighbor_group_ids=neighbor_group_ids,
        session_dataset=session_dataset,
        ages=ages, sexes=sexes, groups=groups,
    )


def train(cfg: DictConfig, extra_callbacks: list[L.Callback] | None = None) -> dict[str, Any]:
    """Execute the full SSL training pipeline given a composed Hydra DictConfig."""
    # Sanitize environment and warnings before Lightning or PyTorch initialization
    _sanitize_environment(accelerator=cfg.trainer.get("accelerator"))

    logger.info("Starting T5-SSL Training with configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set deterministic random seed
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    # 1. Setup unified experiment directory: logs/[loss-params-date-time]/
    # Runs started in the same second get the same timestamped name; claim the
    # directory atomically and suffix it instead of mixing two runs' files.
    # `log_root` / `run_name` (optional) give orchestrated runs fixed locations.
    exp_name = cfg.get("run_name") or get_experiment_dir_name(cfg)
    log_root = Path(cfg.get("log_root") or "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    exp_dir = log_root / exp_name
    suffix = 1
    while True:
        try:
            exp_dir.mkdir()
            break
        except FileExistsError:
            exp_dir = log_root / f"{exp_name}-{suffix}"
            suffix += 1

    # Save resolved config into experiment directory
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 2. Augmentations
    aug_cfg = cfg.augmentation
    image_size = tuple(aug_cfg.get("image_size", (224, 224)))
    loss_name = str(cfg.loss.name).lower()
    transform_kwargs = dict(
        image_size=image_size,
        crop_scale=tuple(aug_cfg.get("crop_scale", (0.3, 1.0))),
        crop_ratio=tuple(aug_cfg.get("crop_ratio", (0.9, 1.1))),
        horizontal_flip_p=aug_cfg.get("horizontal_flip_p", 0.5),
        vertical_flip_p=aug_cfg.get("vertical_flip_p", 0.5),
        affine_degrees=aug_cfg.get("affine_degrees", 15.0),
        translate_frac=tuple(aug_cfg.get("translate_frac", (0.0, 0.0))),
        blur_p=aug_cfg.get("blur_p", 0.5),
        noise_std=aug_cfg.get("noise_std", 0.03),
        intensity_scale=tuple(aug_cfg.get("intensity_scale", (0.9, 1.1))),
        gamma_range=tuple(aug_cfg.get("gamma_range", (1.0, 1.0))),
        bias_field_p=aug_cfg.get("bias_field_p", 0.0),
        bias_field_alpha=aug_cfg.get("bias_field_alpha", 0.15),
        torchio_transforms=(OmegaConf.to_container(aug_cfg.torchio, resolve=True)
                            if aug_cfg.get("torchio") else None),
    )
    eval_view_kwargs = dict(
        horizontal_flip=aug_cfg.get("horizontal_flip_p", 0.5) > 0,
        second_view_angle=min(float(aug_cfg.get("affine_degrees", 15.0)), 5.0),
    )
    if str(cfg.data.get("source", "")) == "roi3d":
        from t5_ssl.data.roi3d import build_roi3d_transforms
        train_transform, eval_transform = build_roi3d_transforms(cfg)
    elif loss_name == "dinov2":
        local_size = tuple(aug_cfg.get("local_size", (32, 32)))
        n_local = aug_cfg.get("n_local_crops", 2)
        train_transform = DINOv2ViewTransform(
            **transform_kwargs, local_size=local_size,
            local_crop_scale=tuple(aug_cfg.get("local_crop_scale", (0.1, 0.3))),
            n_local_crops=n_local,
        )
        eval_transform = DeterministicSSLViews(
            image_size=image_size, local_size=local_size, n_local_crops=n_local,
            **eval_view_kwargs,
        )
    elif loss_name == "byol":
        train_transform = SSLViewTransform(**transform_kwargs)
        eval_transform = DeterministicSSLViews(image_size=image_size, **eval_view_kwargs)
    else:
        train_transform = SSLViewTransform(**transform_kwargs)
        eval_transform = DeterministicSSLViews(image_size=image_size, **eval_view_kwargs)

    # 3. DataModule
    dm = build_datamodule(cfg, train_transform, eval_transform)

    # 4. Backbone, Projector, and Loss
    backbone = build_backbone(cfg)

    proj_cfg = cfg.model.projector

    def dino_head() -> DINOHead:
        # DINO/DINOv2 head: MLP -> L2-normalized bottleneck -> cosine prototypes.
        return DINOHead(
            in_dim=backbone.out_dim,
            out_dim=proj_cfg.get("out_dim", 4096),
            hidden_dim=proj_cfg.get("hidden_dim", 2048),
            bottleneck_dim=proj_cfg.get("bottleneck_dim", 256),
            num_layers=proj_cfg.get("num_layers", 3),
            use_bn=proj_cfg.get("use_bn", False),
        )

    if loss_name == "dinov2":
        projector: nn.Module = dino_head()
    else:
        projector = MLPProjector(
            in_dim=backbone.out_dim,
            hidden_dim=proj_cfg.get("hidden_dim", 2048),
            out_dim=proj_cfg.get("out_dim", 2048),
            num_layers=proj_cfg.get("num_layers", 3),
            use_bn=proj_cfg.get("use_bn", True),
            activation=proj_cfg.get("activation", "relu"),
            norm=proj_cfg.get("norm", None),
            dropout=proj_cfg.get("dropout", 0.0),
        )

    loss_fn = build_loss(cfg) if loss_name != "mae_vicreg" else None

    # 5. Lightning Module
    opt_cfg = cfg.optimizer
    trainer_cfg = cfg.trainer
    max_epochs = trainer_cfg.get("max_epochs", 50)
    # The cosine LR decay horizon defaults to the trainer's epoch ceiling
    # (unchanged behavior for every existing config), but can be set
    # separately so a generous trainer.max_epochs used only as an
    # early-stopping safety net doesn't stretch/slow the whole LR decay
    # curve. Concretely: raising trainer.max_epochs from 30 to 100 without
    # this would keep the LR near its peak for far longer at every epoch
    # number, silently changing training dynamics rather than just "letting
    # it run longer if it wants to."
    schedule_epochs = opt_cfg.get("schedule_epochs", None) or max_epochs

    module_kwargs = dict(
        backbone=backbone, projector=projector, loss_fn=loss_fn,
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-6),
        max_epochs=schedule_epochs,
        warmup_epochs=opt_cfg.get("warmup_epochs", 5),
        min_lr=opt_cfg.get("min_lr", 1e-6),
    )
    if loss_name == "byol":
        pred_cfg = cfg.model.predictor
        predictor = MLPProjector(
            in_dim=proj_cfg.out_dim,
            hidden_dim=pred_cfg.get("hidden_dim", 256),
            out_dim=proj_cfg.out_dim,
            num_layers=pred_cfg.get("num_layers", 2),
            use_bn=pred_cfg.get("use_bn", True),
            norm=pred_cfg.get("norm", None),
            dropout=pred_cfg.get("dropout", 0.0),
        )
        model = BYOLLightningModule(
            **module_kwargs, predictor=predictor,
            ema_base=cfg.loss.get("ema_base", 0.996),
            predictor_lr_mult=opt_cfg.get("predictor_lr_mult", 1.0),
        )
    elif loss_name == "dinov2":
        # Separate (untied) iBOT head for the masked patch tokens, as in DINOv2.
        model = DINOv2LightningModule(
            **module_kwargs, patch_head=dino_head(),
            mask_probability=cfg.loss.get("mask_probability", 0.5),
            mask_ratio_min_max=tuple(cfg.loss.get("mask_ratio_min_max", (0.1, 0.5))),
            min_mask_block=cfg.loss.get("min_mask_block", 4),
            ema_base=cfg.loss.get("ema_base", 0.992),
            teacher_temp_final=cfg.loss.get("teacher_temp_final", None),
            teacher_temp_warmup_epochs=cfg.loss.get("teacher_temp_warmup_epochs", 30),
            freeze_last_layer_epochs=cfg.loss.get("freeze_last_layer_epochs", 1),
            weight_decay_end=opt_cfg.get("weight_decay_end", None),
        )
    elif loss_name == "vicreg":
        model = VICRegLightningModule(**module_kwargs, queue_size=cfg.loss.get("queue_size", 0))
    elif loss_name == "mae":
        patch_size = int(cfg.model.backbone.get("patch_size", 8))
        in_channels = int(cfg.model.backbone.get("in_channels", 1))
        recon_head = nn.Linear(backbone.out_dim, in_channels * patch_size**2)
        module_kwargs["projector"] = nn.Identity()
        model = MAELightningModule(
            **module_kwargs, recon_head=recon_head,
            mask_ratio=cfg.loss.get("mask_ratio", 0.6),
        )
    elif loss_name == "mae_vicreg":
        patch_size = int(cfg.model.backbone.get("patch_size", 8))
        in_channels = int(cfg.model.backbone.get("in_channels", 1))
        recon_head = nn.Linear(backbone.out_dim, in_channels * patch_size**2)
        vicreg_loss_fn = VICRegLoss(
            sim_coeff=cfg.loss.get("sim_coeff", 25.0),
            var_coeff=cfg.loss.get("var_coeff", 25.0),
            cov_coeff=cfg.loss.get("cov_coeff", 1.0),
            gamma=cfg.loss.get("gamma", 1.0),
            eps=cfg.loss.get("eps", 1e-4),
        )
        module_kwargs.pop("loss_fn")
        model = MAEVICRegLightningModule(
            **module_kwargs, recon_head=recon_head,
            mae_loss_fn=SimMIMLoss(), vicreg_loss_fn=vicreg_loss_fn,
            mask_ratio=cfg.loss.get("mask_ratio", 0.6),
            mae_weight=cfg.loss.get("mae_weight", 1.0),
            vicreg_weight=cfg.loss.get("vicreg_weight", 1.0),
            queue_size=cfg.loss.get("queue_size", 0),
        )
    else:
        model = SSLLightningModule(**module_kwargs)

    if aug_cfg.get("gpu_elastic"):
        model.gpu_elastic = dict(OmegaConf.to_container(aug_cfg.gpu_elastic, resolve=True))
    if aug_cfg.get("tio_gpu_elastic"):
        model.tio_gpu_elastic = dict(OmegaConf.to_container(aug_cfg.tio_gpu_elastic, resolve=True))
        model.crop_size = int(cfg.data.roi_size)
    if str(aug_cfg.get("val_views", "fixed")) == "train":  # fixed GPU deformations in validation
        model.augment_val = True
        model.val_view_seed = int(aug_cfg.get("val_view_seed", 12345))
    if cfg.model.get("freeze_backbone_bn", False):
        model.freeze_backbone_bn = True
        model.bn_calibration_batches = cfg.model.get("bn_calibration_batches", None)

    # 6. Logger & Callbacks inside the same experiment directory
    csv_logger = CSVLogger(
        save_dir=str(exp_dir.parent),
        name=exp_dir.name,
        version="",
    )

    callbacks: list[L.Callback] = list(extra_callbacks or [])
    patience = trainer_cfg.get("early_stopping_patience", None)
    if patience is not None:
        callbacks.append(EarlyStopping(
            monitor="val/loss", mode="min", patience=int(patience),
            min_delta=float(trainer_cfg.get("early_stopping_min_delta", 0.0)),
        ))
    # Optimizer state is 1/2-2/3 of every .ckpt and only matters for resuming
    # with trainer.ckpt_path; the probes and finalize scripts read just
    # "state_dict" and "epoch". Set trainer.checkpoint_weights_only=false to
    # keep resumable full checkpoints.
    weights_only = bool(trainer_cfg.get("checkpoint_weights_only", True))
    if trainer_cfg.get("enable_checkpointing", True):
        # Kept for scripts that still glob "best-loss-epoch=*.ckpt"
        # (evaluate_frozen_probe.py, finalize_full50.py, report_pretext_tuning.py).
        # It is NOT the recommended way to pick the final representation for
        # these three anti-collapse-aware runs: a falling SSL loss can coincide
        # with collapse, so treat this only as one reference point among
        # several, and pick the epoch to actually use from the periodic
        # snapshots below plus representation_health.json (effective_rank /
        # mean_feature_std / mean_cosine_similarity_diff_samples) and the loss
        # curves together — never from val/loss alone.
        callbacks.append(ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="best-loss-epoch={epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            save_last=False,
            save_weights_only=weights_only,
        ))
        # Periodic snapshots so any pre-collapse epoch can be recovered even
        # when it is not the val/loss optimum above. `checkpoint_keep_last: k`
        # rotates them (only the k most recent stay on disk); unset keeps all.
        keep_last = trainer_cfg.get("checkpoint_keep_last", None)
        rotation = ({"save_top_k": int(keep_last), "monitor": "step", "mode": "max"}
                    if keep_last else {"save_top_k": -1})
        callbacks.append(ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="ckpt-epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=int(trainer_cfg.get("checkpoint_every_n_epochs", 5)),
            save_last=True,
            save_weights_only=weights_only,
            **rotation,
        ))

    # 7. Trainer with adaptive log interval
    dm.prepare_data()
    dm.setup(stage="fit")
    with (exp_dir / "subject_splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "subject"])
        for split, ids in dm.split_subject_ids.items():
            writer.writerows((split, subject) for subject in ids)
    with (exp_dir / "data_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "subject", "session", "idx", "path", "group"])
        for split in ("train", "val", "test"):
            dataset = getattr(dm, f"{split}_dataset")
            for record in dataset.records:
                writer.writerow([split, record["subject"], record["session"],
                                 record.get("idx", ""), record.get("path", ""), record["group"]])
    n_batches = len(dm.train_dataloader()) if dm.train_dataset is not None else 10
    effective_log_steps = min(trainer_cfg.get("log_every_n_steps", 10), max(1, n_batches))

    if trainer_cfg.get("enable_health_check", True) and isinstance(dm, OASIS3LMDBDataModule):
        callbacks.append(_build_health_callback(cfg, dm, exp_dir))

    # DINOv2's sharpened teacher targets make student gradients spikier than
    # VICReg/BYOL's; clip by default for that method only, unless overridden.
    default_grad_clip = 3.0 if loss_name == "dinov2" else None
    gradient_clip_val = trainer_cfg.get("gradient_clip_val", default_grad_clip)

    # Optional runtime profile (configs/perf, e.g. `+perf=rtx4090`): TF32, fused
    # AdamW, channels_last, torch.compile with fallback, throughput monitor.
    # Without a perf node nothing changes; precision only degrades when the
    # device cannot run what was asked (bf16 without CUDA/Ampere).
    precision = resolve_precision(trainer_cfg.get("precision", "32-true"),
                                  trainer_cfg.get("accelerator", "auto"))
    example = None
    if str(cfg.data.get("source", "")) == "roi3d":
        import torch
        side = int(cfg.data.roi_size)
        example = torch.rand(int(cfg.data.get("batch_size", 16)),
                             int(cfg.model.backbone.get("in_channels", 1)), side, side, side)
    runtime = apply_runtime(cfg, model, precision, example)
    if perf_settings(cfg)["monitor"]:
        callbacks.append(PerfMonitor(exp_dir / "perf.json", extra={
            **runtime, "batch_size": int(cfg.data.get("batch_size", 16)),
            "accumulate_grad_batches": int(trainer_cfg.get("accumulate_grad_batches", 1)),
            "num_workers": int(cfg.data.get("num_workers", 0))}))

    trainer = L.Trainer(
        max_epochs=max_epochs,
        max_steps=trainer_cfg.get("max_steps", -1),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=precision,
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        log_every_n_steps=effective_log_steps,
        deterministic=trainer_cfg.get("deterministic", True),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", True),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", True),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        gradient_clip_val=gradient_clip_val,
        logger=csv_logger,
        callbacks=callbacks,
    )

    # 8. Fit
    trainer.fit(model, datamodule=dm,
                ckpt_path=trainer_cfg.get("ckpt_path", None))
    if trainer_cfg.get("enable_checkpointing", True):
        # Lightning's ModelCheckpoint(save_last=True) can retain the last *best*
        # epoch when no later validation loss improves. Save the final state
        # explicitly so frozen probes always evaluate all requested epochs.
        trainer.save_checkpoint(str(exp_dir / "checkpoints" / "last.ckpt"), weights_only=weights_only)
    logger.info("Training completed successfully. Artifacts saved in: %s", exp_dir)

    return {
        "trainer": trainer,
        "model": model,
        "datamodule": dm,
        "exp_dir": exp_dir,
        "runtime": runtime,
    }


# Path to configs folder relative to this file
_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs")


@hydra.main(version_base="1.3", config_path=_CONFIG_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point."""
    train(cfg)


if __name__ == "__main__":
    main()
