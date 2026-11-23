"""``model.backbone`` of a composed config -> the encoder module (step 4 of t5_ssl.train).

``build_encoder``: the network, by ``model.backbone.variant``:

| variant | module | ``pretrained`` |
|---|---|---|
| ``convnext_*_3d``, ``resnet*_3d`` | ``resnet3d`` (2D weights inflated, ``inflate3d``) | ``true`` = torchvision ImageNet-1k; ConvNeXt also ``in22k`` / ``in22k_ft_in1k`` (``convnext_weights``) |
| ``vit_small_3d`` | ``vit3d`` | ``deit_small_in1k`` / ``dinov2_vits14`` |
| ``medicalnet<depth>`` | ``medicalnet`` (MedicalNet 3D ResNets) | ``medicalnet`` |
| ``convnext_*``, ``resnet*``, ``vit_small`` | 2D ``convnext`` / ``resnet`` / ``vit`` | ``true`` = torchvision ImageNet-1k |

``build_backbone``: that encoder, inside a wrapper when the config says how the input
reaches it (2026-11-22):

- ``model.backbone.ingest``: ``ingest.IngestBackbone`` (normalization, hemispheres as
  separate passes, resampling, 2D slices of the ROI cube);
- ``model.backbone.triplane``: ``triplane.TriPlaneBackbone`` (the whole-brain axial,
  coronal and sagittal planes of ``data.source=brain2d``).

Without either, the plain encoder. ``t5_ssl.recipes.BACKBONES`` names these settings
(``--backbone`` of train.py); ``t5_ssl.train`` re-exports ``build_backbone``.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from t5_ssl.models.backbones.convnext import ConvNeXtBackbone
from t5_ssl.models.backbones.resnet import ResNetBackbone
from t5_ssl.models.backbones.vit import ViTBackbone
from t5_ssl.utils.logging import get_logger

logger = get_logger(__name__)


def build_backbone(cfg: DictConfig) -> nn.Module:
    """Build the encoder of ``model.backbone``, inside its ingestion wrapper if it has one."""
    if cfg.model.backbone.get("triplane"):
        backbone = _triplane_backbone(cfg)
    elif cfg.model.backbone.get("ingest"):
        backbone = _ingest_backbone(cfg)
    else:
        return build_encoder(cfg)
    logger.info("Ingestion: %r", backbone)
    return backbone


def _settings(node: Any) -> dict[str, Any]:
    return dict(OmegaConf.to_container(node, resolve=True))


def _triplane_backbone(cfg: DictConfig) -> nn.Module:
    """Put a 2D encoder on the brain planes; each plane is repeated on its 3 RGB inputs."""
    from t5_ssl.models.backbones.triplane import TriPlaneBackbone
    encoder = build_encoder(cfg, in_channels=3).to(memory_format=torch.channels_last)
    return TriPlaneBackbone(encoder, planes=int(cfg.model.backbone.get("in_channels", 3)),
                            **_settings(cfg.model.backbone.triplane))


def _ingest_backbone(cfg: DictConfig) -> nn.Module:
    """Put the encoder behind the ingestion settings of ``model.backbone.ingest``."""
    from t5_ssl.models.backbones.ingest import IngestBackbone, encoder_in_channels
    spec = _settings(cfg.model.backbone.ingest)
    spec.pop("in_channels", None)
    if "axes" in spec:
        spec["axes"] = tuple(spec["axes"])
    in_channels = int(cfg.model.backbone.get("in_channels", 1))
    variant = str(cfg.model.backbone.get("variant", "convnext_tiny"))
    two_d = not (variant.endswith("_3d") or variant.startswith("medicalnet"))
    encoder = build_encoder(cfg, in_channels=encoder_in_channels(spec, in_channels, two_d))
    if two_d:
        encoder = encoder.to(memory_format=torch.channels_last)
    return IngestBackbone(encoder, in_channels, two_d=two_d, **spec)


def build_encoder(cfg: DictConfig, in_channels: int | None = None) -> nn.Module:
    """Build the network of ``model.backbone`` (``in_channels`` overrides its input channels)."""
    option = cfg.model.backbone.get
    variant = str(option("variant", "convnext_tiny"))
    if in_channels is None:
        in_channels = option("in_channels", 1)
    pretrained = option("pretrained", False)

    logger.info("Initializing backbone: %s (in_channels=%d)", variant, in_channels)
    if variant.startswith("medicalnet"):
        from t5_ssl.models.backbones.medicalnet import MedicalNetBackbone
        depth = int(variant.removeprefix("medicalnet").removeprefix("_") or 18)
        return MedicalNetBackbone(depth=depth, in_channels=in_channels,
                                  pretrained=pretrained in (True, "medicalnet"))
    if variant == "vit_small_3d":
        from t5_ssl.models.backbones.vit3d import ViT3DBackbone
        return ViT3DBackbone(
            in_channels=in_channels, image_size=option("image_size", 32),
            patch_size=option("patch_size", 8), embed_dim=option("embed_dim", 384),
            depth=option("depth", 12), num_heads=option("num_heads", 6),
            mlp_ratio=option("mlp_ratio", 4.0), pool=option("pool", "cls"),
            pretrained=pretrained or None, drop_path=option("drop_path", 0.0),
            layer_scale=option("layer_scale", None),
        )
    if variant.endswith("_3d"):
        from t5_ssl.models.backbones.resnet3d import ConvNeXt3DBackbone, ResNet3DBackbone
        backbone_3d = ConvNeXt3DBackbone if variant.startswith("convnext") else ResNet3DBackbone
        return backbone_3d(variant=variant, in_channels=in_channels, pretrained=pretrained)
    if variant == "vit_small":
        return ViTBackbone(
            in_channels=in_channels, image_size=option("image_size", 64),
            patch_size=option("patch_size", 8), embed_dim=option("embed_dim", 192),
            depth=option("depth", 4), num_heads=option("num_heads", 3),
            pool=option("pool", "cls"),
        )
    if variant.startswith("resnet"):
        return ResNetBackbone(variant=variant, in_channels=in_channels, pretrained=pretrained)
    return ConvNeXtBackbone(variant=variant, in_channels=in_channels, pretrained=pretrained)
