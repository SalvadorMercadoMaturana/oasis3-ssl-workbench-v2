"""Projection heads."""

from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.heads.projector import MLPProjector

__all__ = ["DINOHead", "MLPProjector"]
