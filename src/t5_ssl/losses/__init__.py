"""Self-Supervised Learning loss functions."""

from t5_ssl.losses.base import BaseSSLLoss
from t5_ssl.losses.vicreg import VICRegLoss

__all__ = ["BaseSSLLoss", "VICRegLoss"]
