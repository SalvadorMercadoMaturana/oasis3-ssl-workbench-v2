"""Small vision transformer exposing both CLS and patch tokens for DINOv2."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.models.backbones.base import BaseBackbone


class ViTBackbone(BaseBackbone):
    """ViT for four-channel MRI with interpolated positions for local crops."""

    def __init__(
        self, in_channels: int = 4, image_size: int = 64, patch_size: int = 8,
        embed_dim: int = 192, depth: int = 4, num_heads: int = 3, pool: str = "cls",
    ) -> None:
        super().__init__()
        if image_size % patch_size or embed_dim % num_heads:
            raise ValueError("image_size must divide by patch_size and embed_dim by num_heads.")
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        self.pool = pool
        self.patch_size = patch_size
        self._out_dim = embed_dim
        grid = image_size // patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def out_dim(self) -> int:
        """Return CLS embedding size."""
        return self._out_dim

    def forward_tokens(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CLS ``(B,D)`` and patch tokens ``(B,N,D)``."""
        patches = self.patch_embed(x)
        height, width = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2)
        if mask is not None:
            if mask.shape != patches.shape[:2]:
                raise ValueError("Patch mask shape must match the patch grid.")
            patches = torch.where(mask.unsqueeze(-1), self.mask_token, patches)
        original_grid = int((self.pos_embed.shape[1] - 1) ** 0.5)
        patch_pos = self.pos_embed[:, 1:].reshape(1, self.out_dim, original_grid, original_grid)
        patch_pos = F.interpolate(patch_pos, size=(height, width), mode="bicubic",
                                  align_corners=False).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), patches), dim=1)
        tokens = tokens + torch.cat((self.pos_embed[:, :1], patch_pos), dim=1)
        tokens = self.norm(self.blocks(tokens))
        return tokens[:, 0], tokens[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation used for linear probing and diagnostics.

        MAE/SimMIM never supervise the CLS token directly (only patch
        reconstruction is optimized), so it can stay under-differentiated across
        inputs for a long time; mean-pooling the patch tokens is the standard
        fix used by the MAE paper for exactly this failure mode. DINOv2/BYOL/
        VICReg keep the default CLS pooling since those objectives do supervise it.
        """
        cls, patches = self.forward_tokens(x)
        return patches.mean(dim=1) if self.pool == "mean" else cls
