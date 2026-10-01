"""3D vision transformer exposing CLS and patch tokens, for DINOv2 on ROI cubes.

Volumetric counterpart of ``ViTBackbone``: cubic patches, a learned position
per patch of the ``image_size`` grid (trilinearly resized for smaller local
crops) and pre-norm blocks with the parameter names of timm/DINOv2, so 2D
ViT-S checkpoints load after inflation:

- ``deit_small_in1k``: DeiT-S/16 supervised on ImageNet-1k, the ViT analogue of
  the torchvision ImageNet-1k weights that VICReg and BYOL start from.
- ``dinov2_vits14``: DINOv2 ViT-S/14 (LVD-142M, with LayerScale).

Inflation (``inflate_patch_embed``/``inflate_pos_embed``): the ImageNet input
normalization is folded into the patch embedding for grey [0, 1] inputs, the
2D kernel is resized to the new patch side so that a patch gives the token of
its bilinear upsampling (FlexiViT PI-resize, exact when shrinking), and it is
then repeated along depth and split over the input channels, so a volume that
is constant along depth reproduces the 2D token of its (channel-averaged)
slice, as in the I3D inflation of the CNNs. Positions: the 2D grid is resized
to the in-plane grid, repeated along depth (ViViT) and offset along depth by
the 2D row code, so every 3D position starts distinct.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.models.backbones.base import BaseBackbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRETRAINED_2D = {
    "deit_small_in1k": {
        "url": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        "check_hash": True, "layer_scale": False},
    "dinov2_vits14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "check_hash": False, "layer_scale": True},
}


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    """Per-channel residual scaling (CaiT/DINOv2)."""

    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention with a fused qkv projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block with optional LayerScale and stochastic depth."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 layer_scale: float | None = None, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class PatchEmbed3D(nn.Module):
    """Non-overlapping cubic patches to tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def inflate_patch_embed(weight: torch.Tensor, bias: torch.Tensor, patch_size: int,
                        in_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RGB patch embedding ``(D, 3, P, P)`` -> 3D ``(D, C, p, p, p)`` for grey [0, 1] inputs."""
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    weight, bias = weight.float(), bias.float()
    grey = (weight / std).sum(dim=1)  # (D, P, P): response to (g, g, g) after normalization
    bias = bias - (weight * mean / std).sum(dim=(1, 2, 3))
    side = grey.shape[-1]
    if patch_size != side:
        basis = torch.eye(patch_size * patch_size).reshape(-1, 1, patch_size, patch_size)
        upsample = F.interpolate(basis, size=(side, side), mode="bilinear", align_corners=False)
        grey = grey.reshape(grey.shape[0], -1) @ upsample.reshape(patch_size * patch_size, -1).t()
        grey = grey.reshape(-1, patch_size, patch_size)
    weight3d = grey[:, None, None].repeat(1, in_channels, patch_size, 1, 1)
    return weight3d / (patch_size * in_channels), bias


def inflate_pos_embed(pos_embed: torch.Tensor, grid: int) -> torch.Tensor:
    """2D ``(1, 1 + G*G, D)`` positions -> 3D ``(1, 1 + grid**3, D)``."""
    cls_pos, patch_pos = pos_embed[:, :1].float(), pos_embed[:, 1:].float()
    side, dim = round(math.sqrt(patch_pos.shape[1])), patch_pos.shape[-1]
    plane = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    plane = F.interpolate(plane, size=(grid, grid), mode="bicubic", align_corners=False,
                          antialias=grid < side)[0]  # (D, grid, grid)
    rows = plane.mean(dim=2)  # (D, grid): the code of each row
    depth = rows - rows.mean(dim=1, keepdim=True)
    volume = plane[:, None] + depth[:, :, None, None]  # (D, depth, height, width)
    return torch.cat((cls_pos, volume.reshape(dim, -1).t()[None]), dim=1)


class ViT3DBackbone(BaseBackbone):
    """ViT for (B, C, D, H, W) volumes with interpolated positions for local crops."""

    def __init__(self, in_channels: int = 2, image_size: int = 32, patch_size: int = 8,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, pool: str = "cls", pretrained: str | bool | None = None,
                 drop_path: float = 0.0, layer_scale: float | None = None) -> None:
        super().__init__()
        if image_size % patch_size or embed_dim % num_heads:
            raise ValueError("image_size must divide by patch_size and embed_dim by num_heads.")
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        if pretrained is True:
            pretrained = "deit_small_in1k"  # "ImageNet weights", as for the CNN backbones
        if pretrained and pretrained not in PRETRAINED_2D:
            raise ValueError(f"Unknown pretrained ViT '{pretrained}'. Choose from {list(PRETRAINED_2D)}.")
        if pretrained and PRETRAINED_2D[pretrained]["layer_scale"] and not layer_scale:
            layer_scale = 1e-5  # placeholder, overwritten by the checkpoint
        self.pool, self.patch_size, self._out_dim = pool, patch_size, embed_dim
        self.in_channels, self.grid = in_channels, image_size // patch_size
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid**3, embed_dim))
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.Sequential(*(Block(embed_dim, num_heads, mlp_ratio, layer_scale, rate)
                                      for rate in rates))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        for parameter in (self.cls_token, self.mask_token, self.pos_embed):
            nn.init.trunc_normal_(parameter, std=0.02)
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        self.pretrained = pretrained or None
        if self.pretrained:
            self.load_2d_weights(self.pretrained)

    @property
    def out_dim(self) -> int:
        """Return CLS embedding size."""
        return self._out_dim

    def load_2d_weights(self, name: str) -> None:
        """Load and inflate a 2D ViT-S checkpoint (see module docstring)."""
        spec = PRETRAINED_2D[name]
        state = torch.hub.load_state_dict_from_url(spec["url"], map_location="cpu",
                                                   check_hash=spec["check_hash"])
        state = state.get("model", state)
        own = {k: v for k, v in state.items() if k.startswith(("blocks.", "norm."))}
        own["cls_token"] = state["cls_token"]
        if "mask_token" in state:
            own["mask_token"] = state["mask_token"].reshape(1, 1, -1)
        own["pos_embed"] = inflate_pos_embed(state["pos_embed"], self.grid)
        own["patch_embed.proj.weight"], own["patch_embed.proj.bias"] = inflate_patch_embed(
            state["patch_embed.proj.weight"], state["patch_embed.proj.bias"], self.patch_size,
            self.in_channels)
        missing, unexpected = self.load_state_dict(own, strict=False)
        missing = [key for key in missing if key != "mask_token"]
        if missing or unexpected:
            raise RuntimeError(f"{name}: missing {missing}, unexpected {unexpected}")

    def forward_tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CLS ``(B, D)`` and patch tokens ``(B, N, D)``."""
        patches = self.patch_embed(x)
        size = patches.shape[-3:]
        patches = patches.flatten(2).transpose(1, 2)
        if mask is not None:
            if mask.shape != patches.shape[:2]:
                raise ValueError("Patch mask shape must match the patch grid.")
            patches = torch.where(mask.unsqueeze(-1), self.mask_token.to(patches.dtype), patches)
        grid = (self.grid,) * 3
        patch_pos = self.pos_embed[:, 1:].transpose(1, 2).reshape(1, self.out_dim, *grid)
        if tuple(size) != grid:
            patch_pos = F.interpolate(patch_pos, size=tuple(size), mode="trilinear",
                                      align_corners=False)
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), patches), dim=1)
        tokens = tokens + torch.cat((self.pos_embed[:, :1], patch_pos), dim=1)
        tokens = self.norm(self.blocks(tokens))
        return tokens[:, 0], tokens[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation used for linear probing."""
        cls, patches = self.forward_tokens(x)
        return patches.mean(dim=1) if self.pool == "mean" else cls
