"""ConvNeXt weights of the original release, in the layout of torchvision's ConvNeXt.

torchvision ships one ConvNeXt checkpoint per size, ImageNet-1k (``IMAGENET1K_V1``). The
original release (Liu et al. 2022, facebookresearch/ConvNeXt) also has ImageNet-22k ones:

- ``in22k``: pretrained on ImageNet-22k (21841 classes), without 1k fine-tuning;
- ``in22k_ft_in1k``: the same weights, then fine-tuned on ImageNet-1k.

The two architectures are the same (stem, three downsampling layers, four stages of blocks);
only the parameter names and the layer-scale shape differ. ``load_release_features`` renames
a release checkpoint into ``torchvision.models.convnext_*().features``, so it then goes through
the same 3D inflation as the torchvision weights (t5_ssl.models.backbones.inflate3d). The
classification head is dropped: the backbones keep the features only.
"""

from __future__ import annotations

import torch
from torch import nn

RELEASE = "https://dl.fbaipublicfiles.com/convnext"
URLS = {
    ("convnext_tiny", "in22k"): f"{RELEASE}/convnext_tiny_22k_224.pth",
    ("convnext_tiny", "in22k_ft_in1k"): f"{RELEASE}/convnext_tiny_22k_1k_224.pth",
    ("convnext_small", "in22k"): f"{RELEASE}/convnext_small_22k_224.pth",
    ("convnext_small", "in22k_ft_in1k"): f"{RELEASE}/convnext_small_22k_1k_224.pth",
}
RELEASE_WEIGHTS = ("in22k", "in22k_ft_in1k")
# release block layer -> index in torchvision's CNBlock.block (dwconv, permute, norm,
# linear, GELU, linear, permute)
_BLOCK = {"dwconv": 0, "norm": 2, "pwconv1": 3, "pwconv2": 5}


def to_torchvision(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Release ConvNeXt state dict -> the state dict of torchvision's ``features``.

    - ``downsample_layers.<i>.<k>``: the stem (``i = 0``) and the downsampling layer in
      front of stage ``i``, which are ``features.<2i>.<k>``;
    - ``stages.<i>.<j>``: block ``j`` of stage ``i``, which is ``features.<2i+1>.<j>``.
      The ``(C,)`` ``gamma`` becomes the ``(C, 1, 1)`` ``layer_scale``.
    """
    out = {}
    for key, value in state.items():
        part = key.split(".")
        if part[0] == "downsample_layers":
            out[f"{2 * int(part[1])}.{part[2]}.{part[3]}"] = value
        elif part[0] == "stages":
            block = f"{2 * int(part[1]) + 1}.{part[2]}"
            if part[3] == "gamma":
                out[f"{block}.layer_scale"] = value.reshape(-1, 1, 1)
            else:
                out[f"{block}.block.{_BLOCK[part[3]]}.{part[4]}"] = value
        elif part[0] not in ("norm", "head"):
            raise KeyError(f"unexpected ConvNeXt release parameter {key!r}")
    return out


def load_release_features(features: nn.Sequential, variant: str, weights: str) -> None:
    """Load the ``weights`` release checkpoint of ``variant`` into torchvision ``features``."""
    if (variant, weights) not in URLS:
        raise ValueError(f"no ConvNeXt release weights {weights!r} for {variant!r}; "
                         f"available: {sorted(URLS)}")
    # weights_only: a downloaded file is never unpickled as arbitrary objects
    state = torch.hub.load_state_dict_from_url(URLS[variant, weights], map_location="cpu",
                                               weights_only=True)
    features.load_state_dict(to_torchvision(state.get("model", state)), strict=True)
