"""ImageNet-22k ConvNeXt weights of the original release (t5_ssl.models.backbones.convnext_weights)."""

from __future__ import annotations

import pytest
import torch
from torchvision import models

from t5_ssl.models.backbones import convnext_weights as cw
from t5_ssl.models.backbones.resnet3d import ConvNeXt3DBackbone

_LAYER = {index: name for name, index in cw._BLOCK.items()}


def release_state(features: torch.nn.Sequential) -> dict[str, torch.Tensor]:
    """torchvision ``features`` state dict renamed back to the original release's names."""
    out = {}
    for key, value in features.state_dict().items():
        index, rest = key.split(".", 1)
        if int(index) % 2 == 0:
            out[f"downsample_layers.{int(index) // 2}.{rest}"] = value
            continue
        block, rest = rest.split(".", 1)
        stage = f"stages.{int(index) // 2}.{block}"
        if rest == "layer_scale":
            out[f"{stage}.gamma"] = value.reshape(-1)
        else:
            layer, param = rest.removeprefix("block.").split(".")
            out[f"{stage}.{_LAYER[int(layer)]}.{param}"] = value
    out["norm.weight"], out["head.weight"] = torch.ones(768), torch.zeros(10, 768)
    return out


def test_release_names_map_onto_every_torchvision_parameter() -> None:
    features = models.convnext_tiny(weights=None).features
    with torch.no_grad():
        for p in features.parameters():
            p.normal_()
    state = cw.to_torchvision(release_state(features))
    target = models.convnext_tiny(weights=None).features
    target.load_state_dict(state, strict=True)
    for (name, a), b in zip(features.state_dict().items(), target.state_dict().values(),
                            strict=True):
        assert torch.equal(a, b), name


def test_unknown_release_parameter_is_an_error() -> None:
    with pytest.raises(KeyError):
        cw.to_torchvision({"stem.weight": torch.zeros(1)})


def test_3d_convnext_loads_release_weights_before_inflating(monkeypatch) -> None:
    source = models.convnext_tiny(weights=None).features
    with torch.no_grad():
        for p in source.parameters():
            p.normal_()
    urls = []

    def fake_download(url, **kwargs):
        assert kwargs["weights_only"] is True
        urls.append(url)
        return {"model": release_state(source)}

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_download)
    backbone = ConvNeXt3DBackbone("convnext_tiny_3d", in_channels=2, pretrained="in22k")
    assert urls == [cw.URLS["convnext_tiny", "in22k"]] and backbone.weights == "in22k"
    # a block's depthwise kernel k x k becomes k x k x k: its depth mean is the 2D kernel / k
    k2d = source[1][0].block[0].weight
    k3d = backbone.features[1][0].block[0].weight
    assert torch.allclose(k3d.sum(dim=2), k2d, atol=1e-5)
    assert backbone(torch.randn(1, 2, 32, 32, 32)).shape == (1, 768)


def test_3d_convnext_rejects_unknown_weights() -> None:
    with pytest.raises(ValueError, match="in22k"):
        ConvNeXt3DBackbone("convnext_tiny_3d", pretrained="in21k")
    assert ConvNeXt3DBackbone("convnext_tiny_3d", pretrained=False).weights is None
