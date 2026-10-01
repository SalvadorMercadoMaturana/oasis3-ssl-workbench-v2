"""DINOv2 building blocks: head, Sinkhorn-Knopp, masking, KoLeo and the 3D ViT inflation."""

from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F

from t5_ssl.losses.dinov2 import DINOv2Loss, koleo, sinkhorn_knopp
from t5_ssl.models.backbones import vit3d
from t5_ssl.models.backbones.vit3d import ViT3DBackbone, inflate_patch_embed, inflate_pos_embed
from t5_ssl.models.dinov2_module import DINOv2LightningModule
from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.masking import block_mask, sample_masks


def test_dino_head_scores_are_bounded_cosines():
    """Scores are cosines with unit prototypes: bounded, and blind to prototype norms."""
    head = DINOHead(12, out_dim=32, hidden_dim=16, bottleneck_dim=8)
    x = torch.randn(5, 12) * 100
    scores = head(x)
    assert scores.shape == (5, 32)
    assert scores.abs().max() <= 1 + 1e-5
    with torch.no_grad():
        head.prototypes.mul_(7.0)
    assert torch.allclose(head(x), scores, atol=1e-6)
    assert head.last_layer_parameters() == [head.prototypes]


def test_sinkhorn_knopp_rows_are_distributions_with_balanced_prototypes():
    """Each sample gets a distribution and every prototype the same total mass."""
    probs = sinkhorn_knopp(torch.randn(24, 10), temperature=0.5, iterations=50)
    assert torch.allclose(probs.sum(dim=1), torch.ones(24), atol=1e-4)
    assert torch.allclose(probs.sum(dim=0), torch.full((10,), 2.4), atol=1e-3)


def test_sinkhorn_knopp_is_finite_at_low_temperature_on_cosines():
    """Cosine scores at a 0.01 temperature must not underflow into NaN targets."""
    scores = torch.rand(40, 512) * 2 - 1
    probs = sinkhorn_knopp(scores, temperature=0.01)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(40), atol=1e-3)


@pytest.mark.parametrize("grid", [(4, 4), (4, 4, 4)])
def test_masks_follow_dinov2_sampling_rule(grid):
    """Half of the images are masked, with ratios within range, in 2D and 3D."""
    total = math.prod(grid)
    masks = sample_masks(16, grid, 0.5, (0.1, 0.5), random.Random(0), min_block=2)
    assert masks.shape == (16, total)
    counts = masks.sum(dim=1)
    assert int((counts > 0).sum()) == 8
    assert int(counts.max()) <= total // 2
    again = sample_masks(16, grid, 0.5, (0.1, 0.5), random.Random(0), min_block=2)
    assert torch.equal(masks, again)


def test_block_mask_reaches_requested_count():
    """Blocks are added until the requested number of patches is masked."""
    mask = block_mask((4, 4, 4), 20, random.Random(3), min_block=4)
    assert 16 <= int(mask.sum()) <= 20


def test_koleo_spreads_features_and_backpropagates():
    """KoLeo is lower for spread features and gives gradients to its inputs."""
    spread = koleo(torch.eye(8))
    clumped = koleo(torch.eye(8) * 0.01 + 1.0)
    assert spread < clumped
    features = torch.randn(6, 5, requires_grad=True)
    koleo(features).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_uniform_teacher_is_reported_as_collapse():
    """Identical teacher scores with softmax centering give entropy ln(K) and KL ~ 0."""
    loss_fn = DINOv2Loss(out_dim=16, centering="softmax")
    same = torch.zeros(4, 16)
    student = [same.clone().requires_grad_() for _ in range(3)]
    _, metrics = loss_fn(student, [same, same], [torch.zeros(0, 16)] * 2, [torch.zeros(0, 16)] * 2,
                         [torch.zeros(0)] * 2, [torch.randn(4, 6), torch.randn(4, 6)])
    assert metrics["teacher_entropy_frac"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["dino_kl"].abs() < 1e-5
    assert metrics["ibot"] == 0


def test_patch_embed_inflation_reproduces_the_2d_token():
    """A depth-constant grey patch gives the 2D token of its upsampled, normalized image."""
    torch.manual_seed(0)
    weight, bias = torch.randn(5, 3, 16, 16), torch.randn(5)
    weight3d, bias3d = inflate_patch_embed(weight, bias, patch_size=8, in_channels=2)
    assert weight3d.shape == (5, 2, 8, 8, 8)
    grey = torch.rand(8, 8)
    cube = grey.expand(2, 8, 8, 8)[None]  # both channels, every depth
    token3d = F.conv3d(cube, weight3d, bias3d, stride=8).flatten()
    big = F.interpolate(grey[None, None], size=(16, 16), mode="bilinear", align_corners=False)
    mean = torch.tensor(vit3d.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(vit3d.IMAGENET_STD).view(1, 3, 1, 1)
    token2d = F.conv2d((big.expand(1, 3, 16, 16) - mean) / std, weight, bias, stride=16).flatten()
    assert torch.allclose(token3d, token2d, atol=1e-4)


def test_pos_embed_inflation_gives_distinct_3d_positions():
    """Every 3D position starts from the 2D code and is distinct from the others."""
    pos2d = torch.randn(1, 1 + 14 * 14, 6)
    pos3d = inflate_pos_embed(pos2d, grid=4)
    assert pos3d.shape == (1, 1 + 64, 6)
    assert torch.equal(pos3d[:, 0], pos2d[:, 0])
    distances = torch.cdist(pos3d[0, 1:], pos3d[0, 1:])
    assert distances[~torch.eye(64, dtype=torch.bool)].min() > 1e-4


def _fake_checkpoint(dim: int, layer_scale: bool) -> dict[str, torch.Tensor]:
    """A timm/DINOv2-style 2D ViT state dict with random weights."""
    source = ViT3DBackbone(in_channels=2, image_size=16, patch_size=8, embed_dim=dim, depth=1,
                           num_heads=3, layer_scale=0.5 if layer_scale else None)
    state = {k: v for k, v in source.state_dict().items() if k.startswith(("blocks.", "norm."))}
    state.update({"cls_token": torch.randn(1, 1, dim), "pos_embed": torch.randn(1, 1 + 14 * 14, dim),
                  "patch_embed.proj.weight": torch.randn(dim, 3, 16, 16),
                  "patch_embed.proj.bias": torch.randn(dim), "head.weight": torch.randn(3, dim)})
    if layer_scale:
        state["mask_token"] = torch.randn(1, dim)
    return {"model": state}


@pytest.mark.parametrize(("name", "layer_scale"), [("deit_small_in1k", False),
                                                   ("dinov2_vits14", True)])
def test_vit3d_loads_and_inflates_2d_checkpoints(monkeypatch, name, layer_scale):
    """Pretrained 2D blocks load verbatim; patch and position embeddings are inflated."""
    checkpoint = _fake_checkpoint(12, layer_scale)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: checkpoint)
    model = ViT3DBackbone(in_channels=2, image_size=16, patch_size=8, embed_dim=12, depth=1,
                          num_heads=3, pretrained=name)
    state = checkpoint["model"]
    assert torch.equal(model.blocks[0].attn.qkv.weight, state["blocks.0.attn.qkv.weight"])
    if layer_scale:
        assert torch.equal(model.blocks[0].ls1.gamma, state["blocks.0.ls1.gamma"])
        assert torch.equal(model.mask_token.flatten(), state["mask_token"].flatten())
    cls, patches = model.forward_tokens(torch.rand(2, 2, 16, 16, 16))
    assert cls.shape == (2, 12) and patches.shape == (2, 8, 12)


def _tiny_dinov2_3d() -> DINOv2LightningModule:
    backbone = ViT3DBackbone(in_channels=2, image_size=16, patch_size=4, embed_dim=12, depth=1,
                             num_heads=3, drop_path=0.1)
    return DINOv2LightningModule(
        backbone=backbone, projector=DINOHead(12, 16, hidden_dim=16, bottleneck_dim=8),
        patch_head=DINOHead(12, 16, hidden_dim=16, bottleneck_dim=8),
        loss_fn=DINOv2Loss(out_dim=16), max_epochs=1, weight_decay=0.04,
        weight_decay_end=0.2, min_mask_block=2,
    )


def test_dinov2_3d_step_freezes_prototypes_in_first_epoch():
    """Global 3D views are masked, local crops pass, and prototypes stay frozen at epoch 0."""
    model = _tiny_dinov2_3d()
    views = [torch.rand(4, 2, 16, 16, 16) for _ in range(2)] + \
            [torch.rand(4, 2, 8, 8, 8) for _ in range(2)]
    loss = model.training_step({"image": views}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projector.prototypes.grad is not None
    model.on_after_backward()
    assert model.projector.prototypes.grad is None and model.patch_head.prototypes.grad is None
    assert model.backbone.patch_embed.proj.weight.grad is not None
    assert all(p.grad is None for p in model.teacher_backbone.parameters())
    assert all(m.drop_prob == 0 for m in model.teacher_backbone.modules() if hasattr(m, "drop_prob"))


def test_dinov2_weight_decay_skips_one_dimensional_parameters():
    """Biases, norms and LayerScale are excluded from weight decay, as in DINO."""
    model = _tiny_dinov2_3d()
    regularized, plain = model.optimizer_param_groups()
    assert regularized["regularized"] and regularized["weight_decay"] == 0.04
    assert all(p.ndim > 1 for p in regularized["params"])
    assert plain["weight_decay"] == 0 and all(p.ndim <= 1 for p in plain["params"])
    optimized = {id(p) for group in (regularized, plain) for p in group["params"]}
    assert optimized == {id(p) for p in model.trainable_parameters()}


def test_dino_head_batch_norm_removes_the_shared_component():
    """With use_bn, inputs that differ only by a tiny offset still get distinct scores."""
    torch.manual_seed(0)
    common = torch.randn(1, 12) * 10
    x = common + 0.01 * torch.randn(8, 12)
    plain = DINOHead(12, out_dim=32, hidden_dim=16, bottleneck_dim=8)
    normed = DINOHead(12, out_dim=32, hidden_dim=16, bottleneck_dim=8, use_bn=True)
    spread = {name: head(x).std(dim=0).mean().item() for name, head in
              (("plain", plain), ("bn", normed))}
    assert spread["bn"] > 10 * spread["plain"]
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in normed.mlp)
