"""iBOT/DINOv2 block masking of 2D or 3D patch grids.

``block_mask`` follows the BEiT/iBOT ``MaskingGenerator`` used by DINOv2 —
random blocks of at least ``min_block`` patches with aspect ratios in
[0.3, 1/0.3] until about the requested number of patches is masked —
generalized to any number of spatial dimensions. ``sample_masks`` reproduces
DINOv2's collate rule: a fraction ``probability`` of the images is masked,
with ratios spread over ``ratio_range``, and the rest are left unmasked.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch

LOG_ASPECT = (math.log(0.3), math.log(1 / 0.3))


def _add_block(mask: torch.Tensor, budget: int, rng: random.Random, min_block: int) -> int:
    """Mask one random block of at most ``budget`` new patches; return how many."""
    shape = mask.shape
    for _ in range(10):
        target = rng.uniform(min(min_block, budget), budget)
        # Per-axis log-scales with zero mean: pairwise aspect ratios within [0.3, 1/0.3].
        logs = [rng.uniform(*LOG_ASPECT) / 2 for _ in shape]
        mean = sum(logs) / len(logs)
        sides = [min(size, max(1, round(target ** (1 / len(shape)) * math.exp(log - mean))))
                 for size, log in zip(shape, logs, strict=True)]
        starts = [rng.randint(0, size - side) for size, side in zip(shape, sides, strict=True)]
        region = tuple(slice(start, start + side) for start, side in zip(starts, sides, strict=True))
        new = int((~mask[region]).sum())
        if 0 < new <= budget:
            mask[region] = True
            return new
    return 0


def block_mask(grid: Sequence[int], num_masked: int, rng: random.Random, min_block: int = 4,
               max_block: int | None = None) -> torch.Tensor:
    """Flattened boolean mask of a ``grid`` with about ``num_masked`` patches in blocks."""
    shape = tuple(int(side) for side in grid)
    mask = torch.zeros(shape, dtype=torch.bool)
    num_masked = min(max(0, int(num_masked)), mask.numel())
    max_block = max_block or mask.numel()
    count = 0
    while count < num_masked:
        added = _add_block(mask, min(num_masked - count, max_block), rng, min_block)
        if added == 0:
            break
        count += added
    return mask.flatten()


def sample_masks(batch_size: int, grid: Sequence[int], probability: float,
                 ratio_range: tuple[float, float], rng: random.Random,
                 min_block: int = 4) -> torch.Tensor:
    """``(batch_size, prod(grid))`` masks: DINOv2's per-batch sampling rule."""
    total = math.prod(int(side) for side in grid)
    n_masked = int(batch_size * probability)
    low, high = ratio_range
    bounds = [low + (high - low) * i / max(1, n_masked) for i in range(n_masked + 1)]
    masks = [block_mask(grid, int(total * rng.uniform(bounds[i], bounds[i + 1])), rng,
                        min_block=min_block, max_block=max(1, total // 2))
             for i in range(n_masked)]
    masks += [torch.zeros(total, dtype=torch.bool) for _ in range(batch_size - n_masked)]
    rng.shuffle(masks)
    return torch.stack(masks)
