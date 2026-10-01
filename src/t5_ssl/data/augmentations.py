"""Spatial and intensity augmentations for multi-modal 2D MRI slices."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as F


def _tupleize(value: Any) -> Any:
    """TorchIO parses ranges from tuples; Hydra hands them over as lists."""
    if isinstance(value, list | tuple):
        return tuple(_tupleize(item) for item in value)
    return value


def build_torchio_transform(specs: Sequence[dict[str, Any]] | None, spatial_dims: int = 2
                            ) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Compose TorchIO transforms from specs like ``{"name": "RandomMotion", "p": 0.5}``.

    ``{"name": "OneOf", "transforms": [...], "p": ...}`` picks one child per call.
    Returns a callable on (C, H, W) slices, which TorchIO sees as 2D (C, H, W, 1)
    images with 1 mm spacing, so mm parameters are pixels of the source slice;
    with ``spatial_dims=3`` it takes (C, D, H, W) volumes as they are.
    """
    if not specs:
        return None
    import torchio as tio

    def build(spec: dict[str, Any]) -> Any:
        kwargs = dict(spec)
        name = kwargs.pop("name")
        if name == "OneOf":
            return tio.OneOf([build(child) for child in kwargs.pop("transforms")],
                             **{k: _tupleize(v) for k, v in kwargs.items()})
        return getattr(tio, name)(**{k: _tupleize(v) for k, v in kwargs.items()})

    compose = tio.Compose([build(spec) for spec in specs])

    def apply(image: torch.Tensor) -> torch.Tensor:
        if spatial_dims == 3:
            return compose(image)
        return compose(image.unsqueeze(-1)).squeeze(-1)

    return apply


class SSLViewTransform:
    """Create two independently augmented views of a channel-first MRI tensor.

    Spatial operations are applied to the stacked tensor so every MRI modality
    remains registered with the others.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        crop_scale: tuple[float, float] = (0.3, 1.0),
        crop_ratio: tuple[float, float] = (0.9, 1.1),
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        affine_degrees: float = 15.0,
        translate_frac: tuple[float, float] = (0.0, 0.0),
        blur_p: float = 0.5,
        noise_std: float = 0.03,
        intensity_scale: tuple[float, float] = (0.9, 1.1),
        gamma_range: tuple[float, float] = (1.0, 1.0),
        bias_field_p: float = 0.0,
        bias_field_alpha: float = 0.15,
        torchio_transforms: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.affine_degrees = affine_degrees
        self.translate_frac = translate_frac
        self.blur_p = blur_p
        self.noise_std = noise_std
        self.intensity_scale = intensity_scale
        self.gamma_range = gamma_range
        self.bias_field_p = bias_field_p
        self.bias_field_alpha = bias_field_alpha
        self.torchio_transform = build_torchio_transform(torchio_transforms)

    def _bias_field(self, view: torch.Tensor) -> torch.Tensor:
        """Multiply by a smooth low-frequency field to mimic scanner inhomogeneity."""
        control = torch.rand(1, 1, 4, 4) * 2 - 1
        field = torch.nn.functional.interpolate(
            control, size=view.shape[-2:], mode="bicubic", align_corners=False,
        )[0]
        field = 1.0 + self.bias_field_alpha * field
        return view * field

    def _make_view(self, image: torch.Tensor) -> torch.Tensor:
        if self.torchio_transform is not None:
            # Acquisition artifacts act on the whole field of view, before cropping.
            image = self.torchio_transform(image)
        if self.crop_scale == (1.0, 1.0):
            # A full-field view must not become a center crop on nonsquare MRIs.
            view = F.resize(image, self.image_size,
                            interpolation=InterpolationMode.BILINEAR, antialias=True)
        else:
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=self.crop_scale, ratio=self.crop_ratio,
            )
            view = F.resized_crop(
                image, top, left, height, width, self.image_size,
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )

        if random.random() < self.horizontal_flip_p:
            view = F.hflip(view)
        if random.random() < self.vertical_flip_p:
            view = F.vflip(view)
        tx_frac, ty_frac = self.translate_frac
        translate = [
            random.uniform(-tx_frac, tx_frac) * view.shape[-1],
            random.uniform(-ty_frac, ty_frac) * view.shape[-2],
        ]
        if self.affine_degrees > 0 or tx_frac > 0 or ty_frac > 0:
            angle = random.uniform(-self.affine_degrees, self.affine_degrees)
            view = F.affine(
                view,
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
        if random.random() < self.blur_p:
            view = F.gaussian_blur(view, kernel_size=[5, 5], sigma=[0.1, 1.5])

        scale = random.uniform(*self.intensity_scale)
        view = view * scale
        if self.gamma_range != (1.0, 1.0):
            gamma = random.uniform(*self.gamma_range)
            view = view.clamp_min(1e-6).pow(gamma)
        if random.random() < self.bias_field_p:
            view = self._bias_field(view)
        if self.noise_std > 0:
            view = view + torch.randn_like(view) * self.noise_std
        return view.clamp_(0.0, 1.0)

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two stochastic views without mutating the source tensor."""
        return self._make_view(image), self._make_view(image)


class EvalTransform:
    """Resize a channel-first MRI tensor deterministically for evaluation."""

    def __init__(self, image_size: tuple[int, int] = (224, 224)) -> None:
        self.image_size = image_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Resize the tensor while preserving its value range."""
        return F.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).clamp_(0.0, 1.0)


class DINOv2ViewTransform(SSLViewTransform):
    """Produce two global crops and smaller local crops from one registered MRI slice."""

    def __init__(
        self, *, local_size: tuple[int, int] = (32, 32),
        local_crop_scale: tuple[float, float] = (0.1, 0.3),
        n_local_crops: int = 2, **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if n_local_crops < 0:
            raise ValueError("n_local_crops must be nonnegative.")
        self.local_size = local_size
        self.local_crop_scale = local_crop_scale
        self.n_local_crops = n_local_crops

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Keep modalities aligned inside each independently augmented crop."""
        globals_ = super().__call__(image)
        original_size, original_scale = self.image_size, self.crop_scale
        try:
            self.image_size, self.crop_scale = self.local_size, self.local_crop_scale
            locals_ = tuple(self._make_view(image) for _ in range(self.n_local_crops))
        finally:
            self.image_size, self.crop_scale = original_size, original_scale
        return (*globals_, *locals_)


class DeterministicSSLViews(EvalTransform):
    """Fixed paired validation views, with optional smaller local crops."""

    def __init__(self, *, local_size: tuple[int, int] | None = None,
                 n_local_crops: int = 0, horizontal_flip: bool = True,
                 second_view_angle: float = 0.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.local_size = local_size
        self.n_local_crops = n_local_crops
        self.horizontal_flip = horizontal_flip
        self.second_view_angle = second_view_angle

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Preserve a stable first global view for the linear probe."""
        first = super().__call__(image)
        if self.horizontal_flip:
            second = F.hflip(first)
        elif self.second_view_angle:
            second = F.affine(first, angle=self.second_view_angle,
                              translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
                              interpolation=InterpolationMode.BILINEAR)
        else:
            second = first.clone()
        if self.local_size is None:
            return first, second
        locals_ = tuple(
            F.resize(F.center_crop(first if index % 2 == 0 else second,
                                   [max(1, first.shape[-2] // 2), max(1, first.shape[-1] // 2)]),
                     self.local_size, antialias=True)
            for index in range(self.n_local_crops)
        )
        return first, second, *locals_
