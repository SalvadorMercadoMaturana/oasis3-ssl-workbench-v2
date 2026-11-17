"""Show old and full-field SSL views of one fixed training MRI slice."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from t5_ssl.data.augmentations import EvalTransform, SSLViewTransform
from t5_ssl.train import build_datamodule

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "campaign_mri_fullview20/figures/augmentation_examples.png"


def main() -> None:
    """Use a central slice from train only to inspect field-of-view preservation."""
    baseline = json.loads((ROOT / "campaign_full50/results.json").read_text())
    reference = next(row for row in baseline["results"] if row["method"] == "vicreg")
    cfg = OmegaConf.load(Path(reference["experiment_dir"]) / "config.yaml")
    dm = build_datamodule(cfg, None, None)
    dm.setup()
    image = dm.eval_dataset("train", None)[0]["image"]
    original = EvalTransform((96, 96))(image)
    policies = (
        ("Anterior", SSLViewTransform(
            image_size=(96, 96), crop_scale=(0.3, 1.0),
            horizontal_flip_p=0.5, vertical_flip_p=0.5,
            affine_degrees=15, blur_p=0.5, noise_std=0.03,
        )),
        ("Campo completo", SSLViewTransform(
            image_size=(96, 96), crop_scale=(1.0, 1.0),
            horizontal_flip_p=0, vertical_flip_p=0,
            affine_degrees=5, blur_p=0, noise_std=0.01,
            intensity_scale=(0.95, 1.05),
        )),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=180)
    for row, (name, transform) in enumerate(policies):
        random.seed(42)
        torch.manual_seed(42)
        views = transform(image)
        for col, item in enumerate((original, *views)):
            axes[row, col].imshow(item[0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title("Original" if col == 0 else f"{name} · vista {col}")
            axes[row, col].axis("off")
    fig.suptitle("Mismo corte axial de train; dos vistas SSL por política")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
