"""One real OASIS3 train/validation batch and checkpoint reload per SSL method."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch

from t5_ssl.data.oasis3_datamodule import load_volume
from t5_ssl.train import train


def main() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    results = {}
    methods = {
        "vicreg": [],
        "byol": ["model=byol_convnext", "loss=byol"],
        "dinov2_compact": ["model=dinov2_vit", "loss=dinov2",
                           "augmentation=dinov2_multicrop"],
    }
    common = ["data.batch_size=2", "data.num_workers=0", "data.max_volumes=2",
              "data.slices_per_volume=1", "trainer.max_epochs=1",
              "trainer.limit_train_batches=1", "trainer.limit_val_batches=1",
              "trainer.accelerator=cpu", "trainer.enable_progress_bar=false",
              "optimizer.warmup_epochs=0"]
    for name, overrides in methods.items():
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=common + overrides)
        result = train(cfg)
        dm, model, trainer = result["datamodule"], result["model"], result["trainer"]
        sets = [set(v) for v in dm.split_subject_ids.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
        assert dm.records and dm.train_dataset and dm.val_dataset
        volume = load_volume(dm.train_dataset.records[0]["path"])
        assert volume.ndim == 3 and volume.dtype.name == "float32"
        batch = next(iter(dm.train_dataloader()))
        views = batch["image"]
        assert len(views) == (4 if name == "dinov2_compact" else 2)
        assert all(view.shape[0] == 2 and view.shape[1] == 1 and
                   0 <= view.min() <= view.max() <= 1 for view in views)
        if name == "byol":
            assert all(parameter.grad is None for parameter in model.target_backbone.parameters())
            assert all(parameter.grad is None for parameter in model.target_projector.parameters())
        if name == "dinov2_compact":
            assert all(parameter.grad is None for parameter in model.teacher_backbone.parameters())
            assert all(parameter.grad is None for parameter in model.teacher_projector.parameters())
            assert all(parameter.grad is None for parameter in model.teacher_patch_head.parameters())
        path = result["exp_dir"] / "checkpoints" / "last.ckpt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        val = float(trainer.callback_metrics["val/loss"])
        assert torch.isfinite(torch.tensor(val))
        results[name] = {"volume_shape": list(volume.shape), "volume_range":
                         [float(volume.min()), float(volume.max())],
                         "view_shapes": [list(view.shape) for view in views],
                         "split_subjects": {k: len(v) for k, v in dm.split_subject_ids.items()},
                         "used_volumes": {k: len(getattr(dm, f"{k}_dataset").records)
                                          for k in ("train", "val", "test")},
                         "train_batches": trainer.global_step, "val_loss": val,
                         "checkpoint": str(path), "checkpoint_reload": "passed"}
        print(name, results[name], flush=True)
    output = Path("smoke_results.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
