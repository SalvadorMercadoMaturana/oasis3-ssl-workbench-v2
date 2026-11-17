"""A few real 2.5D batches per new SSL candidate: shapes, gradients, checkpoint reload.

Mirrors scripts/legacy/smoke_oasis3.py's contract but exercises the additions for the
SSL-input-diagnostic follow-up: the 2.5D axial stack (data=oasis3_2p5d), the
tuned intermediate augmentation, the VICReg feature queue, the BYOL GroupNorm
control, and the MAE / MAE+VICReg objectives. Not a baseline: batches and
epochs are capped, and the frozen probe is not run here.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch

from t5_ssl.train import train


def main() -> None:
    """Run a couple of train/val steps per candidate and check the shared contract."""
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    results = {}
    methods = {
        "vicreg_2p5d_queue": [
            "data=oasis3_2p5d", "model=convnext_2p5d", "loss=vicreg",
            "augmentation=mri_2p5d_intermediate", "loss.queue_size=64",
        ],
        "byol_groupnorm": [
            "data=oasis3_2p5d", "model=byol_convnext_groupnorm", "loss=byol",
            "augmentation=mri_2p5d_intermediate", "model.backbone.in_channels=5",
        ],
        "mae_2p5d": [
            "data=oasis3_2p5d", "model=vit_2p5d", "loss=mae",
            "augmentation=mri_2p5d_intermediate",
        ],
        "mae_vicreg_2p5d": [
            "data=oasis3_2p5d", "model=vit_2p5d", "loss=mae_vicreg",
            "augmentation=mri_2p5d_intermediate", "loss.queue_size=64",
        ],
    }
    common = [
        "data.batch_size=4", "data.num_workers=0", "data.views_per_subject_epoch=1",
        "data.train_cache_capacity=16", "data.eval_cache_capacity=16",
        "trainer.max_epochs=1", "trainer.limit_train_batches=3", "trainer.limit_val_batches=2",
        "trainer.accelerator=gpu", "trainer.precision=16-mixed", "trainer.deterministic=false",
        "trainer.enable_progress_bar=false", "trainer.health_check_every_n_epochs=1",
        "optimizer.warmup_epochs=0",
    ]
    for name, overrides in methods.items():
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=common + overrides)
        result = train(cfg)
        dm, model, trainer, exp_dir = (result["datamodule"], result["model"],
                                       result["trainer"], result["exp_dir"])
        sets = [set(v) for v in dm.split_subject_ids.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
        batch = next(iter(dm.train_dataloader()))
        views = batch["image"]
        n_views = len(views) if isinstance(views, list | tuple) else 1
        path = exp_dir / "checkpoints" / "last.ckpt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        val = float(trainer.callback_metrics["val/loss"])
        assert torch.isfinite(torch.tensor(val))
        health_path = exp_dir / "representation_health.json"
        assert health_path.is_file(), f"{name}: representation_health.json missing"
        health = json.loads(health_path.read_text())
        assert len(health) >= 1 and "effective_rank" in health[0]
        results[name] = {
            "n_views": n_views,
            "view_shape": list(views[0].shape) if isinstance(views, list | tuple) else list(views.shape),
            "split_subjects": {k: len(v) for k, v in dm.split_subject_ids.items()},
            "train_steps": trainer.global_step, "val_loss": val,
            "health_last": health[-1], "checkpoint_reload": "passed",
        }
        print(name, results[name], flush=True)
    output = Path("smoke_2p5d_results.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
