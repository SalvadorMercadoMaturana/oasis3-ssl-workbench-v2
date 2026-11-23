"""Frozen probes every N epochs during training (t5_ssl.callbacks.periodic_probe)."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from t5_ssl.callbacks.periodic_probe import PeriodicProbe
from t5_ssl.evaluation import run_scoring


def fake_main() -> dict:
    return {"cv": {"clean": {"metrics": {"auc": 0.9, "f1": 0.6, "aupr": 0.3}}},
            "test": {"test_auc": 0.8, "test_f1_macro": 0.7, "test_average_precision": 0.6},
            "protocol_v2": {"cv_rep": {"mean3": 0.7}}}


def test_scores_a_copy_every_n_epochs_but_not_the_last_and_restores_state(
        monkeypatch, tmp_path) -> None:
    calls = []

    def fake_evaluate(model_name, encoders, dm, cfg, spatial):
        encoder = encoders["main"]
        with torch.no_grad():  # the evaluation may change its copy freely
            encoder.weight.zero_()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        random.random(), np.random.rand(), torch.rand(1)  # and draw random numbers
        calls.append((model_name, spatial, dm))
        return {"main": fake_main()}

    monkeypatch.setattr(run_scoring, "evaluate_encoders", fake_evaluate)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    module = SimpleNamespace(teacher_backbone=nn.Linear(2, 2))
    weight = module.teacher_backbone.weight.detach().clone()
    probe = PeriodicProbe(cfg=None, model_name="dinov2", every_n_epochs=50,
                          encoder_attr="teacher_backbone")
    trainer = SimpleNamespace(max_epochs=200, datamodule="dm", log_dir=str(tmp_path))
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    states = random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()
    for epoch in range(200):
        trainer.current_epoch = epoch
        probe.on_train_epoch_end(trainer, module)
    assert calls == [("dinov2", False, "dm")] * 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "probe_epoch050.json", "probe_epoch100.json", "probe_epoch150.json"]
    doc = json.loads((tmp_path / "probe_epoch100.json").read_text(encoding="utf-8"))
    assert doc["epoch"] == 100 and doc["encoder"] == "teacher_backbone"
    assert abs(doc["cv_mean3"] - 0.6) < 1e-9 and abs(doc["test_mean3"] - 0.7) < 1e-9
    assert torch.equal(module.teacher_backbone.weight, weight)
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1])
    assert torch.equal(torch.get_rng_state(), states[2])
