"""Frozen-probe scores of the encoder every few epochs of an SSL run (``--probe-every``).

At the end of every ``every_n_epochs``-th epoch, except the last one (``result.json`` scores
it), a copy of the encoder that ``train.py`` evaluates (the backbone; DINOv2's EMA
teacher) goes through the evaluation of the end of the run
(``t5_ssl.evaluation.run_scoring.evaluate_encoders``: probe v1 and protocol v2). The
``main`` block it returns, plus ``epoch``, ``cv_mean3`` and ``test_mean3``, is written to
``<exp_dir>/probe_epoch<NNN>.json``, so a long run shows how the representation evolves.

Training is unaffected: only the copy is scored, outside autocast, and the random states
(Python, NumPy, torch CPU and CUDA) and the TF32 switches the evaluation turns off are
restored afterwards.
"""

from __future__ import annotations

import contextlib
import copy
import json
import random
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch


class PeriodicProbe(L.Callback):
    """Score a frozen copy of ``pl_module.<encoder_attr>`` every ``every_n_epochs`` epochs."""

    def __init__(self, cfg: Any, model_name: str, every_n_epochs: int,
                 encoder_attr: str = "backbone") -> None:
        self.cfg, self.model_name, self.encoder_attr = cfg, model_name, encoder_attr
        self.every = max(1, int(every_n_epochs))

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        done = trainer.current_epoch + 1
        if done % self.every == 0 and done < trainer.max_epochs:
            self.score(trainer, pl_module, done)

    def score(self, trainer: L.Trainer, pl_module: L.LightningModule, epoch: int) -> dict:
        """Evaluate a copy of the encoder and write ``probe_epoch<epoch>.json``."""
        from t5_ssl.evaluation.run_scoring import evaluate_encoders, mean3_cv, mean3_test
        encoder = copy.deepcopy(getattr(pl_module, self.encoder_attr))
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda = torch.cuda.is_available()
        no_autocast = (torch.autocast(device_type="cuda", enabled=False) if cuda
                       else contextlib.nullcontext())
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else []), \
                    no_autocast:
                main = evaluate_encoders(self.model_name, {"main": encoder},
                                         trainer.datamodule, self.cfg, spatial=False)["main"]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
            del encoder
            if cuda:
                torch.cuda.empty_cache()
        doc = {"epoch": epoch, "encoder": self.encoder_attr, "cv_mean3": mean3_cv(main),
               "test_mean3": mean3_test(main), **main}
        path = Path(trainer.log_dir or ".") / f"probe_epoch{epoch:03d}.json"
        path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
        print(f"[probe] epoch {epoch}: {path.as_posix()}", flush=True)
        return doc
