"""predict_ad.py --calibrate-site: site statistics only from scans that pass the input QC."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/6_deploy/predict_ad.py"
SPEC = spec_from_file_location("predict_ad", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
predict_ad = module_from_spec(SPEC)
SPEC.loader.exec_module(predict_ad)


def test_qc_rejected_scans_do_not_shift_the_site_statistics(tmp_path):
    rng = np.random.default_rng(0)
    good = rng.normal(size=(30, 4))
    bad = rng.normal(loc=50.0, size=(5, 4))  # another intensity normalization
    seeds = [np.concatenate([good, bad]), np.concatenate([good, bad]) * 2]
    passed = np.r_[np.ones(30, bool), np.zeros(5, bool)]
    names = [f"scan{i}" for i in range(35)]
    out = tmp_path / "site.json"
    stats = predict_ad.calibrate_site(out, names, seeds, passed)
    assert json.loads(out.read_text(encoding="utf-8")) == stats
    assert stats["scans"] == 30 and stats["qc_rejected"] == 5
    np.testing.assert_allclose(stats["seeds"][0]["mean"], good.mean(0))
    np.testing.assert_allclose(stats["seeds"][1]["sd"], (2 * good).std(0) + 1e-6)


def test_calibration_needs_scans_that_pass_the_qc(tmp_path):
    with pytest.raises(SystemExit, match="too few"):
        predict_ad.calibrate_site(tmp_path / "site.json", ["a", "b"], [np.zeros((2, 3))],
                                  np.array([True, False]))
