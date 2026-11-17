"""t5_ssl.evaluation.input_qc: intensity statistic and accepted range of the deployed model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from t5_ssl.evaluation import input_qc


def test_qc_intensity_uses_channel_0_center_cube() -> None:
    cubes = np.ones((1, 2, 6, 6, 6), dtype=np.float32)
    cubes[0, 0, 2:4, 2:4, 2:4] = 0.5  # the evaluated 2^3 center of a 6^3 extracted cube
    cubes[0, 1] = 9.0  # other channels do not count
    assert input_qc.qc_intensity(cubes, roi_size=2) == pytest.approx([0.5])


def _fake_cache(root: Path, values: dict[str, float], studies: dict[str, str]) -> None:
    (root / "data/roi3d").mkdir(parents=True)
    (root / "data/dementia").mkdir(parents=True)
    keys = list(values)
    (root / "data/roi3d/hippo_s32.json").write_text(json.dumps({"idx": keys}), encoding="utf-8")
    cubes = np.stack([np.full((2, 38, 38, 38), values[k], dtype=np.float16) for k in keys])
    np.save(root / "data/roi3d/hippo_s32.npy", cubes)
    with (root / "data/dementia/mni-dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "study_name"])
        writer.writeheader()
        writer.writerows({"idx": k, "study_name": v} for k, v in studies.items())


def test_qc_range_is_oasis3_only_and_widened(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cache(tmp_path, {"a": 0.25, "b": 0.5, "c": 0.875},
                {"a": "OASIS3", "b": " oasis3 ", "c": "ADNI"})
    monkeypatch.setattr(input_qc, "ROOT", tmp_path)
    accepted = input_qc.qc_range("hippo_s32")
    assert accepted["oasis3_sessions"] == 2
    assert accepted["lo"] == pytest.approx(0.25 - input_qc.QC_MARGIN)
    assert accepted["hi"] == pytest.approx(0.5 + input_qc.QC_MARGIN)
    assert input_qc.cache_intensity("hippo_s32", ["c"]) == pytest.approx([0.875])
