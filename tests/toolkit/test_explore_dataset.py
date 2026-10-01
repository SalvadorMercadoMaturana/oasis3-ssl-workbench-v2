"""Regression tests for OASIS3 CSV statistics."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "dataset_oasis3"
    / "scripts"
    / "explore_dataset.py"
)
SPEC = spec_from_file_location("explore_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_subject_is_derived_from_original_code() -> None:
    """Rows without a subject column should still count distinct OASIS subjects."""
    records = [
        {"original_code": "sub-OAS30001ses-d0001", "group": "HC"},
        {"original_code": "sub-OAS30001ses-d0100", "group": "HC"},
        {"original_code": "sub-OAS30002ses-d0001", "group": "AD"},
    ]

    stats = MODULE.analyze_dataset(records)

    assert stats["total_records"] == 3
    assert stats["total_subjects"] == 2
    assert stats["records_per_subject"]["sub-OAS30001"] == 2
    assert stats["unique_sessions_per_subject"]["sub-OAS30001"] == 2
