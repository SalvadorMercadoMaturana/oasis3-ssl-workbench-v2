"""Portable defaults for the local OASIS3 data checkout."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("OASIS3_DATA_ROOT", PROJECT_ROOT / "data" / "dementia"))
DEFAULT_CSV = str(Path(os.environ.get("OASIS3_CSV", DATA_ROOT / "mni-dataset.csv")))
DEFAULT_LMDB = str(Path(os.environ.get("OASIS3_LMDB", DATA_ROOT / "mni-dataset.lmdb")))
DEFAULT_IMAGE_ROOT = str(Path(os.environ.get("OASIS3_IMAGE_ROOT", DATA_ROOT / "oasis3")))


def rebase_missing_lmdb_paths(
    csv_path: str | Path,
    lmdb_path: str | Path,
    manifest_path: str | Path | None,
    *,
    project_root: str | Path = PROJECT_ROOT,
    default_csv: str | Path = DEFAULT_CSV,
    default_lmdb: str | Path = DEFAULT_LMDB,
) -> tuple[str, str, str | None]:
    """Rebase stale server paths while preserving paths that still exist.

    Experiment configs are immutable records and may contain absolute paths from
    the machine that trained them.  Evaluation uses the local dataset defaults
    only when those recorded paths no longer exist.  A missing manifest is
    located by basename inside the project campaigns.
    """
    root = Path(project_root)
    csv = Path(csv_path)
    lmdb = Path(lmdb_path)
    if not csv.is_file() and Path(default_csv).is_file():
        csv = Path(default_csv)
    if not (lmdb / "data.mdb").is_file() and (Path(default_lmdb) / "data.mdb").is_file():
        lmdb = Path(default_lmdb)

    manifest: Path | None = Path(manifest_path) if manifest_path is not None else None
    if manifest is not None and not manifest.is_file():
        direct = root / manifest.name
        candidates = ([direct] if direct.is_file() else []) + sorted(
            path for path in root.glob(f"campaign*/{manifest.name}") if path.is_file()
        )
        if len(candidates) == 1:
            manifest = candidates[0]
    return str(csv), str(lmdb), str(manifest) if manifest is not None else None
