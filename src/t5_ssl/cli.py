"""Small OASIS3 command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_IMAGE_ROOT, DEFAULT_LMDB


def main(argv: list[str] | None = None) -> None:
    """Inspect OASIS3 data or forward Hydra overrides to training."""
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "train":
        from t5_ssl.train import main as train_main
        sys.argv = [sys.argv[0], *args[1:]]
        train_main()
        return
    parser = argparse.ArgumentParser(prog="t5-ssl")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("data-info", help="validate and summarize local OASIS3 data")
    info.add_argument("--source", choices=("auto", "lmdb", "nifti"), default="auto")
    info.add_argument(
        "--csv-path",
        default=DEFAULT_CSV,
    )
    info.add_argument(
        "--lmdb-path",
        default=DEFAULT_LMDB,
    )
    info.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
    )
    info.add_argument("--manifest-path", default=None)
    parsed = parser.parse_args(args)
    if parsed.command == "data-info":
        source = parsed.source
        if source == "auto":
            source = "lmdb" if Path(parsed.lmdb_path, "data.mdb").is_file() else "nifti"
        if source == "lmdb":
            from t5_ssl.data.oasis3_lmdb_datamodule import OASIS3LMDBDataModule

            dm = OASIS3LMDBDataModule(
                csv_path=parsed.csv_path,
                lmdb_path=parsed.lmdb_path,
                manifest_path=parsed.manifest_path,
            )
        else:
            from t5_ssl.data.oasis3_datamodule import OASIS3DataModule

            dm = OASIS3DataModule(csv_path=parsed.csv_path, image_root=parsed.image_root)
        dm.setup()
        print(f"OASIS3 source: {source}")
        for split, ids in dm.split_subject_ids.items():
            print(f"{split}: {len(ids)} subjects")


if __name__ == "__main__":
    main()
