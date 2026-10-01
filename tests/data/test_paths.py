from pathlib import Path

from t5_ssl.data.paths import rebase_missing_lmdb_paths


def test_rebases_stale_server_paths(tmp_path: Path) -> None:
    csv = tmp_path / "data" / "mni-dataset.csv"
    lmdb = tmp_path / "data" / "mni-dataset.lmdb"
    manifest = tmp_path / "campaign_full50" / "split.csv"
    csv.parent.mkdir()
    csv.touch()
    lmdb.mkdir()
    (lmdb / "data.mdb").touch()
    manifest.parent.mkdir()
    manifest.touch()

    actual = rebase_missing_lmdb_paths(
        "/home/datasets/dementia/mni-dataset.csv",
        "/home/datasets/dementia/mni-dataset.lmdb",
        "/home/remote/project/split.csv",
        project_root=tmp_path,
        default_csv=csv,
        default_lmdb=lmdb,
    )

    assert actual == (str(csv), str(lmdb), str(manifest))


def test_preserves_existing_paths(tmp_path: Path) -> None:
    csv = tmp_path / "original.csv"
    lmdb = tmp_path / "original.lmdb"
    manifest = tmp_path / "original-split.csv"
    csv.touch()
    lmdb.mkdir()
    (lmdb / "data.mdb").touch()
    manifest.touch()

    actual = rebase_missing_lmdb_paths(
        csv,
        lmdb,
        manifest,
        project_root=tmp_path,
        default_csv=tmp_path / "unused.csv",
        default_lmdb=tmp_path / "unused.lmdb",
    )

    assert actual == (str(csv), str(lmdb), str(manifest))
