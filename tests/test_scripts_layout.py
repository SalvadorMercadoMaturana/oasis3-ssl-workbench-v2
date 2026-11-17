"""Layout of scripts/: one folder per pipeline stage, no imports across folders, all documented.

A script may import the ``t5_ssl`` package and scripts of its own folder only; anything two
stages share belongs in the package. Every script is listed in its folder's README
(``scripts/README.md`` for the stages, ``scripts/legacy/README.md`` for the archive).
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
STAGES = ("1_data", "2_splits", "3_train", "4_evaluate", "5_reports", "6_deploy")
# Already broken before the 2026-11-16 restructuring (see scripts/legacy/README.md).
KNOWN_BROKEN = {"legacy/report_atrophy.py"}
ALL_PY = sorted(SCRIPTS.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(SCRIPTS).as_posix()


def _imported_names(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_folders() -> None:
    folders = sorted(p.name for p in SCRIPTS.iterdir() if p.is_dir() and p.name != "__pycache__")
    assert folders == sorted((*STAGES, "legacy"))
    assert not list(SCRIPTS.glob("*.py")), "every script lives in a stage folder or legacy/"


def test_script_names_are_unique() -> None:
    repeated = [n for n, k in Counter(p.stem for p in ALL_PY).items() if k > 1]
    assert not repeated, f"same script name in two folders: {repeated}"


@pytest.mark.parametrize("path", ALL_PY, ids=_rel)
def test_imports_stay_in_their_folder(path: Path) -> None:
    if _rel(path) in KNOWN_BROKEN:
        pytest.skip("known broken legacy script")
    folder_of = {p.stem: p.parent.name for p in ALL_PY}
    foreign = sorted(f"{folder_of[name]}/{name}" for name in _imported_names(path)
                     if name in folder_of and folder_of[name] != path.parent.name)
    assert not foreign, f"{_rel(path)} imports scripts of other folders: {foreign}"


@pytest.mark.parametrize("readme, folders", [("README.md", STAGES), ("legacy/README.md", ("legacy",))])
def test_every_script_is_documented(readme: str, folders: tuple[str, ...]) -> None:
    text = (SCRIPTS / readme).read_text(encoding="utf-8")
    scripts = [p for folder in folders for p in sorted((SCRIPTS / folder).iterdir())
               if p.suffix in (".py", ".sh")]
    missing = [_rel(p) for p in scripts if f"`{p.name}`" not in text]
    assert not missing, f"not listed in scripts/{readme}: {missing}"
