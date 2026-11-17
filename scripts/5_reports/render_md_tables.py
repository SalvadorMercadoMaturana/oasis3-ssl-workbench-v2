"""Render a markdown report (title, paragraphs, pipe tables) as a PNG figure.

Keeps every table of the file with the report palette: model names in their colour and
rows grouped by model, recipe names (backticks) in monospace, the current recipe
("(actual)") in grey italics, the best CV/test value per model in bold, and confidence
intervals that exclude zero in green (above) or red (below).

Usage: ``python scripts/5_reports/render_md_tables.py <file.md> [<file.md> ...]`` writes
``<file>.png`` next to each markdown file.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLOR = {"vicreg": "#2a78d6", "byol": "#eb6834", "dinov2": "#1baf7a"}
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GOOD, BAD = "#1e8449", "#b03a2e"
WIDTH = 11.5           # figure width, inches
ROW, HEAD = 0.25, 0.42  # row and header heights, inches
CHAR = 0.068           # inches per character at the cell font size

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8})


def parse(path: Path) -> list[tuple[str, object]]:
    """Blocks in order: ("h1"|"h2"|"p", text) or ("table", (header, rows))."""
    blocks: list[tuple[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    table.append(cells)
                i += 1
            blocks.append(("table", (table[0], table[1:])))
            continue
        if line.startswith("## "):
            blocks.append(("h2", line[3:]))
        elif line.startswith("# "):
            blocks.append(("h1", line[2:]))
        elif line:
            blocks.append(("p", line))
        i += 1
    return blocks


def clean(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\*(\S[^*]*?)\*", r"\1", text)  # *emphasis*, but keep a lone "*"


def model_of(row: list[str]) -> str | None:
    for cell in row[:2]:
        key = clean(cell).lower().split("_")[0].split(" ")[0]
        if key in COLOR:
            return key
    return None


def ci_sign(cell: str) -> int:
    m = re.search(r"\[([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+)\]", cell)
    if not m:
        return 0
    lo, hi = float(m.group(1)), float(m.group(2))
    return 1 if lo > 0 else -1 if hi < 0 else 0


def leading_number(cell: str) -> float | None:
    m = re.match(r"\s*([+-]?\d*\.\d+|\d+)", clean(cell))
    return float(m.group(1)) if m else None


def best_cells(header: list[str], rows: list[list[str]]) -> set[tuple[int, int]]:
    """(row, col) of the highest CV/test value of each model."""
    cols = [j for j, h in enumerate(header) if re.search(r"\b(CV|Test|test)\b", h) and "Δ" not in h]
    best: set[tuple[int, int]] = set()
    for j in cols:
        by_model: dict[str, tuple[float, int]] = {}
        for i, row in enumerate(rows):
            m, v = model_of(row), leading_number(row[j]) if j < len(row) else None
            if m is None or v is None:
                continue
            if m not in by_model or v > by_model[m][0]:
                by_model[m] = (v, i)
        best |= {(i, j) for _, i in by_model.values()}
    return best


def layout(blocks) -> float:
    h = 0.35
    for kind, content in blocks:
        if kind == "h1":
            h += 0.5
        elif kind == "h2":
            h += 0.42
        elif kind == "p":
            h += 0.2 * len(textwrap.wrap(clean(content), 170)) + 0.08
        else:
            h += HEAD + ROW * len(content[1]) + 0.25
    return h + 0.15


def draw(path: Path) -> Path:
    blocks = parse(path)
    height = layout(blocks)
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    x0, x1 = 0.35, WIDTH - 0.35
    y = 0.35
    for kind, content in blocks:
        if kind == "h1":
            ax.text(x0, y + 0.18, clean(content), fontsize=14, fontweight="bold", color="#1B4F72", va="center")
            y += 0.5
        elif kind == "h2":
            ax.text(x0, y + 0.17, clean(content), fontsize=11, fontweight="bold", color="#2E86AB", va="center")
            y += 0.42
        elif kind == "p":
            for line in textwrap.wrap(clean(content), 170):
                ax.text(x0, y + 0.1, line, fontsize=8.3, color=INK2, va="center")
                y += 0.2
            y += 0.08
        else:
            header, rows = content
            n = len(header)
            chars = [max(len(clean(header[j])), *(len(clean(r[j])) for r in rows if j < len(r))) for j in range(n)]
            widths = [c * CHAR + 0.22 for c in chars]
            scale = (x1 - x0) / sum(widths)
            widths = [w * scale for w in widths]
            numeric = [all(leading_number(r[j]) is not None or clean(r[j]) in ("—", "-", "")
                           for r in rows if j < len(r)) and j > 0 for j in range(n)]
            best = best_cells(header, rows)
            lefts = [x0 + sum(widths[:j]) for j in range(n)]
            # header
            ax.plot([x0, x1], [y + HEAD, y + HEAD], color=INK, lw=0.9)
            ax.plot([x0, x1], [y, y], color=INK, lw=0.9)
            for j, h in enumerate(header):
                wrapped = textwrap.fill(clean(h).replace("/", "/ "), max(8, int(widths[j] / CHAR) - 1),
                                        break_long_words=False).replace("/ ", "/")
                xa = lefts[j] + widths[j] - 0.1 if numeric[j] else lefts[j] + 0.1
                ax.text(xa, y + HEAD / 2, wrapped, fontsize=7.8, fontweight="bold", color=INK,
                        ha="right" if numeric[j] else "left", va="center", linespacing=1.0)
            y += HEAD
            prev = None
            for i, row in enumerate(rows):
                model = model_of(row)
                actual = any("(actual)" in c for c in row)
                if model is not None:
                    ax.add_patch(Rectangle((x0, y), x1 - x0, ROW, color=COLOR[model],
                                           alpha=0.05 if i % 2 else 0.1, lw=0))
                if prev is not None and model != prev:
                    ax.plot([x0, x1], [y, y], color=RULE, lw=0.8)
                prev = model
                for j, cell in enumerate(row[:n]):
                    text = clean(cell)
                    mono = "`" in cell
                    color, weight, style = INK, "normal", "normal"
                    if j == 0 and model is not None and text.lower().split("_")[0] in COLOR:
                        color, weight = COLOR[model], "bold"
                    if actual:
                        color, style = (color if j == 0 else INK2), "italic"
                    if (i, j) in best:
                        weight = "bold"
                    sign = ci_sign(cell)
                    if sign:
                        color, weight = (GOOD if sign > 0 else BAD), "bold"
                    if "✓" in text:
                        color = GOOD
                    xa = lefts[j] + widths[j] - 0.1 if numeric[j] else lefts[j] + 0.1
                    ax.text(xa, y + ROW / 2, text, fontsize=8, color=color, fontweight=weight, fontstyle=style,
                            family="monospace" if mono else None, ha="right" if numeric[j] else "left",
                            va="center")
                y += ROW
            ax.plot([x0, x1], [y, y], color=INK, lw=0.9)
            y += 0.25
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        draw(Path(arg))
