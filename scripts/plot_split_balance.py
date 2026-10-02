"""Balance of the 361 / 63 / 84 split: diagnosis, age strata and years of education.

    python scripts/plot_split_balance.py

Recomputes the strata exactly as ``scripts/make_split_361_63_84.py`` did: subject
labels (AD / HC only when every OASIS3 session agrees, else "otros"), the age of the
evaluated session (primary, else earliest) and age tertiles cut *within each
diagnosis* over its 210 HC / 203 AD subjects (``np.quantile`` at 1/3 and 2/3,
``searchsorted(side="right")``); "otros" were not stratified. Education (years,
clinical CSV) was not a stratum: the third panel shows how it fell. Checks the
diagnosis counts against ``SPLIT.md`` and writes
``docs/figures_split361/split_balance.png``.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report_split361 as rep

plt = rep.plt
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = rep.ROOT
ORIGINAL = ROOT / "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
CLINICAL = ROOT / "data/dementia/mni-dataset.csv"
SUBJECT = re.compile(r"^(sub-OAS3\d+)")
DAY = re.compile(r"d(\d+)$")
SPLITS = ("train", "val", "test")
HC_SHADES = ("#a9c8f0", "#5f9ae3", "#1f5fae")
AD_SHADES = ("#f7c2a6", "#f08a58", "#b9471b")
EDU = [("≤ 12 años (secundaria)", lambda y: y <= 12, "#d7ecd9"),
       ("13–15 (universidad incompleta)", lambda y: 13 <= y <= 15, "#9fd0a6"),
       ("16 (título universitario)", lambda y: y == 16, "#4fa66a"),
       ("≥ 17 (posgrado)", lambda y: y >= 17, "#1f6b3c")]


def subjects() -> list[dict]:
    """One record per subject: split, label, age of the evaluated session, tertile, education."""
    split = {r["subject"]: r["split"] for r in csv.DictReader(
        rep.MANIFEST.open(encoding="utf-8"))}
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(ORIGINAL.open(encoding="utf-8")):
        sessions[row["subject"]].append(row)
    clinical = [r for r in csv.DictReader(CLINICAL.open(encoding="utf-8"))
                if r["study_name"].strip().upper() == "OASIS3"]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in clinical:
        groups[SUBJECT.match(row["original_code"]).group(1)].add(row["group"].strip().upper())
    by_idx = {row["idx"]: row for row in clinical}

    def label(s: str) -> str:
        only = next(iter(groups[s]))
        return only if len(groups[s]) == 1 and only in ("AD", "HC") else "otros"

    def evaluated(s: str) -> dict:
        rows = sessions[s]
        primary = [r for r in rows if r["primary"] == "1"]
        return by_idx[min(primary or rows, key=lambda r: int(DAY.search(r["session"]).group(1)))["idx"]]

    out = []
    for s in sorted(sessions):
        row = evaluated(s)
        edu = row["education"].strip()
        out.append({"subject": s, "split": split[s], "label": label(s), "age": float(row["age"]),
                    "education": float(edu) if edu not in ("", "nan") else None,
                    "sessions": sum(1 for _ in sessions[s]) if split[s] == "train" else 1})
    cuts = {}
    for diagnosis in ("HC", "AD"):
        ages = np.array([r["age"] for r in out if r["label"] == diagnosis])
        cuts[diagnosis] = np.quantile(ages, [1 / 3, 2 / 3])
    for r in out:
        r["tertile"] = (int(np.searchsorted(cuts[r["label"]], r["age"], side="right"))
                        if r["label"] in cuts else None)
    return out, cuts


def check(records: list[dict]) -> None:
    expected = rep.split_counts()
    for split in SPLITS:
        got = Counter(r["label"] for r in records if r["split"] == split)
        assert (got["HC"], got["AD"], got["otros"]) == tuple(
            expected[split][k] for k in ("HC", "AD", "otros")), (split, got)


def main() -> None:
    records, cuts = subjects()
    check(records)
    by = {sp: [r for r in records if r["split"] == sp] for sp in SPLITS}
    order = ["test", "val", "train"]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.6), sharex=True)
    fmt = lambda v: f"{v:g}"  # noqa: E731

    def row(ax, y, parts, note):
        left = 0
        for n, color, text_color, hatch in parts:
            ax.barh(y, n, left=left, height=0.58, color=color, edgecolor="white", linewidth=1.6,
                    hatch=hatch)
            if n >= 9:
                ax.text(left + n / 2, y, str(n), ha="center", va="center",
                        fontsize=8.6 if n >= 16 else 7.0, color=text_color, fontweight="semibold")
            left += n
        ax.text(left + 6, y, note, va="center", fontsize=8.6, color=rep.INK2)

    # 1. diagnosis
    ax = axes[0]
    for y, split in enumerate(order):
        c = Counter(r["label"] for r in by[split])
        sessions = sum(r["sessions"] for r in by[split])
        row(ax, y, [(c["HC"], rep.COLOR["vicreg"], "white", None),
                    (c["AD"], rep.COLOR["byol"], "white", None),
                    (c["otros"], rep.OTHER, rep.INK, None)],
            f"{len(by[split])} sujetos · {sessions} sesiones")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (rep.COLOR["vicreg"], rep.COLOR["byol"], rep.OTHER)]
    ax.legend(handles, ["HC (control sano)", "AD (Alzheimer)", "otros diagnósticos (solo SSL)"],
              ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.4)
    ax.set_title("Diagnóstico: cupos fijos en val (32 HC + 31 AD) y test (43 HC + 41 AD)",
                 loc="left", fontsize=9.6, pad=26)

    # 2. age tertiles within each diagnosis
    ax = axes[1]
    for y, split in enumerate(order):
        c = Counter((r["label"], r["tertile"]) for r in by[split])
        parts = [(c[("HC", t)], HC_SHADES[t], "white" if t else rep.INK, None) for t in range(3)]
        parts += [(c[("AD", t)], AD_SHADES[t], "white" if t else rep.INK, None) for t in range(3)]
        parts.append((c[("otros", None)], rep.OTHER, rep.INK, "///"))
        means = {d: np.mean([r["age"] for r in by[split] if r["label"] == d]) for d in ("HC", "AD")}
        row(ax, y, parts, f"edad media HC {means['HC']:.1f} · AD {means['AD']:.1f}")
    hc, ad = (np.round(cuts[d], 1) for d in ("HC", "AD"))
    labels = [f"HC < {fmt(hc[0])}", f"HC {fmt(hc[0])}–{fmt(hc[1])}", f"HC ≥ {fmt(hc[1])}",
              f"AD < {fmt(ad[0])}", f"AD {fmt(ad[0])}–{fmt(ad[1])}", f"AD ≥ {fmt(ad[1])}",
              "otros (sin estrato)"]
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (*HC_SHADES, *AD_SHADES)]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=rep.OTHER, hatch="///", edgecolor="white"))
    blank = Rectangle((0, 0), 1, 1, alpha=0)
    # legend fills column by column: interleave so that row 1 is HC (+ otros) and row 2 is AD
    legend_order = [0, 3, 1, 4, 2, 5, 6]
    ax.legend([handles[i] for i in legend_order] + [blank], [labels[i] for i in legend_order] + [""], ncol=4,
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.0, columnspacing=1.4,
              handlelength=1.4)
    ax.set_title("Edad: terciles calculados dentro de cada diagnóstico (años, sesión evaluada)",
                 loc="left", fontsize=9.6, pad=38)

    # 3. years of education (not a stratum)
    ax = axes[2]
    for y, split in enumerate(order):
        years = [r["education"] for r in by[split]]
        parts = [(sum(1 for v in years if v is not None and test(v)), color,
                  "white" if k >= 2 else rep.INK, None) for k, (_, test, color) in enumerate(EDU)]
        missing = sum(v is None for v in years)
        if missing:
            parts.append((missing, "#ffffff", rep.INK, "..."))
        known = [v for v in years if v is not None]
        row(ax, y, parts, f"media {np.mean(known):.1f} ± {np.std(known):.1f} años"
            + (f" · {missing} sin dato" if missing else ""))
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in EDU]
    ax.legend(handles, [name for name, _, _ in EDU], ncol=4, loc="lower left",
              bbox_to_anchor=(0, 1.0), fontsize=7.8, columnspacing=1.0, handlelength=1.4)
    ax.set_title("Años de estudio (no estratificado: muestra cómo quedó repartido)", loc="left",
                 fontsize=9.6, pad=26)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.set_yticks(range(3), order)
        ax.set_xlim(0, 480)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("sujetos", fontsize=8.6)
    fig.tight_layout(h_pad=1.6)
    rep.save(fig, "split_balance.png")
    for split in SPLITS:
        c = Counter((r["label"], r["tertile"]) for r in by[split])
        print(split, {f"{d}{t + 1}": c[(d, t)] for d in ("HC", "AD") for t in range(3)},
              "otros", c[("otros", None)])
    print("cuts HC", cuts["HC"], "AD", cuts["AD"])


if __name__ == "__main__":
    main()
