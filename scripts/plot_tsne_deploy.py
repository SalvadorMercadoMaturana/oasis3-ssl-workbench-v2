"""t-SNE of the frozen SSL encoders on the representative test of split_deploy.

    python scripts/plot_tsne_deploy.py [--runs campaign_ssl3d_runpod/deploy/runs] [--seed 1]

One panel per method (its best backbone, tag ``dep50_ssl``): the embeddings of the 178
test subjects (41 AD + 137 HC, OASIS3's own prevalence), standardized with the labeled
train+val statistics as the probe does, projected with t-SNE. No label is used to build
the map; the colors are only drawn on top. The title gives the frozen probe's test AUROC
of the same seed. Writes ``campaign_ssl3d_runpod/figures/dep50_tsne.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402

plt = base.plt
RUNS = {"vicreg": ("vicreg_campaign_augval__geo__bbconvnext_tiny__roi32", "ConvNeXt-T"),
        "byol": ("byol_gn_dp0.3_dq0_wd0.01__noise__bbconvnext_tiny__roi32", "ConvNeXt-T"),
        "dinov2": ("dinov2_campaign__elastic_tio_gpu__bbvit_small__roimulti3", "ViT-S")}
AD_COLOR, HC_COLOR = "#c4314b", "#2a78d6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "deploy" / "runs")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, len(RUNS), figsize=(15, 5.2))
    for ax, (method, (name, backbone)) in zip(axes, RUNS.items(), strict=True):
        run = args.runs / "dep50_ssl" / f"{name}_s{args.seed}"
        z = np.load(run / "embeddings.npz")
        auc = json.loads((run / "result.json").read_text(encoding="utf-8"))["test"]["test_auc"]
        fit = np.concatenate([z["train_x"], z["val_x"]])
        x = (z["test_x"] - fit.mean(0)) / (fit.std(0) + 1e-6)
        y = z["test_y"].astype(int)
        xy = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(x)
        ax.scatter(*xy[y == 0].T, s=22, color=HC_COLOR, alpha=0.55, linewidths=0,
                   label=f"HC ({(y == 0).sum()})")
        ax.scatter(*xy[y == 1].T, s=34, color=AD_COLOR, alpha=0.95, edgecolors="white",
                   linewidths=0.6, label=f"AD ({y.sum()})")
        ax.set_title(f"{base.LABEL[method]} · {backbone}\nAUROC del probe en test: {auc:.3f}",
                     fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        base.style(ax)
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10,
               markerscale=1.4)
    fig.suptitle(f"t-SNE del encoder SSL congelado (semilla {args.seed}, sin etiquetas): test "
                 "representativo, 41 AD + 137 HC (23 % AD)", fontsize=11.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out = base.FIGURES / "dep50_tsne.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
