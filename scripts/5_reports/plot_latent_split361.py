"""Figures of ``scripts/5_reports/analyze_latent_split361.py`` (run its ``compute`` stage first).

Writes to ``docs/figures_split361/``:

- ``latent.png``: AD/HC subjects in each champion's latent space (PCA, t-SNE and
  the out-of-fold probe axis);
- ``latent_corr.png``: Spearman correlation of the first principal components
  (and of the whole embedding, cross-validated) with clinical, anatomical and
  image variables;
- ``noise_checks.png``: four checks that the representation is not image noise.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import build_report_split361 as rep
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = rep.RUNPOD / "latent_split361"
SUFFIX = ""  # appended to every figure name (analyze_latent_split361.py --suffix)
MODELS = ("vicreg", "byol", "dinov2")
HC, AD = rep.COLOR["vicreg"], rep.COLOR["byol"]


def save(fig: plt.Figure, name: str) -> None:
    rep.save(fig, name.replace(".png", f"{SUFFIX}.png"))


def load() -> tuple[dict, dict, dict]:
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    v = np.load(OUT / "variables.npz")
    variables = {"subjects": v["subjects"], "names": list(v["names"]), "values": v["values"],
                 "labels": v["labels"], "splits": v["splits"]}
    emb = {(m, role): dict(np.load(OUT / f"{m}_{role}.npz")) for m in MODELS for role in ("ssl", "init")}
    return results, variables, emb


def scatter(ax: plt.Axes, xy: np.ndarray, y: np.ndarray) -> None:
    for label, color, name in ((0, HC, "HC"), (1, AD, "AD")):
        pts = xy[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, alpha=0.75, linewidths=0, label=name)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_latent(results: dict, variables: dict, emb: dict) -> None:
    y = variables["labels"][variables["labels"] >= 0]
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.6))
    for c, model in enumerate(MODELS):
        e = emb[(model, "ssl")]
        r = results["models"][model]["ssl"]
        pv = r["pca_var"]
        ax = axes[0, c]
        scatter(ax, e["pcs"][:, :2], y)
        ax.set_title(f"{rep.LABEL[model]}", fontsize=10, color=rep.COLOR[model])
        ax.set_xlabel(f"PC1 ({100 * pv[0]:.0f} % de la varianza)", fontsize=8)
        ax.set_ylabel(f"PC2 ({100 * pv[1]:.0f} %)", fontsize=8)
        ax = axes[1, c]
        scatter(ax, e["tsne"], y)
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax = axes[2, c]
        s = e["scores"]
        bins = np.linspace(s.min(), s.max(), 26)
        for label, color, name in ((0, HC, "HC"), (1, AD, "AD")):
            ax.hist(s[y == label], bins=bins, color=color, alpha=0.55, label=name)
        ax.axvline(0, color=rep.INK2, linewidth=1, linestyle=":")
        ax.set_xlabel("puntaje del probe (fuera de muestra)", fontsize=8)
        ax.set_ylabel("sujetos", fontsize=8)
        ax.text(0.03, 0.95, f"AUROC {r['oof_auroc']:.2f}", transform=ax.transAxes, fontsize=8.5,
                va="top", color=rep.INK, fontweight="semibold")
    for r, text in enumerate(("1 · Lo que más varía (PCA): ¿AD y HC se separan solos?",
                              "2 · Vecindarios (t-SNE): ¿los vecinos comparten diagnóstico?",
                              "3 · La dirección que encuentra el probe lineal")):
        axes[r, 0].annotate(text, xy=(0, 1.13 if r else 1.2), xycoords="axes fraction", fontsize=9.5,
                            fontweight="semibold", color=rep.INK)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["HC (control sano)", "AD (Alzheimer)"], ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), markerscale=2)
    fig.subplots_adjust(hspace=0.5, wspace=0.25, bottom=0.08, top=0.93)
    save(fig, "latent.png")


def fig_corr(results: dict, variables: dict, emb: dict) -> None:
    adhc = variables["labels"] >= 0
    names = ["diagnóstico (AD = 1)"] + variables["names"]
    values = np.column_stack([variables["labels"][adhc], variables["values"][adhc]])
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        e = emb[(model, "ssl")]
        pv = results["models"][model]["ssl"]["pca_var"]
        cols = [f"PC{k + 1}\n{100 * pv[k]:.0f} %" for k in range(5)] + ["embedding\ncompleto"]
        mat = np.full((len(names), len(cols)), np.nan)
        for i in range(len(names)):
            t = values[:, i]
            ok = np.isfinite(t)
            for k in range(5):
                mat[i, k] = spearmanr(e["pcs"][ok, k], t[ok]).statistic
            pred = e["scores"] if i == 0 else e["ridge"][:, i - 1]
            ok2 = ok & np.isfinite(pred)
            mat[i, 5] = spearmanr(pred[ok2], t[ok2]).statistic
        ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                v = mat[i, k]
                ax.text(k, i, f"{v:+.2f}".replace("+0.", "0.").replace("-0.", "−0."), ha="center",
                        va="center", fontsize=7.5, color="white" if abs(v) > 0.55 else rep.INK)
        ax.axvline(4.5, color="white", linewidth=3)
        ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
        ax.set_yticks(range(len(names)), names, fontsize=8.5)
        ax.set_title(rep.LABEL[model], color=rep.COLOR[model], fontsize=10)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.5, -0.03, "Correlación de Spearman en los 413 sujetos AD/HC (semilla 1). PC = componente principal del "
             "embedding (% de varianza). «Embedding completo»: correlación entre la variable y su predicción\n"
             "lineal desde todo el vector, fuera de muestra (CV 5 pliegues; para el diagnóstico, el puntaje del probe). "
             "Volúmenes en espacio MNI, desde la segmentación.", ha="center", fontsize=8, color=rep.INK2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "latent_corr.png")


def fig_tsne_vars(results: dict, variables: dict, emb: dict, keep: list[str] | None = None,
                  ncols: int = 4, name: str = "latent_tsne_vars.png") -> None:
    """Plot the t-SNE map of each champion coloured by every variable (percentile colour scale).

    ``keep`` restricts the variables (the diagnosis panel is always first); panels whose
    cross-validated r reaches the diagnosis' r get a frame in the model's colour.
    """
    import math

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from scipy.stats import rankdata
    adhc = variables["labels"] >= 0
    y = variables["labels"][adhc]
    cols = [i for i, n in enumerate(variables["names"]) if keep is None or n in keep]
    names = ["diagnóstico"] + [variables["names"][i] for i in cols]
    values = np.column_stack([y, variables["values"][adhc][:, cols]])
    cmap = plt.get_cmap("viridis")
    rpm = math.ceil(len(names) / ncols)  # rows per model
    width = 2.15 * ncols + 0.0
    height = 2.1 * 3 * rpm + 0.9 + 0.4
    fig, axes = plt.subplots(3 * rpm, ncols, figsize=(width, height), squeeze=False)
    for m, model in enumerate(MODELS):
        e = emb[(model, "ssl")]
        xy = e["tsne"]
        rs: list[float] = []
        for v in range(len(names)):
            ax = axes[rpm * m + v // ncols, v % ncols]
            t = values[:, v]
            ok = np.isfinite(t)
            if v == 0:
                scatter(ax, xy, y)
                r = spearmanr(e["scores"], y).statistic
            else:
                ax.scatter(xy[~ok, 0], xy[~ok, 1], s=5, color=rep.BASELINE, linewidths=0)
                pct = (rankdata(t[ok]) - 1) / (ok.sum() - 1)
                ax.scatter(xy[ok, 0], xy[ok, 1], s=6, c=cmap(pct), linewidths=0)
                pred = e["ridge"][:, cols[v - 1]]
                both = ok & np.isfinite(pred)
                r = spearmanr(pred[both], t[both]).statistic
            ax.set_xticks([])
            ax.set_yticks([])
            rs.append(r)
        for v in range(len(names), rpm * ncols):  # unused slots
            axes[rpm * m + v // ncols, v % ncols].axis("off")
        # frame the variables the embedding encodes at least as well as the diagnosis
        for v, (label, r) in enumerate(zip(names, rs, strict=True)):
            ax = axes[rpm * m + v // ncols, v % ncols]
            strong = v > 0 and r >= rs[0]
            for spine in ax.spines.values():
                spine.set_visible(v == 0 or strong)
                spine.set_color(rep.COLOR[model] if strong else rep.INK2)
                spine.set_linewidth(2.4 if strong else 1.2)
                spine.set_linestyle("-" if strong else (0, (3, 2)))
            ax.set_title(label, fontsize=8.5, pad=3, fontweight="bold" if v == 0 or strong else "normal",
                         color=rep.INK if v == 0 or strong else rep.INK2)
            ax.text(0.03, 0.04, f"r = {r:.2f}", transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold" if strong else "normal", color=rep.INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
        axes[rpm * m, 0].annotate(rep.LABEL[model], xy=(-0.12, 0.5 if rpm == 1 else -0.1),
                                  xycoords="axes fraction", rotation=90, fontsize=12, fontweight="bold",
                                  color=rep.COLOR[model], ha="center", va="center")
    fig.subplots_adjust(hspace=0.28, wspace=0.06, left=0.06 * 4 / ncols * 0.75 + 0.015, right=0.99,
                        top=1 - 0.4 / height, bottom=0.94 / height)
    for m in range(1, 3):  # thin rule between models
        yline = (axes[rpm * m - 1, 0].get_position().y0 + axes[rpm * m, 0].get_position().y1) / 2
        fig.add_artist(plt.Line2D([0.03, 0.99], [yline, yline], color=rep.BASELINE, lw=1))
    cax = fig.add_axes((0.27, 0.47 / height, 0.32 if ncols > 3 else 0.27, 0.12 / height))
    bar = fig.colorbar(ScalarMappable(Normalize(0, 100), cmap), cax=cax, orientation="horizontal")
    bar.set_label("percentil de la variable entre los 413 sujetos (gris: dato faltante)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["HC", "AD"], loc="lower left", bbox_to_anchor=(0.04, 0.24 / height), ncol=2,
               fontsize=8, markerscale=2, title="diagnóstico", title_fontsize=8)
    fig.text(0.985, 0.4 / height, "marco de color: el embedding predice esa\nvariable al menos tan bien como "
             "el diagnóstico\n(r ≥ r del diagnóstico, marco punteado)", ha="right", va="center", fontsize=7.8,
             color=rep.INK2)
    save(fig, name)


def mean_cube() -> np.ndarray:
    cache, _, index = rep.cache("hippo_s32")
    test = [r["idx"] for r in rep.manifest() if r["split"] == "test"]
    m = (cache.shape[-1] - 32) // 2
    core = slice(m, m + 32)
    return np.mean([cache[index[i]][0, core, core, core].astype(np.float32) for i in test], axis=0)


def fig_noise(results: dict, variables: dict, emb: dict) -> None:
    fig = plt.figure(figsize=(11.6, 3.7))
    top = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1, 1, 1.1])
    # (a) robustness to added noise
    ax = fig.add_subplot(top[0])
    sigmas = [float(s) for s in results["models"]["vicreg"]["ssl"]["noise_auroc"]]
    for model in MODELS:
        for role, style in (("ssl", "-"), ("init", ":")):
            vals = list(results["models"][model][role]["noise_auroc"].values())
            ax.plot(sigmas, vals, style, color=rep.COLOR[model], marker="o" if role == "ssl" else None,
                    markersize=4, label=f"{rep.LABEL[model]}" if role == "ssl" else None)
    ax.axhline(0.5, color=rep.INK2, linewidth=1, linestyle=":")
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("ruido gaussiano añadido al cubo de test (σ)", fontsize=8)
    ax.set_ylabel("AUROC en test (84 sujetos)", fontsize=8)
    ax.set_title("1 · Más ruido, misma respuesta", fontsize=9.5)
    ax.text(0.0, 0.508, "azar", fontsize=7.5, color=rep.INK2, ha="left")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.1),
              title="continua: con SSL · punteada: sin SSL", title_fontsize=7)
    # (b) same subject, another session
    ax = fig.add_subplot(top[1])
    x = np.arange(3)
    ret = {m: {role: results["models"][m][role]["retrieval"] for role in ("ssl", "init")} for m in MODELS}
    ax.bar(x - 0.18, [ret[m]["init"]["top1"] for m in MODELS], 0.34, color=rep.OTHER, label="sin SSL")
    ax.bar(x + 0.18, [ret[m]["ssl"]["top1"] for m in MODELS], 0.34,
           color=[rep.COLOR[m] for m in MODELS], label="con SSL")
    for i, m in enumerate(MODELS):
        ax.text(i + 0.18, ret[m]["ssl"]["top1"] + 0.02, f"{ret[m]['ssl']['top1']:.2f}", ha="center", fontsize=8)
        ax.text(i - 0.18, ret[m]["init"]["top1"] + 0.02, f"{ret[m]['init']['top1']:.2f}", ha="center",
                fontsize=8, color=rep.INK2)
    chance = ret["vicreg"]["ssl"]["chance"]
    ax.axhline(chance, color=rep.INK2, linewidth=1, linestyle=":")
    ax.set_xticks(x, [rep.LABEL[m] for m in MODELS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1: encuentra al mismo sujeto", fontsize=8)
    r0 = ret["vicreg"]["ssl"]
    ax.set_title("2 · Otro día, otra imagen: mismo sujeto", fontsize=9.5)
    ax.text(0.98, 0.97, f"{r0['subjects']} sujetos con ≥ 2 sesiones\n({r0['sessions']} sesiones); "
            f"azar {chance:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=rep.INK2)
    ax.grid(axis="x", visible=False)
    # (c) partial association of the AD axis with anatomy and with scanner noise
    ax = fig.add_subplot(top[2])
    adhc = variables["labels"] >= 0
    names = variables["names"]
    vals = variables["values"][adhc]
    col = {n: vals[:, names.index(n)] for n in names}
    ok = np.isfinite(vals).all(axis=1)
    noise, hippo = col[NOISE][ok], col["vol. hipocampo"][ok]
    anatomy = np.column_stack([col[n][ok] for n in ("vol. hipocampo", "vol. ventrículos", "vol. cerebral", "edad")])
    age_noise = np.column_stack([col["edad"][ok], noise])
    tests = [("volumen del hipocampo (descontando edad y ruido)", hippo, age_noise),
             ("ruido en sust. blanca (descontando anatomía y edad)", noise, anatomy)]
    rows = np.arange(len(tests)) * 1.45
    for j, model in enumerate(MODELS):
        s = emb[(model, "ssl")]["scores"][ok]
        rho = [abs(partial_rho(s, t, c)) for _, t, c in tests]
        ax.barh(rows + (j - 1) * 0.26, rho, 0.25, color=rep.COLOR[model], label=rep.LABEL[model])
        for r, v in zip(rows, rho, strict=True):
            ax.text(v + 0.01, r + (j - 1) * 0.26, f"{v:.2f}", va="center", fontsize=7, color=rep.INK2)
    yv = variables["labels"][adhc][ok].astype(float)
    truth = [abs(partial_rho(yv, t, c)) for _, t, c in tests]
    ax.scatter(truth, rows, marker="|", s=420, color=rep.INK, zorder=5, linewidths=2,
               label="diagnóstico real")
    for r, (name, _, _) in zip(rows, tests, strict=True):
        ax.text(0.0, r - 0.45, name, fontsize=8, color=rep.INK, va="bottom", fontweight="semibold")
    ax.set_yticks([])
    ax.set_ylim(rows[-1] + 0.5, rows[0] - 0.75)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("|ρ parcial| con el puntaje AD del probe", fontsize=8)
    ax.set_title("3 · El eje AD: sobre todo anatomía", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "noise_checks.png")


NOISE = "ruido (sust. blanca)"


def partial_rho(s: np.ndarray, t: np.ndarray, covariates: np.ndarray) -> float:
    """Spearman partial correlation: ranks, residualised on the ranked covariates."""
    from scipy.stats import pearsonr, rankdata
    design = np.column_stack([np.ones(len(s))] + [rankdata(c) for c in covariates.T])

    def residual(z: np.ndarray) -> np.ndarray:
        rz = rankdata(z)
        return rz - design @ np.linalg.lstsq(design, rz, rcond=None)[0]
    return float(pearsonr(residual(s), residual(t)).statistic)


def main() -> None:
    results, variables, emb = load()
    fig_latent(results, variables, emb)
    fig_corr(results, variables, emb)
    fig_tsne_vars(results, variables, emb)
    fig_tsne_vars(results, variables, emb, keep=["edad", "educación"], ncols=3,
                  name="latent_tsne_native.png")
    fig_noise(results, variables, emb)


if __name__ == "__main__":
    main()
