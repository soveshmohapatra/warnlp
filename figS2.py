from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

FIT = "#E67E22"
SPINE = "#333333"
BASE, TICK = 14.0, 12.0


def apply_style():
    for cand in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            font = cand
            break
    mpl.rcParams.update({
        "font.family": font, "font.size": BASE,
        "axes.labelsize": BASE, "axes.labelweight": "bold",
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4, "axes.edgecolor": SPINE,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "axes.labelcolor": "black", "text.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def sci(p):
    exp = int(np.floor(np.log10(p)))
    mant = round(p / 10 ** exp, 2)
    if mant >= 10:
        mant, exp = mant / 10, exp + 1
    return f"{mant:.2f}$\\times$10$^{{{exp}}}$"


def main():
    apply_style()
    y = pd.read_csv(DATA / "yearly_prevalence.csv")
    y = y[y.pub_year.between(2010, 2025)]

    def union(df):
        d = df.sort_values("pub_year")
        return ((d.total_t1_count + d.total_t2_count + d.total_t3_count)
                / d.total_papers * 100).values

    oa = union(y[y.source == "OpenAlex"])
    pm = union(y[y.source == "PubMed"])
    years = np.sort(y.pub_year.unique())

    r, r_p = pearsonr(oa, pm)
    rho, rho_p = spearmanr(oa, pm)

    fig, ax = plt.subplots(figsize=(5.75, 4.75))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.5, width=1.4, colors=SPINE)

    lims = [min(oa.min(), pm.min()) - 1.5, max(oa.max(), pm.max()) + 1.0]
    ax.plot(lims, lims, "--", color="#AAAAAA", lw=2.0, zorder=1)

    slope, intercept = np.polyfit(oa, pm, 1)
    xf = np.linspace(oa.min(), oa.max(), 50)
    ax.plot(xf, slope * xf + intercept, "-", color=FIT, lw=3.0, zorder=2)

    sc = ax.scatter(oa, pm, c=years, cmap="viridis", s=130, zorder=3,
                    linewidths=0)

    ax.text(0.03, 0.97, f"$r$ = {r:.2f}; p = {sci(r_p)}\n"
                        f"$\\rho$ = {rho:.2f}; p = {sci(rho_p)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=BASE)

    ax.set_xlabel("OpenAlex Prevalence (%)")
    ax.set_ylabel("PubMed Prevalence (%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks([35, 40, 45, 50])
    ax.set_yticks([34, 38, 42, 46, 50])

    cb = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.06)
    cb.set_ticks([2010, 2015, 2020, 2025])
    cb.ax.tick_params(labelsize=TICK, length=3.5, width=1.2)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigS2.{ext}", dpi=300)
    print(f"wrote {OUT/'FigS2.pdf'} and {OUT/'FigS2.png'}")


if __name__ == "__main__":
    main()
