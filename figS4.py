from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

OPENALEX, PUBMED = "#E67E22", "#8E44AD"
SPINE = "#333333"
BASE, TICK = 14.0, 12.5


def apply_style():
    for cand in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            font = cand
            break
    mpl.rcParams.update({
        "font.family": font, "font.size": BASE,
        "axes.labelsize": BASE, "axes.labelweight": "bold",
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.fontsize": BASE - 1,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4, "axes.edgecolor": SPINE,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "axes.labelcolor": "black", "text.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def fmt_p(p):
    if p >= 0.01:
        return f"p = {p:.2f}"
    exp = int(np.floor(np.log10(p)))
    mant = round(p / 10 ** exp, 1)
    if mant >= 10:
        mant, exp = mant / 10, exp + 1
    return f"p = {mant:.1f}$\\times$10$^{{{exp}}}$"


def main():
    apply_style()
    t = pd.read_csv(DATA / "tone_scores_yearly.csv")
    t = t[t.pub_year.between(2010, 2025)]

    fig, ax = plt.subplots(figsize=(5.19, 4.38))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.5, width=1.4, colors=SPINE)

    handles, labels = [], []
    for key, name, colour in (("openalex", "OpenAlex", OPENALEX),
                              ("pubmed", "PubMed", PUBMED)):
        d = t[t.source == key].sort_values("pub_year")
        res = mk.original_test(d.mean_tone.values)
        ax.plot(d.pub_year, d.mean_tone, "-", color=colour, lw=3.4, marker="o",
                markersize=6, zorder=3)
        handles.append(Line2D([], [], color=colour, lw=8, solid_capstyle="butt"))
        labels.append(f"{name} ($\\tau$ = {res.Tau:.2f}; {fmt_p(res.p)})")

    ax.set_ylabel("Mean LLM Tone Score (0-5)")
    ax.set_ylim(1.32, 2.52)
    ax.set_yticks([1.40, 1.65, 1.90, 2.15, 2.40])
    ax.set_xticks([2010, 2015, 2020, 2025])
    ax.set_xlim(2009.5, 2025.5)
    ax.legend(handles, labels, loc="upper left", handlelength=0.9,
              handletextpad=0.5, labelspacing=0.35, borderpad=0.1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigS4.{ext}", dpi=300)
    print(f"wrote {OUT/'FigS4.pdf'} and {OUT/'FigS4.png'}")


if __name__ == "__main__":
    main()
