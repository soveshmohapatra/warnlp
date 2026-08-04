import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

CONFLICT, PEACEFUL = "#C0392B", "#2E86C1"
SPINE = "#333333"
BREAK = 2019
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
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4, "axes.edgecolor": SPINE,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "axes.labelcolor": "black", "text.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def acceleration(series):
    pre = series.loc[:BREAK]
    post = series.loc[BREAK:]
    return (np.polyfit(post.index.values.astype(float), post.values, 1)[0]
            - np.polyfit(pre.index.values.astype(float), pre.values, 1)[0])


def main():
    apply_style()
    cc = json.loads((DATA / "conflict_correlation.json").read_text())
    groups = [("Conflict-involved\nNations", cc["B_country_level"]["conflict_countries"], CONFLICT),
              ("Peaceful\nNations", cc["B_country_level"]["peaceful_countries"], PEACEFUL)]

    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    cp = cp[cp.pub_year.between(2010, 2025)]
    g = cp.groupby(["country", "pub_year"]).sum(numeric_only=True).reset_index()
    g["p"] = (g.total_t1_count + g.total_t2_count + g.total_t3_count) / g.total_papers * 100

    fig, ax = plt.subplots(figsize=(4.90, 4.68))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.5, width=1.4, colors=SPINE)

    rng = np.random.default_rng(3)
    values = []
    for i, (label, countries, colour) in enumerate(groups):
        vals = np.array([acceleration(g[g.country == c].set_index("pub_year")["p"])
                         for c in countries])
        values.append(vals)
        bp = ax.boxplot([vals], positions=[i], widths=0.52, patch_artist=True,
                        showfliers=False, zorder=2)
        bp["boxes"][0].set(facecolor=colour, alpha=0.22, edgecolor=colour, linewidth=1.6)
        for elem in ("whiskers", "caps"):
            for art in bp[elem]:
                art.set(color=colour, linewidth=2.0)
        bp["medians"][0].set(color=colour, linewidth=3.0)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=70,
                   color=colour, alpha=0.85, linewidths=0, zorder=3)

    u_stat, p = mannwhitneyu(values[0], values[1], alternative="greater")
    print(f"conflict median {np.median(values[0]):.2f} vs peaceful "
          f"{np.median(values[1]):.2f}; one-sided Mann-Whitney p = {p:.3f}")

    ax.axhline(0, color="#CCCCCC", lw=1.8, ls="--", zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g_[0] for g_ in groups])
    ax.set_xlim(-0.62, len(groups) - 0.38)
    ax.set_ylabel("Post 2019 acceleration (pp/yr$^2$)")
    ax.set_ylim(-0.45, 2.05)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigS3.{ext}", dpi=300)
    print(f"wrote {OUT/'FigS3.pdf'} and {OUT/'FigS3.png'}")


if __name__ == "__main__":
    main()
