from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NORTH, SOUTH = "#2980B9", "#E67E22"
MISSING = "#E8E8E8"
BORDER = "#AAAAAA"
SPINE = "#333333"

MIN_PAPERS = 10_000
MIN_2010_PAPERS = 500
CMAP, VMIN, VMAX = "OrRd", 20, 60

NORTH_GROUP = {
    "United States", "Canada", "Mexico", "Bahamas", "Barbados", "Belize",
    "Caribbean", "Costa Rica", "Cuba", "Dominica", "Dominican Republic",
    "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica",
    "Nicaragua", "Panama", "Saint Kitts and Nevis", "Trinidad and Tobago",
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo",
    "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway",
    "Poland", "Portugal", "Romania", "Russia", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine",
    "United Kingdom",
    "Australia", "New Zealand", "Fiji", "Kiribati", "Papua New Guinea",
}

NE_ALIAS = {"United States": "United States of America",
            "Czech Republic": "Czechia"}

SHORT = {"United States": "USA", "United Kingdom": "UK"}

BASE, LETTER, TICK, LEG = 14.0, 17.0, 11.5, 11.0


def apply_style():
    for cand in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            font = cand
            break
    mpl.rcParams.update({
        "font.family": font, "font.size": BASE,
        "axes.labelsize": BASE, "axes.labelweight": "bold",
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.fontsize": LEG,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4, "axes.edgecolor": SPINE,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "axes.labelcolor": "black", "text.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.5, width=1.4, colors=SPINE)
    ax.grid(False)


def stamp(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=LETTER, fontweight="bold", va="bottom", ha="left")
    fig.text(x + 0.021, y, title, fontsize=BASE, va="bottom", ha="left")


def group_of(country):
    return "north" if country in NORTH_GROUP else "south"


def load_union():
    v = pd.read_csv(DATA / "country_t1_t2_verification.csv")
    v = v[v.year.between(2010, 2025)].rename(columns={"t1_t2_t3_any_pct": "prev"})
    return v[["country", "year", "total_papers", "prev"]]


def load_pooled():
    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    cp = cp[(cp.country != "Unknown") & cp.pub_year.between(2010, 2025)]
    g = cp.groupby(["country", "pub_year"]).sum(numeric_only=True).reset_index()
    g["prev"] = (g.total_t1_count + g.total_t2_count + g.total_t3_count) / g.total_papers * 100
    return g.rename(columns={"pub_year": "year"})[["country", "year", "total_papers", "prev"]]


def panel_a(fig, gs, union):
    world = gpd.read_file(DATA / "ne_50m_admin_0_countries.zip")[["NAME", "geometry"]]
    eligible = union.groupby("country").total_papers.sum()
    eligible = set(eligible[eligible >= MIN_PAPERS].index)

    axes = []
    for col, year in enumerate((2010, 2024)):
        ax = fig.add_subplot(gs[0, col])
        vals = (union[(union.year == year) & union.country.isin(eligible)]
                .assign(NAME=lambda d: d.country.replace(NE_ALIAS))
                .set_index("NAME")["prev"])
        w = world.copy()
        w["prev"] = w.NAME.map(vals)
        w[w.prev.isna()].plot(ax=ax, color=MISSING, edgecolor=BORDER, linewidth=0.25)
        w[w.prev.notna()].plot(ax=ax, column="prev", cmap=CMAP, vmin=VMIN, vmax=VMAX,
                               edgecolor=BORDER, linewidth=0.25)
        ax.set_title(f"Year {year}", fontsize=BASE, fontweight="bold", pad=4)
        ax.set_xlim(-181, 181)
        ax.set_ylim(-60, 85)
        ax.set_axis_off()
        axes.append(ax)

    cax = fig.add_subplot(gs[0, 2])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(VMIN, VMAX))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Militaristic Terms Prevalence (%)", fontsize=BASE - 1,
                 fontweight="bold", labelpad=8)
    cb.set_ticks([20, 25, 30, 35, 40, 45, 50, 55, 60])
    cb.ax.tick_params(labelsize=TICK, length=3.5, width=1.2)
    cb.outline.set_linewidth(1.0)
    cb.outline.set_edgecolor(SPINE)
    return axes


def panel_b(ax, pooled):
    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    cp = cp[(cp.country != "Unknown") & cp.pub_year.between(2010, 2025)].copy()
    cp["grp"] = cp.country.map(group_of)
    g = cp.groupby(["grp", "pub_year"]).sum(numeric_only=True)
    g["prev"] = (g.total_t1_count + g.total_t2_count + g.total_t3_count) / g.total_papers * 100
    north, south = g.loc["north", "prev"], g.loc["south", "prev"]
    years = north.index.values

    ax.fill_between(years, north.values, south.values, color=NORTH, alpha=0.10,
                    linewidth=0, zorder=1)
    ax.plot(years, north.values, "-", color=NORTH, lw=3.2, zorder=3)
    ax.plot(years, south.values, "-", color=SOUTH, lw=3.2, zorder=3)

    diff = south.values - north.values
    cross = np.argmax(diff >= 0)
    if cross > 0:
        frac = -diff[cross - 1] / (diff[cross] - diff[cross - 1])
        x = years[cross - 1] + frac
        yv = np.interp(x, years, north.values)
    else:
        x, yv = years[cross], north.values[cross]
    ax.plot([x], [yv], "o", markerfacecolor="none", markeredgecolor="black",
            markersize=13, markeredgewidth=2.0, zorder=5)
    ax.annotate(f"{int(round(x))}", xy=(x, yv - 0.5), xytext=(x - 0.35, yv - 4.6),
                ha="center", fontsize=LEG + 0.5,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black",
                                shrinkA=1, shrinkB=2))

    ax.set_ylabel("Militaristic Terms Prevalence (%)")
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_xlim(2010, 2025)
    ax.set_yticks([30, 36, 42, 48])
    ax.set_ylim(27.5, 50)
    ax.legend(handles=[Patch(facecolor=NORTH, label="N. America, Europe, Oceania"),
                       Patch(facecolor=SOUTH, label="S. America, Africa, Asia")],
              loc="upper left", handlelength=1.1, handleheight=1.0,
              handletextpad=0.55, labelspacing=0.4, borderpad=0.1)


def panel_c(ax, union):
    tot = union.groupby("country").total_papers.sum()
    top = tot.sort_values(ascending=False).head(10).index
    sub = union[union.country.isin(top)]
    order = sub.groupby("country")["prev"].mean().sort_values(ascending=False).index

    rng = np.random.default_rng(7)
    for i, country in enumerate(order):
        vals = sub[sub.country == country]["prev"].values
        colour = NORTH if group_of(country) == "north" else SOUTH
        bp = ax.boxplot([vals], positions=[i], widths=0.62, patch_artist=True,
                        showfliers=False, zorder=2)
        bp["boxes"][0].set(facecolor=colour, alpha=0.45, edgecolor="none")
        for elem in ("whiskers", "caps"):
            for art in bp[elem]:
                art.set(color="#666666", linewidth=1.3)
        bp["medians"][0].set(color="black", linewidth=2.2)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(vals)), vals, s=26,
                   color=colour, alpha=0.85, linewidths=0, zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT.get(c, c) for c in order], rotation=90)
    ax.set_xlim(-0.75, len(order) - 0.25)
    ax.set_ylabel("Overall Militaristic Prevalence (%)")
    ax.set_ylim(25, 46)
    ax.set_yticks([25, 30, 35, 40, 45])
    ax.legend(handles=[Patch(facecolor=NORTH, label="N. America, Europe, Oceania"),
                       Patch(facecolor=SOUTH, label="S. America, Africa, Asia")],
              loc="lower left", handlelength=1.1, handleheight=1.0,
              handletextpad=0.55, labelspacing=0.4, borderpad=0.1)


def panel_d(ax, pooled):
    piv = pooled.pivot_table(index="country", columns="year", values="prev")
    papers = pooled.pivot_table(index="country", columns="year", values="total_papers")
    keep = papers[papers[2010] >= MIN_2010_PAPERS].index
    delta = (piv[2025] - piv[2010]).loc[piv.index.isin(keep)]
    top = delta.sort_values(ascending=False).head(10).index

    for i, country in enumerate(top):
        y = len(top) - 1 - i
        start, end = piv.loc[country, 2010], piv.loc[country, 2025]
        colour = NORTH if group_of(country) == "north" else SOUTH
        ax.annotate("", xy=(end, y), xytext=(start, y),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                                    color=colour, lw=3.2, shrinkA=0, shrinkB=0))
        ax.scatter([start], [y], s=70, color="#B7B7B7", zorder=4, linewidths=0)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(list(top)[::-1])
    ax.set_ylim(-0.9, len(top) - 0.4)
    ax.set_xlabel("Militaristic Terms Prevalence (%)")
    ax.set_xticks([30, 45, 60])
    ax.set_xlim(19, 74)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=9,
                              markerfacecolor="#B7B7B7", markeredgecolor="none",
                              label="2010"),
                       Line2D([], [], marker=">", linestyle="none", markersize=9,
                              markerfacecolor="#B7B7B7", markeredgecolor="none",
                              label="2025")],
              loc="lower right", handletextpad=0.5, labelspacing=0.45,
              borderpad=0.1)


def main():
    apply_style()
    union, pooled = load_union(), load_pooled()

    fig = plt.figure(figsize=(14.14, 8.78))
    gs_top = fig.add_gridspec(1, 3, left=0.02, right=0.92, top=0.925, bottom=0.50,
                              width_ratios=[1, 1, 0.022], wspace=0.02)
    gs_bot = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.42, bottom=0.115,
                              wspace=0.42)

    panel_a(fig, gs_top, union)
    ax_b, ax_c, ax_d = (fig.add_subplot(gs_bot[0, i]) for i in range(3))
    for ax in (ax_b, ax_c, ax_d):
        despine(ax)
    panel_b(ax_b, pooled)
    panel_c(ax_c, union)
    panel_d(ax_d, pooled)

    stamp(fig, 0.018, 0.955, "A", "Militarized language goes global")
    stamp(fig, 0.018, 0.452, "B", "Geographic reversal in 2024")
    stamp(fig, 0.343, 0.452, "C", "Biggest producers, by prevalence")
    stamp(fig, 0.665, 0.452, "D", "Steepest climbs, 2010-2025")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Fig3.{ext}", dpi=300)
    print(f"wrote {OUT/'Fig3.pdf'} and {OUT/'Fig3.png'}")


if __name__ == "__main__":
    main()
