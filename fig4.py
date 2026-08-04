from pathlib import Path

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

T1, T2, T3 = "#C0392B", "#E67E22", "#2980B9"
GREY_REF, SPINE = "#B8BEC3", "#333333"

BREAK_MARKER = 2019

SHORT = {
    "Social Sciences, Arts & Humanities": "Social Sci.",
    "Business & Management": "Business",
    "Multidisciplinary & General Science": "Multidisc.",
    "Medical & Health Sciences": "Medical",
    "Engineering & Technology": "Engineering",
    "Life & Earth Sciences": "Life & Earth",
    "Computer Science & Mathematics": "CS & Math",
    "Physical Sciences": "Physical",
}
COLOUR = {
    "Social Sci.": "#C0392B", "Business": "#E67E22", "Multidisc.": "#8E44AD",
    "Medical": "#16A085", "Engineering": "#2980B9", "Life & Earth": "#27AE60",
    "CS & Math": "#D4AC0D", "Physical": "#34495E",
}
D_ORDER = ["Business", "CS & Math", "Engineering", "Life & Earth",
           "Physical", "Medical", "Multidisc.", "Social Sci."]

BASE, LETTER, TICK, LEG = 14.0, 17.0, 11.5, 11.0


def apply_style():
    for cand in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            font = cand
            break
    mpl.rcParams.update({
        "font.family": font, "font.size": BASE,
        "axes.labelsize": BASE, "axes.labelweight": "bold",
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
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


def load():
    c = pd.read_csv(DATA / "category_prevalence.csv")
    c = c[c.pub_year.between(2010, 2025)]
    g = c.groupby(["category", "pub_year"]).sum(numeric_only=True).reset_index()
    g["field"] = g.category.map(SHORT)
    for t in ("t1", "t2", "t3"):
        g[t] = g[f"total_{t}_count"] / g.total_papers * 100
    g["union"] = g.t1 + g.t2 + g.t3
    g["delib"] = (g.t1 + g.t2) / g["union"] * 100
    for part in ("title", "abstract"):
        g[part] = (g[f"{part}_t1_count"] + g[f"{part}_t2_count"]
                   + g[f"{part}_t3_count"]) / g.total_papers * 100
    return g


def panel_a(ax, g):
    st = g.groupby("field")["union"].agg(["mean", "std"]).sort_values("mean")
    for i, (field, row) in enumerate(st.iterrows()):
        colour = COLOUR[field]
        half = row["std"] / 2
        ax.plot([row["mean"] - half, row["mean"] + half], [i, i], "-", color=colour,
                lw=4.2, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter([row["mean"]], [i], s=150, color=colour, zorder=4, linewidths=0)

    ax.axvline(st["mean"].mean(), color=GREY_REF, lw=1.8, ls="--", zorder=1)
    ax.set_yticks(range(len(st)))
    ax.set_yticklabels(st.index)
    ax.set_ylim(-0.7, len(st) - 0.3)
    ax.set_xlabel("Militaristic Terms Prevalence (%)")
    ax.set_xlim(24, 48)
    ax.set_xticks([24, 30, 36, 42, 48])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_b(ax, g):
    m = g.groupby("field")[["t1", "t2", "t3"]].mean().sort_values("t1")
    y = np.arange(len(m))
    left = np.zeros(len(m))
    for t, colour in (("t1", T1), ("t2", T2), ("t3", T3)):
        ax.barh(y, m[t].values, left=left, height=0.66, color=colour,
                edgecolor="none", label=t.upper())
        left += m[t].values

    ax.set_yticks(y)
    ax.set_yticklabels(m.index)
    ax.set_ylim(-0.7, len(m) - 0.3)
    ax.set_xlabel("Mean Militaristic Terms Prevalence (%)")
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(facecolor=c, label=t) for t, c in
                       (("T1", T1), ("T2", T2), ("T3", T3))],
              loc="upper right", bbox_to_anchor=(1.00, 1.07), ncol=3,
              handlelength=1.0, handleheight=1.0, handletextpad=0.4,
              columnspacing=1.0, borderpad=0.0)


def panel_c(ax, g):
    order = g.groupby("field")["delib"].median().sort_values().index
    for i, field in enumerate(order):
        vals = g[g.field == field]["delib"].values
        bp = ax.boxplot([vals], positions=[i], widths=0.58, vert=False,
                        patch_artist=True, showfliers=False, zorder=2)
        bp["boxes"][0].set(facecolor=COLOUR[field], edgecolor="none")
        for elem in ("whiskers", "caps"):
            for art in bp[elem]:
                art.set(color="#8A8A8A", linewidth=1.4)
        bp["medians"][0].set(color="white", linewidth=1.8)

    ax.axvline(g.groupby("field")["delib"].median().mean(), color="#9AA2A8",
               lw=1.8, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Deliberateness Index (%)")
    ax.set_xlim(10, 40)
    ax.set_xticks([10, 20, 30, 40])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_d(fig, gs, g):
    axes = []
    for k, field in enumerate(D_ORDER):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        despine(ax)
        sub = g[g.field == field].sort_values("pub_year")
        colour = COLOUR[field]
        ax.plot(sub.pub_year, sub["union"], "-", color=colour, lw=2.6)
        ax.fill_between(sub.pub_year, 0, sub["union"], color=colour, alpha=0.10,
                        linewidth=0)
        ax.axvline(BREAK_MARKER, color="#333333", lw=1.5, ls=":", zorder=3)

        lo, hi = sub["union"].min() - 1, sub["union"].max() + 1
        ticks = np.linspace(lo, hi, 5)
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.0f}" for v in ticks])
        ax.set_xlim(2009.25, 2025.75)
        ax.set_xticks([2010, 2025])
        if k < 4:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.text(0.06, 0.92, field, transform=ax.transAxes, fontsize=BASE - 1.5,
                va="top", ha="left")
        axes.append(ax)

    fig.text(0.055, 0.265, "Militaristic Terms Prevalence (%)", rotation=90,
             fontsize=BASE, fontweight="bold", va="center", ha="center")
    return axes


def panel_e(ax, g):
    piv = g.pivot_table(index="field", columns="pub_year", values=["title", "abstract"])
    growth = pd.DataFrame({
        "title": 100 * (piv["title"][2025] / piv["title"][2010] - 1),
        "abstract": 100 * (piv["abstract"][2025] / piv["abstract"][2010] - 1),
    }).sort_values("title")

    for i, (field, row) in enumerate(growth.iterrows()):
        colour = COLOUR[field]
        ax.plot([row.abstract, row.title], [i, i], "-", color=colour, lw=3.4,
                alpha=0.65, solid_capstyle="round", zorder=2)
        ax.scatter([row.abstract], [i], s=130, facecolor="white",
                   edgecolor=colour, linewidths=2.4, zorder=4)
        ax.scatter([row.title], [i], s=140, color=colour, zorder=4, linewidths=0)

    ax.set_yticks(range(len(growth)))
    ax.set_yticklabels(growth.index)
    ax.set_ylim(-0.9, len(growth) - 0.35)
    ax.set_xlabel("Growth 2010 $\\rightarrow$ 2025 (%)")
    ax.set_xlim(24, 88)
    ax.set_xticks([30, 45, 60, 75])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor="white", markeredgecolor="#7F8C8D",
               markeredgewidth=2.0, label="Abstracts"),
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor="#7F8C8D", markeredgecolor="none", label="Titles")],
        loc="lower right", handletextpad=0.5, labelspacing=0.45, borderpad=0.1)


def main():
    apply_style()
    g = load()

    fig = plt.figure(figsize=(14.29, 8.82))
    gs_top = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.925, bottom=0.575,
                              wspace=0.46)
    gs_d = fig.add_gridspec(2, 4, left=0.095, right=0.565, top=0.435, bottom=0.095,
                            wspace=0.42, hspace=0.14)
    gs_e = fig.add_gridspec(1, 1, left=0.675, right=0.985, top=0.435, bottom=0.115)

    ax_a, ax_b, ax_c = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    for ax in (ax_a, ax_b, ax_c):
        despine(ax)
    panel_a(ax_a, g)
    panel_b(ax_b, g)
    panel_c(ax_c, g)

    panel_d(fig, gs_d, g)
    ax_e = fig.add_subplot(gs_e[0, 0])
    despine(ax_e)
    panel_e(ax_e, g)

    stamp(fig, 0.008, 0.955, "A", "Militarized language spans all disciplines")
    stamp(fig, 0.330, 0.955, "B", "Anatomy of the rise, by field")
    stamp(fig, 0.655, 0.955, "C", "Deliberate use, field by field")
    stamp(fig, 0.070, 0.475, "D", "Eight fields, one turning point")
    stamp(fig, 0.600, 0.475, "E", "War terms move into titles fastest")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Fig4.{ext}", dpi=300)
    print(f"wrote {OUT/'Fig4.pdf'} and {OUT/'Fig4.png'}")


if __name__ == "__main__":
    main()
