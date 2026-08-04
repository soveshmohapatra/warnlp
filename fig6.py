import json
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
PANDEMIC, OTHER = "#E05A4B", "#16A085"
ENGLISH_C, NON_ENGLISH_C = "#E67E22", "#8E44AD"
SPINE = "#333333"

ENGLISH_SPEAKING = {"United States", "United Kingdom", "Canada", "Australia",
                    "New Zealand", "Ireland"}
MIN_PAPERS_F = 100_000
HIGH_CONFLICT_DEATHS = 10_000
PRE_LLM, POST_LLM = (2010, 2022), (2023, 2025)
PRE_LLM_WINDOW_F = (2019, 2022)

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
    fig.text(x + 0.019, y, title, fontsize=BASE, va="bottom", ha="left")


def legend_swatches(ax, entries, loc="upper left", **kw):
    handles = [Line2D([], [], color=c, lw=7, solid_capstyle="butt") for _, c in entries]
    return ax.legend(handles, [t for t, _ in entries], loc=loc, handlelength=1.0,
                     handletextpad=0.55, borderpad=0.1, labelspacing=0.42, **kw)


def slope(years, values):
    return np.polyfit(np.asarray(years, float), np.asarray(values, float), 1)[0]


def pandemic_split():
    y = pd.read_csv(DATA / "yearly_prevalence.csv")
    s = pd.read_csv(DATA / "sensitivity_covid.csv")
    m = y.merge(s, on=["pub_year", "source"], suffixes=("_all", "_nc"))
    g = m[m.pub_year.between(2010, 2025)].groupby("pub_year").sum(numeric_only=True)

    out = pd.DataFrame(index=g.index)
    pandemic_papers = g.total_papers_all - g.total_papers_nc
    for t in ("t1", "t2", "t3"):
        out[f"pandemic_{t}"] = (g[f"total_{t}_count_all"] - g[f"total_{t}_count_nc"]) \
            / pandemic_papers * 100
        out[f"other_{t}"] = g[f"total_{t}_count_nc"] / g.total_papers_nc * 100
    out["pandemic"] = out[[f"pandemic_{t}" for t in ("t1", "t2", "t3")]].sum(axis=1)
    out["other"] = out[[f"other_{t}" for t in ("t1", "t2", "t3")]].sum(axis=1)
    return out


def country_frame():
    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    return cp[(cp.country != "Unknown") & cp.pub_year.between(2010, 2025)].copy()


def high_conflict_countries():
    bd = pd.read_csv(DATA / "UCDP/BattleDeaths_v25_1_conf.csv", low_memory=False)
    bd = bd[bd.year.between(2010, 2024)]
    cc = json.loads((DATA / "conflict_correlation.json").read_text())
    return {c for c in cc["B_country_level"]["conflict_countries"]
            if bd[bd.location_inc.astype(str).str.contains(c, case=False, na=False)]
            .bd_best.sum() > HIGH_CONFLICT_DEATHS}


def panel_a(ax, split):
    for col, colour, marker, label in (("pandemic", PANDEMIC, "o", "Pandemic papers"),
                                       ("other", OTHER, "s", "Other papers")):
        ax.plot(split.index, split[col], "-", color=colour, lw=3.0, marker=marker,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2)
    ax.set_ylabel("Militaristic Terms Prevalence (%)")
    ax.set_ylim(30, 62)
    ax.set_yticks(range(30, 61, 5))
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_xlim(2010, 2025)
    legend_swatches(ax, [("Pandemic papers", PANDEMIC), ("Other papers", OTHER)])


def panel_b(ax, split):
    for i, t in enumerate(("t1", "t2", "t3")):
        for off, col, colour in ((-0.17, f"other_{t}", OTHER),
                                 (0.17, f"pandemic_{t}", PANDEMIC)):
            bp = ax.boxplot([split[col].values], positions=[i + off], widths=0.28,
                            patch_artist=True, showfliers=False, zorder=2)
            bp["boxes"][0].set(facecolor=colour, alpha=0.30, edgecolor=colour,
                               linewidth=1.2)
            for elem in ("whiskers", "caps"):
                for art in bp[elem]:
                    art.set(color="#8A8A8A", linewidth=1.3)
            bp["medians"][0].set(color="black", linewidth=2.4)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Militaristic Terms Prevalence (%)")
    ax.set_ylim(0, 48)
    ax.set_yticks([10, 20, 30, 40])
    ax.legend(handles=[Patch(facecolor=PANDEMIC, alpha=0.30, edgecolor=PANDEMIC,
                             label="Pandemic papers"),
                       Patch(facecolor=OTHER, alpha=0.30, edgecolor=OTHER,
                             label="Other papers")],
              loc="upper left", handlelength=1.1, handleheight=1.0,
              handletextpad=0.55, labelspacing=0.42, borderpad=0.1)


def panel_c(ax):
    c = pd.read_csv(DATA / "category_prevalence.csv")
    c = c[c.pub_year.between(2010, 2025)]
    g = c.groupby(["category", "pub_year"]).sum(numeric_only=True).reset_index()
    g["field"] = g.category.map(SHORT)
    g["p"] = (g.total_t1_count + g.total_t2_count + g.total_t3_count) / g.total_papers * 100

    rows = {}
    for field, sub in g.groupby("field"):
        pre = sub[sub.pub_year < 2020]
        post = sub[sub.pub_year.between(2020, 2024)]
        rows[field] = (slope(pre.pub_year, pre.p), slope(post.pub_year, post.p))
    order = sorted(rows, key=lambda f: rows[f][1])

    for i, field in enumerate(order):
        pre, post = rows[field]
        colour = COLOUR[field]
        ax.plot([pre, post], [i, i], "-", color=colour, lw=3.4, alpha=0.85,
                solid_capstyle="round", zorder=2)
        ax.scatter([pre], [i], s=130, facecolor="white", edgecolor=colour,
                   linewidths=2.4, zorder=4)
        ax.scatter([post], [i], s=140, color=colour, zorder=4, linewidths=0)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.9, len(order) - 0.35)
    ax.set_xlabel("Prevalence Growth (pp/year)")
    ax.set_xlim(0, 2.6)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor="white", markeredgecolor="#7F8C8D",
               markeredgewidth=2.0, label="Pre 2020"),
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor="#7F8C8D", markeredgecolor="none", label="Post 2020")],
        loc="lower right", handletextpad=0.5, labelspacing=0.45, borderpad=0.1)


def language_series(cp, tier=None):
    d = cp.copy()
    d["grp"] = np.where(d.country.isin(ENGLISH_SPEAKING), "English", "Non-English")
    g = d.groupby(["grp", "pub_year"]).sum(numeric_only=True)
    cols = [f"total_{tier}_count"] if tier else [f"total_{t}_count" for t in ("t1", "t2", "t3")]
    g["p"] = g[cols].sum(axis=1) / g.total_papers * 100
    return g["p"]


def panel_d(ax, cp):
    p = language_series(cp)
    entries = []
    for grp, colour, style in (("English", ENGLISH_C, "-"), ("Non-English", NON_ENGLISH_C, "--")):
        s = p.loc[grp]
        ax.plot(s.index, s.values, style, color=colour, lw=3.2, zorder=3)
        entries.append((f"{grp}-speaking countries (+{100 * (s.iloc[-1] / s.iloc[0] - 1):.0f}%)",
                        colour))
    ax.fill_between(p.loc["English"].index, p.loc["Non-English"].values,
                    p.loc["English"].values, color=ENGLISH_C, alpha=0.12, linewidth=0)

    ax.set_ylabel("Militaristic Terms Prevalence (%)")
    ax.set_ylim(29.5, 47.5)
    ax.set_yticks([30, 35, 40, 45])
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_xlim(2010, 2025)
    legend_swatches(ax, entries)


def panel_e(ax, cp):
    rows = []
    for t, colour in (("t1", T1), ("t2", T2), ("t3", T3)):
        p = language_series(cp, t)
        for grp, alpha in (("English", 0.45), ("Non-English", 1.0)):
            s = p.loc[grp]
            pre = slope(s.loc[PRE_LLM[0]:PRE_LLM[1]].index, s.loc[PRE_LLM[0]:PRE_LLM[1]].values)
            post = slope(s.loc[POST_LLM[0]:POST_LLM[1]].index, s.loc[POST_LLM[0]:POST_LLM[1]].values)
            rows.append((f"{t.upper()} - {grp}", colour, alpha, pre, post))

    for i, (label, colour, alpha, pre, post) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + 0.17, post, height=0.32, color=colour, alpha=alpha,
                edgecolor="none", zorder=2)
        ax.barh(y - 0.19, pre, height=0.22, color="#D5D8DC", edgecolor="#8A8A8A",
                linewidth=0.9, hatch="///", zorder=2)

    ax.axvline(0, color="#AAAAAA", lw=1.4, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Prevalence Slope (pp/year)")
    ax.set_xlim(-0.95, 3.15)
    ax.set_xticks([0, 1, 2, 3])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(facecolor="#D5D8DC", edgecolor="#8A8A8A", hatch="///",
                             label="Pre-LLM"),
                       Patch(facecolor="#B3B6B7", edgecolor="none", label="Post-LLM")],
              loc="upper right", handlelength=1.1, handleheight=1.0,
              handletextpad=0.55, labelspacing=0.42, borderpad=0.1)


def panel_f(ax, cp):
    pooled = cp.groupby(["country", "pub_year"]).agg(
        t1=("total_t1_count", "sum"), t2=("total_t2_count", "sum"),
        t3=("total_t3_count", "sum"), papers=("total_papers", "sum")).reset_index()
    pooled["p"] = (pooled[["t1", "t2", "t3"]].sum(axis=1).clip(upper=pooled.papers)) \
        / pooled.papers * 100

    volume = pooled.groupby("country").papers.sum()
    eligible = (set(volume[volume >= MIN_PAPERS_F].index)
                - ENGLISH_SPEAKING - high_conflict_countries())
    sub = pooled[pooled.country.isin(eligible)]
    pre = sub[sub.pub_year.between(*PRE_LLM_WINDOW_F)].groupby("country")["p"].mean()
    post = sub[sub.pub_year.between(*POST_LLM)].groupby("country")["p"].mean()
    top = (post - pre).sort_values(ascending=False).head(10).index

    recent = pooled[pooled.country.isin(top) & pooled.pub_year.between(*POST_LLM)]
    order = recent.groupby("country")["p"].median().sort_values().index

    for i, country in enumerate(order):
        vals = recent[recent.country == country]["p"].values
        bp = ax.boxplot([vals], positions=[i], widths=0.62, vert=False,
                        patch_artist=True, showfliers=False, zorder=2)
        bp["boxes"][0].set(facecolor=PANDEMIC, alpha=0.35, edgecolor="none")
        for elem in ("whiskers", "caps"):
            for art in bp[elem]:
                art.set(color=PANDEMIC, linewidth=1.6)
        bp["medians"][0].set(color="white", linewidth=1.8)
        ax.scatter(vals, [i] * len(vals), s=45, color=PANDEMIC, zorder=3, linewidths=0)
        ax.scatter([np.median(vals)], [i], s=45, color="white", zorder=4, linewidths=0)

    eng = cp[cp.country.isin(ENGLISH_SPEAKING)].groupby("pub_year").sum(numeric_only=True)
    eng_p = ((eng[["total_t1_count", "total_t2_count", "total_t3_count"]].sum(axis=1)
              .clip(upper=eng.total_papers)) / eng.total_papers * 100).loc[POST_LLM[0]:POST_LLM[1]]
    ax.axvspan(eng_p.min(), eng_p.max(), color=T3, alpha=0.12, zorder=0)
    ax.axvline(eng_p.median(), color=T3, lw=2.4, ls="--", zorder=1)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Militaristic Terms Prevalence, 2023-25 (%)")
    ax.set_xlim(35.5, 54)
    ax.set_xticks([40, 45, 50])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    apply_style()
    split, cp = pandemic_split(), country_frame()

    fig = plt.figure(figsize=(14.82, 8.74))
    gs_top = fig.add_gridspec(1, 3, left=0.070, right=0.985, top=0.925, bottom=0.575,
                              wspace=0.50)
    gs_bot = fig.add_gridspec(1, 3, left=0.070, right=0.985, top=0.435, bottom=0.100,
                              wspace=0.50)

    axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)] + \
           [fig.add_subplot(gs_bot[0, i]) for i in range(3)]
    for ax in axes:
        despine(ax)

    panel_a(axes[0], split)
    panel_b(axes[1], split)
    panel_c(axes[2])
    panel_d(axes[3], cp)
    panel_e(axes[4], cp)
    panel_f(axes[5], cp)

    stamp(fig, 0.008, 0.955, "A", "The language of pandemics is militarized")
    stamp(fig, 0.320, 0.955, "B", "Every tier climbs in pandemic papers")
    stamp(fig, 0.640, 0.955, "C", "The pandemic era reshaped every field")
    stamp(fig, 0.008, 0.470, "D", "Non-English countries closed the gap")
    stamp(fig, 0.320, 0.470, "E", "The LLM era accelerated the dual-use rise")
    stamp(fig, 0.640, 0.470, "F", "Top non-English nations have caught up")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Fig6.{ext}", dpi=300)
    print(f"wrote {OUT/'Fig6.pdf'} and {OUT/'Fig6.png'}")


if __name__ == "__main__":
    main()
