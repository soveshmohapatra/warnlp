from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

T1, T2, T3 = "#C0392B", "#E67E22", "#2980B9"
OPENALEX, PUBMED = "#E67E22", "#8E44AD"
E_DARK, E_LIGHT = "#3D8CC0", "#B1D8F0"
PLACEBO = "#95A5A6"
GREY_REF, SPINE = "#CCCCCC", "#333333"

YEARS = range(2010, 2026)
BREAK = 2019

LEGEND_MILITARISTIC_PCT = 69

BASE, LETTER, TICK, LEG = 14.0, 17.0, 11.5, 11.0


def apply_style():
    for cand in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            font = cand
            break
    mpl.rcParams.update({
        "font.family": font,
        "font.size": BASE,
        "axes.labelsize": BASE,
        "axes.labelweight": "bold",
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEG,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4,
        "axes.edgecolor": SPINE,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "axes.labelcolor": "black",
        "text.color": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.5, width=1.4, colors=SPINE)
    ax.grid(False)


def stamp(ax, letter, title):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes, fontsize=LETTER,
            fontweight="bold", va="bottom", ha="left")
    ax.text(-0.09, 1.10, title, transform=ax.transAxes, fontsize=BASE,
            fontweight="normal", va="bottom", ha="left")


def year_axis(ax):
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_xlim(2010, 2025)


def sci_p(p):
    if p == 0 or not np.isfinite(p):
        return "p < 1.0$\\times$10$^{-16}$"
    exp = int(np.floor(np.log10(p)))
    mant = round(p / 10 ** exp, 1)
    if mant >= 10:
        mant, exp = mant / 10, exp + 1
    return f"p = {mant:.1f}$\\times$10$^{{{exp}}}$"


def legend_swatches(ax, entries, loc="upper left", **kw):
    handles = [Line2D([], [], color=c, lw=7, solid_capstyle="butt") for _, c in entries]
    return ax.legend(handles, [t for t, _ in entries], loc=loc,
                     handlelength=1.0, handletextpad=0.55, borderpad=0.1,
                     labelspacing=0.42, **kw)


def load():
    y = pd.read_csv(DATA / "yearly_prevalence.csv")
    y = y[y.pub_year.between(2010, 2025)].copy()
    for t in ("t1", "t2", "t3"):
        for part in ("total", "title", "abstract"):
            y[f"{part}_{t}_pct"] = y[f"{part}_{t}_count"] / y.total_papers * 100
    return y


def pooled_by_count(y, part="total"):
    g = y.groupby("pub_year").sum(numeric_only=True)
    return pd.DataFrame({t: g[f"{part}_{t}_count"] / g.total_papers * 100
                         for t in ("t1", "t2", "t3")})


def pooled_by_mean(y, part="total"):
    piv = y.pivot_table(index="pub_year", columns="source",
                        values=[f"{part}_{t}_pct" for t in ("t1", "t2", "t3")])
    return pd.DataFrame({t: piv[f"{part}_{t}_pct"].mean(axis=1)
                         for t in ("t1", "t2", "t3")})


def union_prevalence(df):
    return (df.total_t1_count + df.total_t2_count + df.total_t3_count) / df.total_papers * 100


def segmented(years, values, brk=BREAK):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    pre, post = years <= brk, years >= brk
    b_pre = np.polyfit(years[pre], values[pre], 1)
    b_post = np.polyfit(years[post], values[post], 1)
    return b_pre, b_post


def panel_a(ax, y):
    for src, colour, style in (("OpenAlex", OPENALEX, "-"), ("PubMed", PUBMED, "--")):
        s = y[y.source == src].sort_values("pub_year")
        u = union_prevalence(s)
        r = mk.original_test(u.values)
        ax.plot(s.pub_year, u, style, color=colour, lw=3.2,
                label=f"{src} ($\\tau$ = {r.Tau:.2f}, {sci_p(r.p)})",
                dash_capstyle="butt")

    ax.axvspan(BREAK, 2025, color="#EFEFEF", zorder=0)
    ax.set_ylabel("Militaristic Terms Prevalence (%)")
    ax.set_ylim(29, 53)
    ax.set_yticks([30, 35, 40, 45, 50])
    year_axis(ax)
    legend_swatches(ax, [(t.get_label(), t.get_color()) for t in ax.get_lines()[:2]])
    stamp(ax, "A", "Two databases, one upward trend")


def panel_b(ax, y):
    prev = pooled_by_mean(y)
    entries = []
    for t, colour in (("t1", T1), ("t2", T2), ("t3", T3)):
        s = prev[t]
        idx = 100 * s / s.iloc[0]
        r = mk.original_test(s.values)
        ax.plot(idx.index, idx.values, "-", color=colour, lw=3.2)
        entries.append((f"{t.upper()} ($\\tau$ = {r.Tau:.2f}, {sci_p(r.p)})", colour))

    ax.axhline(100, color=GREY_REF, lw=1.6, ls="--", zorder=0)
    ax.set_ylabel("Prevalence (2010 = 100)")
    ax.set_ylim(70, 160)
    ax.set_yticks(range(70, 161, 10))
    year_axis(ax)
    legend_swatches(ax, entries)
    stamp(ax, "B", "T2 and T3 rise as T1 declines")


def panel_c(ax, y):
    prev = pooled_by_count(y)
    years = prev.index.values
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color(T3)
    ax_r.spines["right"].set_linewidth(1.4)
    ax_r.tick_params(axis="y", colors=T3, direction="out", length=4.5, width=1.4,
                     labelsize=TICK)

    entries = []
    for t, colour, marker, target in (("t1", T1, "o", ax), ("t2", T2, "v", ax),
                                      ("t3", T3, "s", ax_r)):
        vals = prev[t].values
        target.scatter(years, vals, s=42, color=colour, alpha=0.45,
                       marker=marker, linewidths=0, zorder=2)
        b_pre, b_post = segmented(years, vals)
        pre_x = np.array([2010, BREAK], dtype=float)
        post_x = np.array([BREAK, 2025], dtype=float)
        target.plot(pre_x, np.polyval(b_pre, pre_x), "-", color=colour, lw=3.4, zorder=3)
        target.plot(post_x, np.polyval(b_post, post_x), "-", color=colour, lw=3.4, zorder=3)
        entries.append((f"{t.upper()} ($\\beta$ = {b_pre[0]:.2f} $\\rightarrow$ "
                        f"{b_post[0]:.2f} pp/yr)", colour))

    ax.axvline(BREAK, color="#444444", lw=1.8, ls=":", zorder=1)
    ax.set_ylabel("T1 & T2 Prevalence (%)")
    ax.set_ylim(0, 8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax_r.set_ylabel("T3 Prevalence (%)", color=T3, fontweight="bold")
    ax_r.set_ylim(20, 42)
    ax_r.set_yticks([20, 25, 30, 35, 40])
    year_axis(ax)
    legend_swatches(ax, entries)
    stamp(ax, "C", "Rise accelerates after 2019")


def panel_d(ax, y):
    g = y.groupby("pub_year").sum(numeric_only=True)
    entries = []
    for t, colour in (("t1", T1), ("t2", T2), ("t3", T3)):
        growth = []
        for part in ("abstract", "title"):
            s = g[f"{part}_{t}_count"] / g.total_papers * 100
            growth.append(100 * (s.iloc[-1] / s.iloc[0] - 1))
        ax.plot([0, 1], growth, "-o", color=colour, lw=4.0, markersize=11,
                markeredgecolor=colour, zorder=3)
        entries.append((f"{t.upper()} (abstract {growth[0]:.0f}% $\\rightarrow$ "
                        f"title {growth[1]:.0f}%)".replace("-", "−"), colour))

    ax.axhline(0, color=GREY_REF, lw=1.6, ls="--", zorder=0)
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Abstract", "Title"])
    ax.set_ylabel("Growth 2010 $\\rightarrow$ 2025 (%)")
    ax.set_ylim(-27, 78)
    ax.set_yticks([-20, 0, 20, 40, 60])
    legend_swatches(ax, entries, loc="lower left", bbox_to_anchor=(0.015, 0.13))
    stamp(ax, "D", "Uniform rise, abstracts to titles")


def panel_e(ax, y):
    llm = pd.read_csv(DATA / "tier3_llm_disambiguation.csv")
    llm = llm[llm.pub_year.between(2010, 2025)]
    g = llm.groupby("pub_year")[["t3_total", "t3_metaphor"]].sum()
    share = (g.t3_metaphor / g.t3_total).reindex(YEARS)

    prev = pooled_by_count(y)["t3"]
    militaristic = prev * share

    ax.fill_between(prev.index, 0, militaristic, color=E_DARK, linewidth=0)
    ax.fill_between(prev.index, militaristic, prev, color=E_LIGHT, linewidth=0)

    ax.set_ylabel("T3 Prevalence (%)")
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 10, 20, 30, 40])
    year_axis(ax)
    legend_swatches(ax, [(f"Scientific (~{100 - LEGEND_MILITARISTIC_PCT}%)", E_LIGHT),
                         (f"Militaristic (~{LEGEND_MILITARISTIC_PCT}%)", E_DARK)])
    stamp(ax, "E", "T3 rise isn't semantic drift")


def panel_f(ax, y):
    pl = pd.read_csv(DATA / "placebo_prevalence.csv")
    pl["pub_year"] = pl.pub_year.astype(int)
    pl = pl[pl.pub_year.between(2010, 2025)]
    gw = y.groupby("pub_year").sum(numeric_only=True)
    gp = pl.groupby("pub_year").sum(numeric_only=True)

    for i, (t, colour) in enumerate([("t1", T1), ("t2", T2), ("t3", T3)]):
        war = gw[f"total_{t}_count"] / gw.total_papers * 100
        plc = gp[f"total_p{i + 1}_count"] / gp.total_papers * 100
        war_g = 100 * (war.iloc[-1] / war.iloc[0] - 1)
        plc_g = 100 * (plc.iloc[-1] / plc.iloc[0] - 1)

        ax.plot([plc_g, war_g], [i, i], "-", color=GREY_REF, lw=4.5, zorder=1,
                solid_capstyle="round")
        ax.scatter([plc_g], [i], s=150, color=PLACEBO, zorder=3, linewidths=0)
        ax.scatter([war_g], [i], s=150, color=colour, zorder=3, linewidths=0)
        ax.annotate(f"{war_g:.0f}%".replace("-", "−"), (war_g, i),
                    textcoords="offset points", xytext=(11, 0), va="center",
                    fontsize=LEG + 1)

    ax.axvline(0, color=GREY_REF, lw=1.6, ls="--", zorder=0)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["T1", "T2", "T3"])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-52, 72)
    ax.set_xticks([-40, -20, 0, 20, 40, 60])
    ax.set_xlabel("Growth 2010 $\\rightarrow$ 2025 (%)")

    dot = lambda c: Line2D([], [], marker="o", linestyle="none",
                           markerfacecolor=c, markeredgecolor="none", markersize=11)
    ax.legend([dot(PLACEBO), (dot(T1), dot(T2), dot(T3))],
              ["Placebo", "Militaristic"], loc="upper left",
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
              handletextpad=0.5, labelspacing=0.45, handlelength=2.2)
    stamp(ax, "F", "Rise survives the placebo")


def main():
    apply_style()
    y = load()

    fig, axes = plt.subplots(2, 3, figsize=(13.96, 8.74))
    fig.subplots_adjust(left=0.072, right=0.945, top=0.90, bottom=0.085,
                        wspace=0.50, hspace=0.52)
    for ax in axes.ravel():
        despine(ax)

    panel_a(axes[0, 0], y)
    panel_b(axes[0, 1], y)
    panel_c(axes[0, 2], y)
    panel_d(axes[1, 0], y)
    panel_e(axes[1, 1], y)
    panel_f(axes[1, 2], y)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Fig2.{ext}", dpi=300)
    print(f"wrote {OUT/'Fig2.pdf'} and {OUT/'Fig2.png'}")


if __name__ == "__main__":
    main()
