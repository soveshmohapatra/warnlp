import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

OPENALEX, PUBMED = "#E67E22", "#8E44AD"
CONFLICT, PEACEFUL = "#E05A4B", "#5D6D7D"
BAR_GREY, GREY_REF, SPINE = "#DADDE2", "#CCCCCC", "#333333"
TYPE_COLOUR = {"Interstate": "#C64C40", "Intrastate": "#E36A5D",
               "Internationalized": "#F19E95"}
TYPE_CODE = {"Interstate": 2, "Intrastate": 3, "Internationalized": 4}

CONFLICT_END = 2024
N_BOOTSTRAP, SEED = 2000, 42

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


def sci_p(p, digits=0):
    exp = int(np.floor(np.log10(p)))
    mant = round(p / 10 ** exp, digits)
    if mant >= 10:
        mant, exp = mant / 10, exp + 1
    return f"p = {mant:.{digits}f}$\\times$10$^{{{exp}}}$"


def legend_swatches(ax, entries, loc="upper left", **kw):
    handles = [Line2D([], [], color=c, lw=7, solid_capstyle="butt") for _, c in entries]
    return ax.legend(handles, [t for t, _ in entries], loc=loc, handlelength=1.0,
                     handletextpad=0.55, borderpad=0.1, labelspacing=0.42, **kw)


def load():
    y = pd.read_csv(DATA / "yearly_prevalence.csv")
    y = y[y.pub_year.between(2010, 2025)]
    acd = pd.read_csv(DATA / "UCDP/UcdpPrioConflict_v25_1.csv", low_memory=False)
    brd = pd.read_csv(DATA / "UCDP/BattleDeaths_v25_1_conf.csv", low_memory=False)
    cc = json.loads((DATA / "conflict_correlation.json").read_text())
    return y, acd, brd, cc


def prevalence(df):
    g = df.groupby("pub_year").sum(numeric_only=True)
    return (g.total_t1_count + g.total_t2_count + g.total_t3_count) / g.total_papers * 100


def conflict_indicators(acd, brd, years):
    u = acd[acd.year.isin(years)]
    n_conf = u.groupby("year").conflict_id.nunique().reindex(years).fillna(0)
    n_wars = (u[u.intensity_level == 2].groupby("year").conflict_id.nunique()
              .reindex(years).fillna(0))
    deaths = brd[brd.year.isin(years)].groupby("year").bd_best.sum().reindex(years).fillna(0)
    return n_conf, deaths, n_wars


def panel_a(ax, y, acd, brd):
    years = list(range(2010, CONFLICT_END + 1))
    n_conf, _, _ = conflict_indicators(acd, brd, years)

    ax.bar(years, n_conf.values, width=0.72, color=BAR_GREY, zorder=1)
    ax.set_ylabel("Active Conflicts")
    ax.set_ylim(0, 70)
    ax.set_yticks(range(0, 71, 10))

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_linewidth(1.4)
    ax_r.spines["right"].set_color(SPINE)
    ax_r.tick_params(direction="out", length=4.5, width=1.4, labelsize=TICK)

    entries = []
    for src, colour, style, marker in (("OpenAlex", OPENALEX, "-", "o"),
                                       ("PubMed", PUBMED, "--", "s")):
        p = prevalence(y[(y.source == src) & y.pub_year.isin(years)])
        r, pv = pearsonr(p.values, n_conf.values)
        ax_r.plot(years, p.values, style, color=colour, lw=3.0, marker=marker,
                  markersize=6, zorder=3)
        entries.append((f"{src} (r = {r:.2f}; {sci_p(pv)})", colour))

    ax_r.set_ylabel("Militaristic Terms Prevalence (%)")
    ax_r.set_ylim(30, 48)
    ax_r.set_yticks(range(30, 49, 2))
    ax.set_xticks([2010, 2013, 2016, 2019, 2022])
    ax.set_xlim(2009.2, CONFLICT_END + 0.8)
    legend_swatches(ax, entries)


def panel_b(ax, y, acd, brd):
    years = list(range(2010, CONFLICT_END + 1))
    n_conf, deaths, n_wars = conflict_indicators(acd, brd, years)
    rows = [("Active\nConflicts", n_conf), ("Battle\nDeaths", deaths),
            ("Major\nWars", n_wars)]

    for i, (label, ind) in enumerate(rows):
        pos = len(rows) - 1 - i
        pts = []
        for src, colour in (("OpenAlex", OPENALEX), ("PubMed", PUBMED)):
            p = prevalence(y[(y.source == src) & y.pub_year.isin(years)])
            pts.append((pearsonr(p.values, ind.values)[0], colour))
        ax.plot([pts[0][0], pts[1][0]], [pos, pos], "-", color=GREY_REF, lw=4.5,
                solid_capstyle="round", zorder=1)
        for r, colour in pts:
            ax.scatter([r], [pos], s=160, color=colour, zorder=3, linewidths=0)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl for lbl, _ in rows][::-1])
    ax.set_ylim(-0.45, len(rows) - 0.55)
    ax.set_xlabel("Correlation with Prevalence (r)")
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=10,
                              markerfacecolor=c, markeredgecolor="none", label=t)
                       for t, c in (("OpenAlex", OPENALEX), ("PubMed", PUBMED))],
              loc="upper left", handletextpad=0.4, labelspacing=0.4, borderpad=0.1)


def panel_c(ax, cc):
    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    cp = cp[cp.pub_year.between(2010, 2025)]

    def indexed(names):
        d = cp[cp.country.isin(names)].groupby(["country", "pub_year"]).sum(numeric_only=True)
        d = d.reset_index()
        d["p"] = (d.total_t1_count + d.total_t2_count + d.total_t3_count) / d.total_papers * 100
        piv = d.pivot_table(index="pub_year", columns="country", values="p")
        return (100 * piv / piv.iloc[0]).mean(axis=1)

    groups = [("Conflict-involved Nations", cc["B_country_level"]["conflict_countries"],
               CONFLICT, "-", "o"),
              ("Peaceful Nations", cc["B_country_level"]["peaceful_countries"],
               PEACEFUL, "--", "s")]
    entries = []
    for label, names, colour, style, marker in groups:
        s = indexed(names)
        ax.plot(s.index, s.values, style, color=colour, lw=3.0, marker=marker,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2, zorder=3)
        entries.append((f"{label} (+{s.iloc[-1] - 100:.0f}%)", colour))

    ax.axhline(100, color=GREY_REF, lw=1.6, zorder=1)
    ax.set_ylabel("Prevalence (2010 = 100)")
    ax.set_ylim(97, 172)
    ax.set_yticks(range(100, 171, 10))
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_xlim(2010, 2025)
    legend_swatches(ax, entries)


def panel_d(fig, gs, brd):
    cp = pd.read_csv(DATA / "country_prevalence.csv").rename(columns={"countries": "country"})
    cp = cp[cp.pub_year.between(2010, CONFLICT_END)]
    years = list(range(2010, CONFLICT_END + 1))

    for k, country in enumerate(("Ukraine", "Russia")):
        ax = fig.add_subplot(gs[0, k])
        despine(ax)
        d = cp[cp.country == country].groupby("pub_year").sum(numeric_only=True)
        p = ((d.total_t1_count + d.total_t2_count + d.total_t3_count)
             / d.total_papers * 100).reindex(years)
        ax.plot(years, p.values, "-", color=CONFLICT, lw=3.0, marker="o",
                markersize=5, markerfacecolor="white", markeredgewidth=1.2, zorder=3)
        ax.fill_between(years, 0, p.values, color=CONFLICT, alpha=0.12, linewidth=0)
        ax.set_ylim(0, 80)
        ax.set_yticks(range(0, 81, 10))
        ax.set_xlim(2009.2, CONFLICT_END + 0.8)
        ax.set_xticks([2010, 2014, 2018, 2022])
        ax.set_title(country, fontsize=BASE, fontweight="bold", pad=4)

        deaths = (brd[brd.location_inc.astype(str).str.contains(country, na=False)
                      & brd.year.isin(years)].groupby("year").bd_best.sum()
                  .reindex(years).fillna(0) / 1000)
        ax_r = ax.twinx()
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(True)
        ax_r.spines["right"].set_linewidth(1.4)
        ax_r.spines["right"].set_color(PEACEFUL)
        ax_r.bar(years, deaths.values, width=0.20, color=PEACEFUL, alpha=0.62,
                 zorder=2, linewidth=0)
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks(range(0, 101, 20))
        ax_r.tick_params(axis="y", colors=PEACEFUL, direction="out", length=4.5,
                         width=1.4, labelsize=TICK)

        if k == 0:
            ax.set_ylabel("Militaristic Terms Prevalence (%)", color=CONFLICT)
            ax.tick_params(axis="y", colors=CONFLICT)
            ax.spines["left"].set_color(CONFLICT)
            ax_r.set_yticklabels([])
            ax_r.tick_params(axis="y", length=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax_r.set_ylabel("Battle Deaths (1000s)", color=PEACEFUL)


def panel_e(ax, y, acd):
    years = list(range(2010, CONFLICT_END + 1))
    glob = prevalence(y[y.pub_year.isin(years)]).reindex(years).values

    u = acd[acd.year.isin(years)]
    counts = (u.groupby(["year", "type_of_conflict"]).conflict_id.nunique()
              .unstack(fill_value=0).reindex(years).fillna(0))

    rng = np.random.default_rng(SEED)
    n = len(years)
    order = ["Internationalized", "Intrastate", "Interstate"]
    for pos, name in enumerate(order):
        series = counts[TYPE_CODE[name]].values.astype(float)
        rs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            if np.std(series[idx]) > 0 and np.std(glob[idx]) > 0:
                rs.append(pearsonr(series[idx], glob[idx])[0])
        bp = ax.boxplot([np.array(rs)], positions=[pos], widths=0.44, vert=False,
                        patch_artist=True, showfliers=False, zorder=2)
        bp["boxes"][0].set(facecolor=TYPE_COLOUR[name], edgecolor="none")
        for elem in ("whiskers", "caps"):
            for art in bp[elem]:
                art.set(color="#8A8A8A", linewidth=1.5)
        bp["medians"][0].set(color="black", linewidth=2.4)

    ax.axvline(0, color="#333333", lw=1.6, ls=":", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Bootstrap Correlation with Prevalence")
    ax.set_xlim(-0.32, 1.05)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    apply_style()
    y, acd, brd, cc = load()

    fig = plt.figure(figsize=(14.19, 8.82))
    gs_top = fig.add_gridspec(1, 3, left=0.075, right=0.945, top=0.925, bottom=0.575,
                              wspace=0.62)
    gs_d = fig.add_gridspec(1, 2, left=0.105, right=0.475, top=0.435, bottom=0.095,
                            wspace=0.06)
    gs_e = fig.add_gridspec(1, 1, left=0.635, right=0.985, top=0.435, bottom=0.115)

    ax_a, ax_b, ax_c = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    for ax in (ax_a, ax_b, ax_c):
        despine(ax)
    panel_a(ax_a, y, acd, brd)
    panel_b(ax_b, y, acd, brd)
    panel_c(ax_c, cc)

    panel_d(fig, gs_d, brd)
    ax_e = fig.add_subplot(gs_e[0, 0])
    despine(ax_e)
    panel_e(ax_e, y, acd)

    stamp(fig, 0.008, 0.955, "A", "Militarized language tracks armed conflict")
    stamp(fig, 0.343, 0.955, "B", "Conflicts spread, language follows")
    stamp(fig, 0.665, 0.955, "C", "Conflict-involved nations rise faster")
    stamp(fig, 0.048, 0.475, "D", "Both sides of the war show the coupling")
    stamp(fig, 0.560, 0.475, "E", "Interstate wars resonate most")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Fig5.{ext}", dpi=300)
    print(f"wrote {OUT/'Fig5.pdf'} and {OUT/'Fig5.png'}")


if __name__ == "__main__":
    main()
