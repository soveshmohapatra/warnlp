# warnlp

Figure code for *War in the Abstract: The Rise and Consequences of Militarized
Language in Scientific Communication*.

One self-contained script per figure. Each script reads only from `data/`,
writes a PDF and a PNG into `figures/`, and shares no code with the others, so
any figure can be run, edited or copied on its own.

## Quick start

```bash
pip install -r requirements.txt
python fig2.py
```

Run every figure:

```bash
for f in fig2 fig3 fig4 fig5 fig6 figS2 figS3 figS4; do python $f.py; done
```

## Figures

| Script | Output | Content |
| --- | --- | --- |
| `fig2.py` | `figures/Fig2.pdf` | Temporal dynamics, 2010–2025: cross-database trends, indexed tiers, the 2019 structural break, title-vs-abstract growth, Tier-3 LLM disambiguation, placebo comparison |
| `fig3.py` | `figures/Fig3.pdf` | Geographic variation: 2010 and 2024 choropleths, hemispheric trajectories, highest-volume producers, steepest climbs |
| `fig4.py` | `figures/Fig4.pdf` | Disciplinary patterns: level, tier composition, deliberateness index, per-field trajectories, growth in titles versus abstracts |
| `fig5.py` | `figures/Fig5.pdf` | Armed conflict: UCDP overlay, indicator correlations, conflict versus peaceful nations, Ukraine and Russia, conflict-type bootstrap |
| `fig6.py` | `figures/Fig6.pdf` | Pandemic and LLM eras: pandemic versus other papers, tier decomposition, per-field acceleration, language groups, pre/post-LLM slopes |
| `figS2.py` | `figures/FigS2.pdf` | Cross-database concordance |
| `figS3.py` | `figures/FigS3.pdf` | Post-2019 acceleration, conflict-involved versus peaceful nations |
| `figS4.py` | `figures/FigS4.pdf` | Independent LLM tone validation, by database |

## Data

Everything the scripts need is in `data/`.

| File | Description |
| --- | --- |
| `yearly_prevalence.csv` | Tier counts and paper totals per year and database, 2010–2025 |
| `country_prevalence.csv` | The same, per country |
| `country_t1_t2_verification.csv` | Per-country union prevalence (any tier in title or abstract) |
| `category_prevalence.csv` | The same, per super-discipline |
| `placebo_prevalence.csv` | Frequency-matched placebo lexicon counts |
| `sensitivity_covid.csv` | Corpus with pandemic-related papers removed |
| `tier3_llm_disambiguation.csv` | Per-year Tier-3 occurrences classified as metaphorical, literal or uncertain |
| `tone_scores_yearly.csv` | Mean LLM militarization tone (0–5) per year and database |
| `conflict_correlation.json` | UCDP-derived conflict / peaceful country classification |
| `UCDP/UcdpPrioConflict_v25_1.csv` | UCDP/PRIO Armed Conflict Dataset v25.1 |
| `UCDP/BattleDeaths_v25_1_conf.csv` | UCDP Battle-Related Deaths Dataset v25.1 |
| `ne_50m_admin_0_countries.zip` | Natural Earth 1:50m country boundaries (public domain) |

`tier3_llm_disambiguation.csv` and `tone_scores_yearly.csv` are per-year
aggregates of the paper-level classifier outputs, which are too large to
distribute here.

## Requirements

Python 3.10 or newer. See `requirements.txt`.
