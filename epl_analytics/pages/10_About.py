import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="About | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

st.markdown('<div class="eyebrow">About</div>', unsafe_allow_html=True)
st.title("About This Project")

st.markdown("""
### What this is

A team-level analytics platform for the English Premier League 2025-26 season, built from
`Soccer-Stats-Premier-League-2025-2026_R02.xlsx` (380 completed matches, 20 teams). It covers
league standings, per-team analytics, head-to-head comparison, a leakage-free match-outcome
predictor, Elo ratings, team-strength and playing-style clustering, and full model-performance
reporting.

### Scope: why there's no player analytics

The spec this dashboard was built from called for player-level pages (player search, goals/
assists/cards, percentile rankings, player comparison). Those were **intentionally left out** —
not because the data doesn't exist, but because that was the explicit choice made when scoping
this build: keep it strictly team-level. `PlayerStatsExport` and `TeamRosterExport` do exist
in the source workbook and could support a future player module.

### Data-quality issues found and fixed

1. **`AllFixturesExport['Away Score']` is 0 for every EPL row.** Dead placeholder data. Real
   scores are sourced from `TeamStatsExport['Home Goal'/'Away Goal']` instead — cross-checked
   against known results.
2. **Free-text fixture strings don't match between sheets** (`"X vs Y"` vs `"Y at X"`). The only
   join key used anywhere in this pipeline is `date + home_team + away_team`.
3. **`LeagueTableExport` is missing 3 of the 20 teams.** Standings are rebuilt directly from the
   380 verified match results (`src/data/load_data.py::build_standings`), not read from that sheet.

### Methodology: why the predictor is leakage-free

`TeamStatsExport`'s possession/shots/passing columns are **post-match** stats — a consequence of
the result, not a predictor of it. Feeding them into a classifier for the same fixture they
describe is target leakage, and produces a model that looks like 90%+ accuracy while being
worthless for real prediction. Every feature used by the match predictor is a **shifted rolling
average of each team's prior matches only** (`ROLL_WINDOW = 5`), computed before the fixture it's
predicting. The train/test split is chronological (first ~75% of the season vs. the final
gameweeks), not random K-fold, because a random split would let the model train on gameweek 30
and validate on gameweek 10 — something that can never happen in production use.

### Honest result

None of the three trained models (Logistic Regression, Random Forest, XGBoost) meaningfully beat
"always predict the home team" on raw accuracy, on this single season of data. XGBoost has the
best macro-F1 (it's the only one that predicts draws and away wins at all), but that's a fairer,
not a better, result. See the **Model Performance** page for full numbers. This is the expected
outcome for 38 games with only 5-match rolling form as signal — real match-outcome models rely on
multi-season Elo history, head-to-head record, and market odds as features, none of which exist
in this single-season workbook.

### Stretch goals: what was built vs. skipped, and why

| Feature | Status | Why |
|---|---|---|
| Team Elo rating system | **Built** | Computable directly and honestly from match results — no fabricated inputs needed. |
| Historical form tracker | **Built** | Rolling-form and cumulative-points charts on Team Analytics / League Table. |
| Match momentum / cumulative points | **Built** | League Table and Team Analytics pages. |
| Expected Goals (xG) approximation | **Skipped** | Would require shot-location or shot-quality data this workbook doesn't have — any xG model built on season aggregates alone would be a guess dressed up as a metric. |
| Fantasy Football points predictor | **Skipped** | No player-level minutes/goals/assists were scoped into this build (see above). |
| Team Playing Style classifier | **Built** | PCA + KMeans on possession/passing/tackling profile, on the Clustering page. |
| Interactive match-event timeline | **Skipped** | Out of scope for this pass; would need `PlaysExport`/`LineUpExport` integration, which is a reasonable next addition. |
| Dark/Light theme toggle | **Built** | Sidebar toggle on every page. |

### Tech stack

Streamlit, Plotly, scikit-learn, XGBoost, SHAP, pandas. All heavy computation (data loading,
feature engineering, model training) runs offline via `scripts/build_all.py` — the app only reads
pre-computed CSV/JSON/pickle artifacts, so pages stay fast.

### Project structure

```
epl_analytics/
├── dashboard.py            # Home page / entry point
├── pages/                  # Streamlit multipage app
├── src/
│   ├── data/                # load_data.py, features.py
│   ├── models/               # train.py, clustering.py, elo.py
│   └── utils/                # styling.py, charts.py, data_loader.py
├── data/
│   ├── raw/                  # source workbook
│   └── processed/            # generated CSV/JSON artifacts
├── models/                  # pickled trained models
├── scripts/build_all.py    # runs the full offline pipeline
├── tests/
└── README.md
```
""")
