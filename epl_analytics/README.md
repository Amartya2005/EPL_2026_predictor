# EPL 2025-26 Analytics Dashboard

A team-level analytics platform for the English Premier League 2025-26 season — built as a
production-style upgrade of a small leakage-free ML project into a full Streamlit application.
380 verified match results, 20 teams, honest model reporting, no fabricated data.

> **Scope note:** this build is deliberately team-level only. The original brief asked for
> player analytics, fantasy points, and xG approximation as well — those were scoped out because
> either the data doesn't honestly support them (xG, fantasy points) or it was an explicit
> decision to keep this pass team-only (player analytics; see [About](pages/10_About.py) in the
> app for the full breakdown of what was built vs. skipped, and why).

---

## Features

- **Home dashboard** — league KPIs, top-of-table snapshot, current Elo top 5, playing-style map.
- **League Table** — sortable/filterable standings (rebuilt from match results, not the buggy
  source export), home/away splits, points progression, goal difference, win % charts.
- **Team Analytics** — per-team radar profile, goal/possession/shot trends, home vs away split,
  rolling 5-match form, full match log, CSV export.
- **Team Comparison** — side-by-side stat comparison, radar overlay, Elo-based win probability.
- **Match Predictor** — pick any two teams, get outcome probabilities from 3 trained models, a
  live SHAP explanation of that specific prediction, and global feature importance.
- **Elo Ratings** — a real, chronologically-computed Elo system (K=24, home advantage, margin-of-
  victory weighting) with full rating history and biggest single-match swings.
- **Clustering** — 4-tier team-strength KMeans and a 5-cluster PCA playing-style map, both with
  plain-language cluster explanations.
- **Feature Importance** — top-N features by XGBoost gain, Random Forest impurity, or SHAP.
- **Model Performance** — accuracy/precision/recall/F1, confusion matrices, ROC curves, learning
  curves, and time-series cross-validation for all 3 classifiers, plus goal-regression MAE.
- **Data Explorer** — search/sort/filter/download every dataset behind the app.
- **Dark / light theme toggle**, responsive layout, cached data loading throughout.

## Screenshots

*(Run the app locally and drop screenshots here — `streamlit run dashboard.py`, then capture the
Home, Match Predictor, and Model Performance pages.)*

## Architecture

```
                    ┌─────────────────────────────┐
                    │  Soccer-Stats-...R02.xlsx    │
                    │  (raw workbook, data/raw/)   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   src/data/load_data.py        │  EPL isolation, home/away join,
                    │   src/data/features.py         │  standings rebuild, leakage-safe
                    └───────────────┬───────────────┘  rolling features
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   src/models/clustering.py  src/models/elo.py    src/models/train.py
   (strength tiers,          (chronological Elo)   (LogReg / RF / XGBoost,
   playing style PCA)                               CV, ROC, learning curves,
                                                      SHAP)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  data/processed/*.csv, *.json  │
                    │  models/*.pkl                  │
                    └───────────────┬───────────────┘
                                    │  (read-only, cached)
                    ┌───────────────▼───────────────┐
                    │  dashboard.py + pages/*.py      │
                    │  src/utils/{data_loader,        │
                    │  styling,charts}.py             │
                    └───────────────────────────────┘
```

The app never retrains models or reloads the workbook at runtime — everything under
`data/processed/` and `models/` is generated once by `scripts/build_all.py` and read through
`@st.cache_data` / `@st.cache_resource` wrappers in `src/utils/data_loader.py`.

## Installation

```bash
git clone <this-repo>
cd epl_analytics
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Place the source workbook at `data/raw/Soccer-Stats-Premier-League-2025-2026_R02.xlsx` if it
isn't already there (it ships with this repo).

## Usage

Build the data/model artifacts once (or whenever the source workbook changes):

```bash
python scripts/build_all.py
```

Then launch the dashboard:

```bash
streamlit run dashboard.py
```

Run the test suite:

```bash
pytest tests/ -v
```

## Model explanation

Match-outcome features are **shifted rolling averages of each team's prior matches only** — the
raw `TeamStatsExport` possession/shots/passing columns are post-match outcomes for that same
fixture, and using them directly would be target leakage (a model trained on them looks like 90%+
accuracy and is worthless for real prediction). The train/test split is chronological (first ~75%
of the season vs. the final gameweeks), matching how the model would actually be used —
predicting upcoming games from past ones, never the reverse.

**Result:** none of the three trained models (Logistic Regression, Random Forest, XGBoost)
meaningfully beat "always predict the home team" (44.8% baseline accuracy) on this single season.
XGBoost has the best macro-F1 — it's the only one that predicts draws and away wins at all — but
that's a fairer result, not a better one. Full breakdown on the Model Performance page and in
[About](pages/10_About.py). Real match-outcome models need multi-season Elo history, head-to-head
record, and market odds as features; none of those exist in a single-season workbook.

## Dashboard walkthrough

1. **Home** — start here for the league-wide picture and KPIs.
2. **League Table → Team Analytics → Team Comparison** — drill from league-wide to per-team to
   head-to-head.
3. **Match Predictor** — pick a hypothetical fixture, see model probabilities and why the model
   said what it said (SHAP).
4. **Elo Ratings / Clustering / Feature Importance / Model Performance** — the analytical/ML
   layer, each with its own honest caveats.
5. **Data Explorer** — the underlying data, searchable and downloadable.
6. **About** — data-quality issues found and fixed, full methodology, and an honest scorecard of
   what was built vs. skipped from the original brief.

## Future improvements

- Player-level module (`PlayerStatsExport` / `TeamRosterExport` exist in the source workbook but
  weren't scoped into this build).
- Multi-season history to give the Elo system and match predictor real signal to work with.
- Match-event timeline using `PlaysExport` / `LineUpExport`.
- Team crests/kit colors if a logo asset source is added.

## Deployment

**Streamlit Community Cloud:** push this repo (including `data/processed/` and `models/`, or run
`build_all.py` in a build step) to GitHub, then deploy directly from
[share.streamlit.io](https://share.streamlit.io) pointing at `dashboard.py`.

**Docker:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && python scripts/build_all.py
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Technologies used

Streamlit · Plotly · pandas · scikit-learn · XGBoost · SHAP · pytest

## Project structure

```
epl_analytics/
├── dashboard.py
├── pages/                    # 1-10, Streamlit multipage app
├── src/
│   ├── data/                 # load_data.py, features.py
│   ├── models/                # train.py, clustering.py, elo.py
│   └── utils/                 # data_loader.py, styling.py, charts.py
├── data/
│   ├── raw/                  # source workbook
│   └── processed/            # generated CSV/JSON artifacts (gitignore-able)
├── models/                   # pickled trained models (gitignore-able)
├── scripts/build_all.py      # offline pipeline entrypoint
├── tests/                    # pytest suite
├── assets/                   # favicon, legacy static dashboard.html
├── requirements.txt
└── README.md
```

## License

MIT.
