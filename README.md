# ⚽ EPL 2025-26 Analytics Dashboard

This repository contains a full-stack, team-level analytics platform for the English Premier League 2025-26 season.

It includes:
- a 10-page Streamlit dashboard
- a real chronological Elo system
- leakage-safe feature engineering
- trained match-outcome classifiers and model-performance views
- honest reporting with no fabricated numbers

The detailed project documentation, installation steps, architecture, and usage instructions are in [epl_analytics/README.md](epl_analytics/README.md).

## Quick Start

```bash
git clone https://github.com/Amartya2005/EPL_2026_predictor.git
cd EPL_2026_predictor/epl_analytics
pip install -r requirements.txt
python scripts/build_all.py
streamlit run dashboard.py
```

## Project Scope

This build is intentionally team-level only. Player analytics, fantasy points, and xG approximation were scoped out because the available workbook data does not support them reliably enough for honest reporting.

See the in-app About page and the linked project README for the full methodology and caveats.
