"""
EPL 2025-26 Analytics -- main entry point (Home page).

Run with: streamlit run dashboard.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import charts
from src.utils.data_loader import (artifacts_ready, league_kpis, load_elo_current,
                                    load_standings, load_style_clusters)
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="EPL 2025-26 Analytics", page_icon="assets/favicon.svg",
                    layout="wide", initial_sidebar_state="expanded")

theme_toggle_sidebar()
inject_css()

st.sidebar.markdown('<div class="eyebrow">EPL 2025-26 Analytics</div>', unsafe_allow_html=True)
st.sidebar.caption("Team-level match analytics, honest ML, and playing-style clustering "
                    "for the English Premier League 2025-26 season.")

if not artifacts_ready():
    st.error(
        "Data artifacts not found. Run `python scripts/build_all.py` from the project "
        "root first to generate the processed data and trained models this dashboard reads."
    )
    st.stop()

kpis = league_kpis()
standings = load_standings()
elo = load_elo_current()
style = load_style_clusters()

st.markdown('<div class="eyebrow">Premier League &middot; 2025-26 Season &middot; Final Table</div>',
            unsafe_allow_html=True)
st.title("EPL Analytics Dashboard")
st.caption(
    "A leakage-free, honestly-reported analytics platform built on 380 verified match "
    "results. No player data, fabricated stats, or inflated model accuracy -- "
    "see the **About** page for the data-quality issues found and fixed."
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Teams", kpis["total_teams"])
c2.metric("Matches Played", kpis["total_matches"])
c3.metric("Avg Goals / Match", kpis["avg_goals_per_match"])
c4.metric("League Leader", kpis["leader"], f"{kpis['leader_points']} pts")
c5.metric("Best Attack", kpis["best_attack"], f"{kpis['best_attack_goals']} GF")
c6.metric("Best Defense", kpis["best_defense"], f"{kpis['best_defense_conceded']} GA")

st.divider()

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Top of the Table")
    top10 = standings.head(10)[["Rank", "Team", "MP", "Win", "Draw", "Loss", "GF", "GA", "GD", "Points"]]
    st.dataframe(top10, use_container_width=True, hide_index=True)
    st.caption("Full sortable, filterable table on the **League Table** page.")

with right:
    st.subheader("Current Elo Top 5")
    st.dataframe(elo.head(5)[["Rank", "team", "elo"]].rename(columns={"team": "Team", "elo": "Elo"}),
                 use_container_width=True, hide_index=True)
    st.caption("Elo carried match-by-match across the season. See **Elo Ratings**.")

st.divider()
st.subheader("Playing Style at a Glance")
st.plotly_chart(charts.pca_scatter(style), use_container_width=True)
st.caption("Full breakdown with cluster explanations on the **Clustering** page.")

st.divider()
with st.container():
    st.markdown(
        '<div class="honest-box"><b>Reality check:</b> the match-outcome model in this '
        "dashboard does not meaningfully beat \u201calways predict the home team\u201d on a single "
        "season of data \u2014 that result is shown plainly on the Model Performance page rather "
        "than hidden. See About for the full explanation.</div>",
        unsafe_allow_html=True,
    )

st.sidebar.divider()
st.sidebar.caption("Navigate using the pages above.")
