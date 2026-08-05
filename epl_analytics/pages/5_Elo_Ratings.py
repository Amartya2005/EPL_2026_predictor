import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import artifacts_ready, load_elo_current, load_elo_history
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Elo Ratings | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

elo_current = load_elo_current()
elo_history = load_elo_history()

st.markdown('<div class="eyebrow">Stretch Feature &middot; Elo Rating System</div>', unsafe_allow_html=True)
st.title("Team Elo Ratings")
st.caption(
    "A standard football Elo system (K=24, +60 home advantage, margin-of-victory multiplier), "
    "carried match-by-match in chronological order across the season. All teams start at 1500. "
    "This is a real, computed rating system — not a stand-in for one of the fabricated stretch "
    "features (xG approximation, fantasy points) that were skipped because this dataset can't "
    "honestly support them. See **About**."
)

left, right = st.columns([1, 1.6])
with left:
    st.subheader("Current Ratings")
    st.dataframe(elo_current.rename(columns={"team": "Team", "elo": "Elo"}),
                 use_container_width=True, hide_index=True, height=650)

with right:
    st.subheader("Rating History")
    teams = sorted(elo_current["team"].unique())
    default_teams = elo_current.head(5)["team"].tolist()
    sel = st.multiselect("Teams to plot", teams, default=default_teams)
    if sel:
        st.plotly_chart(charts.elo_timeline(elo_history, sel), use_container_width=True)
    else:
        st.info("Select at least one team.")

st.divider()
st.subheader("Biggest Elo Swings")
elo_history["home_delta"] = elo_history["elo_home_post"] - elo_history["elo_home_pre"]
biggest = elo_history.reindex(elo_history["home_delta"].abs().sort_values(ascending=False).index).head(10)
biggest_display = biggest[["date", "home_team", "away_team", "result", "home_delta"]].copy()
biggest_display["home_delta"] = biggest_display["home_delta"].round(1)
biggest_display.columns = ["Date", "Home", "Away", "Result", "Home Elo Change"]
st.dataframe(biggest_display, use_container_width=True, hide_index=True)
