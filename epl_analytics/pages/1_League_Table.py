import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import artifacts_ready, load_match_features, load_matches, load_standings
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="League Table | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

st.markdown('<div class="eyebrow">League Table</div>', unsafe_allow_html=True)
st.title("2025-26 Premier League Table")
st.caption("Rebuilt directly from the 380 verified match results, not the source workbook's "
           "league table export (which is missing 3 of the 20 teams).")

standings = load_standings()
matches = load_matches()

# ---------------- Filters ----------------
with st.expander("Filters", expanded=True):
    fc1, fc2, fc3, fc4 = st.columns(4)
    team_filter = fc1.multiselect("Team", sorted(standings["Team"]), default=[])
    home_away = fc2.selectbox("Split", ["Overall", "Home only", "Away only"])
    top_n = fc3.slider("Show top N", 5, 20, 20)
    min_clean_sheets = fc4.slider("Min clean sheets", 0, int(standings["Clean Sheets"].max()), 0)

view = standings.copy()
if home_away == "Home only":
    view = view.assign(
        MP=view["Home_MP"],
        Win=view["Home_Win"],
        GF=view["Home_GF"],
        GA=view["Home_GA"],
    )
    view["GD"] = view["GF"] - view["GA"]
elif home_away == "Away only":
    view = view.assign(
        MP=view["Away_MP"],
        Win=view["Away_Win"],
        GF=view["Away_GF"],
        GA=view["Away_GA"],
    )
    view["GD"] = view["GF"] - view["GA"]

if team_filter:
    view = view[view["Team"].isin(team_filter)]
view = view[view["Clean Sheets"] >= min_clean_sheets]
view = view.sort_values("Points" if home_away == "Overall" else "GF", ascending=False).head(top_n)

st.subheader("Standings")
display_cols = ["Rank", "Team", "MP", "Win", "Draw", "Loss", "GF", "GA", "GD", "Points",
                 "Clean Sheets", "Win %"] if home_away == "Overall" else \
    ["Team", "MP", "Win", "GF", "GA", "GD"]
st.dataframe(view[[c for c in display_cols if c in view.columns]], use_container_width=True,
             hide_index=True, height=min(760, 40 + 35 * len(view)))

csv = view.to_csv(index=False).encode("utf-8")
st.download_button("Download table as CSV", csv, "league_table.csv", "text/csv")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Points Progression", "Goal Difference", "Goals Scored vs Conceded", "Win %"])

feats = load_match_features()

def _points_progression(matches: pd.DataFrame, teams: list[str]):
    rows = []
    for _, m in matches.sort_values("date").iterrows():
        rows.append((m.date, m.home_team, 3 if m.result == "H" else (1 if m.result == "D" else 0)))
        rows.append((m.date, m.away_team, 3 if m.result == "A" else (1 if m.result == "D" else 0)))
    long_df = pd.DataFrame(rows, columns=["date", "team", "points"])
    long_df["cum_points"] = long_df.groupby("team")["points"].cumsum()
    return long_df[long_df["team"].isin(teams)] if teams else long_df

with tab1:
    default_teams = standings.head(5)["Team"].tolist()
    sel = st.multiselect("Teams to plot", sorted(standings["Team"]), default=default_teams, key="pp_teams")
    prog = _points_progression(matches, sel)
    import plotly.graph_objects as go
    from src.utils.styling import plotly_layout_kwargs, team_color
    fig = go.Figure()
    for i, team in enumerate(sel):
        sub = prog[prog.team == team]
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["cum_points"], mode="lines", name=team,
                                  line=dict(color=team_color(i))))
    fig.update_layout(title="Cumulative Points Over the Season", xaxis_title="Date",
                       yaxis_title="Points", **plotly_layout_kwargs())
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.plotly_chart(charts.bar_comparison(view["Team"].tolist(), {"Goal Difference": view["GD"].tolist()},
                                           title="Goal Difference by Team"), use_container_width=True)

with tab3:
    st.plotly_chart(charts.bar_comparison(
        view["Team"].tolist(), {"Goals For": view["GF"].tolist(), "Goals Against": view["GA"].tolist()},
        title="Goals Scored vs Conceded"), use_container_width=True)

with tab4:
    if "Win %" in view.columns:
        st.plotly_chart(charts.bar_comparison(view["Team"].tolist(), {"Win %": view["Win %"].tolist()},
                                               title="Win Percentage"), use_container_width=True)
    else:
        st.info("Win % only available in Overall split.")
