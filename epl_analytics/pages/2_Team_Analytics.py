import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import artifacts_ready, load_matches, load_standings, load_style_clusters
from src.utils.styling import inject_css, plotly_layout_kwargs, team_color, theme_toggle_sidebar

st.set_page_config(page_title="Team Analytics | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

standings = load_standings()
matches = load_matches()
style = load_style_clusters()

st.markdown('<div class="eyebrow">Team Analytics</div>', unsafe_allow_html=True)
st.title("Team Deep Dive")

team = st.selectbox("Select a team", sorted(standings["Team"]))
row = standings[standings["Team"] == team].iloc[0]
style_row = style[style["Team"] == team].iloc[0] if team in style["Team"].values else None

# Build long-format per-match rows for this team (home + away combined)
home_m = matches[matches.home_team == team].copy()
home_m["is_home"] = True
home_m = home_m.rename(columns={"away_team": "opponent", "home_goals": "goals_for",
                                 "away_goals": "goals_against", "possessionPct_home": "possession",
                                 "passPct_home": "pass_pct", "shotPct_home": "shot_pct",
                                 "totalShots_home": "shots", "shotsOnTarget_home": "shots_on_target",
                                 "wonCorners_home": "corners", "foulsCommitted_home": "fouls"})
away_m = matches[matches.away_team == team].copy()
away_m["is_home"] = False
away_m = away_m.rename(columns={"home_team": "opponent", "away_goals": "goals_for",
                                 "home_goals": "goals_against", "possessionPct_away": "possession",
                                 "passPct_away": "pass_pct", "shotPct_away": "shot_pct",
                                 "totalShots_away": "shots", "shotsOnTarget_away": "shots_on_target",
                                 "wonCorners_away": "corners", "foulsCommitted_away": "fouls"})
keep = ["date", "opponent", "is_home", "goals_for", "goals_against", "possession", "pass_pct",
        "shot_pct", "shots", "shots_on_target", "corners", "fouls", "result"]
team_matches = pd.concat([home_m[keep], away_m[keep]], ignore_index=True).sort_values("date")
team_matches["points"] = team_matches.apply(
    lambda r: 3 if (r.goals_for > r.goals_against) else (1 if r.goals_for == r.goals_against else 0), axis=1)
team_matches["team"] = team

# ---------------- KPI row ----------------
st.markdown(f"### {team}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Matches", int(row["MP"]))
c2.metric("Record", f"{int(row['Win'])}W-{int(row['Draw'])}D-{int(row['Loss'])}L")
c3.metric("Goals For / Against", f"{int(row['GF'])} / {int(row['GA'])}")
c4.metric("Clean Sheets", int(row["Clean Sheets"]))
c5.metric("Points", int(row["Points"]))
if style_row is not None:
    c6.metric("Avg Possession", f"{style_row['possessionPct']:.1f}%")
else:
    c6.metric("Avg Possession", "n/a")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Avg Passing Accuracy", f"{style_row['passPct']:.1f}%" if style_row is not None else "n/a")
c8.metric("Avg Shot Accuracy", f"{team_matches['shot_pct'].mean():.1f}%")
c9.metric("Avg Corners / Match", f"{team_matches['corners'].mean():.1f}")
c10.metric("Avg Fouls / Match", f"{team_matches['fouls'].mean():.1f}")

st.divider()

tab_overview, tab_charts, tab_home_away, tab_form = st.tabs(
    ["Radar Profile", "Trends", "Home vs Away", "Rolling Form"])

with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.radar_chart(standings, [team], title=f"{team} — Percentile Profile"),
                         use_container_width=True)
    with right:
        st.markdown("**Passing heatmap (season match-by-match accuracy)**")
        # Simple heatmap: matches on x, pass accuracy bucketed as a single row --
        # honest substitute for pitch-zone data, which this dataset doesn't have.
        heat_fig = go.Figure(data=go.Heatmap(
            z=[team_matches["pass_pct"].tolist()], x=[d.strftime("%b %d") for d in team_matches["date"]],
            colorscale=[[0, "#3A241C"], [0.5, "#4A3E22"], [1, "#C7A24C"]], showscale=True,
        ))
        heat_fig.update_layout(title="Pass Accuracy by Match", yaxis=dict(showticklabels=False),
                                height=220, **plotly_layout_kwargs())
        st.plotly_chart(heat_fig, use_container_width=True)
        st.caption("Match-level accuracy shown as a strip heatmap — pitch-zone passing data "
                   "isn't present in this dataset, so this is the honest equivalent.")

with tab_charts:
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=team_matches["date"], y=team_matches["goals_for"], mode="lines+markers",
                                  name="Goals For", line=dict(color=team_color(0))))
        fig.add_trace(go.Scatter(x=team_matches["date"], y=team_matches["goals_against"], mode="lines+markers",
                                  name="Goals Against", line=dict(color=team_color(2))))
        fig.update_layout(title="Goal Trend", **plotly_layout_kwargs())
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig2 = go.Figure(go.Scatter(x=team_matches["date"], y=team_matches["possession"], mode="lines+markers",
                                     line=dict(color=team_color(1))))
        fig2.update_layout(title="Possession Trend (%)", **plotly_layout_kwargs())
        st.plotly_chart(fig2, use_container_width=True)
    g3, g4 = st.columns(2)
    with g3:
        fig3 = go.Figure(go.Scatter(x=team_matches["date"], y=team_matches["shots"], mode="lines+markers",
                                     line=dict(color=team_color(3))))
        fig3.update_layout(title="Shots per Match", **plotly_layout_kwargs())
        st.plotly_chart(fig3, use_container_width=True)
    with g4:
        team_matches["cum_points"] = team_matches["points"].cumsum()
        fig4 = go.Figure(go.Scatter(x=team_matches["date"], y=team_matches["cum_points"], mode="lines",
                                     fill="tozeroy", line=dict(color=team_color(0))))
        fig4.update_layout(title="Cumulative Points", **plotly_layout_kwargs())
        st.plotly_chart(fig4, use_container_width=True)

with tab_home_away:
    home_stats = {
        "Matches": int(row["Home_MP"]), "Wins": int(row["Home_Win"]),
        "Goals For": int(row["Home_GF"]), "Goals Against": int(row["Home_GA"]),
    }
    away_stats = {
        "Matches": int(row["Away_MP"]), "Wins": int(row["Away_Win"]),
        "Goals For": int(row["Away_GF"]), "Goals Against": int(row["Away_GA"]),
    }
    hc1, hc2 = st.columns(2)
    hc1.markdown("**Home**")
    hc1.table(pd.DataFrame([home_stats]).T.rename(columns={0: "Value"}))
    hc2.markdown("**Away**")
    hc2.table(pd.DataFrame([away_stats]).T.rename(columns={0: "Value"}))
    st.plotly_chart(charts.bar_comparison(
        ["Wins", "Goals For", "Goals Against"],
        {"Home": [home_stats["Wins"], home_stats["Goals For"], home_stats["Goals Against"]],
         "Away": [away_stats["Wins"], away_stats["Goals For"], away_stats["Goals Against"]]},
        title="Home vs Away Comparison"), use_container_width=True)

with tab_form:
    st.markdown("**Rolling 5-match form** (points won in the last 5 matches, capped at 15)")
    team_matches["rolling_form"] = team_matches["points"].rolling(5, min_periods=1).sum()
    fig5 = go.Figure(go.Scatter(x=team_matches["date"], y=team_matches["rolling_form"], mode="lines+markers",
                                 fill="tozeroy", line=dict(color=team_color(0))))
    fig5.update_layout(title=f"{team} — Rolling Form (last 5 matches)", yaxis=dict(range=[0, 15]),
                        **plotly_layout_kwargs())
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**Match log**")
    log = team_matches[["date", "opponent", "is_home", "goals_for", "goals_against", "result", "points"]].copy()
    log["venue"] = log["is_home"].map({True: "Home", False: "Away"})
    log = log.drop(columns="is_home").sort_values("date", ascending=False)
    st.dataframe(log, use_container_width=True, hide_index=True)
