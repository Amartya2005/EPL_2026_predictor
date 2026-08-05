import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import (artifacts_ready, load_elo_current, load_standings,
                                    load_style_clusters)
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Team Comparison | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

standings = load_standings()
style = load_style_clusters()
elo = load_elo_current()

st.markdown('<div class="eyebrow">Team Comparison</div>', unsafe_allow_html=True)
st.title("Compare Two Teams")

teams = sorted(standings["Team"])
c1, c2 = st.columns(2)
team_a = c1.selectbox("Team A", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
team_b = c2.selectbox("Team B", teams, index=teams.index("Liverpool") if "Liverpool" in teams else 1)

row_a = standings[standings.Team == team_a].iloc[0]
row_b = standings[standings.Team == team_b].iloc[0]
style_a = style[style.Team == team_a].iloc[0] if team_a in style.Team.values else None
style_b = style[style.Team == team_b].iloc[0] if team_b in style.Team.values else None
elo_a = elo[elo.team == team_a]["elo"].iloc[0] if team_a in elo.team.values else None
elo_b = elo[elo.team == team_b]["elo"].iloc[0] if team_b in elo.team.values else None

st.divider()
metric_rows = [
    ("Points", row_a["Points"], row_b["Points"]),
    ("Goals For", row_a["GF"], row_b["GF"]),
    ("Goals Against", row_a["GA"], row_b["GA"]),
    ("Goal Difference", row_a["GD"], row_b["GD"]),
    ("Clean Sheets", row_a["Clean Sheets"], row_b["Clean Sheets"]),
]
if style_a is not None and style_b is not None:
    metric_rows += [
        ("Possession %", round(style_a["possessionPct"], 1), round(style_b["possessionPct"], 1)),
        ("Passing Accuracy %", round(style_a["passPct"], 1), round(style_b["passPct"], 1)),
        ("Shots / Match", round(style_a["totalShots"], 1), round(style_b["totalShots"], 1)),
        ("Tackles / Match", round(style_a["totalTackles"], 1), round(style_b["totalTackles"], 1)),
    ]
if elo_a is not None:
    metric_rows.append(("Elo Rating", round(elo_a, 0), round(elo_b, 0)))

cols = st.columns(3)
cols[0].markdown(f"### {team_a}")
cols[1].markdown("### vs")
cols[2].markdown(f"### {team_b}")
for label, va, vb in metric_rows:
    c = st.columns(3)
    c[0].metric(label, va)
    c[1].markdown(f"<div style='text-align:center;color:var(--sage);padding-top:14px'>{label}</div>",
                  unsafe_allow_html=True)
    c[2].metric(label, vb)

st.divider()
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(charts.radar_chart(standings, [team_a, team_b], title="Radar Comparison"),
                     use_container_width=True)
with g2:
    labels = [m[0] for m in metric_rows[:5]]
    st.plotly_chart(charts.bar_comparison(
        labels, {team_a: [m[1] for m in metric_rows[:5]], team_b: [m[2] for m in metric_rows[:5]]},
        title="Bar Comparison"), use_container_width=True)

st.divider()
st.subheader("Win Probability Estimate")
if elo_a is not None and elo_b is not None:
    HOME_ADV = 60.0
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a - HOME_ADV) / 400))
    # crude draw allowance derived from season-wide draw rate, split from the binary Elo estimate
    draw_share = 0.24
    home_win = expected_a * (1 - draw_share)
    away_win = (1 - expected_a) * (1 - draw_share)
    st.plotly_chart(
        charts.win_probability_gauge(home_win, draw_share, away_win, team_a, team_b),
        use_container_width=True)
    st.caption(
        f"Estimated from current Elo ratings assuming {team_a} hosts, plus a home-advantage "
        "term and the season's overall draw rate. This is a simple Elo estimate, not the "
        "trained ML classifier — see **Match Predictor** for the full model-based prediction."
    )
else:
    st.info("Elo ratings unavailable for one of the selected teams.")
