"""Reusable Plotly chart builders. Every chart returns a go.Figure styled
with the app's theme tokens (src/utils/styling.py) and is rendered by the
caller via st.plotly_chart(fig, use_container_width=True)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.styling import plotly_layout_kwargs, team_color

RADAR_METRICS = {
    "Win %": ("Win %", 0, 100),
    "Goals For": ("GF", 0, None),
    "Goals Against (inv)": ("GA_inv", 0, None),
    "Clean Sheets": ("Clean Sheets", 0, None),
    "Points/Game": ("pts_per_game", 0, 3),
    "Goal Diff/Game": ("gd_per_game", -2, 2),
}


def radar_chart(standings: pd.DataFrame, teams: list[str], title: str = "") -> go.Figure:
    """Radar comparing 1+ teams across normalized 0-100 percentile scores
    so metrics on very different scales (goals vs percentages) are
    visually comparable."""
    df = standings.copy()
    df["GA_inv"] = df["GA"].max() - df["GA"]
    metrics = ["Win %", "GF", "GA_inv", "Clean Sheets", "pts_per_game"]
    labels = ["Win %", "Goals For", "Defense", "Clean Sheets", "Pts/Game"]

    norm = df.copy()
    for m in metrics:
        lo, hi = df[m].min(), df[m].max()
        norm[m] = 50 if hi == lo else (df[m] - lo) / (hi - lo) * 100

    fig = go.Figure()
    for i, team in enumerate(teams):
        row = norm[norm["Team"] == team]
        if row.empty:
            continue
        values = row[metrics].iloc[0].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]],
            fill="toself", name=team, line_color=team_color(i), opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True, title=title, **plotly_layout_kwargs(),
    )
    return fig


def bar_comparison(labels: list[str], series: dict[str, list[float]], title: str = "",
                    orientation: str = "v") -> go.Figure:
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        if orientation == "v":
            fig.add_trace(go.Bar(x=labels, y=values, name=name, marker_color=team_color(i)))
        else:
            fig.add_trace(go.Bar(y=labels, x=values, name=name, marker_color=team_color(i), orientation="h"))
    fig.update_layout(title=title, barmode="group", **plotly_layout_kwargs())
    return fig


def rolling_form_line(matches_long: pd.DataFrame, team: str, metric_col: str, title: str) -> go.Figure:
    sub = matches_long[matches_long["team"] == team].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub[metric_col], mode="lines+markers",
                              line=dict(color=team_color(0)), name=team))
    fig.update_layout(title=title, **plotly_layout_kwargs())
    return fig


def pca_scatter(style_df: pd.DataFrame, highlight: list[str] | None = None) -> go.Figure:
    highlight = highlight or []
    fig = go.Figure()
    clusters = sorted(style_df["style_name"].unique())
    for i, cname in enumerate(clusters):
        sub = style_df[style_df["style_name"] == cname]
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers+text", text=sub["Team"],
            textposition="top center", name=cname, marker=dict(size=12, color=team_color(i)),
            textfont=dict(size=10),
        ))
    fig.update_layout(
        title="Playing Style Map (PCA of possession / passing / tackling profile)",
        xaxis_title="PC1 (possession-directness axis)", yaxis_title="PC2",
        **plotly_layout_kwargs(),
    )
    return fig


def confusion_matrix_heatmap(cm: list[list[int]], labels: list[str], title: str = "") -> go.Figure:
    z = cm
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
        colorscale=[[0, "#1C2A21"], [1, "#C7A24C"]], text=z, texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(title=title, **plotly_layout_kwargs())
    return fig


def roc_curves(roc_data: dict, title: str = "") -> go.Figure:
    fig = go.Figure()
    for i, (cls, d) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                                  name=f"{cls} (AUC={d['auc']:.2f})", line=dict(color=team_color(i))))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                       **plotly_layout_kwargs())
    return fig


def learning_curve_plot(lc: dict, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["train_mean"], mode="lines+markers",
                              name="Train accuracy", line=dict(color=team_color(0))))
    fig.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["val_mean"], mode="lines+markers",
                              name="Validation accuracy (time-series CV)", line=dict(color=team_color(1))))
    fig.update_layout(title=title, xaxis_title="Training examples", yaxis_title="Accuracy",
                       **plotly_layout_kwargs())
    return fig


def feature_importance_bar(importance: dict, top_n: int = 20, title: str = "") -> go.Figure:
    s = pd.Series(importance).sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=team_color(0)))
    fig.update_layout(title=title, xaxis_title="Importance", height=max(400, top_n * 24),
                       **plotly_layout_kwargs())
    return fig


def elo_timeline(elo_history: pd.DataFrame, teams: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, team in enumerate(teams):
        home = elo_history[elo_history.home_team == team][["date", "elo_home_post"]].rename(
            columns={"elo_home_post": "elo"})
        away = elo_history[elo_history.away_team == team][["date", "elo_away_post"]].rename(
            columns={"elo_away_post": "elo"})
        combined = pd.concat([home, away]).sort_values("date")
        fig.add_trace(go.Scatter(x=combined["date"], y=combined["elo"], mode="lines",
                                  name=team, line=dict(color=team_color(i))))
    fig.update_layout(title="Elo Rating Over the Season", xaxis_title="Date", yaxis_title="Elo",
                       **plotly_layout_kwargs())
    return fig


def win_probability_gauge(home_prob: float, draw_prob: float, away_prob: float,
                           home_team: str, away_team: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[home_prob, draw_prob, away_prob], y=["Outcome"], orientation="h",
        marker=dict(color=["#C7A24C", "#8FA290", "#B65C3D"]),
        text=[f"{home_team} {home_prob:.0%}", f"Draw {draw_prob:.0%}", f"{away_team} {away_prob:.0%}"],
        textposition="inside",
    ))
    fig.update_layout(barmode="stack", showlegend=False, height=140,
                       xaxis=dict(range=[0, 1], showticklabels=False),
                       yaxis=dict(showticklabels=False), **plotly_layout_kwargs())
    return fig
