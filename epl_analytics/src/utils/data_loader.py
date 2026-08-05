"""
Cached loaders for every pre-computed artifact the dashboard reads.
All heavy work (loading the workbook, training models) happens offline via
scripts/build_all.py -- the app only ever reads CSV/JSON/pickle from disk,
so pages stay fast and don't retrain on every interaction.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def artifacts_ready() -> bool:
    required = ["standings.csv", "match_features.csv", "team_strength_tiers.csv",
                "playing_style_clusters.csv", "elo_current.csv", "model_metrics.json"]
    return all((DATA_DIR / f).exists() for f in required)


@st.cache_data
def load_standings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "standings.csv")


@st.cache_data
def load_matches() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "matches_raw.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_match_features() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "match_features.csv", parse_dates=["date"])


@st.cache_data
def load_team_form_snapshot() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "team_form_snapshot.csv")


@st.cache_data
def load_strength_tiers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "team_strength_tiers.csv")


@st.cache_data
def load_style_clusters() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "playing_style_clusters.csv")


@st.cache_data
def load_clustering_meta() -> dict:
    with open(DATA_DIR / "clustering_meta.json") as f:
        return json.load(f)


@st.cache_data
def load_elo_current() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "elo_current.csv")


@st.cache_data
def load_elo_history() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "elo_history.csv", parse_dates=["date"])


@st.cache_data
def load_model_metrics() -> dict:
    with open(DATA_DIR / "model_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_curves() -> dict:
    with open(DATA_DIR / "roc_learning_curves.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance() -> dict:
    with open(DATA_DIR / "feature_importance.json") as f:
        return json.load(f)


@st.cache_resource
def load_classifiers():
    with open(MODEL_DIR / "classifiers.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_regressors():
    with open(MODEL_DIR / "regressors.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def team_list() -> list[str]:
    return sorted(load_standings()["Team"].unique().tolist())


@st.cache_data
def league_kpis() -> dict:
    standings = load_standings()
    matches = load_matches()
    total_goals = matches["home_goals"].sum() + matches["away_goals"].sum()
    leader = standings.iloc[0]
    best_attack = standings.loc[standings["GF"].idxmax()]
    best_defense = standings.loc[standings["GA"].idxmin()]
    return {
        "total_teams": standings["Team"].nunique(),
        "total_matches": len(matches),
        "avg_goals_per_match": round(total_goals / len(matches), 2),
        "leader": leader["Team"],
        "leader_points": int(leader["Points"]),
        "best_attack": best_attack["Team"],
        "best_attack_goals": int(best_attack["GF"]),
        "best_defense": best_defense["Team"],
        "best_defense_conceded": int(best_defense["GA"]),
    }
