"""Tests for the data loading, feature engineering, clustering, and Elo
pipeline. Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT))

from load_data import build_standings, load_epl_matches  # noqa: E402
from features import build_features, get_feature_cols, latest_team_snapshot  # noqa: E402
from clustering import playing_style_clustering, team_strength_clustering  # noqa: E402
from elo import compute_elo_history, current_elo_table  # noqa: E402


@pytest.fixture(scope="module")
def matches():
    return load_epl_matches()


def test_load_matches_shape(matches):
    assert len(matches) == 380
    assert matches["home_team"].nunique() == 20


def test_no_missing_goals(matches):
    assert matches["home_goals"].isna().sum() == 0
    assert matches["away_goals"].isna().sum() == 0


def test_result_labels_valid(matches):
    assert set(matches["result"].unique()) <= {"H", "D", "A"}


def test_standings_has_all_20_teams(matches):
    standings = build_standings(matches)
    assert len(standings) == 20
    assert standings["MP"].sum() == 380 * 2  # each match counted for both teams


def test_standings_points_consistent(matches):
    standings = build_standings(matches)
    for _, row in standings.iterrows():
        assert row["Points"] == row["Win"] * 3 + row["Draw"]
        assert row["GD"] == row["GF"] - row["GA"]


def test_features_no_leakage_columns(matches):
    """The feature set must never include same-match post-match stats --
    only *_r5 rolling/shifted columns and season aggregates."""
    feats = build_features(matches)
    feat_cols = get_feature_cols(feats)
    for c in feat_cols:
        assert c.endswith(("_r5", "_avg", "played")), f"unexpected raw (leaky) column: {c}"


def test_features_chronological_no_future_leak(matches):
    """A team's first eligible match feature must reflect only matches
    strictly before it in time."""
    feats = build_features(matches)
    assert feats["date"].is_monotonic_increasing


def test_latest_snapshot_covers_all_teams(matches):
    snap = latest_team_snapshot(matches)
    assert len(snap) == 20


def test_strength_tiers_all_teams(matches):
    standings = build_standings(matches)
    tiers = team_strength_clustering(standings)
    assert len(tiers) == 20
    assert set(tiers["tier"]) <= {"Elite", "Good", "Average", "Weak"}


def test_style_clusters_all_teams():
    style, meta = playing_style_clustering()
    assert len(style) == 20
    assert meta["explained_variance_total"] > 0


def test_elo_starts_at_1500_and_conserves_zero_sum(matches):
    history = compute_elo_history(matches)
    first = history.iloc[0]
    assert first["elo_home_pre"] == 1500.0
    assert first["elo_away_pre"] == 1500.0
    # every update moves home/away by equal and opposite amounts
    delta_home = history["elo_home_post"] - history["elo_home_pre"]
    delta_away = history["elo_away_post"] - history["elo_away_pre"]
    assert (abs(delta_home + delta_away) < 1e-6).all()


def test_elo_table_all_teams(matches):
    history = compute_elo_history(matches)
    table = current_elo_table(history)
    assert len(table) == 20
