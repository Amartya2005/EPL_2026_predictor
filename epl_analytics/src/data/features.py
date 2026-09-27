"""
Feature engineering for match outcome / goal prediction.

THE LEAKAGE TRAP IN THIS DATASET: TeamStatsExport's possession/shots/passing
columns are POST-match stats -- a *consequence* of the result, not a
predictor of it (a team that wins 4-0 will show high shot accuracy for that
same match, but nobody knows that before kickoff). Feeding them directly
into a classifier for the same fixture is target leakage and produces a
model that looks like 90%+ accuracy and is worthless for real prediction.

Every feature here is computed from each team's PRIOR matches only (shifted
before the current fixture) -- the only information actually known before
kickoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROLL_WINDOW = 5  # last-5-match form, standard choice for a 38-game season

STAT_COLS = [
    "possessionPct", "totalShots", "shotsOnTarget", "shotPct", "accuratePasses",
    "totalPasses", "passPct", "totalCrosses", "crossPct", "totalLongBalls",
    "longballPct", "effectiveTackles", "totalTackles", "interceptions",
    "totalClearance", "foulsCommitted", "yellowCards",
]


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home and away unstacked), chronological."""
    home = matches[["date", "home_team", "away_team", "home_goals", "away_goals"] +
                    [c + "_home" for c in STAT_COLS]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against"] + STAT_COLS
    home["is_home"] = 1

    away = matches[["date", "away_team", "home_team", "away_goals", "home_goals"] +
                    [c + "_away" for c in STAT_COLS]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against"] + STAT_COLS
    away["is_home"] = 0

    # Stable sorting makes feature generation deterministic when the source
    # workbook contains multiple fixtures on the same calendar date.
    long_df = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "date"],
        kind="mergesort",
    )
    long_df["points"] = np.select(
        [long_df.goals_for > long_df.goals_against, long_df.goals_for == long_df.goals_against],
        [3, 1], default=0)
    return long_df.reset_index(drop=True)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    long_df = _long_format(matches)

    # Shift(1) is the whole point: rolling stats computed up to and NOT
    # including the current match, per team, in chronological order.
    grp = long_df.groupby("team", group_keys=False)
    roll_cols = {}
    for col in ["goals_for", "goals_against", "points"] + STAT_COLS:
        roll_cols[f"{col}_r{ROLL_WINDOW}"] = grp[col].apply(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    for name, series in roll_cols.items():
        long_df[name] = series

    long_df["season_pts_avg"] = grp["points"].apply(lambda s: s.shift(1).expanding().mean())
    long_df["season_gd_avg"] = grp.apply(
        lambda g: (g["goals_for"] - g["goals_against"]).shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)
    long_df["matches_played"] = grp.cumcount()

    feat_cols = [f"{c}_r{ROLL_WINDOW}" for c in ["goals_for", "goals_against", "points"] + STAT_COLS]
    feat_cols += ["season_pts_avg", "season_gd_avg", "matches_played"]

    home_feats = long_df[long_df.is_home == 1][["date", "team", "opponent"] + feat_cols]
    home_feats = home_feats.rename(columns={"team": "home_team", "opponent": "away_team"})
    home_feats.columns = ["date", "home_team", "away_team"] + [f"home_{c}" for c in feat_cols]

    away_feats = long_df[long_df.is_home == 0][["date", "team", "opponent"] + feat_cols]
    away_feats = away_feats.rename(columns={"team": "away_team", "opponent": "home_team"})
    # rename() relabels values, it does NOT reorder columns -- reindex
    # explicitly before renaming or home/away values get silently swapped.
    away_feats = away_feats[["date", "home_team", "away_team"] + feat_cols]
    away_feats.columns = ["date", "home_team", "away_team"] + [f"away_{c}" for c in feat_cols]

    out = matches[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].merge(
        home_feats, on=["date", "home_team", "away_team"], how="left").merge(
        away_feats, on=["date", "home_team", "away_team"], how="left")

    # Drop the first few gameweeks per team where rolling form is thin/
    # undefined (matches_played < 3) -- a team's "form" after 1 match is
    # noise, not signal.
    out = out[(out["home_matches_played"] >= 3) & (out["away_matches_played"] >= 3)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("home_") and c not in ("home_team", "home_goals")] + \
           [c for c in df.columns if c.startswith("away_") and c not in ("away_team", "away_goals")]


def latest_team_snapshot(matches: pd.DataFrame) -> pd.DataFrame:
    """Each team's most recent rolling-form snapshot -- used by the live
    match predictor to featurize a hypothetical upcoming fixture."""
    long_df = _long_format(matches)
    grp = long_df.groupby("team", group_keys=False)
    feat_source = {}
    for col in ["goals_for", "goals_against", "points"] + STAT_COLS:
        feat_source[f"{col}_r{ROLL_WINDOW}"] = grp[col].apply(
            lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    for name, series in feat_source.items():
        long_df[name] = series
    long_df["season_pts_avg"] = grp["points"].apply(lambda s: s.expanding().mean())
    long_df["season_gd_avg"] = grp.apply(
        lambda g: (g["goals_for"] - g["goals_against"]).expanding().mean()
    ).reset_index(level=0, drop=True)
    long_df["matches_played"] = grp.cumcount() + 1

    feat_cols = [f"{c}_r{ROLL_WINDOW}" for c in ["goals_for", "goals_against", "points"] + STAT_COLS]
    feat_cols += ["season_pts_avg", "season_gd_avg", "matches_played"]
    latest = long_df.sort_values("date").groupby("team").tail(1)
    return latest[["team"] + feat_cols].reset_index(drop=True)


if __name__ == "__main__":
    from load_data import load_epl_matches

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = load_epl_matches()
    feats = build_features(matches)
    print(feats.shape)
    print(feats["result"].value_counts(normalize=True))
    feats.to_csv(out_dir / "match_features.csv", index=False)

    snapshot = latest_team_snapshot(matches)
    snapshot.to_csv(out_dir / "team_form_snapshot.csv", index=False)
    print(f"snapshot: {len(snapshot)} teams")
