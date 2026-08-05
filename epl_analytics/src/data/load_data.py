"""
Load the raw workbook, isolate the English Premier League 2025-26 season,
and produce a single match-level table (one row per fixture, home + away
stats side by side).

Known data-quality issues in the source workbook, and how each is handled:

1. ``AllFixturesExport['Away Score']`` is 0 for every EPL row -- dead /
   placeholder data. Real scores are sourced from
   ``TeamStatsExport['Home Goal'/'Away Goal']`` instead.
2. The free-text ``Fixture`` field is formatted inconsistently between
   sheets ("X vs Y" in one, "Y at X" in another). The only safe join key
   used throughout this module is ``date + home_team + away_team``.
3. ``LeagueTableExport`` is missing 3 of the 20 teams (only 17 rows).
   Standings are therefore never read from that sheet -- they are rebuilt
   directly from the 380 verified match results in :func:`build_standings`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_XLSX = Path(__file__).resolve().parents[2] / "data" / "raw" / \
    "Soccer-Stats-Premier-League-2025-2026_R02.xlsx"

STAT_COLS = [
    "foulsCommitted", "yellowCards", "redCards", "offsides", "wonCorners",
    "saves", "possessionPct", "totalShots", "shotsOnTarget", "shotPct",
    "penaltyKickGoals", "penaltyKickShots", "accuratePasses", "totalPasses",
    "passPct", "accurateCrosses", "totalCrosses", "crossPct",
    "accurateLongBalls", "totalLongBalls", "longballPct", "blockedShots",
    "effectiveTackles", "totalTackles", "tacklePct", "interceptions",
    "effectiveClearance", "totalClearance",
]


def load_epl_matches(src: Path = RAW_XLSX) -> pd.DataFrame:
    """One row per EPL fixture, home + away team stats side by side."""
    xl = pd.ExcelFile(src)
    fixtures = xl.parse("AllFixturesExport")
    teamstats = xl.parse("TeamStatsExport")

    # Isolate EPL only -- the workbook mixes in Champions League, Club World
    # Cup, Carabao Cup, Community Shield, etc. Every downstream feature
    # (rolling form, standings, clustering) breaks if cup form leaks in.
    epl = fixtures[fixtures["League"] == "English Premier League"].copy()
    epl = epl[epl["Status"] == "STATUS_FULL_TIME"].copy()

    home_rows = teamstats[teamstats["homeAway"] == "home"].copy()
    away_rows = teamstats[teamstats["homeAway"] == "away"].copy()

    join_keys = ["Date Time (US Eastern)", "Home Team", "Away Team"]
    home_stats = home_rows[join_keys + STAT_COLS].add_suffix("_home")
    home_stats = home_stats.rename(columns={
        "Date Time (US Eastern)_home": "date", "Home Team_home": "home_team",
        "Away Team_home": "away_team"})
    away_stats = away_rows[join_keys + STAT_COLS].add_suffix("_away")
    away_stats = away_stats.rename(columns={
        "Date Time (US Eastern)_away": "date", "Home Team_away": "home_team",
        "Away Team_away": "away_team"})

    match_stats = home_stats.merge(away_stats, on=["date", "home_team", "away_team"], how="inner")

    epl = epl.rename(columns={
        "Date Time (US Eastern)": "date", "Home Team": "home_team",
        "Away Team": "away_team"})[["date", "home_team", "away_team"]]

    # Real scores come from TeamStatsExport, not AllFixturesExport (see
    # module docstring, issue 1).
    goals = home_rows[join_keys + ["Home Goal", "Away Goal"]].rename(columns={
        "Date Time (US Eastern)": "date", "Home Team": "home_team",
        "Away Team": "away_team", "Home Goal": "home_goals", "Away Goal": "away_goals"})

    matches = epl.merge(match_stats, on=["date", "home_team", "away_team"], how="left")
    matches = matches.merge(goals, on=["date", "home_team", "away_team"], how="left")
    matches = matches.sort_values("date").reset_index(drop=True)

    n_missing = matches["possessionPct_home"].isna().sum()
    assert len(matches) == 380, f"expected 380 EPL matches, got {len(matches)}"
    assert n_missing == 0, f"{n_missing} matches failed to join to TeamStatsExport"

    def result(row):
        if row.home_goals > row.away_goals:
            return "H"
        if row.home_goals < row.away_goals:
            return "A"
        return "D"

    matches["result"] = matches.apply(result, axis=1)
    matches["date"] = pd.to_datetime(matches["date"])
    return matches


def build_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the full 20-team league table directly from match results.

    Deliberately does not read ``LeagueTableExport`` -- that sheet is
    missing 3 of the 20 teams. This aggregates the verified 380 match
    results instead, so every team that played a match is guaranteed to
    appear.
    """
    rows = []
    for _, m in matches.iterrows():
        rows.append({"team": m.home_team, "opponent": m.away_team, "is_home": 1,
                      "gf": m.home_goals, "ga": m.away_goals, "result": m.result})
        rows.append({"team": m.away_team, "opponent": m.home_team, "is_home": 0,
                      "gf": m.away_goals, "ga": m.home_goals,
                      "result": {"H": "A", "A": "H", "D": "D"}[m.result]})
    long_df = pd.DataFrame(rows)
    long_df["win"] = (long_df["gf"] > long_df["ga"]).astype(int)
    long_df["draw"] = (long_df["gf"] == long_df["ga"]).astype(int)
    long_df["loss"] = (long_df["gf"] < long_df["ga"]).astype(int)
    long_df["points"] = long_df["win"] * 3 + long_df["draw"]
    long_df["clean_sheet"] = (long_df["ga"] == 0).astype(int)

    table = long_df.groupby("team").agg(
        MP=("result", "count"), Win=("win", "sum"), Draw=("draw", "sum"),
        Loss=("loss", "sum"), GF=("gf", "sum"), GA=("ga", "sum"),
        Points=("points", "sum"), Clean_Sheets=("clean_sheet", "sum"),
    ).reset_index().rename(columns={"team": "Team", "Clean_Sheets": "Clean Sheets"})
    table["GD"] = table["GF"] - table["GA"]
    table["pts_per_game"] = table["Points"] / table["MP"]
    table["gd_per_game"] = table["GD"] / table["MP"]
    table["Win %"] = (table["Win"] / table["MP"] * 100).round(1)

    home = long_df[long_df.is_home == 1].groupby("team").agg(
        Home_MP=("result", "count"), Home_Win=("win", "sum"), Home_GF=("gf", "sum"),
        Home_GA=("ga", "sum")).reset_index().rename(columns={"team": "Team"})
    away = long_df[long_df.is_home == 0].groupby("team").agg(
        Away_MP=("result", "count"), Away_Win=("win", "sum"), Away_GF=("gf", "sum"),
        Away_GA=("ga", "sum")).reset_index().rename(columns={"team": "Team"})
    table = table.merge(home, on="Team", how="left").merge(away, on="Team", how="left")

    table = table.sort_values(["Points", "GD", "GF"], ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = load_epl_matches()
    print(f"loaded {len(matches)} matches, {matches.home_team.nunique()} home teams")
    matches.to_csv(out_dir / "matches_raw.csv", index=False)

    standings = build_standings(matches)
    print(f"standings built for {len(standings)} teams (should be 20)")
    assert len(standings) == 20, f"expected 20 teams, got {len(standings)}"
    standings.to_csv(out_dir / "standings.csv", index=False)
