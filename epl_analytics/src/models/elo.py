"""
Team Elo rating system, carried match-by-match across the season.

Standard football Elo update:
    expected_home = 1 / (1 + 10 ** ((elo_away - elo_home - HOME_ADV) / 400))
    elo_home' = elo_home + K * (actual_home - expected_home)

All teams start at 1500. This is computed strictly in chronological order
so each team's rating before a given match reflects only matches already
played -- same no-lookahead discipline as the rolling-form features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

START_ELO = 1500.0
K_FACTOR = 24.0
HOME_ADVANTAGE = 60.0  # Elo points added to home team's effective rating


def compute_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per match with each team's Elo rating BEFORE that
    match was played, plus the post-match updated ratings."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    ratings = {t: START_ELO for t in teams}

    records = []
    for _, m in matches.sort_values("date").iterrows():
        home, away = m.home_team, m.away_team
        elo_home_pre, elo_away_pre = ratings[home], ratings[away]

        expected_home = 1 / (1 + 10 ** ((elo_away_pre - elo_home_pre - HOME_ADVANTAGE) / 400))
        if m.home_goals > m.away_goals:
            actual_home = 1.0
        elif m.home_goals < m.away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Margin-of-victory multiplier -- a 4-0 win says more than a 1-0 win.
        goal_diff = abs(m.home_goals - m.away_goals)
        mov_mult = 1 if goal_diff <= 1 else (1.5 if goal_diff == 2 else 1.75)

        delta = K_FACTOR * mov_mult * (actual_home - expected_home)
        ratings[home] = elo_home_pre + delta
        ratings[away] = elo_away_pre - delta

        records.append({
            "date": m.date, "home_team": home, "away_team": away,
            "elo_home_pre": elo_home_pre, "elo_away_pre": elo_away_pre,
            "expected_home_win_prob": expected_home,
            "elo_home_post": ratings[home], "elo_away_post": ratings[away],
            "result": m.result,
        })
    return pd.DataFrame(records)


def current_elo_table(elo_history: pd.DataFrame) -> pd.DataFrame:
    """Final Elo rating per team after the last match in the dataset."""
    last_home = elo_history.sort_values("date").groupby("home_team").tail(1)[["home_team", "elo_home_post"]]
    last_home.columns = ["team", "elo_home"]
    last_away = elo_history.sort_values("date").groupby("away_team").tail(1)[["away_team", "elo_away_post"]]
    last_away.columns = ["team", "elo_away"]

    # A team's true current rating is whichever of its last home/away
    # appearance happened most recently -- merge by taking the max date.
    all_last = []
    for team in pd.unique(elo_history[["home_team", "away_team"]].values.ravel()):
        home_rows = elo_history[elo_history.home_team == team]
        away_rows = elo_history[elo_history.away_team == team]
        candidates = []
        if len(home_rows):
            r = home_rows.sort_values("date").iloc[-1]
            candidates.append((r["date"], r["elo_home_post"]))
        if len(away_rows):
            r = away_rows.sort_values("date").iloc[-1]
            candidates.append((r["date"], r["elo_away_post"]))
        candidates.sort(key=lambda x: x[0])
        all_last.append({"team": team, "elo": candidates[-1][1]})
    table = pd.DataFrame(all_last).sort_values("elo", ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["elo"] = table["elo"].round(1)
    return table


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from load_data import load_epl_matches

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    history = compute_elo_history(matches)
    history.to_csv(out_dir / "elo_history.csv", index=False)

    table = current_elo_table(history)
    table.to_csv(out_dir / "elo_current.csv", index=False)
    print(table.to_string(index=False))
