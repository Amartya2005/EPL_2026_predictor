"""
Team strength tiers and playing-style clustering.

Unlike match outcome prediction, these use SEASON-AGGREGATE stats -- no
temporal leakage concern, since we're clustering finished-season profiles,
not predicting a future match. Much cleaner problem than the predictor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW_XLSX = Path(__file__).resolve().parents[2] / "data" / "raw" / \
    "Soccer-Stats-Premier-League-2025-2026_R02.xlsx"


def team_strength_clustering(standings: pd.DataFrame) -> pd.DataFrame:
    """4-tier KMeans on points/goal-difference/clean-sheets, built from the
    corrected 20-team standings (not the buggy 17-row LeagueTableExport)."""
    features = standings[["Team", "pts_per_game", "gd_per_game", "GF", "GA", "Clean Sheets"]].copy()

    X = features[["pts_per_game", "gd_per_game", "GF", "GA", "Clean Sheets"]]
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features["cluster"] = km.fit_predict(Xs)

    order = features.groupby("cluster")["pts_per_game"].mean().sort_values(ascending=False).index
    tier_names = ["Elite", "Good", "Average", "Weak"]
    tier_map = {cluster_id: tier_names[i] for i, cluster_id in enumerate(order)}
    features["tier"] = features["cluster"].map(tier_map)

    return features[["Team", "pts_per_game", "gd_per_game", "GF", "GA", "Clean Sheets", "tier"]] \
        .sort_values("pts_per_game", ascending=False).reset_index(drop=True)


def playing_style_clustering(src: Path = RAW_XLSX, n_clusters: int = 5) -> tuple[pd.DataFrame, dict]:
    ts = pd.ExcelFile(src).parse("TeamStatsExport")
    ts = ts[ts["Team"].notna()]
    agg = ts.groupby("Team").agg(
        possessionPct=("possessionPct", "mean"),
        passPct=("passPct", "mean"),
        totalLongBalls=("totalLongBalls", "mean"),
        totalCrosses=("totalCrosses", "mean"),
        totalTackles=("totalTackles", "mean"),
        effectiveTackles=("effectiveTackles", "mean"),
        totalShots=("totalShots", "mean"),
        interceptions=("interceptions", "mean"),
    ).reset_index()
    agg["tackle_success_pct"] = agg["effectiveTackles"] / agg["totalTackles"] * 100

    feat_cols = ["possessionPct", "passPct", "totalLongBalls", "totalCrosses",
                 "totalTackles", "totalShots", "interceptions"]
    X = agg[feat_cols]
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    agg["pc1"], agg["pc2"] = pcs[:, 0], pcs[:, 1]
    explained = pca.explained_variance_ratio_

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg["style_cluster"] = km.fit_predict(Xs)

    # Name clusters from their centroid characteristics rather than guessing.
    profile = agg.groupby("style_cluster")[["possessionPct", "totalLongBalls", "totalTackles"]].mean()
    league_avg_poss = agg["possessionPct"].mean()
    league_avg_tackles = agg["totalTackles"].mean()
    names = {}
    for cid, row in profile.iterrows():
        if row["possessionPct"] > league_avg_poss + 3:
            names[cid] = "Possession-Based"
        elif row["possessionPct"] < league_avg_poss - 3 and row["totalLongBalls"] > profile["totalLongBalls"].mean():
            names[cid] = "Direct / Counter-Attacking"
        elif row["totalTackles"] > league_avg_tackles + 1:
            names[cid] = "High-Press / Aggressive"
        else:
            names[cid] = "Balanced / Mixed"
    # Disambiguate duplicate names
    seen: dict[str, int] = {}
    final_names = {}
    for cid, name in names.items():
        seen[name] = seen.get(name, 0) + 1
        final_names[cid] = name if seen[name] == 1 else f"{name} ({seen[name]})"
    agg["style_name"] = agg["style_cluster"].map(final_names)

    meta = {
        "explained_variance": explained.tolist(),
        "explained_variance_total": float(explained.sum()),
        "cluster_names": final_names,
    }
    return agg[["Team", "possessionPct", "passPct", "totalLongBalls", "totalCrosses",
                "totalTackles", "totalShots", "interceptions", "tackle_success_pct",
                "pc1", "pc2", "style_cluster", "style_name"]].sort_values("style_cluster").reset_index(drop=True), meta


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from load_data import load_epl_matches, build_standings

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    standings = build_standings(matches)

    strength = team_strength_clustering(standings)
    assert len(strength) == 20
    strength.to_csv(out_dir / "team_strength_tiers.csv", index=False)
    print("TEAM STRENGTH TIERS")
    print(strength.to_string(index=False))

    style, meta = playing_style_clustering()
    assert len(style) == 20
    style.to_csv(out_dir / "playing_style_clusters.csv", index=False)
    with open(out_dir / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nPCA explained variance: {[round(v, 3) for v in meta['explained_variance']]} "
          f"(total {meta['explained_variance_total']:.1%})")
    print("\nPLAYING STYLE CLUSTERS")
    print(style.to_string(index=False))
