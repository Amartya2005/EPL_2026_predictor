import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import (artifacts_ready, load_clustering_meta, load_strength_tiers,
                                    load_style_clusters)
from src.utils.styling import TIER_COLOR, inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Clustering | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

tiers = load_strength_tiers()
style = load_style_clusters()
meta = load_clustering_meta()

st.markdown('<div class="eyebrow">Clustering</div>', unsafe_allow_html=True)
st.title("Team Strength Tiers & Playing Style")
st.caption(
    "Both clusterings use finished-season aggregate stats, so there's no temporal-leakage "
    "concern here (unlike the match predictor) — these describe teams as they actually "
    "performed, not a pre-match prediction."
)

tab1, tab2 = st.tabs(["Team Strength Tiers", "Playing Style Clusters"])

with tab1:
    st.subheader("4-Tier KMeans (points/game, goal-diff/game, clean sheets)")
    tier_order = ["Elite", "Good", "Average", "Weak"]
    cols = st.columns(4)
    for i, tname in enumerate(tier_order):
        teams_in_tier = tiers[tiers.tier == tname]["Team"].tolist()
        with cols[i]:
            st.markdown(
                f'<div class="pitch-card"><span class="tier-pill tier-{tname}">{tname}</span>'
                f'<div style="margin-top:8px;font-size:13px;color:var(--chalk-dim)">'
                + "<br>".join(teams_in_tier) + "</div></div>",
                unsafe_allow_html=True)
    st.divider()
    st.dataframe(tiers, use_container_width=True, hide_index=True)
    st.download_button("Download tiers as CSV", tiers.to_csv(index=False).encode(),
                        "team_strength_tiers.csv", "text/csv")

with tab2:
    st.subheader(f"PCA + KMeans on possession/passing/tackling profile "
                 f"({meta['explained_variance_total']:.1%} variance explained by 2 components)")
    st.plotly_chart(charts.pca_scatter(style), use_container_width=True)

    st.markdown("**Cluster profiles**")
    cluster_summ = style.groupby("style_name").agg(
        teams=("Team", lambda s: ", ".join(s)),
        avg_possession=("possessionPct", "mean"),
        avg_pass_pct=("passPct", "mean"),
        avg_long_balls=("totalLongBalls", "mean"),
        avg_tackles=("totalTackles", "mean"),
    ).reset_index().round(1)
    for _, r in cluster_summ.iterrows():
        with st.expander(f"{r['style_name']} — {r['teams']}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Possession", f"{r['avg_possession']}%")
            c2.metric("Avg Pass Accuracy", f"{r['avg_pass_pct']}%")
            c3.metric("Avg Long Balls", r["avg_long_balls"])
            c4.metric("Avg Tackles", r["avg_tackles"])

    st.divider()
    st.dataframe(style, use_container_width=True, hide_index=True)
    st.download_button("Download style clusters as CSV", style.to_csv(index=False).encode(),
                        "playing_style_clusters.csv", "text/csv")
