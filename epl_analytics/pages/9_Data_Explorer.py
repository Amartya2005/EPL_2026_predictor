import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.data_loader import (DATA_DIR, artifacts_ready, load_elo_current, load_match_features,
                                    load_matches, load_standings, load_strength_tiers,
                                    load_style_clusters)
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Data Explorer | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

st.markdown('<div class="eyebrow">Data Explorer</div>', unsafe_allow_html=True)
st.title("Raw & Processed Data")
st.caption("Every dataset behind this dashboard, previewable and downloadable.")

DATASETS = {
    "Match Results (380 fixtures)": load_matches,
    "Standings (rebuilt, 20 teams)": load_standings,
    "Match Features (leakage-safe, model input)": load_match_features,
    "Team Strength Tiers": load_strength_tiers,
    "Playing Style Clusters": load_style_clusters,
    "Elo Current Ratings": load_elo_current,
}

choice = st.selectbox("Dataset", list(DATASETS.keys()))
df = DATASETS[choice]()

c1, c2 = st.columns([2, 1])
search = c1.text_input("Search (matches any cell, case-insensitive)")
sort_col = c2.selectbox("Sort by", df.columns.tolist())

view = df.copy()
if search:
    mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
    view = view[mask]

sort_desc = st.checkbox("Descending", value=False)
view = view.sort_values(sort_col, ascending=not sort_desc)

st.caption(f"{len(view)} of {len(df)} rows")
st.dataframe(view, use_container_width=True, hide_index=True, height=560)

st.download_button(f"Download '{choice}' as CSV", view.to_csv(index=False).encode("utf-8"),
                    f"{choice.split(' (')[0].lower().replace(' ', '_')}.csv", "text/csv")

with st.expander("Column reference"):
    st.write(f"{len(df.columns)} columns:")
    st.code(", ".join(df.columns.tolist()))
