import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import artifacts_ready, load_feature_importance
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Feature Importance | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

importance = load_feature_importance()

st.markdown('<div class="eyebrow">Feature Importance</div>', unsafe_allow_html=True)
st.title("What the Model Is Actually Using")
st.caption(
    "All features are pre-match rolling averages of each team's prior matches only — none of "
    "them are post-match outcome stats for the fixture being predicted (that would be target "
    "leakage; see the Match Predictor and About pages for why that matters here)."
)

model_choice = st.selectbox("Model", ["xgboost", "random_forest", "shap_xgboost"],
                             format_func=lambda x: {"xgboost": "XGBoost (gain-based)",
                                                     "random_forest": "Random Forest (impurity-based)",
                                                     "shap_xgboost": "XGBoost (mean |SHAP|, test set)"}[x])
top_n = st.slider("Show top N features", 5, 30, 20)

imp_dict = importance[model_choice]
st.plotly_chart(charts.feature_importance_bar(imp_dict, top_n=top_n, title=f"Top {top_n} Features"),
                 use_container_width=True)

import pandas as pd
imp_df = pd.Series(imp_dict).sort_values(ascending=False).head(top_n).reset_index()
imp_df.columns = ["Feature", "Importance"]
imp_df.index += 1
st.dataframe(imp_df, use_container_width=True)
