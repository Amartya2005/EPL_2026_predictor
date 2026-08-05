import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import (artifacts_ready, load_classifiers, load_standings,
                                    load_team_form_snapshot)
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Match Predictor | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

standings = load_standings()
snapshot = load_team_form_snapshot()
bundle = load_classifiers()
models, scaler, feat_cols = bundle["models"], bundle["scaler"], bundle["feat_cols"]
LABEL_NAMES = ["H", "D", "A"]

st.markdown('<div class="eyebrow">Match Predictor</div>', unsafe_allow_html=True)
st.title("Match Outcome Predictor")
st.markdown(
    '<div class="honest-box"><b>Honest framing:</b> these models are trained on leakage-free, '
    "pre-match features only (each team's prior-match rolling form), with a chronological "
    "train/test split. On this single season of data, none of them meaningfully beat the "
    "\u201calways predict the home team\u201d baseline \u2014 treat the probabilities below as a "
    "transparent methodology demo, not a betting edge. Full numbers on the Model Performance "
    "page.</div>", unsafe_allow_html=True)
st.write("")

teams = sorted(standings["Team"])
c1, c2, c3 = st.columns([2, 2, 1])
home_team = c1.selectbox("Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away_team = c2.selectbox("Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)
model_choice = c3.selectbox("Model", ["xgboost", "random_forest", "logistic_regression"])

if home_team == away_team:
    st.warning("Pick two different teams.")
    st.stop()

home_snap = snapshot[snapshot.team == home_team]
away_snap = snapshot[snapshot.team == away_team]
if home_snap.empty or away_snap.empty:
    st.error("No form snapshot available for one of these teams.")
    st.stop()

# Build the single-row feature vector the way features.py does: home_<col>,
# away_<col> using each team's most recent rolling-form snapshot.
home_row = home_snap.iloc[0]
away_row = away_snap.iloc[0]
feat_row = {}
for c in feat_cols:
    if c.startswith("home_"):
        src_col = c[len("home_"):]
        feat_row[c] = home_row.get(src_col, np.nan)
    else:
        src_col = c[len("away_"):]
        feat_row[c] = away_row.get(src_col, np.nan)
X_pred = pd.DataFrame([feat_row])[feat_cols]

model = models[model_choice]
if model_choice == "logistic_regression":
    X_use = scaler.transform(X_pred)
else:
    X_use = X_pred

proba = model.predict_proba(X_use)[0]
pred_idx = int(np.argmax(proba))
pred_label = LABEL_NAMES[pred_idx]
confidence = float(proba[pred_idx])

st.divider()
result_map = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
st.subheader(f"Prediction: {result_map[pred_label]}")
st.plotly_chart(
    charts.win_probability_gauge(float(proba[0]), float(proba[1]), float(proba[2]), home_team, away_team),
    use_container_width=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Home Win Probability", f"{proba[0]:.1%}")
k2.metric("Draw Probability", f"{proba[1]:.1%}")
k3.metric("Away Win Probability", f"{proba[2]:.1%}")
k4.metric("Model Confidence", f"{confidence:.1%}")

st.divider()
st.subheader("Prediction Explanation")

exp_tab1, exp_tab2 = st.tabs(["Top Contributing Features (this prediction)", "Global Feature Importance"])

with exp_tab1:
    if model_choice == "xgboost":
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_pred)
        arr = np.asarray(sv)
        if arr.ndim == 3:
            # (1, n_features, n_classes) -> take the predicted class's slice
            class_axis = [i for i, s in enumerate(arr.shape) if s == len(LABEL_NAMES)][0]
            contrib = np.take(arr, pred_idx, axis=class_axis).reshape(-1)
        else:
            contrib = arr.reshape(-1)
        contrib_s = pd.Series(contrib, index=feat_cols).sort_values(key=np.abs, ascending=False).head(12)
        st.caption(f"SHAP contribution toward the predicted outcome ({result_map[pred_label]}) "
                   "for this specific matchup.")
        import plotly.graph_objects as go
        from src.utils.styling import plotly_layout_kwargs
        colors = ["#C7A24C" if v > 0 else "#B65C3D" for v in contrib_s.values]
        fig = go.Figure(go.Bar(x=contrib_s.values, y=contrib_s.index, orientation="h", marker_color=colors))
        fig.update_layout(title="SHAP Feature Contributions", **plotly_layout_kwargs())
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Per-prediction SHAP explanation is shown for the XGBoost model. "
                "Switch the model selector above to 'xgboost' to see it.")

with exp_tab2:
    st.caption(f"Overall feature importance for {model_choice}, learned across the whole training set "
               "(not specific to this matchup).")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(20)
        st.plotly_chart(charts.feature_importance_bar(imp.to_dict(), top_n=20,
                                                        title="Top 20 Features"), use_container_width=True)
    else:
        coef = pd.Series(np.abs(model.coef_).mean(axis=0), index=feat_cols).sort_values(ascending=False).head(20)
        st.plotly_chart(charts.feature_importance_bar(coef.to_dict(), top_n=20,
                                                        title="Top 20 |Coefficients|"), use_container_width=True)

st.divider()
with st.expander("Raw feature vector used for this prediction"):
    st.dataframe(X_pred.T.rename(columns={0: "value"}), use_container_width=True)
