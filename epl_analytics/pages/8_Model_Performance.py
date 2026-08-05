import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import charts
from src.utils.data_loader import artifacts_ready, load_curves, load_model_metrics
from src.utils.styling import inject_css, theme_toggle_sidebar

st.set_page_config(page_title="Model Performance | EPL Analytics", layout="wide")
theme_toggle_sidebar()
inject_css()

if not artifacts_ready():
    st.error("Run `python scripts/build_all.py` first.")
    st.stop()

metrics = load_model_metrics()
curves = load_curves()
clf = metrics["classification"]
reg = metrics["regression"]

st.markdown('<div class="eyebrow">Model Performance</div>', unsafe_allow_html=True)
st.title("Match Outcome Classification — Models Trained")
st.caption(f"Chronological split — train: {metrics['n_train']} matches "
           f"({metrics['train_range'][0]} to {metrics['train_range'][1]}), "
           f"test: {metrics['n_test']} matches ({metrics['test_range'][0]} to {metrics['test_range'][1]}). "
           "Not a random K-fold split — a random split would let the model train on gameweek 30 "
           "and validate on gameweek 10, which never happens in reality.")

st.markdown(
    f'<div class="honest-box"><b>Honest headline:</b> the majority-class baseline '
    f'("always predict {clf["baseline_majority"]["majority_class"]}") scores '
    f'{clf["baseline_majority"]["accuracy"]:.1%} accuracy. None of the three trained models beat '
    "that meaningfully on raw accuracy — this is the expected result for a single 38-game season "
    "with only 5-match rolling form as signal, not a bug in the pipeline.</div>",
    unsafe_allow_html=True)
st.write("")

model_names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
               "xgboost": "XGBoost"}

summary_rows = []
for key, label in model_names.items():
    m = clf[key]
    summary_rows.append({
        "Model": label, "Accuracy": m["accuracy"], "Log Loss": m["log_loss"],
        "Precision (macro)": m["precision_macro"], "Recall (macro)": m["recall_macro"],
        "F1 (macro)": m["f1_macro"], "CV Accuracy": m["cv_accuracy_mean"],
    })
summary_df = pd.DataFrame(summary_rows).round(3)
best = metrics["best_model"]
st.subheader(f"Best-performing model (by test accuracy): **{model_names[best]}**")
st.dataframe(summary_df.style.highlight_max(subset=["Accuracy", "F1 (macro)"], color="#4A3E22"),
             use_container_width=True, hide_index=True)

st.divider()
model_choice = st.selectbox("Inspect model", list(model_names.keys()), format_func=lambda k: model_names[k])
m = clf[model_choice]
c = curves[model_choice]

t1, t2, t3, t4 = st.tabs(["Confusion Matrix", "ROC Curve", "Learning Curve", "Cross-Validation"])

with t1:
    st.plotly_chart(charts.confusion_matrix_heatmap(m["confusion_matrix"], ["H", "D", "A"],
                                                      title=f"{model_names[model_choice]} — Confusion Matrix"),
                     use_container_width=True)
    st.caption("Rows = true outcome, columns = predicted outcome.")

with t2:
    st.plotly_chart(charts.roc_curves(c["roc"], title=f"{model_names[model_choice]} — ROC (one-vs-rest)"),
                     use_container_width=True)

with t3:
    if c.get("learning_curve"):
        st.plotly_chart(charts.learning_curve_plot(c["learning_curve"],
                                                     title=f"{model_names[model_choice]} — Learning Curve"),
                         use_container_width=True)
    else:
        st.info("Learning curve unavailable for this model/split size.")

with t4:
    cv_scores = m["cv_scores"]
    st.metric("Time-Series CV Accuracy (mean ± std)", f"{m['cv_accuracy_mean']:.1%} ± {m['cv_accuracy_std']:.1%}")
    import plotly.graph_objects as go
    from src.utils.styling import plotly_layout_kwargs, team_color
    fig = go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores,
                            marker_color=team_color(0)))
    fig.update_layout(title="Accuracy per Time-Series CV Fold", yaxis_title="Accuracy",
                       **plotly_layout_kwargs())
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forward-chaining folds (TimeSeriesSplit) — each fold trains only on matches "
               "before its validation window, same no-lookahead discipline as the main split.")

st.divider()
st.subheader("Goal Prediction (Regression)")
reg_rows = [{"Target": "Home Goals", "MAE": reg["home_goals"]["mae"],
             "Baseline MAE (predict mean)": reg["home_goals"]["baseline_mae"]},
            {"Target": "Away Goals", "MAE": reg["away_goals"]["mae"],
             "Baseline MAE (predict mean)": reg["away_goals"]["baseline_mae"]}]
st.dataframe(pd.DataFrame(reg_rows).round(3), use_container_width=True, hide_index=True)
st.caption("Essentially no lift over predicting the league-average goal count for either target — "
           "goals are high-variance and 5-match form alone is weak signal for them.")
