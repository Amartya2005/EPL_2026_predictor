"""
Match outcome (H/D/A) and goal prediction, trained on leakage-free
pre-match features from features.py.

Split strategy: chronological, not random. This is a season of matches --
random K-fold would let the model train on gameweek 30 and validate on
gameweek 10, which never happens in reality and inflates validation scores.
Train = first ~75% of matches by date, test = final ~25% (most recent
gameweeks), matching how the model would actually be used in production.

All artifacts (trained models, metrics, curves, SHAP values) are saved to
disk so the Streamlit app loads pre-computed results instead of retraining
on every page view.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              log_loss, precision_score, recall_score,
                              roc_curve, auc)
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = ["H", "D", "A"]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25):
    n_test = int(len(df) * test_frac)
    train = df.iloc[: len(df) - n_test].copy()
    test = df.iloc[len(df) - n_test:].copy()
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("home_") and c not in ("home_team", "home_goals")] + \
           [c for c in df.columns if c.startswith("away_") and c not in ("away_team", "away_goals")]


def run_classification(train, test, feat_cols):
    Xtr, Xte = train[feat_cols], test[feat_cols]
    ytr, yte = train["result"].map(LABEL_MAP), test["result"].map(LABEL_MAP)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    majority = ytr.value_counts().idxmax()
    baseline_acc = float((yte == majority).mean())

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                  min_samples_leaf=8, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, eval_metric="mlogloss"),
    }

    metrics = {"baseline_majority": {"accuracy": baseline_acc,
                                      "majority_class": LABEL_NAMES[majority]}}
    curves = {}
    trained = {}

    yte_bin = label_binarize(yte, classes=[0, 1, 2])

    for name, model in models.items():
        Xtr_use, Xte_use = (Xtr_s, Xte_s) if name == "logistic_regression" else (Xtr, Xte)
        model.fit(Xtr_use, ytr)
        trained[name] = model
        pred = model.predict(Xte_use)
        proba = model.predict_proba(Xte_use)

        acc = accuracy_score(yte, pred)
        ll = log_loss(yte, proba, labels=[0, 1, 2])
        f1_macro = f1_score(yte, pred, average="macro", zero_division=0)
        precision_macro = precision_score(yte, pred, average="macro", zero_division=0)
        recall_macro = recall_score(yte, pred, average="macro", zero_division=0)
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])

        metrics[name] = {
            "accuracy": float(acc), "log_loss": float(ll),
            "f1_macro": float(f1_macro), "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "confusion_matrix": cm.tolist(),
        }

        # One-vs-rest ROC per class
        roc_data = {}
        for i, cls in enumerate(LABEL_NAMES):
            fpr, tpr, _ = roc_curve(yte_bin[:, i], proba[:, i])
            roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        curves[name] = {"roc": roc_data}

        # Learning curve (accuracy vs training-set size, using a
        # forward-chaining TimeSeriesSplit so no future data leaks into
        # any fold's training set)
        tscv = TimeSeriesSplit(n_splits=5)
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, Xtr_use, ytr, cv=tscv, scoring="accuracy",
                train_sizes=np.linspace(0.3, 1.0, 6), random_state=RANDOM_STATE)
            curves[name]["learning_curve"] = {
                "train_sizes": train_sizes.tolist(),
                "train_mean": train_scores.mean(axis=1).tolist(),
                "val_mean": val_scores.mean(axis=1).tolist(),
            }
        except ValueError:
            curves[name]["learning_curve"] = None

        # Time-series cross-validation accuracy (on the training portion)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(Xtr_use):
            m = type(model)(**model.get_params())
            m.fit(Xtr_use[tr_idx] if not hasattr(Xtr_use, "iloc") else Xtr_use.iloc[tr_idx],
                  ytr.iloc[tr_idx])
            Xv = Xtr_use[val_idx] if not hasattr(Xtr_use, "iloc") else Xtr_use.iloc[val_idx]
            cv_scores.append(accuracy_score(ytr.iloc[val_idx], m.predict(Xv)))
        metrics[name]["cv_accuracy_mean"] = float(np.mean(cv_scores))
        metrics[name]["cv_accuracy_std"] = float(np.std(cv_scores))
        metrics[name]["cv_scores"] = [float(s) for s in cv_scores]

    return metrics, curves, trained, scaler


def run_goal_regression(train, test, feat_cols):
    Xtr, Xte = train[feat_cols], test[feat_cols]
    results = {}
    models = {}
    for target in ["home_goals", "away_goals"]:
        ytr, yte = train[target], test[target]
        baseline_mae = float(np.mean(np.abs(yte - ytr.mean())))

        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = float(np.mean(np.abs(yte - pred)))
        results[target] = {"mae": mae, "baseline_mae": baseline_mae}
        models[target] = model
    return results, models


def compute_shap(model, X: pd.DataFrame, sample_size: int = 150):
    """SHAP values for the XGBoost classifier (tree explainer -- fast,
    exact for tree ensembles)."""
    X_sample = X.sample(min(sample_size, len(X)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Multiclass XGBoost via the modern shap API returns a single array
    # shaped (n_samples, n_features, n_classes); older/other cases may
    # return a list of per-class (n_samples, n_features) arrays instead.
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # find the axis that matches n_features, average over the other two
            n_features = X.shape[1]
            feat_axis = [i for i, s in enumerate(arr.shape) if s == n_features][0]
            other_axes = tuple(i for i in range(3) if i != feat_axis)
            mean_abs = np.abs(arr).mean(axis=other_axes)
        else:
            mean_abs = np.abs(arr).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src" / "data"))
    from load_data import load_epl_matches
    from features import build_features

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_epl_matches()
    feats = build_features(matches)
    feats = feats.sort_values("date").reset_index(drop=True)
    train, test = temporal_split(feats)
    feat_cols = get_feature_cols(feats)

    print(f"train: {len(train)} matches ({train.date.min().date()} to {train.date.max().date()})")
    print(f"test:  {len(test)} matches ({test.date.min().date()} to {test.date.max().date()})")
    print(f"{len(feat_cols)} features used")

    clf_metrics, clf_curves, clf_models, scaler = run_classification(train, test, feat_cols)
    reg_metrics, reg_models = run_goal_regression(train, test, feat_cols)

    # Feature importance from XGBoost + Random Forest
    importances = {}
    for name in ["xgboost", "random_forest"]:
        imp = pd.Series(clf_models[name].feature_importances_, index=feat_cols).sort_values(ascending=False)
        importances[name] = imp.to_dict()

    # SHAP values (XGBoost classifier)
    shap_importance = compute_shap(clf_models["xgboost"], test[feat_cols])

    best_model_name = max(
        [k for k in clf_metrics if k != "baseline_majority"],
        key=lambda k: clf_metrics[k]["accuracy"])

    # ---- persist everything ----
    with open(MODEL_DIR / "classifiers.pkl", "wb") as f:
        pickle.dump({"models": clf_models, "scaler": scaler, "feat_cols": feat_cols}, f)
    with open(MODEL_DIR / "regressors.pkl", "wb") as f:
        pickle.dump({"models": reg_models, "feat_cols": feat_cols}, f)

    metrics_out = {
        "classification": clf_metrics,
        "regression": reg_metrics,
        "best_model": best_model_name,
        "feature_cols": feat_cols,
        "train_range": [str(train.date.min().date()), str(train.date.max().date())],
        "test_range": [str(test.date.min().date()), str(test.date.max().date())],
        "n_train": len(train), "n_test": len(test),
    }
    with open(DATA_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(DATA_DIR / "roc_learning_curves.json", "w") as f:
        json.dump(clf_curves, f, indent=2)
    with open(DATA_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "xgboost": importances["xgboost"],
            "random_forest": importances["random_forest"],
            "shap_xgboost": shap_importance.to_dict(),
        }, f, indent=2)

    test.to_csv(DATA_DIR / "test_predictions_input.csv", index=False)

    print(f"\nbest model: {best_model_name} (accuracy {clf_metrics[best_model_name]['accuracy']:.3f})")
    print(f"baseline (majority='{metrics_out['classification']['baseline_majority']['majority_class']}'): "
          f"{metrics_out['classification']['baseline_majority']['accuracy']:.3f}")
    print("\nTop 10 SHAP features:")
    print(shap_importance.head(10))
    print("\nSaved: models/classifiers.pkl, models/regressors.pkl, "
          "data/processed/model_metrics.json, roc_learning_curves.json, feature_importance.json")


if __name__ == "__main__":
    main()
