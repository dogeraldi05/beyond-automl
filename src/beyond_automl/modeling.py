"""Builds N problems × 4 models and returns technical metrics + probabilities."""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .guardrails import apply_condition

warnings.filterwarnings("ignore", category=UserWarning)

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                   class_weight="balanced", n_jobs=-1, random_state=42),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                     random_state=42, n_jobs=-1),
    "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                       random_state=42, verbose=-1),
}


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def prepare(problem: dict, df: pd.DataFrame, split_cfg: dict):
    sub = df[apply_condition(df, problem["population"])] if problem.get("population") else df
    y = apply_condition(sub, problem["target"]).astype(int)
    X = sub[problem["features"]].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    return train_test_split(X, y, stratify=y, **split_cfg)


def leakage_diagnosis(problem: dict, df: pd.DataFrame, split_cfg: dict, top: int = 3) -> list[tuple]:
    """Drop-one: how much AUC falls when each feature is removed (logistic regression).

    Used when AUC is too good: points to the columns that, combined,
    practically define the target.
    """
    X_tr, X_te, y_tr, y_te = prepare(problem, df, split_cfg)

    def auc_without(cols):
        pipe = Pipeline([("prep", _preprocessor(X_tr[cols])), ("model", MODELS["LogisticRegression"]())])
        pipe.fit(X_tr[cols], y_tr)
        return roc_auc_score(y_te, pipe.predict_proba(X_te[cols])[:, 1])

    base = auc_without(list(X_tr.columns))
    drops = [(c, base - auc_without([x for x in X_tr.columns if x != c])) for c in X_tr.columns]
    return sorted(drops, key=lambda t: -t[1])[:top]


def run_problem(problem: dict, df: pd.DataFrame, split_cfg: dict) -> list[dict]:
    X_tr, X_te, y_tr, y_te = prepare(problem, df, split_cfg)
    out = []
    for name, make in MODELS.items():
        pipe = Pipeline([("prep", _preprocessor(X_tr)), ("model", make())])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "model": name,
            "AUC": roc_auc_score(y_te, proba),
            "F1": f1_score(y_te, pred, zero_division=0),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "_proba": proba,
            "_y": y_te.to_numpy(),
            "_index": X_te.index.to_numpy(),
        })
    return out
