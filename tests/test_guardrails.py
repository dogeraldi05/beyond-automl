import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from beyond_automl import business, guardrails  # noqa: E402

G = {"min_prevalence": 0.03, "max_prevalence": 0.97, "min_positives": 5, "leakage_auc": 0.95,
     "max_value_arpu_months": 24, "sensitive_columns": ["gender"]}


@pytest.fixture
def df():
    n = 200
    return pd.DataFrame({
        "customerID": [f"id{i}" for i in range(n)],
        "gender": ["F", "M"] * (n // 2),
        "MonthlyCharges": [50.0 + (i % 7) for i in range(n)],
        "Plan": ["A", "B", "C", "D"] * (n // 4),
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
    })


def problem(**kw):
    p = {"id": "x", "problem": "x", "target": {"column": "Churn", "operator": "in", "values": ["Yes"]},
         "population": None, "leakage_columns": [], "value_per_hit_brl": 100,
         "cost_per_action_brl": 10, "impact_score": 5}
    p.update(kw)
    return p


def test_hallucinated_column_is_rejected(df):
    p = problem(target={"column": "PaymentDefault", "operator": "in", "values": ["Yes"]})
    assert guardrails.validate_problem(p, df, G, ["customerID"])["status"] == "rejected"


def test_implausible_value_is_rejected(df):
    p = problem(value_per_hit_brl=1_000_000)
    assert guardrails.validate_problem(p, df, G, ["customerID"])["status"] == "rejected"


def test_sensitive_target_is_rejected(df):
    p = problem(target={"column": "gender", "operator": "in", "values": ["F"]})
    assert guardrails.validate_problem(p, df, G, ["customerID"])["status"] == "rejected"


def test_leaking_feature_is_dropped(df):
    # Plan == "A" defines Churn exactly → must be dropped from features
    out = guardrails.validate_problem(problem(), df, G, ["customerID"])
    assert out["status"] == "approved"
    assert "Plan" not in out["features"]
    assert "customerID" not in out["features"] and "gender" not in out["features"]


def test_best_threshold_profit():
    import numpy as np
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    profit, thr, n_act, tp = business.best_threshold_profit(y, proba, value=100, cost=10)
    assert (profit, n_act, tp) == (180, 2, 2)
