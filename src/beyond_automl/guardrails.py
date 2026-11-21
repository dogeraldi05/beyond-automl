"""Guardrails: the LLM proposes, the code validates.

Each check is tied to a known shortcoming of generative AI:
hallucinates, unreliable, still learning, privacy, bias,
complexity and resource intensive.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED = ["id", "problem", "target", "value_per_hit_brl", "cost_per_action_brl", "impact_score"]
OPERATORS = {"in", "not_in", "gt", "lt"}


def _check(risk, name, status, detail):
    return {"risk": risk, "check": name, "status": status, "detail": detail}


def apply_condition(df: pd.DataFrame, cond: dict) -> pd.Series:
    """Evaluates {"column","operator","values"} without eval()."""
    s = df[cond["column"]]
    vals = cond["values"]
    op = cond["operator"]
    if op == "in":
        return s.isin(vals)
    if op == "not_in":
        return ~s.isin(vals)
    if op == "gt":
        return s > vals[0]
    if op == "lt":
        return s < vals[0]
    raise ValueError(op)


def _condition_errors(df: pd.DataFrame, cond: dict) -> list[str]:
    errs = []
    if not isinstance(cond, dict) or cond.get("operator") not in OPERATORS:
        return ["malformed condition"]
    col = cond.get("column")
    if col not in df.columns:
        return [f"column '{col}' does not exist in the dataset"]
    if cond["operator"] in {"in", "not_in"}:
        existing = set(df[col].unique().tolist())
        missing = [v for v in cond["values"] if v not in existing]
        if missing:
            errs.append(f"values {missing} do not exist in '{col}'")
    return errs


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """AUC of a single variable on its own (target mean per category/bin)."""
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > 20:
        x = pd.qcut(x, 20, duplicates="drop").astype(str)
    score = y.groupby(x.astype(str)).transform("mean")
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)


def validate_problem(p: dict, df: pd.DataFrame, g: dict, id_cols: list[str]) -> dict:
    """Returns the problem enriched with checks, status and final features."""
    checks = []

    # Unreliable: response structure
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        checks.append(_check("Unreliable", "JSON schema", "fail", f"missing fields: {missing}"))
        return {**p, "checks": checks, "status": "rejected"}
    checks.append(_check("Unreliable", "JSON schema", "pass", "all required fields present"))

    # Hallucinates: target and population really exist
    errs = _condition_errors(df, p["target"])
    if p.get("population"):
        errs += _condition_errors(df, p["population"])
    if errs:
        checks.append(_check("Hallucinates", "Columns and values exist", "fail", "; ".join(errs)))
        return {**p, "checks": checks, "status": "rejected"}
    checks.append(_check("Hallucinates", "Columns and values exist", "pass",
                         "target and population verified in the dataset"))

    # Unreliable: plausible financial assumptions
    arpu = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df else None
    v, c = float(p["value_per_hit_brl"]), float(p["cost_per_action_brl"])
    cap = arpu * g["max_value_arpu_months"] if arpu else np.inf
    if v <= 0 or c < 0 or v > cap or c >= v:
        checks.append(_check("Unreliable", "Plausible financial assumptions", "fail",
                             f"value BRL {v:,.0f} / cost BRL {c:,.0f} out of bounds "
                             f"(cap BRL {cap:,.0f} = {g['max_value_arpu_months']} months of ARPU)"))
    else:
        checks.append(_check("Unreliable", "Plausible financial assumptions", "pass",
                             f"value BRL {v:,.0f} ≤ cap BRL {cap:,.0f}; cost BRL {c:,.0f} < value"))

    # Bias: target cannot be a sensitive attribute
    sens = g["sensitive_columns"]
    if p["target"]["column"] in sens:
        checks.append(_check("Bias", "Target is not a sensitive attribute", "fail",
                             f"'{p['target']['column']}' is a protected attribute"))
    else:
        checks.append(_check("Bias", "Sensitive attributes excluded from features", "pass",
                             f"{sens} removed from features; recall gap measured after training"))

    # Population slice and target
    sub = df[apply_condition(df, p["population"])] if p.get("population") else df
    y = apply_condition(sub, p["target"]).astype(int)

    # Complexity: prevalence and minimum volume
    prev, pos = float(y.mean()), int(y.sum())
    if not (g["min_prevalence"] <= prev <= g["max_prevalence"]) or pos < g["min_positives"]:
        checks.append(_check("Complexity", "Prevalence and volume", "fail",
                             f"prevalence {prev:.1%}, {pos} positives"))
    else:
        checks.append(_check("Complexity", "Prevalence and volume", "pass",
                             f"prevalence {prev:.1%}, {pos} positives out of {len(sub)} customers"))

    # Privacy: IDs never become features nor go to the LLM
    checks.append(_check("Privacy", "No individual data sent to the LLM", "pass",
                         f"LLM received only the aggregated profile; {id_cols} removed"))

    # Still learning: leakage the LLM did not declare
    drop = set(id_cols) | set(sens) | {p["target"]["column"]} | set(p.get("leakage_columns", []))
    if p.get("population"):
        drop.add(p["population"]["column"])
    candidates = [col for col in df.columns if col not in drop]
    leaks = {}
    if 2 <= pos < len(y):
        for col in candidates:
            auc = single_feature_auc(sub[col], y)
            if auc >= g["leakage_auc"]:
                leaks[col] = round(auc, 3)
    if leaks:
        checks.append(_check("Still learning", "Undeclared leakage", "warn",
                             f"LLM did not list {list(leaks)} (single-feature AUC {leaks}) — dropped automatically"))
    else:
        checks.append(_check("Still learning", "Undeclared leakage", "pass",
                             f"no single variable with AUC ≥ {g['leakage_auc']}"))
    features = [col for col in candidates if col not in leaks]

    status = "rejected" if any(ch["status"] == "fail" for ch in checks) else "approved"
    return {**p, "checks": checks, "status": status, "features": features,
            "prevalence": prev, "n_population": int(len(sub))}


def recall_gap(y_true, y_pred, groups: pd.Series) -> tuple[float, dict]:
    """Largest recall difference between groups of a sensitive attribute."""
    recalls = {}
    for gval in groups.unique():
        m = (groups == gval).to_numpy() & (y_true == 1)
        if m.sum() > 0:
            recalls[str(gval)] = float(y_pred[m].mean())
    gap = max(recalls.values()) - min(recalls.values()) if len(recalls) > 1 else 0.0
    return gap, recalls
