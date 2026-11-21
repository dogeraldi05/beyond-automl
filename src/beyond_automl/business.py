"""Business ranking: how much is acting on each model worth?

For each model, picks the threshold that maximizes profit on the test set:
    profit = TP × value_per_hit − (TP + FP) × cost_per_action
and compares it with the best no-model alternative (act on everyone or no one).
The gain per customer is extrapolated to the configured customer base.
"""
import numpy as np


def best_threshold_profit(y: np.ndarray, proba: np.ndarray, value: float, cost: float):
    order = np.argsort(-proba)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    n_act = np.arange(1, len(y) + 1)
    profit = tp * value - n_act * cost
    k = int(np.argmax(profit))
    best = float(profit[k])
    if best <= 0:
        return 0.0, 1.0, 0, 0  # best not to act
    threshold = float(proba[order][k])
    return best, threshold, int(n_act[k]), int(tp[k])


def score(run: dict, problem: dict, customer_base: int, n_population: int, n_total: int) -> dict:
    y, proba = run["_y"], run["_proba"]
    v, c = float(problem["value_per_hit_brl"]), float(problem["cost_per_action_brl"])
    profit, thr, n_act, tp = best_threshold_profit(y, proba, v, c)
    act_all = float(y.sum() * v - len(y) * c)
    baseline = max(act_all, 0.0)
    uplift = profit - baseline
    # extrapolate: profit per customer of the population × population size in the base
    pop_share = n_population / n_total
    scale = customer_base * pop_share / len(y)
    return {
        "threshold": thr,
        "contacted_pct": n_act / len(y),
        "precision_at_thr": tp / n_act if n_act else 0.0,
        "recall_at_thr": tp / y.sum() if y.sum() else 0.0,
        "profit_test_brl": profit,
        "baseline_test_brl": baseline,
        "impact_brl": uplift * scale,
    }
