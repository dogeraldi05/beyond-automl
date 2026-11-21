"""beyond-automl dashboard.   streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="beyond-automl", page_icon="🧭", layout="wide")
R = Path("results")
if not (R / "leaderboard.csv").exists():
    st.error("Run the notebook first: `beyond_automl.ipynb`")
    st.stop()

lb = pd.read_csv(R / "leaderboard.csv")
biz = pd.read_csv(R / "business_ranking.csv")
rec = json.loads((R / "recommendation.json").read_text(encoding="utf-8"))
probs = json.loads((R / "problems_validated.json").read_text(encoding="utf-8"))
brl = lambda v: f"BRL {v/1e6:,.1f}M"

st.title("🧭 beyond-automl")
st.caption("AutoML finds the best model. LLM + human find the best problem.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Problems proposed by the LLM", len(probs))
c2.metric("Approved by guardrails", sum(p["status"] == "approved" for p in probs))
c3.metric("Models trained", rec["n_models"])
c4.metric("LLM source", rec["llm"]["source"])

t1, t2, t3, t4, t5 = st.tabs(["1 · Problems & guardrails", "2 · 20 models",
                              "3 · Technical leaderboard", "4 · Business leaderboard",
                              "5 · Recommendation"])

with t1:
    st.subheader("The LLM proposes, the code validates")
    icon = {"approved": "✅", "review": "🟡", "rejected": "❌"}
    st.dataframe(pd.DataFrame([{
        "": icon[p["status"]], "Problem": p["problem"], "Target": p["target"]["column"],
        "Impact (LLM)": p["impact_score"], "Status": p["status"],
    } for p in probs]), hide_index=True, width="stretch")
    st.markdown("**Checks by generative-AI shortcoming**")
    chk = pd.DataFrame([{"Problem": p["problem"], **c} for p in probs for c in p["checks"]])
    only_issues = st.toggle("Show only warnings and failures", value=True)
    if only_issues:
        chk = chk[chk["status"] != "pass"]
    st.dataframe(chk.rename(columns={"risk": "Shortcoming", "check": "Check",
                                     "status": "Result", "detail": "Detail"}),
                 hide_index=True, width="stretch")

with t2:
    st.subheader(f"{lb['problem'].nunique()} problems × {lb['model'].nunique()} models")
    piv = lb.pivot(index="problem", columns="model", values="AUC").round(3)
    st.dataframe(piv, width="stretch")

with t3:
    st.subheader("Technical ranking (AUC)")
    st.dataframe(lb.sort_values("rank_technical")[["rank_technical", "problem", "model", "AUC", "F1",
                                                    "Precision", "Recall", "status"]].round(3),
                 hide_index=True, width="stretch")

with t4:
    st.subheader("Business ranking (gain over not using a model)")
    fig = px.scatter(lb, x="AUC", y="impact_brl", color="problem", symbol="model",
                     labels={"impact_brl": "Impact (BRL/cycle)", "problem": "Problem"},
                     height=450)
    st.plotly_chart(fig, width="stretch")
    show = biz[["problem", "model", "AUC", "impact_brl", "contacted_pct", "precision_at_thr",
                "fairness_attr", "fairness_gap", "status"]].copy()
    show["impact_brl"] = show["impact_brl"].map(brl)
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    st.caption("Impact = (model profit − best no-model alternative) extrapolated to the customer base. "
               "Threshold chosen to maximize profit.")

with t5:
    st.subheader("Final recommendation")
    st.success(f"**{rec['recommended_problem']}** with **{rec['recommended_model']}** — "
               f"{brl(rec['impact_brl'])} per cycle (AUC {rec['auc']:.3f})")
    bt = rec["best_technical"]
    if not rec["same_as_technical"]:
        st.info(f"The highest AUC was **{bt['problem']}** ({bt['model']}, AUC {bt['auc']:.3f}), "
                f"but it is not the best business decision.")
    if rec["fairness_flag"]:
        st.warning("The recommended model has a recall gap between sensitive groups above the limit.")
    st.caption(f"Runtime: {rec['runtime_s']} s · LLM: {rec['llm']['model']}")
