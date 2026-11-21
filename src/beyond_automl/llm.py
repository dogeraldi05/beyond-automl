"""LLM Analyst: proposes business problems from the dataset profile.

Uses OpenRouter (OpenAI-compatible API). Without OPENROUTER_API_KEY it
falls back to the versioned offline cache in problems/llm_problems_cache.json.
"""
import json
import os
import re
from pathlib import Path

PROMPT = """You are a senior data scientist at a telecom company.
Below is the PROFILE of a customer dataset (schema and statistics only, no rows).

{profile}

Propose {n} business problems that could be modeled as BINARY CLASSIFICATION.
Reply ONLY with valid JSON in this format:
{{"problems": [
  {{
    "id": "short_snake_case",
    "problem": "short name",
    "objective": "one sentence",
    "target": {{"column": "<column>", "operator": "in|not_in|gt|lt", "values": [...]}},
    "population": null or {{"column": "<column>", "operator": "in|not_in", "values": [...]}},
    "leakage_columns": ["columns that must not be used as features"],
    "metric": "AUC|F1|Recall|Precision",
    "value_per_hit_brl": <expected value in BRL of acting on a true positive>,
    "cost_per_action_brl": <cost in BRL of each action/contact>,
    "impact_score": <1-10>,
    "difficulty": <1-10>
  }}
]}}
Assume MonthlyCharges is in BRL."""


def build_prompt(profile: dict, n: int) -> str:
    return PROMPT.format(profile=json.dumps(profile, ensure_ascii=False, indent=1), n=n)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM response contains no JSON")
    return json.loads(match.group(0))


def ask_llm(profile: dict, cfg: dict) -> dict:
    """Returns {"source", "model", "problems", "usage"}."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    cache = Path(cfg["cache_path"])
    if not api_key:
        print("OPENROUTER_API_KEY not set -> using offline cache:", cache)
        data = json.loads(cache.read_text(encoding="utf-8"))
        data.setdefault("usage", None)
        return data

    from openai import OpenAI

    model = os.getenv("OPENROUTER_MODEL", cfg["model"])
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(profile, cfg["n_problems"])}],
        temperature=0.4,
    )
    text = resp.choices[0].message.content
    data = _extract_json(text)
    out = {
        "source": "openrouter (live)",
        "model": resp.model,
        "problems": data.get("problems", []),
        "usage": resp.usage.model_dump() if resp.usage else None,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/llm_raw_response.txt").write_text(text, encoding="utf-8")
    return out
