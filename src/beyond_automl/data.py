"""Dataset loading and aggregated profile (what is sent to the LLM)."""
from pathlib import Path
import urllib.request

import pandas as pd


def load_data(path: str, url: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if url is None:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    # TotalCharges comes as text with blanks for brand-new customers
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def id_columns(df: pd.DataFrame) -> list[str]:
    """Columns that identify the customer (non-numeric, one value per row)."""
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == len(df)]


def profile(df: pd.DataFrame, top_n: int = 6) -> dict:
    """Schema summary WITHOUT individual rows (privacy guardrail).

    The LLM only sees column names, types and aggregated statistics.
    """
    ids = set(id_columns(df))
    cols = []
    for c in df.columns:
        if c in ids:
            cols.append({"name": c, "kind": "id (omitted)"})
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            cols.append({
                "name": c, "kind": "numeric",
                "min": round(float(s.min()), 2), "mean": round(float(s.mean()), 2),
                "max": round(float(s.max()), 2),
            })
        else:
            freq = s.value_counts(normalize=True).head(top_n).round(3)
            cols.append({
                "name": c, "kind": "categorical", "n_unique": int(s.nunique()),
                "top_values": {str(k): float(v) for k, v in freq.items()},
            })
    return {"n_rows": int(len(df)), "columns": cols}
