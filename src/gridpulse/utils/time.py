"""Utilities: time."""
from __future__ import annotations
import pandas as pd

def ensure_datetime_utc(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    # Key: shared utilities used across the pipeline
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    return out
