"""
Utilities: Time and Datetime Helpers.

This module provides helper functions for working with timestamps in the
energy forecasting context. Consistent timezone handling is critical since:

- OPSD data uses UTC timestamps
- Grid operations often use local time (Europe/Berlin for DE)
- Forecasts must align temporally with market schedules

Key Functions:
    - ensure_datetime_utc: Standardize timestamps to UTC timezone-aware

Usage:
    >>> from gridpulse.utils.time import ensure_datetime_utc
    >>> df = ensure_datetime_utc(df, 'timestamp')
    >>> # Now df['timestamp'] is guaranteed to be UTC timezone-aware

Note:
    All internal processing should use UTC. Convert to local time only
    for display purposes or market-specific logic.
"""
from __future__ import annotations
import pandas as pd


def ensure_datetime_utc(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Ensure a timestamp column is parsed as UTC-aware datetimes.
    
    This is a defensive function that handles various input formats:
    - String timestamps (ISO format, etc.)
    - Naive datetime objects (assumes UTC)
    - Already UTC-aware datetimes (no change)
    
    Args:
        df: DataFrame containing the timestamp column
        ts_col: Name of the column to convert
        
    Returns:
        Copy of DataFrame with ts_col as UTC-aware datetime64[ns, UTC]
        
    Example:
        >>> df = pd.DataFrame({'timestamp': ['2024-01-01 00:00:00']})
        >>> df = ensure_datetime_utc(df, 'timestamp')
        >>> df['timestamp'].dtype
        datetime64[ns, UTC]
    """
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    return out
