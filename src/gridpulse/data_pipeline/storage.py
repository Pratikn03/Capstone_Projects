"""Storage helpers for processed datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

def write_sql(df: pd.DataFrame, path: Path, table: str = "features", engine: str = "duckdb") -> None:
    """Write a DataFrame into a SQL database (DuckDB or SQLite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine == "duckdb":
        try:
            import duckdb
        except Exception as exc:
            raise RuntimeError("duckdb not installed; add it to requirements or use --sql-engine sqlite") from exc
        # DuckDB handles Parquet-scale tables efficiently for analytics.
        con = duckdb.connect(str(path))
        con.register("df_view", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_view")
        con.close()
    elif engine == "sqlite":
        import sqlite3
        # SQLite is lighter but slower for large datasets.
        con = sqlite3.connect(str(path))
        df.to_sql(table, con, if_exists="replace", index=False)
        con.close()
    else:
        raise ValueError(f"Unsupported SQL engine: {engine}")
