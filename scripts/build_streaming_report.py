"""Generate a streaming ingestion report from DuckDB."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="data/interim/streaming.duckdb")
    ap.add_argument("--table", default="telemetry_opsd")
    ap.add_argument("--out-md", default="reports/streaming_ingestion_report.md")
    args = ap.parse_args()

    con = duckdb.connect(args.duckdb)
    rowcount = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
    first = con.execute(f"SELECT MIN(utc_timestamp) FROM {args.table}").fetchone()[0]
    last = con.execute(f"SELECT MAX(utc_timestamp) FROM {args.table}").fetchone()[0]

    lines = [
        "# Streaming Ingestion Report\n",
        f"- DuckDB: `{args.duckdb}`\n",
        f"- Table: `{args.table}`\n",
        f"- Rows ingested: **{rowcount}**\n",
        f"- First timestamp: **{first}**\n",
        f"- Last timestamp: **{last}**\n",
        "\n## Notes\n",
        "- Validate contract failures via consumer logs.\n",
        "- Add per-minute ingestion rate and lag if you store producer timestamps.\n",
    ]

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
