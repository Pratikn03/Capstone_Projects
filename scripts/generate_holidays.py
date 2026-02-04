"""Generate a holiday calendar CSV for feature engineering."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import holidays  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: holidays. Install with `pip install holidays`.") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="DE", help="Country code (default: DE)")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--out", default="data/raw/holidays_de.csv")
    args = parser.parse_args()

    hset = holidays.country_holidays(args.country, years=range(args.start_year, args.end_year + 1))
    dates = sorted(hset.keys())
    out = pd.DataFrame({"date": [d.isoformat() for d in dates], "name": [hset[d] for d in dates]})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()

