#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.forecasting.uncertainty.shift_aware import AdaptiveQuantileState, update_adaptive_quantile


def main() -> None:
    p = argparse.ArgumentParser(description="Replay adaptive quantile updates over time")
    p.add_argument("--input", required=True, help="CSV with covered column (0/1)")
    p.add_argument("--out", default="reports/publication/adaptive_quantile_trace.csv")
    p.add_argument("--mode", choices=["fixed", "aci_basic", "aci_clipped"], default="aci_clipped")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--step", type=float, default=0.01)
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"missing input: {src}")
    df = pd.read_csv(src)
    if "covered" not in df.columns:
        raise SystemExit("input must contain covered column")

    state = AdaptiveQuantileState(
        mode=args.mode,
        base_alpha=args.alpha,
        effective_alpha=args.alpha,
        learning_rate=args.step,
    )
    rows: list[dict] = []
    for t, covered in enumerate(df["covered"].tolist()):
        update_adaptive_quantile(state, miss=not bool(covered))
        rows.append(
            {
                "t": t,
                "covered": int(bool(covered)),
                "effective_alpha": state.effective_alpha,
                "miss_streak": state.miss_streak,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
