#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from orius.forecasting.uncertainty.shift_aware import (
    ShiftAwareConfig,
    make_aci_state,
    update_adaptive_quantile,
    compute_validity_score,
    write_shift_aware_artifacts,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay adaptive quantile updates over ordered predictions.")
    ap.add_argument("--input", required=True, help="CSV with covered,reliability,residual,drift,under_coverage_gap")
    ap.add_argument("--out-dir", default="reports/publication")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"missing input: {path}")

    cfg = ShiftAwareConfig(enabled=True, aci_mode="aci_clipped")
    aci = make_aci_state(cfg)
    q_rows = []
    v_rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            covered = str(row.get("covered", "1")).strip().lower() in {"1", "true", "yes"}
            aci = update_adaptive_quantile(aci, is_miss=not covered, config=cfg)
            val = compute_validity_score(
                reliability_score=float(row.get("reliability", 1.0)),
                drift_magnitude=float(row.get("drift", 0.0)),
                normalized_residual=float(row.get("residual", 0.0)),
                under_coverage_gap=float(row.get("under_coverage_gap", 0.0)),
                adaptation_instability=float(aci.instability),
                config=cfg,
            )
            q_rows.append({"step": i, **aci.to_dict()})
            v_rows.append({"step": i, **val.to_dict()})

    out = write_shift_aware_artifacts(
        reliability_rows=[],
        volatility_rows=[],
        fault_rows=[],
        validity_trace_rows=v_rows,
        quantile_trace_rows=q_rows,
        out_dir=args.out_dir,
    )
    print(f"steps={len(q_rows)} final_quantile={aci.effective_quantile:.4f} final_status={v_rows[-1]['validity_status'] if v_rows else 'nominal'}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
