#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from orius.forecasting.uncertainty.conformal import build_runtime_interval
from orius.forecasting.uncertainty.shift_aware import ShiftAwareConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark shift-aware interval runtime overhead.")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    cfg = ShiftAwareConfig(enabled=True)
    t0 = time.perf_counter()
    for i in range(args.steps):
        build_runtime_interval(
            y_hat=10.0,
            base_half_width=1.0,
            reliability_score=max(0.1, 1.0 - i / max(1, args.steps)),
            drift_flag=(i % 23 == 0),
            residual_features={"abs_residual": (i % 7) / 10.0, "normalized_residual": (i % 5) / 4.0},
            subgroup_context={"volatility": (i % 4) / 4.0, "hour": i % 24},
            fault_context={"fault_type": "sensor" if i % 17 == 0 else "none"},
            config=cfg,
        )
    dt = time.perf_counter() - t0
    per_step_us = 1e6 * dt / max(1, args.steps)
    print(f"steps={args.steps} total_s={dt:.6f} per_step_us={per_step_us:.2f}")


if __name__ == "__main__":
    main()
