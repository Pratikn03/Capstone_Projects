#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from orius.forecasting.uncertainty.conformal import build_runtime_interval
from orius.forecasting.uncertainty.shift_aware import ShiftAwareConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark shift-aware runtime interval overhead")
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()

    cfg = ShiftAwareConfig(enabled=True, policy_mode="shift_aware_linear", aci_mode="aci_clipped")
    start = time.perf_counter()
    for i in range(args.steps):
        build_runtime_interval(
            y_hat=100.0,
            base_half_width=5.0,
            reliability_score=max(0.1, 1.0 - (i % 10) * 0.08),
            drift_flag=(i % 17 == 0),
            residual_features={
                "abs_residual": float(i % 11),
                "covered": bool(i % 5),
                "volatility": float(i % 7) / 7.0,
            },
            subgroup_context={"timestamp": "2026-01-01T00:00:00Z", "custom_group_key": "bench"},
            fault_context={"fault_type": "none"},
            config=cfg,
        )
    elapsed = time.perf_counter() - start
    print(f"steps={args.steps} total_s={elapsed:.6f} per_step_us={(elapsed / max(args.steps, 1)) * 1e6:.2f}")


if __name__ == "__main__":
    main()
