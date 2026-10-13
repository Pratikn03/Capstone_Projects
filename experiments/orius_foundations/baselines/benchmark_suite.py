"""Reproducible Baseline Comparison Suite for ORIUS Foundational Paper.

Benchmarks ORIUS against:
1. Nominal (Unfiltered AI)
2. Fixed-Margin CBF (Static margin, uncalibrated)
3. Tube-MPC Proxy (Fixed geometric tube)
4. Always-Fallback (Trivial safety baseline)
5. ORIUS DC3S (Flagship reliability-aware conformal shield)
6. ORIUS Ablations (No-Calibrate, Shuffled-w causal controls)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from orius.universal_theory.dynamical_theorems import DoubleIntegratorRobot, Interval, TheoremEngine


def run_comparative_benchmark(
    n_episodes: int = 100,
    horizon: int = 15,
    noise_std: float = 0.25,
    dropout_rate: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    robot = DoubleIntegratorRobot(dt=0.1, p_max=10.0, u_max=3.0)
    rng = np.random.default_rng(seed)

    methods = [
        "Nominal",
        "FixedMargin_CBF",
        "TubeMPC_Proxy",
        "AlwaysFallback",
        "ORIUS_DC3S",
        "Ablation_NoCalibrate",
        "Control_Shuffled_W",
    ]

    records: dict[str, dict[str, Any]] = {
        m: {
            "violations": 0,
            "total_steps": 0,
            "total_distance": 0.0,
            "interventions": 0,
            "fallbacks": 0,
            "latencies_us": [],
        }
        for m in methods
    }

    import time

    for ep in range(n_episodes):
        # Initial state in middle of corridor
        x_init = np.array([4.0, 1.5])

        # Pre-generate noise and drops for this episode so all methods experience identical telemetry
        drops = [bool(rng.uniform() < dropout_rate) for _ in range(horizon)]
        noises = [rng.normal(0, noise_std, size=2) for _ in range(horizon)]
        shuffled_scores = list(rng.uniform(0.1, 1.0, size=horizon))

        for method in methods:
            x_curr = x_init.copy()
            ep_failed = False

            for t in range(horizon):
                records[method]["total_steps"] += 1
                is_drop = drops[t]
                nu = noises[t]

                # Raw telemetry
                o_t = robot.observe(x_curr, nu, is_dropped=is_drop)

                t0 = time.perf_counter_ns()
                # Nominal requested action: accelerate towards target
                a_nom = np.array([2.5])

                # Policy execution
                if method == "Nominal":
                    a_exec = a_nom

                elif method == "AlwaysFallback":
                    a_exec = robot.fallback_action()
                    records[method]["fallbacks"] += 1

                elif method == "FixedMargin_CBF":
                    # Fixed static margin: assumes fixed observation error of 0.2m
                    obs_p = (o_t[0] if o_t is not None else x_curr[0]) + 0.2
                    obs_v = max(0.0, o_t[1] if o_t is not None else x_curr[1])
                    stopping_d = (obs_v ** 2) / (2.0 * robot.u_max)
                    if obs_p + stopping_d > robot.p_max:
                        a_exec = robot.fallback_action()
                        records[method]["interventions"] += 1
                    else:
                        a_exec = a_nom

                elif method == "TubeMPC_Proxy":
                    # Fixed tube width of 0.5m regardless of dropout
                    obs_p = (o_t[0] if o_t is not None else x_curr[0]) + 0.5
                    obs_v = max(0.0, o_t[1] if o_t is not None else x_curr[1])
                    stopping_d = (obs_v ** 2) / (2.0 * robot.u_max)
                    if obs_p + stopping_d > robot.p_max:
                        a_exec = robot.fallback_action()
                        records[method]["interventions"] += 1
                    else:
                        a_exec = a_nom

                elif method == "ORIUS_DC3S":
                    # Stage 1: Detect
                    w_t = 0.2 if is_drop else 1.0
                    # Stage 2: Calibrate
                    hw = 1.96 * noise_std * (1.0 + (1.0 - w_t) * 2.5)
                    u_center = o_t if o_t is not None else x_curr
                    u_box = [Interval(u_center[i] - hw, u_center[i] + hw) for i in range(2)]
                    # Stage 3 & 4: Constrain & Shield
                    feas = TheoremEngine.theorem_a_feasibility(robot, u_box, action_grid_points=30)
                    if feas["feasible"] and feas["admissible_interval"] is not None:
                        iv = feas["admissible_interval"]
                        a_exec = np.array([np.clip(a_nom[0], iv.low, iv.high)])
                        if a_exec[0] < a_nom[0]:
                            records[method]["interventions"] += 1
                    else:
                        a_exec = robot.fallback_action()
                        records[method]["fallbacks"] += 1
                        records[method]["interventions"] += 1

                elif method == "Ablation_NoCalibrate":
                    # Ignore reliability score w_t; fix inflation to 1.0
                    hw = 1.96 * noise_std
                    u_center = o_t if o_t is not None else x_curr
                    u_box = [Interval(u_center[i] - hw, u_center[i] + hw) for i in range(2)]
                    feas = TheoremEngine.theorem_a_feasibility(robot, u_box, action_grid_points=30)
                    if feas["feasible"] and feas["admissible_interval"] is not None:
                        iv = feas["admissible_interval"]
                        a_exec = np.array([np.clip(a_nom[0], iv.low, iv.high)])
                    else:
                        a_exec = robot.fallback_action()
                        records[method]["fallbacks"] += 1

                elif method == "Control_Shuffled_W":
                    # Causal negative control: shuffled reliability score
                    w_shuffled = shuffled_scores[t]
                    hw = 1.96 * noise_std * (1.0 + (1.0 - w_shuffled) * 2.5)
                    u_center = o_t if o_t is not None else x_curr
                    u_box = [Interval(u_center[i] - hw, u_center[i] + hw) for i in range(2)]
                    feas = TheoremEngine.theorem_a_feasibility(robot, u_box, action_grid_points=30)
                    if feas["feasible"] and feas["admissible_interval"] is not None:
                        iv = feas["admissible_interval"]
                        a_exec = np.array([np.clip(a_nom[0], iv.low, iv.high)])
                    else:
                        a_exec = robot.fallback_action()

                t1 = time.perf_counter_ns()
                records[method]["latencies_us"].append((t1 - t0) / 1000.0)

                # Step dynamics
                x_next = robot.step(x_curr, a_exec, np.zeros(1))
                records[method]["total_distance"] += max(0.0, float(x_next[0] - x_curr[0]))

                if not robot.is_safe(x_next):
                    ep_failed = True

                x_curr = x_next

            if ep_failed:
                records[method]["violations"] += 1

    # Summarize results
    summary: dict[str, Any] = {}
    for m in methods:
        tot_dist = records[m]["total_distance"]
        max_dist_possible = horizon * 3.0 * robot.dt * n_episodes  # approximate max progress
        norm_utility = min(1.0, tot_dist / (max_dist_possible + 1e-6))
        summary[m] = {
            "trajectory_violation_rate": float(records[m]["violations"] / n_episodes),
            "normalized_utility": float(norm_utility),
            "intervention_rate": float(records[m]["interventions"] / records[m]["total_steps"]),
            "fallback_rate": float(records[m]["fallbacks"] / records[m]["total_steps"]),
            "latency_p95_us": float(np.percentile(records[m]["latencies_us"], 95)),
            "latency_median_us": float(np.median(records[m]["latencies_us"])),
        }

    return summary


if __name__ == "__main__":
    out_dir = Path("reports/orius_foundations")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_comparative_benchmark(n_episodes=100, horizon=15)
    out_file = out_dir / "baselines_comparison.json"
    out_file.write_text(json.dumps(res, indent=2))
    print(f"Benchmark completed successfully. Output saved to {out_file}")
    for k, v in res.items():
        print(f"  {k:22s} -> TSVR: {v['trajectory_violation_rate']:.3f} | Utility: {v['normalized_utility']:.3f} | Latency p95: {v['latency_p95_us']:.1f}us")
