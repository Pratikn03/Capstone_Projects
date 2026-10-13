#!/usr/bin/env python3
"""
ORIUS Foundations - Reliability Controls Causal Ablation (OR-006)
Demonstrates the causal necessity of real-time Observation Quality Estimation (OQE).
Compares:
1. Adaptive_ORIUS: Real-time dynamic w_t
2. Shuffled_OQE: Identical marginal distribution of w_t, but temporally scrambled
3. Constant_Optimistic: w_t = 1.0 (blind trust)
4. Constant_Pessimistic: w_t = 0.0 (maximum conservatism)
5. Lagged_OQE: w_t delayed by tau steps
"""

import json
import time
import numpy as np
from pathlib import Path

from orius.universal_theory.dynamical_theorems import DoubleIntegratorRobot

def run_causal_control_trial(mode: str, num_steps: int = 1500, lag_steps: int = 5, seed: int = 101):
    np.random.seed(seed)
    plant = DoubleIntegratorRobot(p_max=5.0, u_max=1.0, dt=0.1, xi_bound=0.01)
    
    # Pre-generate telemetry dropout sequence: bursts of packet loss
    drop_seq = []
    t = 0
    while t < num_steps:
        burst_len = np.random.geometric(p=0.2)
        is_drop = (np.random.rand() < 0.25)
        for _ in range(burst_len):
            if t < num_steps:
                drop_seq.append(is_drop)
                t += 1
                
    # Compute ground-truth w_t sequence
    w_true = []
    curr_w = 1.0
    for d in drop_seq:
        if d:
            curr_w = max(0.0, curr_w - 0.3)
        else:
            curr_w = min(1.0, curr_w + 0.2)
        w_true.append(curr_w)
        
    # Construct perceived w_t sequence per ablation mode
    if mode == "Adaptive_ORIUS":
        w_perceived = list(w_true)
    elif mode == "Shuffled_OQE":
        w_perceived = list(w_true)
        np.random.shuffle(w_perceived)
    elif mode == "Constant_Optimistic":
        w_perceived = [1.0] * num_steps
    elif mode == "Constant_Pessimistic":
        w_perceived = [0.0] * num_steps
    elif mode == "Lagged_OQE":
        w_perceived = [1.0] * lag_steps + w_true[:-lag_steps]
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    x_true = np.array([0.0, 0.0])
    x_est = np.array([0.0, 0.0])
    true_violations = 0
    interventions = 0
    fallbacks = 0
    utilities = []
    latencies = []
    
    for t in range(num_steps):
        t0 = time.perf_counter()
        is_dropped = drop_seq[t]
        w_t = w_perceived[t]
        
        # Sensor observation
        nu = np.random.normal(0, 0.02, size=2)
        x_obs = plant.observe(x_true, nu, is_dropped=is_dropped)
        
        if x_obs is not None:
            x_est = np.array(x_obs)
            r_obs = 0.08 + 0.30 * (1.0 - w_t)
        else:
            # Dead reckon forward under packet drop and expand uncertainty
            x_est[0] += x_est[1] * plant.dt
            r_obs = 0.25 + 0.45 * (1.0 - w_t)
            
        u_nom = np.array([1.0])
        
        if is_dropped and w_t < 0.25:
            u_rel = plant.fallback_action()
            fallbacks += 1
            interventions += 1
        else:
            worst_p = x_est[0] + r_obs
            worst_v = max(0.0, x_est[1] + r_obs)
            stopping_dist = (worst_v ** 2) / (2.0 * plant.u_max)
            
            if worst_p + stopping_dist >= plant.p_max - 0.05:
                u_rel = plant.fallback_action()
                interventions += 1
            else:
                u_rel = u_nom
                
        latencies.append((time.perf_counter() - t0) * 1e6)
        
        # Execute true physical dynamics
        xi = np.random.uniform(-plant.xi_bound, plant.xi_bound, size=1)
        x_true = plant.step(x_true, u_rel, xi)
        
        if not plant.is_safe(x_true):
            true_violations += 1
            x_true = np.array([0.0, 0.0])
            x_est = np.array([0.0, 0.0])
        elif x_true[1] < 1e-3 and x_true[0] > 3.0:
            # Safely arrived and stopped: dispatch next navigation segment
            x_true = np.array([0.0, 0.0])
            x_est = np.array([0.0, 0.0])
            
        utilities.append(float(x_true[1]))
        
    return {
        "mode": mode,
        "num_steps": num_steps,
        "true_violations": true_violations,
        "violation_rate": true_violations / num_steps,
        "interventions": interventions,
        "intervention_rate": interventions / num_steps,
        "fallback_rate": fallbacks / num_steps,
        "mean_utility": float(np.mean(utilities)),
        "latency_p95_us": float(np.percentile(latencies, 95))
    }

def main():
    print("Executing OR-006: Reliability Controls Causal Ablation Suite...")
    modes = [
        "Adaptive_ORIUS",
        "Shuffled_OQE",
        "Constant_Optimistic",
        "Constant_Pessimistic",
        "Lagged_OQE"
    ]
    
    results = {}
    for m in modes:
        res = run_causal_control_trial(m)
        results[m] = res
        print(f"[{m:<20}] Violations: {res['true_violations']:3d} | Interv: {res['intervention_rate']*100:4.1f}% | Utility: {res['mean_utility']:.3f}")
        
    out_path = Path("reports/orius_foundations/reliability_controls_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
