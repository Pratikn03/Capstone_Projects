#!/usr/bin/env python3
"""
ORIUS Foundations - Cross-Domain Empirical Evaluation
Evaluates the unified certified action release contract across three heterogeneous physical plants:
1. Battery Energy Storage System (BESS)
2. Power Grid Inverter (Frequency / Voltage Droop)
3. ROS 2 Mobile Robotics (Obstacle Collision & Safe Braking)

Demonstrates:
- Theorem A (Viability invariance enforcement)
- Theorem C (Trajectory safety bound under degraded observation)
- Theorem E (Cross-domain transferability under common DomainAdapter interface)
"""

import sys
import json
import csv
import time
import numpy as np
from pathlib import Path

from orius.universal_theory.dynamical_theorems import BatteryStoragePlant
from orius.adapters.power_grid.adapter import PowerGridInverterAdapter
from orius.adapters.robotics.adapter import ROS2RoboticsAdapter

def evaluate_battery_domain(num_steps=1000, seed=42):
    np.random.seed(seed)
    plant = BatteryStoragePlant(soc_min=0.10, soc_max=0.90, t_max=50.0, p_max_mw=25.0)
    
    # State: [soc, temp_c]
    x_true = np.array([0.50, 25.0])
    
    true_violations = 0
    interventions = 0
    fallbacks = 0
    latencies = []
    
    for t in range(num_steps):
        t0 = time.perf_counter()
        
        # Telemetry degradation: 20% dropout, gaussian noise, thermal drift
        has_telemetry = np.random.rand() > 0.20
        obs_noise = np.random.normal(0, [0.02, 1.0])
        x_obs = (x_true + obs_noise) if has_telemetry else (x_true + np.random.normal(0, [0.08, 4.0]))
        
        # Calibrated conformal uncertainty radius r_soc, r_temp (Theorem B/C coverage)
        r_soc = 0.08 if has_telemetry else 0.22
        r_temp = 4.0 if has_telemetry else 12.0
        
        # Nominal proposed action: aggressive discharge/charge in MW
        u_nom = 20.0 * np.sin(t / 20.0)
        
        # Tightened viable boundaries under observation uncertainty
        tight_soc_max = max(plant.soc_min, plant.soc_max - r_soc)
        tight_soc_min = min(plant.soc_max, plant.soc_min + r_soc)
        
        # Certified safe headroom
        headroom_charge = max(0.0, (tight_soc_max - x_obs[0]) * plant.capacity_mwh / (plant.eta * plant.dt_hours))
        headroom_discharge = max(0.0, (x_obs[0] - tight_soc_min) * plant.capacity_mwh * plant.eta / plant.dt_hours)
        
        if not has_telemetry or tight_soc_max <= tight_soc_min:
            # Fallback standby under severe degradation
            u_safe = 0.0
            fallbacks += 1
            interventions += 1
        else:
            if u_nom >= 0:
                u_safe = min(u_nom, plant.p_max_mw, headroom_charge)
            else:
                u_safe = -min(abs(u_nom), plant.p_max_mw, headroom_discharge)
                
            if abs(u_safe - u_nom) > 1e-4:
                interventions += 1
                
        u_released = np.array([u_safe])
        x_true = plant.step(x_true, u_released, xi=np.array([0.0]))
        latencies.append((time.perf_counter() - t0) * 1e6)
        
        if (x_true[0] < plant.soc_min or x_true[0] > plant.soc_max or x_true[1] > plant.t_max):
            true_violations += 1
            
    return {
        "domain": "Battery_BESS",
        "num_steps": num_steps,
        "true_violations": true_violations,
        "violation_rate": true_violations / num_steps,
        "interventions": interventions,
        "intervention_rate": interventions / num_steps,
        "fallback_rate": fallbacks / num_steps,
        "latency_median_us": float(np.median(latencies)),
        "latency_p95_us": float(np.percentile(latencies, 95))
    }

def evaluate_grid_domain(num_steps=1000, seed=43):
    np.random.seed(seed)
    adapter = PowerGridInverterAdapter(f_nom=60.0, f_min=59.5, f_max=60.5, v_min=0.95, v_max=1.05)
    
    true_violations = 0
    interventions = 0
    fallbacks = 0
    latencies = []
    
    true_f = 60.0
    true_v = 1.00
    
    for t in range(num_steps):
        t0 = time.perf_counter()
        
        is_stale = np.random.rand() < 0.20
        noise_f = np.random.normal(0, 0.02)
        noise_v = np.random.normal(0, 0.005)
        
        raw_telemetry = {
            "f_hz": true_f + noise_f if not is_stale else true_f + np.random.normal(0, 0.15),
            "v_pu": true_v + noise_v if not is_stale else true_v + np.random.normal(0, 0.04),
            "delay_ms": 150.0 if is_stale else 15.0,
            "is_stale": is_stale
        }
        
        pkt = adapter.ingest_telemetry(raw_telemetry)
        oqe_score, meta_oqe = adapter.compute_oqe([pkt])
        u_set = adapter.build_uncertainty_set(pkt, oqe_score)
        admissible = adapter.tighten_action_set(u_set)
        
        # Nominal aggressive action
        u_nom = {"p_mw": float(12.0 * np.sin(t / 15.0)), "q_mvar": float(4.0 * np.cos(t / 15.0))}
        safe_a, meta_rep = adapter.repair_action(u_nom, admissible)
        cert = adapter.emit_certificate({"safe_a": safe_a, "oqe": oqe_score})
        
        latencies.append((time.perf_counter() - t0) * 1e6)
        
        if meta_rep.get("repaired") or meta_rep.get("fallback"):
            interventions += 1
        if meta_rep.get("fallback"):
            fallbacks += 1
            
        # Physical dynamic step response
        dp = safe_a["p_mw"]
        dq = safe_a["q_mvar"]
        true_f += 0.002 * (dp / 10.0) - 0.05 * (true_f - 60.0) + np.random.normal(0, 0.005)
        true_v += 0.003 * (dq / 5.0) - 0.08 * (true_v - 1.00) + np.random.normal(0, 0.001)
        
        true_state = {"frequency_hz": true_f, "voltage_pu": true_v}
        if adapter.true_constraint_violated(true_state):
            true_violations += 1
            
    return {
        "domain": "PowerGrid_Inverter",
        "num_steps": num_steps,
        "true_violations": true_violations,
        "violation_rate": true_violations / num_steps,
        "interventions": interventions,
        "intervention_rate": interventions / num_steps,
        "fallback_rate": fallbacks / num_steps,
        "latency_median_us": float(np.median(latencies)),
        "latency_p95_us": float(np.percentile(latencies, 95))
    }

def evaluate_robotics_domain(num_steps=1000, seed=44):
    np.random.seed(seed)
    adapter = ROS2RoboticsAdapter(v_max=1.5, a_brake=2.0, d_margin=0.30)
    
    true_violations = 0
    interventions = 0
    fallbacks = 0
    latencies = []
    
    true_dist = 5.0
    true_v = 0.0
    dt = 0.05
    
    for t in range(num_steps):
        t0 = time.perf_counter()
        
        lidar_drop = np.random.rand() < 0.20
        odom_noise = np.random.normal(0, 0.02)
        
        raw_telemetry = {
            "odom_v": max(0.0, true_v + odom_noise),
            "scan_min_dist": (true_dist + np.random.normal(0, 0.04)) if not lidar_drop else 999.0,
            "is_dropped": lidar_drop
        }
        
        pkt = adapter.ingest_telemetry(raw_telemetry)
        oqe_score, meta_oqe = adapter.compute_oqe([pkt])
        u_set = adapter.build_uncertainty_set(pkt, oqe_score)
        admissible = adapter.tighten_action_set(u_set)
        
        u_nom = {"v_cmd": 1.5, "w_cmd": 0.0}
        safe_a, meta_rep = adapter.repair_action(u_nom, admissible)
        cert = adapter.emit_certificate({"safe_a": safe_a, "oqe": oqe_score})
        
        latencies.append((time.perf_counter() - t0) * 1e6)
        
        if meta_rep.get("repaired") or meta_rep.get("fallback"):
            interventions += 1
        if meta_rep.get("fallback"):
            fallbacks += 1
            
        true_v = safe_a["v_cmd"]
        true_dist -= true_v * dt
        
        true_state = {"scan_min_dist": true_dist, "odom_v": true_v}
        if adapter.true_constraint_violated(true_state):
            true_violations += 1
            
        if true_dist <= adapter.d_margin:
            # obstacle encounter complete, reset to continue run
            true_dist = 4.0
            true_v = 0.0
            
    return {
        "domain": "Robotics_ROS2",
        "num_steps": num_steps,
        "true_violations": true_violations,
        "violation_rate": true_violations / num_steps,
        "interventions": interventions,
        "intervention_rate": interventions / num_steps,
        "fallback_rate": fallbacks / num_steps,
        "latency_median_us": float(np.median(latencies)),
        "latency_p95_us": float(np.percentile(latencies, 95))
    }

def main():
    print("Executing ORIUS Cross-Domain Empirical Synthesis...")
    
    results = [
        evaluate_battery_domain(),
        evaluate_grid_domain(),
        evaluate_robotics_domain()
    ]
    
    out_dir = Path("reports/orius_foundations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = out_dir / "cross_domain_synthesis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    csv_path = out_dir / "cross_domain_synthesis.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results written to {json_path} and {csv_path}")
    for r in results:
        print(f"[{r['domain']}] Violations: {r['true_violations']} ({r['violation_rate']*100:.2f}%) | Interventions: {r['intervention_rate']*100:.1f}% | Latency p95: {r['latency_p95_us']:.1f}us")

if __name__ == "__main__":
    main()
