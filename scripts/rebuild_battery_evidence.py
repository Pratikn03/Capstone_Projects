#!/usr/bin/env python3
"""Rebuild battery evidence artifacts for the ORIUS framework proof.

Runs the DC3S pipeline on synthetic or real telemetry data to generate
dispatch certificates and shield intervention statistics.

Usage:
    python scripts/rebuild_battery_evidence.py [--steps 200] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orius.dc3s.pipeline import run_dc3s_step
from orius.dc3s.drift import PageHinkleyDetector, AdaptivePageHinkleyDetector
from orius.dc3s.rac_cert import RACCertConfig, RACCertModel
from orius.utils.config import load_config


def _make_synthetic_state(soc: float, capacity: float):
    """Create a minimal state object with the required attributes."""
    class _State:
        pass
    s = _State()
    s.current_soc_mwh = soc
    s.last_net_mw = 0.0
    s.min_soc_mwh = 0.0
    s.max_soc_mwh = capacity
    s.capacity_mwh = capacity
    return s


def main():
    parser = argparse.ArgumentParser(description="Rebuild battery evidence")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="artifacts/runs",
                        help="Directory for evidence artifacts")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dc3s_cfg = load_config("configs/dc3s.yaml").get("dc3s", {})
    opt_cfg = load_config("configs/optimization.yaml")
    battery_cfg = opt_cfg.get("battery", {})

    # Resolve active profile
    profile_name = battery_cfg.get("active_profile", "pdf_reference")
    profiles = battery_cfg.get("profiles", {})
    if profile_name in profiles:
        battery_cfg = {**battery_cfg, **profiles[profile_name]}

    capacity = float(battery_cfg.get("capacity_mwh", 10000.0))
    max_power = float(battery_cfg.get("max_power_mw", 200.0))

    # Calibrate a tiny RACCert model
    n_cal = 100
    y_cal = rng.normal(500, 50, n_cal)
    noise = rng.normal(0, 30, n_cal)
    q_lo_cal = y_cal - 60 + noise * 0.3
    q_hi_cal = y_cal + 60 + noise * 0.3
    rac = RACCertModel(cfg=RACCertConfig())
    rac.fit(y_cal, q_lo_cal, q_hi_cal)

    # Simple domain adapter (lightweight mock)
    from orius.dc3s import BatteryDomainAdapter
    adapter = BatteryDomainAdapter()

    drift_detector = AdaptivePageHinkleyDetector.from_state(
        None, dc3s_cfg.get("drift", {})
    )

    soc = capacity * 0.5
    certificates = []
    stats = {
        "interventions": 0,
        "drift_events": 0,
        "total_steps": args.steps,
        "soc_violations": 0,
    }

    prev_event = None
    prev_cert_hash = None

    for t in range(args.steps):
        # Synthetic telemetry
        load_true = 500 + 100 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 20)
        yhat = load_true + rng.normal(0, 15)
        q_half = max(10.0, 30 + rng.normal(0, 5))
        residual = abs(load_true - yhat)

        event = {
            "ts_utc": f"2024-01-01T{t % 24:02d}:00:00Z",
            "load_mw": float(load_true),
            "temperature_c": 25.0 + rng.normal(0, 2),
        }

        # Candidate action: simple heuristic
        net_load = load_true - 100  # 100 MW renewables
        if net_load > 0 and soc > capacity * 0.1:
            candidate = {"charge_mw": 0.0, "discharge_mw": min(net_load, max_power)}
        elif net_load < 0 and soc < capacity * 0.9:
            candidate = {"charge_mw": min(-net_load, max_power), "discharge_mw": 0.0}
        else:
            candidate = {"charge_mw": 0.0, "discharge_mw": 0.0}

        state = _make_synthetic_state(soc, capacity)

        result = run_dc3s_step(
            event=event,
            last_event=prev_event,
            yhat=yhat,
            q=q_half,
            candidate_action=candidate,
            domain_adapter=adapter,
            state=state,
            drift_detector=drift_detector,
            residual=residual,
            cfg=dc3s_cfg,
            prev_cert_hash=prev_cert_hash,
        )

        safe = result["safe_action"]
        charge_eff = float(battery_cfg.get("charge_efficiency", 0.95))
        discharge_eff = float(battery_cfg.get("discharge_efficiency", 0.95))
        soc = soc + charge_eff * safe["charge_mw"] - safe["discharge_mw"] / discharge_eff
        soc = max(0.0, min(capacity, soc))

        if result["shield_meta"].get("repaired", False):
            stats["interventions"] += 1
        if result["drift_flag"]:
            stats["drift_events"] += 1
        if soc < 0 or soc > capacity:
            stats["soc_violations"] += 1

        cert = result["certificate"]
        certificates.append({
            "step": t,
            "command_id": cert["command_id"],
            "reliability_w": result["reliability_w"],
            "drift_flag": result["drift_flag"],
            "inflation": result["inflation"],
            "repaired": result["shield_meta"].get("repaired", False),
            "soc_mwh": float(soc),
            "charge_mw": safe["charge_mw"],
            "discharge_mw": safe["discharge_mw"],
        })

        prev_event = event
        prev_cert_hash = cert.get("certificate_hash")

    # Save evidence
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "battery_evidence.json"
    evidence_path.write_text(json.dumps({
        "stats": stats,
        "certificates": certificates,
    }, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"Battery Evidence Rebuild Complete")
    print(f"{'='*60}")
    print(f"  Steps:        {stats['total_steps']}")
    print(f"  Interventions: {stats['interventions']} ({100*stats['interventions']/max(1,stats['total_steps']):.1f}%)")
    print(f"  Drift events:  {stats['drift_events']}")
    print(f"  SOC violations:{stats['soc_violations']}")
    print(f"  Output:        {evidence_path}")
    print(f"{'='*60}")

    return 0 if stats["soc_violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
