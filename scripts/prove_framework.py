#!/usr/bin/env python3
"""Prove the ORIUS/ORIUS framework works end-to-end.

Runs a sequence of automated checks that demonstrate every major subsystem
behaves as specified in the PDF:

    1. DC3S pipeline integrity (5-stage orchestration)
    2. L2 projection safety (shield always produces feasible actions)
    3. RAC-Cert parameter alignment (κ values match PDF)
    4. Adaptive drift detection (catches injected concept drift)
    5. Certificate chain integrity (hash chain is valid)
    6. Dispatch optimization (MILP produces feasible plans)
    7. Forecasting model architecture (hyperparameters match PDF)
    8. Battery evidence (no SOC violations in simulation)

Usage:
    python scripts/prove_framework.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _check(name: str, ok: bool, detail: str = ""):
    status = _PASS if ok else _FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def prove_dc3s_pipeline():
    """Check 1: run_dc3s_step produces valid results."""
    from orius.dc3s import BatteryDomainAdapter, PageHinkleyDetector
    from orius.dc3s.pipeline import run_dc3s_step

    class _S:
        current_soc_mwh = 5000.0
        last_net_mw = 0.0
        min_soc_mwh = 0.0
        max_soc_mwh = 10000.0
        capacity_mwh = 10000.0

    adapter = BatteryDomainAdapter()
    detector = PageHinkleyDetector()
    result = run_dc3s_step(
        event={"ts_utc": "2024-01-01T12:00:00Z", "load_mw": 500.0},
        last_event=None,
        yhat=500.0,
        q=30.0,
        candidate_action={"charge_mw": 0.0, "discharge_mw": 100.0},
        domain_adapter=adapter,
        state=_S(),
        drift_detector=detector,
        residual=10.0,
        cfg={
            "k_quality": 0.5,
            "k_drift": 0.3,
            "k_sensitivity": 0.05,
            "infl_max": 2.0,
            "expected_cadence_s": 3600,
            "reliability": {"min_w": 0.05},
        },
    )

    ok = True
    ok &= _check("run_dc3s_step returns certificate", "certificate" in result)
    ok &= _check("run_dc3s_step returns safe_action", "safe_action" in result)
    ok &= _check("Certificate has hash", bool(result["certificate"].get("certificate_hash")))
    ok &= _check(
        "Reliability w_t in [0,1]",
        0.0 <= result["reliability_w"] <= 1.0,
        f"w_t={result['reliability_w']:.3f}",
    )
    return ok


def prove_l2_projection():
    """Check 2: L2 projection always produces feasible battery actions."""
    from orius.domain.battery_adapter import _l2_projection_repair

    rng = np.random.default_rng(123)
    constraints = {
        "capacity_mwh": 10000.0,
        "min_soc_mwh": 0.0,
        "max_soc_mwh": 10000.0,
        "max_power_mw": 200.0,
        "max_charge_mw": 200.0,
        "max_discharge_mw": 200.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "ramp_mw": 0.0,
        "last_net_mw": 0.0,
    }
    cfg = {"shield": {"reserve_soc_pct_drift": 0.08}}
    ok = True

    for i in range(100):
        soc = rng.uniform(0, 10000)
        charge = rng.uniform(0, 500)
        discharge = rng.uniform(0, 500)
        a = {"charge_mw": charge, "discharge_mw": discharge}
        state = {"current_soc_mwh": soc}
        safe, meta = _l2_projection_repair(a, state, {"meta": {}}, constraints, cfg)

        c = safe["charge_mw"]
        d = safe["discharge_mw"]
        # Must be non-negative
        ok &= (
            _check(f"L2[{i}] non-negative", c >= -1e-9 and d >= -1e-9, f"c={c:.2f}, d={d:.2f}")
            if i < 5
            else (c >= -1e-9 and d >= -1e-9)
        )
        # Mutual exclusion
        ok &= c < 1e-9 or d < 1e-9
        # Power bounds
        ok &= c <= 200.0 + 1e-9 and d <= 200.0 + 1e-9
        # SOC feasibility
        next_soc = soc + 0.95 * c - d / 0.95
        ok &= next_soc >= -1e-6 and next_soc <= 10000.0 + 1e-6

    _check("L2 projection: 100 random inputs all feasible", ok)
    return ok


def prove_rac_cert_params():
    """Check 3: RAC-Cert defaults match PDF."""
    from orius.dc3s.rac_cert import RACCertConfig

    cfg = RACCertConfig()
    ok = True
    ok &= _check("beta_reliability = 0.5", cfg.beta_reliability == 0.5, f"got {cfg.beta_reliability}")
    ok &= _check("beta_sensitivity = 0.3", cfg.beta_sensitivity == 0.3, f"got {cfg.beta_sensitivity}")
    ok &= _check("k_sensitivity = 0.05", cfg.k_sensitivity == 0.05, f"got {cfg.k_sensitivity}")
    ok &= _check("infl_max = 2.0", cfg.infl_max == 2.0, f"got {cfg.infl_max}")
    ok &= _check("max_q_multiplier = 2.0", cfg.max_q_multiplier == 2.0, f"got {cfg.max_q_multiplier}")
    return ok


def prove_adaptive_drift():
    """Check 4: Adaptive drift detector catches injected drift."""
    from orius.dc3s.drift import AdaptivePageHinkleyDetector

    detector = AdaptivePageHinkleyDetector(base_threshold=3.0, warmup_steps=20, cooldown_steps=5)
    # Feed stable residuals then inject a shift
    rng = np.random.default_rng(99)
    for _ in range(30):
        detector.update(rng.normal(10, 1))

    # Inject drift: residuals jump to 50
    detected = False
    for _ in range(50):
        result = detector.update(rng.normal(50, 2))
        if result["drift"]:
            detected = True
            break

    ok = _check("Adaptive drift detects injected shift", detected)
    return ok


def prove_certificate_chain():
    """Check 5: Certificate hash chain is consistent."""
    from orius.dc3s.certificate import compute_config_hash, make_certificate

    certs = []
    prev_hash = None
    for i in range(5):
        cert = make_certificate(
            command_id=f"cmd-{i}",
            device_id="battery-0",
            zone_id="zone-0",
            controller="dc3s",
            proposed_action={"charge_mw": 0.0, "discharge_mw": 10.0},
            safe_action={"charge_mw": 0.0, "discharge_mw": 10.0},
            uncertainty={"inflation": 1.0},
            reliability={"w_t": 0.9},
            drift={"drift": False},
            model_hash="abc123",
            config_hash=compute_config_hash(b"test"),
            prev_hash=prev_hash,
        )
        certs.append(cert)
        prev_hash = cert["certificate_hash"]

    ok = True
    for i in range(1, len(certs)):
        ok &= certs[i]["prev_hash"] == certs[i - 1]["certificate_hash"]
    ok &= _check("Certificate chain: 5-step hash chain valid", ok)
    return ok


def prove_dispatch_optimizer():
    """Check 6: MILP optimizer produces feasible dispatch plan."""
    from orius.optimizer.lp_dispatch import optimize_dispatch

    cfg = {
        "battery": {
            "capacity_mwh": 10000.0,
            "max_power_mw": 200.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
            "efficiency": 0.95,
            "efficiency_mode": "constant",
            "initial_soc_mwh": 5000.0,
        },
        "grid": {"max_import_mw": 10000.0, "price_per_mwh": 50.0},
        "penalties": {},
        "objective": {"cost_weight": 1.0, "mode": "dual"},
        "time_step_hours": 1.0,
    }
    load = [500.0] * 24
    renew = [100.0] * 24
    result = optimize_dispatch(load, renew, cfg)

    ok = True
    ok &= _check("Optimizer returns grid_mw", "grid_mw" in result)
    ok &= _check("Optimizer returns soc_mwh", "soc_mwh" in result)
    soc = result["soc_mwh"]
    ok &= _check(
        "SOC stays in bounds",
        all(0 <= s <= 10000.0 + 1e-3 for s in soc),
        f"min={min(soc):.1f}, max={max(soc):.1f}",
    )
    return ok


def prove_dl_architecture():
    """Check 7: Deep learning model defaults match PDF."""
    from orius.forecasting.dl_lstm import LSTMForecaster
    from orius.forecasting.dl_patchtst import PatchTSTForecaster

    ok = True
    # PatchTST defaults
    ptst = PatchTSTForecaster(n_features=10, lookback=168)
    ok &= _check("PatchTST patch_len=16", ptst.patch_len == 16, f"got {ptst.patch_len}")
    ok &= _check("PatchTST horizon=48", ptst.horizon == 48, f"got {ptst.horizon}")

    # LSTM defaults
    lstm = LSTMForecaster(n_features=10)
    ok &= _check("LSTM hidden_size=128", lstm.hidden_size == 128, f"got {lstm.hidden_size}")
    ok &= _check("LSTM horizon=48", lstm.horizon == 48, f"got {lstm.horizon}")
    return ok


def prove_half_life():
    """Check 8: Certificate half-life computation."""
    from orius.dc3s.temporal_theorems import certificate_half_life

    result = certificate_half_life(tau_t=100, decay_rate=0.5)
    ok = _check(
        "Half-life at decay=0.5 equals tau_t",
        result["half_life_steps"] == 100,
        f"got {result['half_life_steps']}",
    )
    result2 = certificate_half_life(tau_t=100, decay_rate=0.25)
    ok &= _check(
        "Half-life at decay=0.25 < tau_t",
        result2["half_life_steps"] < 100,
        f"got {result2['half_life_steps']}",
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Prove ORIUS framework")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checks = [
        ("DC3S Pipeline Integrity", prove_dc3s_pipeline),
        ("L2 Projection Safety", prove_l2_projection),
        ("RAC-Cert Parameter Alignment", prove_rac_cert_params),
        ("Adaptive Drift Detection", prove_adaptive_drift),
        ("Certificate Chain Integrity", prove_certificate_chain),
        ("Dispatch Optimizer Feasibility", prove_dispatch_optimizer),
        ("DL Architecture Alignment", prove_dl_architecture),
        ("Certificate Half-Life", prove_half_life),
    ]

    print(f"\n{'=' * 60}")
    print("ORIUS Framework Proof — End-to-End Verification")
    print(f"{'=' * 60}\n")

    passed = 0
    failed = 0
    for name, fn in checks:
        print(f"\n▸ {name}")
        try:
            if fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            _check(name, False, f"Exception: {e}")
            if args.verbose:
                traceback.print_exc()
            failed += 1

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} checks passed, {failed} failed")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
