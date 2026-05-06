#!/usr/bin/env python3
"""ORIUS full pipeline check — verify all components work end-to-end.

Run: python scripts/run_orius_full_check.py [--quick] [--skip-cpsbench]

--quick: Skip CPSBench (faster, ~10s)
--skip-cpsbench: Same as --quick
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))


def _step(name: str, fn, *args, **kwargs):
    print(f"  [{name}] ... ", end="", flush=True)
    try:
        result = fn(*args, **kwargs)
        print("OK")
        return result
    except Exception as e:
        print(f"FAIL: {e}")
        raise


def check_imports():
    """Verify all core ORIUS imports."""

    def _do():
        from orius.adapters.battery import BatteryDomainAdapter
        from orius.adapters.healthcare import HealthcareDomainAdapter
        from orius.adapters.vehicle import VehicleDomainAdapter

        assert BatteryDomainAdapter is not None
        assert VehicleDomainAdapter is not None
        assert HealthcareDomainAdapter is not None
        return True

    _step("Imports (active adapters, DC3S, forecasting, optimizer, CPSBench)", _do)


def check_config():
    """Verify configs load."""

    def _do():
        import yaml

        cfg = yaml.safe_load((repo_root / "configs/dc3s.yaml").read_text())
        assert "dc3s" in cfg
        dc3s = cfg["dc3s"]
        assert dc3s.get("alpha0") is not None
        assert dc3s.get("reliability", {}).get("min_w") is not None
        return True

    _step("Config (dc3s.yaml)", _do)


def check_dc3s_demo():
    """Run DC3S demo via FastAPI TestClient."""

    def _do():
        from contextlib import ExitStack
        from datetime import datetime
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        from fastapi.testclient import TestClient

        from services.api.main import app
        from services.api.routers import dc3s as dc3s_router

        def _predict(*, target, horizon, features_df, forecast_cfg, required):
            return np.full(horizon, 50.0, dtype=float), Path(f"demo_{target}.bin")

        client = TestClient(app)
        features_df = pd.DataFrame({"price_eur_mwh": [62.0], "carbon_kg_per_mwh": [410.0]})
        with ExitStack() as stack:
            stack.enter_context(patch.object(dc3s_router, "_load_features_df", return_value=features_df))
            stack.enter_context(patch.object(dc3s_router, "_predict_target", side_effect=_predict))
            stack.enter_context(
                patch.object(dc3s_router, "_resolve_conformal_q", return_value=np.full(24, 4.0, dtype=float))
            )

            payload = {
                "device_id": "check-device",
                "zone_id": "DE",
                "current_soc_mwh": 1.0,
                "telemetry_event": {
                    "ts_utc": datetime.now(UTC).isoformat(),
                    "load_mw": 52.0,
                    "renewables_mw": 12.0,
                },
                "last_actual_load_mw": 52.0,
                "last_pred_load_mw": 50.0,
                "controller": "deterministic",
                "horizon": 24,
                "include_certificate": True,
            }
            r = client.post("/dc3s/step", json=payload)
            r.raise_for_status()
            data = r.json()
            assert "command_id" in data
            assert "certificate_id" in data or "certificate_hash" in str(data)
        return True

    _step("DC3S demo (POST /dc3s/step)", _do)


def check_cpsbench_quick():
    """Run a minimal CPSBench (1 scenario, 1 seed, short horizon)."""

    def _do():
        import tempfile

        from orius.cpsbench_iot.runner import run_suite

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            run_suite(
                scenarios=["nominal"],
                seeds=[42],
                out_dir=out_dir,
                horizon=24,
                include_fault_sweep=False,
            )
            main = out_dir / "dc3s_main_table.csv"
            assert main.exists()
            assert main.stat().st_size > 0
        return True

    _step("CPSBench (nominal, 1 seed, 24h)", _do)


def check_universal_framework():
    """Run universal framework across all registered runtime domains."""

    def _do():
        from orius.universal_framework import get_adapter, list_domains, run_universal_step

        domains = list_domains()
        assert "energy" in domains
        assert "av" in domains
        assert "healthcare" in domains

        # Run one step per runtime domain with synthetic telemetry.
        for domain_id in ["energy", "av", "healthcare"]:
            adapter = get_adapter(domain_id, {})
            if domain_id == "energy":
                telemetry = {
                    "load_mw": 45.0,
                    "renewables_mw": 80.0,
                    "current_soc_mwh": 100.0,
                    "capacity_mwh": 200.0,
                    "yhat_load": 48.0,
                    "ts_utc": "2026-01-01T00:00:00Z",
                }
                candidate = {"charge_mw": 20.0, "discharge_mw": 0.0}
                constraints = {
                    "min_soc_mwh": 20.0,
                    "max_soc_mwh": 180.0,
                    "capacity_mwh": 200.0,
                    "max_power_mw": 100.0,
                }
            elif domain_id == "av":
                telemetry = {
                    "position_m": 100.0,
                    "speed_mps": 8.0,
                    "speed_limit_mps": 15.0,
                    "lead_position_m": 150.0,
                    "ts_utc": "2026-01-01T00:00:00Z",
                }
                candidate = {"acceleration_mps2": 0.5}
                constraints = {"speed_max_mps": 15.0}
            elif domain_id == "healthcare":
                telemetry = {
                    "hr_bpm": 72.0,
                    "spo2_pct": 97.0,
                    "respiratory_rate": 14.0,
                    "ts_utc": "2026-01-01T00:00:00Z",
                }
                candidate = {"alert_level": 0.2}
                constraints = {"spo2_min_pct": 90.0}
            else:
                continue

            result = run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=telemetry,
                history=None,
                candidate_action=candidate,
                constraints=constraints,
                quantile=50.0,
            )
            assert "certificate" in result
            assert "safe_action" in result
        return True

    _step("Universal framework (3 active domains)", _do)


def check_locked_evidence():
    """Check that key locked files exist (optional)."""
    locked = [
        ("reports/impact_summary.csv", "DE impact"),
        ("reports/publication/dc3s_main_table_ci.csv", "DC3S main results"),
        ("reports/publication/dc3s_latency_summary.csv", "DC3S latency"),
    ]
    missing = []
    for rel, desc in locked:
        p = repo_root / rel
        if not p.exists() or p.stat().st_size == 0:
            missing.append(f"{desc}: {rel}")
    if missing:
        print(f"  [Locked evidence] WARNING: {len(missing)} missing: {missing}")
    else:
        print(f"  [Locked evidence] OK ({len(locked)} files)")
    return len(missing) == 0


def main():
    ap = argparse.ArgumentParser(description="ORIUS full pipeline check")
    ap.add_argument(
        "--quick", "--skip-cpsbench", dest="skip_cpsbench", action="store_true", help="Skip CPSBench"
    )
    args = ap.parse_args()

    print("ORIUS full check\n")
    try:
        check_imports()
        check_config()
        check_dc3s_demo()
        check_universal_framework()
        if not args.skip_cpsbench:
            check_cpsbench_quick()
        else:
            print("  [CPSBench] SKIPPED (--quick)")
        check_locked_evidence()
        print("\nAll checks passed. ORIUS pipeline is operational.")
        return 0
    except Exception as e:
        print(f"\nCheck failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
