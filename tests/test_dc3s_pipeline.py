"""Tests for the DC3S run_dc3s_step pipeline entry point."""

from __future__ import annotations

from orius.dc3s import BatteryDomainAdapter
from orius.dc3s.drift import AdaptivePageHinkleyDetector, PageHinkleyDetector
from orius.dc3s.pipeline import run_dc3s_step


class _MockState:
    current_soc_mwh = 5000.0
    last_net_mw = 0.0
    min_soc_mwh = 0.0
    max_soc_mwh = 10000.0
    capacity_mwh = 10000.0


_BASE_CFG = {
    "k_quality": 0.5,
    "k_drift": 0.3,
    "k_sensitivity": 0.05,
    "infl_max": 2.0,
    "expected_cadence_s": 3600,
    "reliability": {"min_w": 0.05},
    "shield": {"mode": "l2_projection", "reserve_soc_pct_drift": 0.08},
}


class TestRunDC3SStep:
    def test_basic_invocation(self):
        adapter = BatteryDomainAdapter()
        result = run_dc3s_step(
            event={"ts_utc": "2024-01-01T00:00:00Z", "load_mw": 500.0},
            last_event=None,
            yhat=500.0,
            q=30.0,
            candidate_action={"charge_mw": 0.0, "discharge_mw": 50.0},
            domain_adapter=adapter,
            state=_MockState(),
            cfg=_BASE_CFG,
        )
        assert "certificate" in result
        assert "safe_action" in result
        assert "reliability_w" in result
        assert 0.0 <= result["reliability_w"] <= 1.0

    def test_certificate_has_hash(self):
        adapter = BatteryDomainAdapter()
        result = run_dc3s_step(
            event={"ts_utc": "2024-01-01T12:00:00Z", "load_mw": 400.0},
            last_event=None,
            yhat=400.0,
            q=20.0,
            candidate_action={"charge_mw": 10.0, "discharge_mw": 0.0},
            domain_adapter=adapter,
            state=_MockState(),
            cfg=_BASE_CFG,
        )
        cert = result["certificate"]
        assert "certificate_hash" in cert
        assert len(cert["certificate_hash"]) == 64  # SHA-256 hex

    def test_with_drift_detector(self):
        adapter = BatteryDomainAdapter()
        detector = PageHinkleyDetector()
        result = run_dc3s_step(
            event={"ts_utc": "2024-01-01T00:00:00Z", "load_mw": 500.0},
            last_event=None,
            yhat=500.0,
            q=25.0,
            candidate_action={"charge_mw": 0.0, "discharge_mw": 0.0},
            domain_adapter=adapter,
            state=_MockState(),
            drift_detector=detector,
            residual=5.0,
            cfg=_BASE_CFG,
        )
        assert result["drift_flag"] is False

    def test_with_adaptive_drift(self):
        adapter = BatteryDomainAdapter()
        detector = AdaptivePageHinkleyDetector()
        result = run_dc3s_step(
            event={"ts_utc": "2024-01-01T00:00:00Z", "load_mw": 500.0},
            last_event=None,
            yhat=500.0,
            q=25.0,
            candidate_action={"charge_mw": 0.0, "discharge_mw": 0.0},
            domain_adapter=adapter,
            state=_MockState(),
            drift_detector=detector,
            residual=5.0,
            cfg=_BASE_CFG,
        )
        assert "drift_flag" in result

    def test_safe_action_feasible(self):
        adapter = BatteryDomainAdapter()
        result = run_dc3s_step(
            event={"ts_utc": "2024-01-01T00:00:00Z", "load_mw": 500.0},
            last_event=None,
            yhat=500.0,
            q=30.0,
            candidate_action={"charge_mw": 999.0, "discharge_mw": 999.0},
            domain_adapter=adapter,
            state=_MockState(),
            cfg=_BASE_CFG,
        )
        safe = result["safe_action"]
        assert safe["charge_mw"] >= 0.0
        assert safe["discharge_mw"] >= 0.0
        # Mutual exclusion
        assert safe["charge_mw"] < 1e-9 or safe["discharge_mw"] < 1e-9

    def test_certificate_chain(self):
        adapter = BatteryDomainAdapter()
        prev_hash = None
        for i in range(3):
            result = run_dc3s_step(
                event={"ts_utc": f"2024-01-01T{i:02d}:00:00Z", "load_mw": 500.0},
                last_event=None,
                yhat=500.0,
                q=30.0,
                candidate_action={"charge_mw": 0.0, "discharge_mw": 10.0},
                domain_adapter=adapter,
                state=_MockState(),
                prev_cert_hash=prev_hash,
                cfg=_BASE_CFG,
            )
            cert = result["certificate"]
            if i > 0:
                assert cert["prev_hash"] == prev_hash
            prev_hash = cert["certificate_hash"]
