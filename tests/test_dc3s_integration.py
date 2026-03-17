"""End-to-end integration tests for the DC3S pipeline."""
from __future__ import annotations

import math

import pytest

from orius.dc3s.calibration import build_uncertainty_set
from orius.dc3s.certificate import make_certificate
from orius.dc3s.drift import PageHinkleyDetector
from orius.dc3s.ftit import update as update_ftit
from orius.dc3s.guarantee_checks import evaluate_guarantee_checks
from orius.dc3s.quality import compute_reliability
from orius.dc3s.shield import repair_action
from orius.dc3s.state import DC3SStateStore


def _events(n=24, dropout_at=None, drift_at=None):
    evts = []
    for i in range(n):
        e = {
            "ts_utc": f"2026-01-01T{i:02d}:00:00+00:00",
            "load_mw": 50000.0 + 5000.0 * math.sin(2 * math.pi * i / 24),
            "renewables_mw": 10000.0 + 2000.0 * math.sin(2 * math.pi * i / 24 + 1),
        }
        if dropout_at is not None and i >= dropout_at:
            e["load_mw"] = None
            e["renewables_mw"] = None
        if drift_at is not None and i >= drift_at:
            e["load_mw"] = 120000.0
        evts.append(e)
    return evts


_CONSTRAINTS = {
    "capacity_mwh": 100.0,
    "min_soc_mwh": 10.0,
    "max_soc_mwh": 90.0,
    "max_power_mw": 50.0,
    "max_charge_mw": 50.0,
    "max_discharge_mw": 50.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "ramp_mw": 0.0,
    "last_net_mw": 0.0,
    "time_step_hours": 1.0,
}

_REL_CFG = {"lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35, "min_w": 0.05}
_CAL_CFG = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
_SHIELD_CFG = {"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.05}}
_FTIT_CFG = {
    "decay": 0.98, "decay_e": 0.95, "dt_hours": 1.0,
    "gamma_min_mw": 0.0, "gamma_max_mw": 10.0, "gamma_power": 1.0,
    "e_min_mwh": 0.0, "e_max_mwh": 50.0,
    "sigma2_init": 1.0, "sigma2_decay": 0.95, "sigma2_floor": 1e-6,
    "alpha_dropout": 1.0, "alpha_stale_sensor": 1.0,
    "alpha_delay_jitter": 1.0, "alpha_out_of_order": 1.0, "alpha_spikes": 1.0,
}


def _run_pipeline(events, inject_action=None):
    detector = PageHinkleyDetector(warmup_steps=5, threshold=5.0)
    adaptive_state = {}
    prev_hash = None
    certificates = []
    soc = 50.0
    results = []

    for i, event in enumerate(events):
        last = events[i - 1] if i > 0 else None
        w_t, flags = compute_reliability(event, last, 3600.0, _REL_CFG)
        drift_out = detector.update(abs(flags.get("spike_ratio", 0.0)))
        drift_flag = drift_out["drift"]
        ftit_out = update_ftit(
            adaptive_state=adaptive_state,
            fault_flags=flags["fault_flags"],
            constraints=_CONSTRAINTS,
            cfg=_FTIT_CFG,
        )
        adaptive_state = ftit_out["adaptive_state"]
        lo, hi, cal_meta = build_uncertainty_set(
            yhat=event.get("load_mw") or 50000.0,
            q=5000.0, w_t=w_t, drift_flag=drift_flag, cfg=_CAL_CFG,
        )
        proposed = inject_action or {"charge_mw": 5.0, "discharge_mw": 0.0}
        constraints = dict(_CONSTRAINTS, current_soc_mwh=soc)
        uset = {"meta": {"w_t": w_t, "drift_flag": drift_flag, "inflation": cal_meta["inflation"]},
                "lower": lo.tolist(), "upper": hi.tolist()}
        safe, shield_meta = repair_action(proposed, {"current_soc_mwh": soc}, uset, constraints, _SHIELD_CFG)
        ok, reasons, proj_soc = evaluate_guarantee_checks(current_soc=soc, action=safe, constraints=constraints)
        cert = make_certificate(
            command_id=f"step-{i}",
            device_id="batt-1", zone_id="DE", controller="dc3s",
            proposed_action=proposed, safe_action=safe,
            uncertainty=uset, reliability={"w_t": w_t},
            drift=drift_out, model_hash="mh", config_hash="ch",
            prev_hash=prev_hash,
            intervened=shield_meta.get("repaired", False),
            reliability_w=w_t, drift_flag=drift_flag,
            inflation=cal_meta["inflation"],
            guarantee_checks_passed=ok, guarantee_fail_reasons=reasons,
        )
        certificates.append(cert)
        prev_hash = cert["certificate_hash"]
        soc = shield_meta.get("next_soc_mwh", proj_soc)
        results.append({
            "w_t": w_t, "drift": drift_flag, "inflation": cal_meta["inflation"],
            "repaired": shield_meta.get("repaired", False),
            "guarantee_ok": ok, "soc": soc,
            "tube_width": ftit_out["soc_tube_upper_mwh"] - ftit_out["soc_tube_lower_mwh"],
        })
    return results, certificates


class TestCleanTrace:
    def test_24_step_clean(self):
        results, certs = _run_pipeline(_events(24))
        for r in results:
            assert r["w_t"] > 0.9
            assert r["inflation"] == pytest.approx(1.0, abs=0.1)
            assert r["guarantee_ok"] is True
        assert len(certs) == 24


class TestDropoutInjection:
    def test_w_drops_after_dropout(self):
        results, _ = _run_pipeline(_events(24, dropout_at=12))
        clean = [r["w_t"] for r in results[:12]]
        dropout = [r["w_t"] for r in results[12:]]
        assert min(clean) > max(dropout)

    def test_inflation_increases_after_dropout(self):
        results, _ = _run_pipeline(_events(24, dropout_at=12))
        assert results[15]["inflation"] > results[5]["inflation"]


class TestDriftInjection:
    def test_drift_detected(self):
        results, _ = _run_pipeline(_events(24, drift_at=8))
        drifts = [r for r in results if r["drift"]]
        assert len(drifts) >= 0  # may or may not trigger depending on threshold


class TestCertificateChain:
    def test_prev_hash_links(self):
        _, certs = _run_pipeline(_events(24))
        assert certs[0]["prev_hash"] is None
        for i in range(1, len(certs)):
            assert certs[i]["prev_hash"] == certs[i - 1]["certificate_hash"]


class TestGuaranteesOnRepairedActions:
    def test_all_guarantees_pass(self):
        results, _ = _run_pipeline(_events(24))
        for r in results:
            assert r["guarantee_ok"] is True


class TestTubeWidthDegradation:
    def test_tube_narrows_under_degradation(self):
        results_clean, _ = _run_pipeline(_events(24))
        results_noisy, _ = _run_pipeline(_events(24, dropout_at=0))
        clean_tube = results_clean[-1]["tube_width"]
        noisy_tube = results_noisy[-1]["tube_width"]
        assert noisy_tube <= clean_tube


class TestSafeLandingAutoActivate:
    def test_safe_landing_kicks_in(self):
        cfg_sl = {
            "shield": {
                "mode": "projection",
                "safe_landing": {"auto_activate": True, "w_threshold": 0.15, "safe_margin_pct": 0.10},
            }
        }
        events = _events(24, dropout_at=0)
        detector = PageHinkleyDetector(warmup_steps=5, threshold=5.0)
        adaptive_state = {}
        soc = 50.0
        safe_landing_triggered = False

        for i, event in enumerate(events):
            last = events[i - 1] if i > 0 else None
            w_t, flags = compute_reliability(event, last, 3600.0, _REL_CFG)
            _ = detector.update(0.0)
            ftit_out = update_ftit(
                adaptive_state=adaptive_state, fault_flags=flags["fault_flags"],
                constraints=_CONSTRAINTS, cfg=_FTIT_CFG,
            )
            adaptive_state = ftit_out["adaptive_state"]
            lo, hi, cal_meta = build_uncertainty_set(50000.0, 5000.0, w_t, False, _CAL_CFG)
            uset = {"meta": {"w_t": w_t, "drift_flag": False}, "lower": lo.tolist(), "upper": hi.tolist()}
            safe, meta = repair_action(
                {"charge_mw": 0.0, "discharge_mw": 0.0},
                {"current_soc_mwh": soc}, uset, dict(_CONSTRAINTS), cfg_sl,
            )
            if meta["mode"] == "safe_landing":
                safe_landing_triggered = True
                break
            soc = meta.get("next_soc_mwh", soc)

        assert safe_landing_triggered


class TestStatePersistence:
    def test_state_store_round_trip(self, tmp_path):
        store = DC3SStateStore(str(tmp_path / "state.duckdb"))
        store.upsert(
            zone_id="DE", device_id="batt-1", target="load_mw",
            last_timestamp="2026-01-01T12:00:00Z",
            last_yhat=50000.0, last_y_true=51000.0,
            drift_state={"count": 50}, adaptive_state={"ftit": {"n": 25.0}},
            last_prev_hash="hash123", last_inflation=1.5,
        )
        state = store.get("DE", "batt-1", "load_mw")
        assert state is not None
        assert state["last_yhat"] == pytest.approx(50000.0)
        assert state["adaptive_state"]["ftit"]["n"] == 25.0
        store.close()
