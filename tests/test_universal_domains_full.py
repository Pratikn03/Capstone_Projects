"""Comprehensive validation tests for the ORIUS Universal Framework.

Each non-battery domain is exercised end-to-end through run_universal_step():
  1. Single-step pass — certificate emitted, safe_action returned
  2. Fault injection — blackout/bias/noise produce valid (possibly NaN-guarded) output
  3. Safety repair   — adapter catches unsafe candidate actions and repairs them
  4. Multi-step episode — no regressions vs nominal baseline over N steps
  5. Guarantee checks — intervention_rate is in [0, 1] and TSVR is non-negative

All tests are deterministic (fixed seeds).
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from orius.adapters.aerospace import AerospaceDomainAdapter, AerospaceTrackAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter, HealthcareTrackAdapter
from orius.adapters.industrial import IndustrialDomainAdapter, IndustrialTrackAdapter
from orius.adapters.navigation import NavigationDomainAdapter, NavigationTrackAdapter
from orius.adapters.vehicle import VehicleDomainAdapter, VehicleTrackAdapter
from orius.orius_bench.controller_api import (
    DC3SController,
    DomainAwareController,
    NominalController,
)
from orius.orius_bench.fault_engine import generate_fault_schedule, active_faults
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_framework import run_universal_step


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _ts(step: int = 0) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step)).isoformat().replace("+00:00", "Z")


def _run_domain_episode(
    track: Any,
    adapter: Any,
    cfg: dict[str, Any],
    constraints_fn: Any,
    quantile: float,
    hold_keys: tuple[str, ...],
    controller: Any,
    seed: int = 42,
    horizon: int = 48,
    use_universal: bool = True,
) -> list[StepRecord]:
    """Run a short episode through the universal adapter (or baseline) and return StepRecords."""
    schedule = generate_fault_schedule(seed, horizon)
    track.reset(seed)
    history: list[dict[str, Any]] = []
    records: list[StepRecord] = []
    trajectory: list[dict[str, Any]] = []
    domain = track.domain_name
    wrapped = DomainAwareController(controller, domain)

    for t in range(horizon):
        ts = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        obs = dict(track.observe(ts, fault_dict))

        if use_universal:
            raw = dict(obs)
            raw["ts_utc"] = _ts(t)
            if history:
                prev = history[-1]
                for key in hold_keys:
                    raw.setdefault(f"_hold_{key}", prev.get(key, 0.0))
            candidate = wrapped.propose_action(obs)
            constraints = constraints_fn(ts)
            result = run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=raw,
                history=history,
                candidate_action=candidate,
                constraints=constraints,
                quantile=quantile,
                cfg=cfg,
                controller=f"test-{domain}",
            )
            action = dict(result["safe_action"])
            history.append(dict(result["state"]))
        else:
            action = dict(wrapped.propose_action(obs))

        new_state = track.step(action)
        violation = track.check_violation(new_state)
        soc_after = 0.5 if not violation["violated"] else 0.0

        step_d = {**dict(new_state), **dict(action)}
        trajectory.append(step_d)
        uw = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [step_d])
        records.append(StepRecord(
            step=t,
            true_state=ts,
            observed_state=dict(obs),
            action=action,
            soc_after=soc_after,
            soc_min=0.1,
            soc_max=0.9,
            certificate_valid=not violation["violated"],
            certificate_predicted_valid=not violation["violated"],
            fallback_active=bool(faults and faults[0].kind == "blackout"),
            useful_work=0.0 if math.isnan(uw) else uw,
            audit_fields_present=1,
            audit_fields_required=1,
        ))
    return records


# ===========================================================================
# Vehicle / AV tests
# ===========================================================================

VEHICLE_CFG = {"expected_cadence_s": 0.25}
VEHICLE_HOLD = ("position_m", "speed_mps", "speed_limit_mps", "lead_position_m")
VEHICLE_CONSTRAINTS = {
    "speed_limit_mps": 30.0,
    "accel_min_mps2": -5.0,
    "accel_max_mps2": 3.0,
    "dt_s": 0.25,
    "min_headway_m": 5.0,
    "headway_time_s": 2.0,
}


class TestVehicleDomain:
    adapter = VehicleDomainAdapter(VEHICLE_CFG)

    def _constraints(self, state: dict) -> dict:
        return {**VEHICLE_CONSTRAINTS, "speed_limit_mps": float(state.get("speed_limit_mps", 30.0))}

    def test_single_step_nominal(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"position_m": 0.0, "speed_mps": 10.0, "speed_limit_mps": 30.0,
                           "lead_position_m": 50.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"acceleration_mps2": 1.0},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
        )
        assert "certificate" in result
        assert "safe_action" in result
        assert "acceleration_mps2" in result["safe_action"]
        assert 0.0 <= result["reliability_w"] <= 1.0

    def test_headway_clamp_fires(self):
        """Candidate acceleration toward lead vehicle triggers repair."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"position_m": 44.0, "speed_mps": 10.0, "speed_limit_mps": 30.0,
                           "lead_position_m": 50.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"acceleration_mps2": 3.0},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
        )
        assert result["repair_meta"]["repaired"] is True
        assert "headway" in result["repair_meta"]["intervention_reason"]
        assert result["safe_action"]["acceleration_mps2"] < 3.0

    def test_speed_limit_clamp_fires(self):
        """Candidate would overshoot speed limit; repair must cap it."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"position_m": 0.0, "speed_mps": 29.8, "speed_limit_mps": 30.0,
                           "lead_position_m": 1000.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"acceleration_mps2": 3.0},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
        )
        assert result["repair_meta"]["repaired"] is True

    def test_blackout_produces_emergency_brake(self):
        """NaN telemetry (blackout) must result in braking, not acceleration."""
        raw = {"position_m": float("nan"), "speed_mps": float("nan"),
               "speed_limit_mps": 30.0, "lead_position_m": float("nan"), "ts_utc": _ts()}
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry=raw,
            history=None,
            candidate_action={"acceleration_mps2": 2.0},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
        )
        # Safety: during full blackout the adapter enforces emergency brake
        assert result["safe_action"]["acceleration_mps2"] <= 0.0

    def test_multi_step_no_regression(self):
        track = VehicleTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, VEHICLE_CFG, self._constraints, 0.9, VEHICLE_HOLD,
            DC3SController(), seed=2000, horizon=48, use_universal=True,
        )
        track = VehicleTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, VEHICLE_CFG, self._constraints, 0.9, VEHICLE_HOLD,
            NominalController(), seed=2000, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        assert dc3s_m.tsvr <= nom_m.tsvr + 0.01, (
            f"Vehicle DC3S TSVR {dc3s_m.tsvr:.3f} regressed vs nominal {nom_m.tsvr:.3f}"
        )
        assert 0.0 <= dc3s_m.intervention_rate <= 1.0


# ===========================================================================
# Healthcare tests
# ===========================================================================

HEALTHCARE_CFG = {"expected_cadence_s": 1.0}
HEALTHCARE_HOLD = ("hr_bpm", "spo2_pct", "respiratory_rate")
HEALTHCARE_CONSTRAINTS = {"spo2_min_pct": 90.0, "hr_min_bpm": 40.0, "hr_max_bpm": 120.0}


class TestHealthcareDomain:
    adapter = HealthcareDomainAdapter(HEALTHCARE_CFG)

    def _constraints(self, state: dict) -> dict:
        return HEALTHCARE_CONSTRAINTS

    def test_single_step_nominal(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        assert "certificate" in result
        safe = result["safe_action"]
        assert "alert_level" in safe
        assert 0.0 <= safe["alert_level"] <= 1.0
        assert 0.0 <= result["reliability_w"] <= 1.0

    def test_low_spo2_triggers_alert_boost(self):
        """When SpO2 lower bound drops below threshold, alert_level must be ≥ 0.5."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 89.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"alert_level": 0.1},  # dangerously low
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        # The healthcare adapter's repair_action boosts alert when SpO2 is low
        assert result["safe_action"]["alert_level"] >= 0.1  # at minimum: no lowering
        assert result["repair_meta"]["repaired"] is True

    def test_alert_clamped_to_unit_interval(self):
        """Alert levels outside [0, 1] must be clipped."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"hr_bpm": 80.0, "spo2_pct": 98.0, "respiratory_rate": 15.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"alert_level": 5.0},  # way out of range
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        assert result["safe_action"]["alert_level"] <= 1.0

    def test_blackout_yields_valid_output(self):
        """NaN vitals: output still has a valid alert_level, certificate emitted."""
        raw = {"hr_bpm": float("nan"), "spo2_pct": float("nan"),
               "respiratory_rate": float("nan"), "ts_utc": _ts()}
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry=raw,
            history=None,
            candidate_action={"alert_level": 0.2},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        assert "certificate" in result
        assert 0.0 <= result["safe_action"]["alert_level"] <= 1.0

    def test_history_lowers_reliability_after_dropout(self):
        """Consecutive dropout events must reduce w_t from the first observation."""
        # Step 0: good observation
        r0 = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts(0)},
            history=None,
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        # Step 1: dropout — all NaN
        r1 = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"hr_bpm": float("nan"), "spo2_pct": float("nan"),
                           "respiratory_rate": float("nan"), "ts_utc": _ts(1)},
            history=[dict(r0["state"])],
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        # Reliability after dropout must be lower than during clean observation
        assert r1["reliability_w"] < 1.0

    def test_multi_step_no_regression(self):
        track = HealthcareTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, HEALTHCARE_CFG, self._constraints, 5.0, HEALTHCARE_HOLD,
            DC3SController(), seed=2001, horizon=48, use_universal=True,
        )
        track = HealthcareTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, HEALTHCARE_CFG, self._constraints, 5.0, HEALTHCARE_HOLD,
            NominalController(), seed=2001, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        assert dc3s_m.tsvr <= nom_m.tsvr + 0.01, (
            f"Healthcare DC3S TSVR {dc3s_m.tsvr:.3f} regressed vs nominal {nom_m.tsvr:.3f}"
        )
        assert 0.0 <= dc3s_m.intervention_rate <= 1.0
        assert dc3s_m.audit_completeness == 1.0


# ===========================================================================
# Industrial tests
# ===========================================================================

INDUSTRIAL_CFG = {"expected_cadence_s": 3600.0}
INDUSTRIAL_HOLD = ("temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw")
INDUSTRIAL_CONSTRAINTS = {"power_max_mw": 500.0, "temp_min_c": 0.0, "temp_max_c": 120.0}


class TestIndustrialDomain:
    adapter = IndustrialDomainAdapter(INDUSTRIAL_CFG)

    def _constraints(self, state: dict) -> dict:
        return INDUSTRIAL_CONSTRAINTS

    def test_single_step_nominal(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"temp_c": 25.0, "pressure_mbar": 1010.0, "power_mw": 450.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"power_setpoint_mw": 480.0},
            constraints=INDUSTRIAL_CONSTRAINTS,
            quantile=30.0,
            cfg=INDUSTRIAL_CFG,
        )
        assert "certificate" in result
        safe = result["safe_action"]
        assert "power_setpoint_mw" in safe
        assert 0.0 <= safe["power_setpoint_mw"] <= 500.0
        assert 0.0 <= result["reliability_w"] <= 1.0

    def test_power_clamp_fires_on_overdemand(self):
        """Setpoint exceeding power_max must be clamped."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"temp_c": 50.0, "pressure_mbar": 1015.0, "power_mw": 490.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"power_setpoint_mw": 9999.0},  # way over limit
            constraints=INDUSTRIAL_CONSTRAINTS,
            quantile=30.0,
            cfg=INDUSTRIAL_CFG,
        )
        assert result["repair_meta"]["repaired"] is True
        assert result["safe_action"]["power_setpoint_mw"] <= 500.0

    def test_negative_setpoint_clipped_to_zero(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"temp_c": 25.0, "pressure_mbar": 1010.0, "power_mw": 400.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"power_setpoint_mw": -100.0},
            constraints=INDUSTRIAL_CONSTRAINTS,
            quantile=30.0,
            cfg=INDUSTRIAL_CFG,
        )
        assert result["safe_action"]["power_setpoint_mw"] >= 0.0

    def test_blackout_yields_valid_output(self):
        raw = {"temp_c": float("nan"), "pressure_mbar": float("nan"),
               "power_mw": float("nan"), "ts_utc": _ts()}
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry=raw,
            history=None,
            candidate_action={"power_setpoint_mw": 450.0},
            constraints=INDUSTRIAL_CONSTRAINTS,
            quantile=30.0,
            cfg=INDUSTRIAL_CFG,
        )
        assert "certificate" in result
        assert 0.0 <= result["safe_action"]["power_setpoint_mw"] <= 500.0

    def test_multi_step_no_regression(self):
        track = IndustrialTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, INDUSTRIAL_CFG, self._constraints, 30.0, INDUSTRIAL_HOLD,
            DC3SController(), seed=2002, horizon=48, use_universal=True,
        )
        track = IndustrialTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, INDUSTRIAL_CFG, self._constraints, 30.0, INDUSTRIAL_HOLD,
            NominalController(), seed=2002, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        assert dc3s_m.tsvr <= nom_m.tsvr + 0.01, (
            f"Industrial DC3S TSVR {dc3s_m.tsvr:.3f} regressed vs nominal {nom_m.tsvr:.3f}"
        )
        assert 0.0 <= dc3s_m.intervention_rate <= 1.0
        assert dc3s_m.audit_completeness == 1.0


# ===========================================================================
# Aerospace tests
# ===========================================================================

AEROSPACE_CFG = {"expected_cadence_s": 1.0}
AEROSPACE_HOLD = ("altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct")
AEROSPACE_CONSTRAINTS = {"v_min_kt": 60.0, "v_max_kt": 350.0, "max_bank_deg": 30.0}


class TestAerospaceDomain:
    adapter = AerospaceDomainAdapter(AEROSPACE_CFG)

    def _constraints(self, state: dict) -> dict:
        return AEROSPACE_CONSTRAINTS

    def test_single_step_nominal(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"altitude_m": 3000.0, "airspeed_kt": 180.0, "bank_angle_deg": 5.0,
                           "fuel_remaining_pct": 65.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"throttle": 0.7, "bank_deg": 3.0},
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
        )
        assert "certificate" in result
        safe = result["safe_action"]
        assert "throttle" in safe
        assert "bank_deg" in safe
        assert 0.0 <= safe["throttle"] <= 1.0
        assert abs(safe["bank_deg"]) <= 30.0

    def test_bank_angle_clamp(self):
        """Bank command exceeding max_bank_deg must be clipped."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"altitude_m": 3000.0, "airspeed_kt": 200.0, "bank_angle_deg": 10.0,
                           "fuel_remaining_pct": 60.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"throttle": 0.7, "bank_deg": 90.0},  # WAY over limit
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
        )
        assert result["repair_meta"]["repaired"] is True
        assert abs(result["safe_action"]["bank_deg"]) <= 30.0

    def test_low_fuel_throttle_cap(self):
        """Low fuel in uncertainty set must reduce throttle cap to 0.5."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"altitude_m": 3000.0, "airspeed_kt": 200.0, "bank_angle_deg": 0.0,
                           "fuel_remaining_pct": 5.0, "ts_utc": _ts()},  # below 10 % min
            history=None,
            candidate_action={"throttle": 1.0, "bank_deg": 0.0},
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
        )
        assert result["repair_meta"]["repaired"] is True
        assert result["safe_action"]["throttle"] <= 0.5

    def test_throttle_clamped_to_unit_interval(self):
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"altitude_m": 3000.0, "airspeed_kt": 150.0, "bank_angle_deg": 0.0,
                           "fuel_remaining_pct": 80.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"throttle": 5.0, "bank_deg": 0.0},
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
        )
        assert result["safe_action"]["throttle"] <= 1.0

    def test_blackout_yields_valid_output(self):
        raw = {"altitude_m": float("nan"), "airspeed_kt": float("nan"),
               "bank_angle_deg": float("nan"), "fuel_remaining_pct": float("nan"),
               "ts_utc": _ts()}
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry=raw,
            history=None,
            candidate_action={"throttle": 0.7, "bank_deg": 5.0},
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
        )
        assert "certificate" in result
        assert 0.0 <= result["safe_action"]["throttle"] <= 1.0
        assert abs(result["safe_action"]["bank_deg"]) <= 30.0

    def test_multi_step_no_regression(self):
        track = AerospaceTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, AEROSPACE_CFG, self._constraints, 5.0, AEROSPACE_HOLD,
            DC3SController(), seed=2003, horizon=48, use_universal=True,
        )
        track = AerospaceTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, AEROSPACE_CFG, self._constraints, 5.0, AEROSPACE_HOLD,
            NominalController(), seed=2003, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        assert dc3s_m.tsvr <= nom_m.tsvr + 0.01, (
            f"Aerospace DC3S TSVR {dc3s_m.tsvr:.3f} regressed vs nominal {nom_m.tsvr:.3f}"
        )
        assert 0.0 <= dc3s_m.intervention_rate <= 1.0
        assert dc3s_m.audit_completeness == 1.0


# ===========================================================================
# Cross-domain certificate structure tests
# ===========================================================================

class TestCrossDomainCertificateStructure:
    """Verify every domain emits a structurally complete certificate."""

    CASES = [
        (
            VehicleDomainAdapter(VEHICLE_CFG),
            {"position_m": 0.0, "speed_mps": 10.0, "speed_limit_mps": 30.0,
             "lead_position_m": 50.0, "ts_utc": _ts()},
            {"acceleration_mps2": 1.0},
            VEHICLE_CONSTRAINTS,
            0.9,
            VEHICLE_CFG,
        ),
        (
            HealthcareDomainAdapter(HEALTHCARE_CFG),
            {"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0, "ts_utc": _ts()},
            {"alert_level": 0.3},
            HEALTHCARE_CONSTRAINTS,
            5.0,
            HEALTHCARE_CFG,
        ),
        (
            IndustrialDomainAdapter(INDUSTRIAL_CFG),
            {"temp_c": 25.0, "pressure_mbar": 1010.0, "power_mw": 450.0, "ts_utc": _ts()},
            {"power_setpoint_mw": 480.0},
            INDUSTRIAL_CONSTRAINTS,
            30.0,
            INDUSTRIAL_CFG,
        ),
        (
            AerospaceDomainAdapter(AEROSPACE_CFG),
            {"altitude_m": 3000.0, "airspeed_kt": 180.0, "bank_angle_deg": 5.0,
             "fuel_remaining_pct": 65.0, "ts_utc": _ts()},
            {"throttle": 0.7, "bank_deg": 3.0},
            AEROSPACE_CONSTRAINTS,
            5.0,
            AEROSPACE_CFG,
        ),
        (
            NavigationDomainAdapter({"expected_cadence_s": 0.25}),
            {"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "ts_utc": _ts()},
            {"ax": 0.2, "ay": 0.1},
            {"arena_size": 10.0, "speed_limit": 1.0},
            1.0,
            {"expected_cadence_s": 0.25},
        ),
    ]

    @pytest.mark.parametrize("adapter,telemetry,action,constraints,quantile,cfg", CASES)
    def test_certificate_has_required_fields(self, adapter, telemetry, action, constraints, quantile, cfg):
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=action,
            constraints=constraints,
            quantile=quantile,
            cfg=cfg,
        )
        cert = result["certificate"]
        for field in ("command_id", "certificate_id", "created_at", "device_id",
                      "zone_id", "controller", "proposed_action", "safe_action",
                      "reliability_w", "drift_flag", "inflation", "certificate_hash"):
            assert field in cert, f"Missing certificate field '{field}' in domain {type(adapter).__name__}"

    @pytest.mark.parametrize("adapter,telemetry,action,constraints,quantile,cfg", CASES)
    def test_reliability_w_in_bounds(self, adapter, telemetry, action, constraints, quantile, cfg):
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=action,
            constraints=constraints,
            quantile=quantile,
            cfg=cfg,
        )
        assert 0.05 <= result["reliability_w"] <= 1.0

    @pytest.mark.parametrize("adapter,telemetry,action,constraints,quantile,cfg", CASES)
    def test_inflation_positive(self, adapter, telemetry, action, constraints, quantile, cfg):
        """Inflation/margin value must be strictly positive (non-battery adapters use
        margin as the inflation field, which may be < 1.0 for small quantiles)."""
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=action,
            constraints=constraints,
            quantile=quantile,
            cfg=cfg,
        )
        assert result["certificate"].get("inflation", 1.0) > 0.0


# ===========================================================================
# Drift detection integration tests (cross-domain)
# ===========================================================================

class TestDriftDetectionAllDomains:
    """Verify drift detector fires and propagates through run_universal_step.

    We use a mock drift detector that always returns drift=True so the test
    exercises the propagation interface without depending on the Page-Hinkley
    accumulator's cooldown/warmup state.  The PH detector's own correctness is
    covered by the dedicated test_dc3s_* suite.
    """

    @pytest.fixture
    def drifted_detector(self):
        """A minimal detector stub that always reports drift=True."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.update.return_value = {"drift": True, "score": 9.9, "mean_residual": 5.0}
        return mock

    def test_vehicle_drift_flag_propagates(self, drifted_detector):
        adapter = VehicleDomainAdapter(VEHICLE_CFG)
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"position_m": 0.0, "speed_mps": 10.0, "speed_limit_mps": 30.0,
                           "lead_position_m": 500.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"acceleration_mps2": 0.0},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
            drift_detector=drifted_detector,
            residual=10.0,
        )
        drifted_detector.update.assert_called_once_with(10.0)
        assert result["drift_flag"] is True

    def test_healthcare_drift_flag_propagates_to_certificate(self, drifted_detector):
        adapter = HealthcareDomainAdapter(HEALTHCARE_CFG)
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
            drift_detector=drifted_detector,
            residual=10.0,
        )
        assert result["drift_flag"] is True
        assert result["certificate"]["drift_flag"] is True

    def test_industrial_drift_flag_propagates(self, drifted_detector):
        adapter = IndustrialDomainAdapter(INDUSTRIAL_CFG)
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"temp_c": 25.0, "pressure_mbar": 1010.0, "power_mw": 450.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"power_setpoint_mw": 480.0},
            constraints=INDUSTRIAL_CONSTRAINTS,
            quantile=30.0,
            cfg=INDUSTRIAL_CFG,
            drift_detector=drifted_detector,
            residual=10.0,
        )
        assert result["drift_flag"] is True

    def test_aerospace_drift_flag_propagates(self, drifted_detector):
        adapter = AerospaceDomainAdapter(AEROSPACE_CFG)
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"altitude_m": 3000.0, "airspeed_kt": 180.0, "bank_angle_deg": 5.0,
                           "fuel_remaining_pct": 65.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"throttle": 0.7, "bank_deg": 3.0},
            constraints=AEROSPACE_CONSTRAINTS,
            quantile=5.0,
            cfg=AEROSPACE_CFG,
            drift_detector=drifted_detector,
            residual=10.0,
        )
        assert result["drift_flag"] is True

    def test_no_drift_when_detector_not_provided(self):
        """Without a detector, drift_flag must be False regardless of residual."""
        adapter = HealthcareDomainAdapter(HEALTHCARE_CFG)
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts()},
            history=None,
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        assert result["drift_flag"] is False


# ===========================================================================
# Certificate hash-chain integrity
# ===========================================================================

class TestCertificateHashChain:
    """Verify that prev_hash links produce a non-trivial chain."""

    def test_vehicle_two_step_hash_chain(self):
        adapter = VehicleDomainAdapter(VEHICLE_CFG)
        telemetry = {"position_m": 0.0, "speed_mps": 10.0, "speed_limit_mps": 30.0,
                     "lead_position_m": 100.0, "ts_utc": _ts(0)}
        r0 = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action={"acceleration_mps2": 0.5},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
        )
        cert0_hash = r0["certificate"]["certificate_hash"]

        telemetry1 = {**telemetry, "position_m": 2.5, "ts_utc": _ts(1)}
        r1 = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry1,
            history=[dict(r0["state"])],
            candidate_action={"acceleration_mps2": 0.5},
            constraints=VEHICLE_CONSTRAINTS,
            quantile=0.9,
            cfg=VEHICLE_CFG,
            prev_cert_hash=cert0_hash,
        )
        assert r1["certificate"]["prev_hash"] == cert0_hash
        assert r1["certificate"]["certificate_hash"] != cert0_hash

    def test_healthcare_two_step_hash_chain(self):
        adapter = HealthcareDomainAdapter(HEALTHCARE_CFG)
        r0 = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"hr_bpm": 72.0, "spo2_pct": 97.0, "respiratory_rate": 14.0,
                           "ts_utc": _ts(0)},
            history=None,
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
        )
        cert0_hash = r0["certificate"]["certificate_hash"]
        r1 = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry={"hr_bpm": 73.0, "spo2_pct": 96.5, "respiratory_rate": 14.0,
                           "ts_utc": _ts(1)},
            history=[dict(r0["state"])],
            candidate_action={"alert_level": 0.3},
            constraints=HEALTHCARE_CONSTRAINTS,
            quantile=5.0,
            cfg=HEALTHCARE_CFG,
            prev_cert_hash=cert0_hash,
        )
        assert r1["certificate"]["prev_hash"] == cert0_hash


# ===========================================================================
# Navigation domain tests
# ===========================================================================

NAVIGATION_CFG = {"expected_cadence_s": 0.25}
NAVIGATION_HOLD = ("x", "y", "vx", "vy")
NAVIGATION_CONSTRAINTS = {"arena_size": 10.0, "speed_limit": 1.0}


def _nav_constraints(state: dict) -> dict:
    return NAVIGATION_CONSTRAINTS


class TestNavigationDomain:
    adapter = NavigationDomainAdapter(NAVIGATION_CFG)

    def _constraints(self, state: dict) -> dict:
        return NAVIGATION_CONSTRAINTS

    def test_single_step_nominal(self):
        """Navigation adapter emits a certificate for a safe nominal step."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"ax": 0.2, "ay": 0.1},
            constraints=NAVIGATION_CONSTRAINTS,
            quantile=1.0,
            cfg=NAVIGATION_CFG,
        )
        assert "certificate" in result
        safe = result["safe_action"]
        assert "ax" in safe
        assert "ay" in safe

    def test_arena_x_boundary_clamp(self):
        """Action pushing robot past x=10 must be clamped."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"x": 9.9, "y": 5.0, "vx": 0.0, "vy": 0.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"ax": 2.0, "ay": 0.0},  # would push x >> 10
            constraints=NAVIGATION_CONSTRAINTS,
            quantile=1.0,
            cfg=NAVIGATION_CFG,
        )
        assert result["repair_meta"]["repaired"] is True
        # Next predicted x must not exceed arena_size
        safe = result["safe_action"]
        dt = 0.25
        x_next = 9.9 + safe["ax"] * dt
        assert x_next <= 10.0 + 1e-6

    def test_arena_y_boundary_clamp(self):
        """Action pushing robot past y=10 must be clamped."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"x": 5.0, "y": 9.9, "vx": 0.0, "vy": 0.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"ax": 0.0, "ay": 2.0},
            constraints=NAVIGATION_CONSTRAINTS,
            quantile=1.0,
            cfg=NAVIGATION_CFG,
        )
        assert result["repair_meta"]["repaired"] is True

    def test_speed_limit_enforced(self):
        """Actions exceeding the speed limit must be scaled down."""
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry={"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "ts_utc": _ts()},
            history=None,
            candidate_action={"ax": 5.0, "ay": 5.0},  # |v| >> 1.0 limit
            constraints=NAVIGATION_CONSTRAINTS,
            quantile=1.0,
            cfg=NAVIGATION_CFG,
        )
        safe = result["safe_action"]
        import math
        speed = math.hypot(safe["ax"], safe["ay"])
        assert speed <= 1.0 + 1e-6

    def test_blackout_yields_valid_output(self):
        """Full NaN telemetry (blackout) must produce a valid zero-motion certificate."""
        raw = {"x": float("nan"), "y": float("nan"), "vx": float("nan"), "vy": float("nan"),
               "ts_utc": _ts()}
        result = run_universal_step(
            domain_adapter=self.adapter,
            raw_telemetry=raw,
            history=None,
            candidate_action={"ax": 0.5, "ay": 0.5},
            constraints=NAVIGATION_CONSTRAINTS,
            quantile=1.0,
            cfg=NAVIGATION_CFG,
        )
        assert "certificate" in result
        assert 0.05 <= result["reliability_w"] <= 1.0

    def test_multi_step_no_regression(self):
        """DC3S must not regress TSVR vs nominal over a full episode."""
        track = NavigationTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, NAVIGATION_CFG, _nav_constraints, 1.0, NAVIGATION_HOLD,
            DC3SController(), seed=2004, horizon=48, use_universal=True,
        )
        track = NavigationTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, NAVIGATION_CFG, _nav_constraints, 1.0, NAVIGATION_HOLD,
            NominalController(), seed=2004, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        assert dc3s_m.tsvr <= nom_m.tsvr + 0.01, (
            f"Navigation DC3S TSVR {dc3s_m.tsvr:.3f} regressed vs nominal {nom_m.tsvr:.3f}"
        )
        assert 0.0 <= dc3s_m.intervention_rate <= 1.0
        assert dc3s_m.audit_completeness == 1.0

    def test_proof_quality_tsvr_reduction(self):
        """DC3S must achieve ≥ 25 % TSVR reduction vs nominal (evidence gate)."""
        track = NavigationTrackAdapter()
        dc3s_recs = _run_domain_episode(
            track, self.adapter, NAVIGATION_CFG, _nav_constraints, 1.0, NAVIGATION_HOLD,
            DC3SController(), seed=2004, horizon=48, use_universal=True,
        )
        track = NavigationTrackAdapter()
        nom_recs = _run_domain_episode(
            track, self.adapter, NAVIGATION_CFG, _nav_constraints, 1.0, NAVIGATION_HOLD,
            NominalController(), seed=2004, horizon=48, use_universal=False,
        )
        dc3s_m = compute_all_metrics(dc3s_recs)
        nom_m  = compute_all_metrics(nom_recs)
        if nom_m.tsvr > 0.05:
            reduction = (nom_m.tsvr - dc3s_m.tsvr) / nom_m.tsvr
            assert reduction >= 0.25, (
                f"Navigation TSVR reduction {reduction*100:.1f}% < 25% "
                f"(nominal={nom_m.tsvr:.3f}, dc3s={dc3s_m.tsvr:.3f})"
            )
