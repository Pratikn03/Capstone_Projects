"""ORIUS-Bench tests (Paper 4).

Covers:
- Fault engine determinism and replay
- Metrics computation correctness
- Battery track adapter state transitions
- Navigation track adapter safety violations
- Controller API contract
- Leaderboard export schema
- Rank reversal: naive vs robust across domains
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from orius.orius_bench.adapter import BenchmarkAdapter
from orius.orius_bench.battery_track import BatteryTrackAdapter
from orius.orius_bench.controller_api import (
    DC3SController,
    FallbackController,
    NaiveController,
    NominalController,
    RobustController,
)
from orius.orius_bench.export import (
    controller_contract_schema,
    fault_schema,
    leaderboard_row,
    metrics_schema,
    write_bundle_json,
    write_leaderboard_csv,
    write_schemas,
)
from orius.orius_bench.fault_engine import (
    FaultEvent,
    FaultSchedule,
    active_faults,
    apply_faults,
    generate_fault_schedule,
    replay_schedule,
)
from orius.orius_bench.metrics_engine import (
    BenchmarkMetrics,
    StepRecord,
    compute_all_metrics,
    compute_audit_completeness,
    compute_cva,
    compute_gdq,
    compute_intervention_rate,
    compute_recovery_latency,
    compute_tsvr,
)
from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter
from orius.orius_bench.healthcare_track import HealthcareTrackAdapter
from orius.orius_bench.aerospace_track import AerospaceTrackAdapter
from orius.orius_bench.vehicle_track import VehicleTrackAdapter


# ── Fault Engine ──────────────────────────────────────────────────────

class TestFaultEngineDeterminism:
    def test_same_seed_same_schedule(self):
        s1 = generate_fault_schedule(42, horizon=96)
        s2 = generate_fault_schedule(42, horizon=96)
        assert s1.digest == s2.digest
        assert len(s1.events) == len(s2.events)
        for a, b in zip(s1.events, s2.events):
            assert a.step == b.step
            assert a.kind == b.kind

    def test_different_seed_different_schedule(self):
        s1 = generate_fault_schedule(42, horizon=96)
        s2 = generate_fault_schedule(99, horizon=96)
        assert s1.digest != s2.digest

    def test_replay_alias(self):
        s1 = generate_fault_schedule(123, horizon=48)
        s2 = replay_schedule(123, horizon=48)
        assert s1.digest == s2.digest

    def test_active_faults_at_step(self):
        sched = FaultSchedule(
            seed=0,
            events=[
                FaultEvent(step=5, kind="blackout", duration=3),
                FaultEvent(step=10, kind="bias", params={"magnitude": 2.0}),
            ],
            horizon=20,
        )
        assert len(active_faults(sched, 4)) == 0
        assert len(active_faults(sched, 5)) == 1
        assert active_faults(sched, 5)[0].kind == "blackout"
        assert len(active_faults(sched, 7)) == 1  # step 5+3-1=7
        assert len(active_faults(sched, 8)) == 0
        assert len(active_faults(sched, 10)) == 1

    def test_apply_blackout_produces_nan(self):
        state = {"soc": 0.5, "voltage": 3.7}
        faults = [FaultEvent(step=0, kind="blackout", duration=5)]
        obs = apply_faults(state, faults)
        assert all(math.isnan(v) for v in obs.values())

    def test_apply_bias(self):
        state = {"soc": 0.5}
        faults = [FaultEvent(step=0, kind="bias", params={"magnitude": 0.1})]
        obs = apply_faults(state, faults)
        assert abs(obs["soc"] - 0.6) < 1e-9


# ── Metrics Engine ────────────────────────────────────────────────────

class TestMetricsEngine:
    @staticmethod
    def _make_records(n=20, soc=0.5, violated=False, fallback=False):
        return [
            StepRecord(
                step=i,
                true_state={"soc": soc},
                observed_state={"soc": soc},
                action={"charge_mw": 0, "discharge_mw": 50},
                soc_after=soc,
                certificate_valid=not violated,
                certificate_predicted_valid=not violated,
                fallback_active=fallback,
                useful_work=1.0,
                audit_fields_present=1,
                audit_fields_required=1,
            )
            for i in range(n)
        ]

    def test_tsvr_no_violations(self):
        recs = self._make_records(20, soc=0.5)
        assert compute_tsvr(recs) == 0.0

    def test_tsvr_all_violations(self):
        recs = self._make_records(10, soc=0.05)  # below min 0.1
        assert compute_tsvr(recs) == 1.0

    def test_cva_perfect(self):
        recs = self._make_records(10)
        assert compute_cva(recs) == 1.0

    def test_gdq_no_fallback(self):
        recs = self._make_records(10)
        gdq = compute_gdq(recs)
        assert gdq > 0

    def test_intervention_rate_zero(self):
        recs = self._make_records(10)
        assert compute_intervention_rate(recs) == 0.0

    def test_intervention_rate_all_fallback(self):
        recs = self._make_records(10, fallback=True)
        assert compute_intervention_rate(recs) == 1.0

    def test_audit_completeness_full(self):
        recs = self._make_records(10)
        assert compute_audit_completeness(recs) == 1.0

    def test_recovery_latency_no_expiry(self):
        recs = self._make_records(10)
        assert compute_recovery_latency(recs) == 0.0

    def test_compute_all_returns_dataclass(self):
        recs = self._make_records(10)
        m = compute_all_metrics(recs)
        assert isinstance(m, BenchmarkMetrics)
        assert m.n_steps == 10


# ── Battery Track ─────────────────────────────────────────────────────

class TestBatteryTrack:
    def test_reset_returns_state(self):
        bt = BatteryTrackAdapter()
        s = bt.reset(42)
        assert "soc" in s
        assert 0.4 < s["soc"] < 0.6  # starts at 50%

    def test_step_changes_soc(self):
        bt = BatteryTrackAdapter()
        bt.reset(42)
        s1 = bt.true_state()
        s2 = bt.step({"charge_mw": 0, "discharge_mw": 50})
        assert s2["soc"] < s1["soc"]

    def test_violation_below_min(self):
        bt = BatteryTrackAdapter()
        v = bt.check_violation({"soc": 0.05})
        assert v["violated"]
        assert v["severity"] > 0

    def test_observe_with_blackout(self):
        bt = BatteryTrackAdapter()
        bt.reset(42)
        ts = bt.true_state()
        obs = bt.observe(ts, {"kind": "blackout"})
        assert all(math.isnan(v) for v in obs.values())

    def test_domain_name(self):
        bt = BatteryTrackAdapter()
        assert bt.domain_name == "battery"


# ── Navigation Track ──────────────────────────────────────────────────

class TestNavigationTrack:
    def test_reset_and_state(self):
        nt = NavigationTrackAdapter()
        s = nt.reset(42)
        assert "x" in s and "y" in s

    def test_step_moves_robot(self):
        nt = NavigationTrackAdapter()
        nt.reset(42)
        s1 = nt.true_state()
        s2 = nt.step({"ax": 2.0, "ay": 0.0})
        assert s2["x"] != s1["x"]

    def test_obstacle_violation(self):
        nt = NavigationTrackAdapter(obstacle_centres=[(1.0, 1.0)], obstacle_radius=0.5)
        v = nt.check_violation({"x": 1.0, "y": 1.0})
        assert v["violated"]

    def test_arena_bounds_violation(self):
        nt = NavigationTrackAdapter(arena_size=10.0)
        v = nt.check_violation({"x": -1.0, "y": 5.0})
        assert v["violated"]
        assert v["severity"] > 0

    def test_domain_name(self):
        nt = NavigationTrackAdapter()
        assert nt.domain_name == "navigation"


# ── Controller API ────────────────────────────────────────────────────

class TestControllerAPI:
    def test_nominal_returns_action(self):
        c = NominalController()
        a = c.propose_action({"soc": 0.5})
        assert "discharge_mw" in a
        assert c.name == "nominal"

    def test_robust_reduces_action(self):
        c = RobustController(safety_factor=0.5)
        a = c.propose_action({"soc": 0.5})
        assert a["discharge_mw"] <= 50.0

    def test_dc3s_fallback_on_cert_invalid(self):
        c = DC3SController()
        a = c.propose_action({"soc": 0.5}, certificate_state={"fallback_required": True})
        assert a["discharge_mw"] == 0.0

    def test_fallback_decays(self):
        c = FallbackController()
        a1 = c.propose_action({"soc": 0.5})
        a2 = c.propose_action({"soc": 0.5})
        assert a2["discharge_mw"] <= a1["discharge_mw"]

    def test_naive_always_max(self):
        c = NaiveController(max_discharge=200.0)
        a = c.propose_action({"soc": 0.5})
        assert a["discharge_mw"] == 200.0


# ── Export ────────────────────────────────────────────────────────────

class TestExport:
    def test_leaderboard_csv_roundtrip(self):
        m = BenchmarkMetrics(
            tsvr=0.05, oasg=0.9, cva=0.95, gdq=0.8,
            intervention_rate=0.1, audit_completeness=1.0,
            recovery_latency=2.0, n_steps=96,
        )
        row = leaderboard_row("test_ctrl", "battery", 42, m)
        with tempfile.TemporaryDirectory() as d:
            p = write_leaderboard_csv([row], Path(d) / "lb.csv")
            assert p.exists()
            lines = p.read_text().strip().split("\n")
            assert len(lines) == 2  # header + 1 row

    def test_bundle_json_valid(self):
        m = BenchmarkMetrics(
            tsvr=0.0, oasg=1.0, cva=1.0, gdq=1.0,
            intervention_rate=0.0, audit_completeness=1.0,
            recovery_latency=0.0, n_steps=96,
        )
        row = leaderboard_row("ctrl", "battery", 42, m)
        with tempfile.TemporaryDirectory() as d:
            p = write_bundle_json([row], {42: "abc123"}, Path(d) / "bundle.json")
            bundle = json.loads(p.read_text())
            assert "results" in bundle
            assert "fault_digests" in bundle

    def test_schemas_written(self):
        with tempfile.TemporaryDirectory() as d:
            write_schemas(d)
            assert (Path(d) / "controller_contract.json").exists()
            assert (Path(d) / "fault_schema.json").exists()
            assert (Path(d) / "metrics_schema.json").exists()


# ── Rank Reversal ─────────────────────────────────────────────────────

class TestRankReversal:
    """Naive controller should have higher TSVR than robust on battery."""

    def _run_episode(self, track, ctrl, seed=42, horizon=48):
        from orius.orius_bench.fault_engine import active_faults, generate_fault_schedule

        schedule = generate_fault_schedule(seed, horizon)
        track.reset(seed)
        records = []
        for t in range(horizon):
            ts = track.true_state()
            faults = active_faults(schedule, t)
            fault_dict = None
            if faults:
                fault_dict = {"kind": faults[0].kind, **faults[0].params}
            obs = track.observe(ts, fault_dict)
            action = ctrl.propose_action(obs)
            new_state = track.step(action)
            v = track.check_violation(new_state)
            soc_after = new_state.get("soc", 0.5)
            records.append(
                StepRecord(
                    step=t, true_state=dict(ts), observed_state=dict(obs),
                    action=dict(action), soc_after=soc_after,
                    certificate_valid=not v["violated"],
                    certificate_predicted_valid=not v["violated"],
                    useful_work=1.0,
                    audit_fields_present=1, audit_fields_required=1,
                )
            )
        return compute_all_metrics(records)

    def test_naive_worse_than_robust_on_battery(self):
        bt = BatteryTrackAdapter()
        m_naive = self._run_episode(bt, NaiveController(max_discharge=200.0))
        m_robust = self._run_episode(bt, RobustController(safety_factor=0.3))
        # Naive should cause more violations
        assert m_naive.tsvr >= m_robust.tsvr


# ── Universal Tracks (Gap A, C) ────────────────────────────────────────

class TestUniversalTracks:
    """Industrial, Healthcare, Aerospace, Vehicle tracks run and produce metrics."""

    @pytest.mark.parametrize("track", [
        IndustrialTrackAdapter(),
        HealthcareTrackAdapter(),
        AerospaceTrackAdapter(),
        VehicleTrackAdapter(),
    ])
    def test_track_reset_step_violation(self, track):
        state = track.reset(seed=42)
        assert track.true_state() == state
        obs = track.observe(state, None)
        # Each adapter uses action.get(key, default) so empty dict uses safe defaults
        new_state = track.step({})
        v = track.check_violation(new_state)
        assert "violated" in v
        assert "severity" in v

    def test_all_tracks_have_domain_name(self):
        tracks = [
            BatteryTrackAdapter(),
            NavigationTrackAdapter(),
            IndustrialTrackAdapter(),
            HealthcareTrackAdapter(),
            AerospaceTrackAdapter(),
            VehicleTrackAdapter(),
        ]
        names = [t.domain_name for t in tracks]
        assert names == ["battery", "navigation", "industrial", "healthcare", "aerospace", "vehicle"]
