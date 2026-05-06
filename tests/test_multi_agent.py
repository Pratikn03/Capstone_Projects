"""Tests for multi-agent shared-constraint protocol (Paper 5).

Covers:
- SharedFeederPlant physics + joint violation detection
- Three coordination protocols
- Non-composition counterexample (independent violates, centralised does not)
- Margin allocation schemes + fairness
- Distributed negotiation converges
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from orius.multi_agent.margin_allocation import (
    allocate_margins,
    allocate_margins_fairness,
)
from orius.multi_agent.plant import SharedFeederPlant
from orius.multi_agent.protocol import (
    CentralizedCoordinatorProtocol,
    DistributedNegotiationProtocol,
    IndependentLocalProtocol,
)
from orius.multi_agent.scenarios import run_transformer_capacity_scenario

_TWO_BATTERIES = [
    {"capacity_mwh": 100.0, "initial_soc_frac": 0.5},
    {"capacity_mwh": 100.0, "initial_soc_frac": 0.5},
]


class TestSharedFeederPlant:
    def test_reset_returns_state(self):
        p = SharedFeederPlant(batteries=_TWO_BATTERIES, feeder_capacity_mw=80.0)
        state = p.reset()
        assert "socs_mwh" in state
        assert len(state["socs_mwh"]) == 2

    def test_step_changes_soc(self):
        p = SharedFeederPlant(batteries=_TWO_BATTERIES, feeder_capacity_mw=80.0)
        p.reset()
        s0 = list(p.state()["socs_mwh"])
        actions = [{"charge_mw": 0, "discharge_mw": 30} for _ in range(2)]
        p.step(actions)
        s1 = list(p.state()["socs_mwh"])
        assert all(s1[i] < s0[i] for i in range(2))

    def test_joint_violation_when_exceeding_feeder(self):
        p = SharedFeederPlant(batteries=_TWO_BATTERIES, feeder_capacity_mw=80.0)
        p.reset()
        actions = [{"charge_mw": 0, "discharge_mw": 60} for _ in range(2)]
        v = p.check_joint_violation(actions)
        assert v["violated"]

    def test_no_joint_violation_within_limits(self):
        p = SharedFeederPlant(batteries=_TWO_BATTERIES, feeder_capacity_mw=80.0)
        p.reset()
        actions = [{"charge_mw": 0, "discharge_mw": 30} for _ in range(2)]
        v = p.check_joint_violation(actions)
        assert not v["violated"]


class TestProtocols:
    def test_independent_passes_through(self):
        proto = IndependentLocalProtocol()
        actions = [{"charge_mw": 0, "discharge_mw": 60} for _ in range(2)]
        result = proto.compute_actions({}, actions, feeder_capacity_mw=80.0)
        # Independent should not modify actions
        for r, a in zip(result, actions, strict=False):
            assert r["discharge_mw"] == a["discharge_mw"]

    def test_centralized_respects_feeder(self):
        proto = CentralizedCoordinatorProtocol()
        actions = [{"charge_mw": 0, "discharge_mw": 60} for _ in range(2)]
        result = proto.compute_actions({}, actions, feeder_capacity_mw=80.0)
        total = sum(r["discharge_mw"] for r in result)
        assert total <= 80.0 + 1e-9

    def test_distributed_respects_feeder(self):
        proto = DistributedNegotiationProtocol()
        actions = [{"charge_mw": 0, "discharge_mw": 60} for _ in range(2)]
        result = proto.compute_actions({}, actions, feeder_capacity_mw=80.0)
        total = sum(r["discharge_mw"] for r in result)
        assert total <= 80.0 + 1e-9


class TestMarginAllocation:
    def test_equal_allocation(self):
        alloc = allocate_margins(80.0, 3, scheme="equal")
        assert len(alloc) == 3
        assert abs(sum(alloc) - 80.0) < 1e-9

    def test_proportional_allocation(self):
        alloc = allocate_margins(100.0, 2, scheme="proportional", weights=[1, 3])
        assert abs(alloc[0] - 25.0) < 1e-9
        assert abs(alloc[1] - 75.0) < 1e-9

    def test_demand_allocation(self):
        alloc = allocate_margins(80.0, 2, scheme="demand", demands=[60.0, 60.0])
        assert abs(sum(alloc) - 80.0) < 1e-9
        # Should be equal since demands are equal
        assert abs(alloc[0] - alloc[1]) < 1e-9

    def test_fairness_perfect(self):
        f = allocate_margins_fairness([40.0, 40.0], [40.0, 40.0])
        assert abs(f - 1.0) < 1e-6

    def test_fairness_decreases_with_imbalance(self):
        f1 = allocate_margins_fairness([40.0, 40.0], [40.0, 40.0])
        f2 = allocate_margins_fairness([10.0, 70.0], [40.0, 40.0])
        assert f2 < f1


class TestNonCompositionCounterexample:
    def test_independent_violates_centralized_does_not(self):
        """Core Paper 5 result: local certificates do NOT compose."""
        plant = SharedFeederPlant(batteries=_TWO_BATTERIES, feeder_capacity_mw=80.0)
        plant.reset()
        proposals = [{"charge_mw": 0, "discharge_mw": 60} for _ in range(2)]

        # Independent: total 120 MW > 80 MW feeder → violation
        ind = IndependentLocalProtocol()
        ind_actions = ind.compute_actions({}, proposals, feeder_capacity_mw=80.0)
        v_ind = plant.check_joint_violation(ind_actions)
        assert v_ind["violated"]

        # Centralized: scales down to 80 MW → safe
        cent = CentralizedCoordinatorProtocol()
        cent_actions = cent.compute_actions({}, proposals, feeder_capacity_mw=80.0)
        v_cent = plant.check_joint_violation(cent_actions)
        assert not v_cent["violated"]

    def test_scenario_runner_produces_results(self):
        with tempfile.TemporaryDirectory() as d:
            results = run_transformer_capacity_scenario(out_dir=d)
            assert "independent" in results
            assert "centralized" in results
            assert results["independent"]["joint_violations"] > 0
            assert results["centralized"]["joint_violations"] == 0

    def test_scenario_writes_step_log_when_out_dir_set(self):
        """Multi-agent step logs exist when out_dir is set."""
        with tempfile.TemporaryDirectory() as d:
            run_transformer_capacity_scenario(out_dir=d, n_steps=6)
            step_log = Path(d) / "multi_agent_step_log.csv"
            assert step_log.exists()
            rows = step_log.read_text().strip().split("\n")
            assert len(rows) >= 2  # header + at least one row
            assert "step" in rows[0] and "protocol" in rows[0]
            # 6 steps * 3 protocols = 18 rows
            assert len(rows) == 19  # header + 18

    def test_heterogeneous_degradation_changes_output(self):
        """Heterogeneous per-agent degradation produces different joint_violations."""
        r_none = run_transformer_capacity_scenario(out_dir=None)
        # [0.5, 0.5]: both propose 30 MW, total 60 < 80 → no violation
        r_deg = run_transformer_capacity_scenario(out_dir=None, agent_degradation=[0.5, 0.5])
        assert r_none["independent"]["joint_violations"] > 0
        assert r_deg["independent"]["joint_violations"] == 0

    def test_heterogeneous_degradation_injectable(self):
        """Per-agent efficiency degradation (degradation_per_agent) is injectable."""
        # [1.0, 0.85]: agent 1 has 85% charge/discharge efficiency (0.95*0.85)
        results = run_transformer_capacity_scenario(out_dir=None, degradation_per_agent=[1.0, 0.85])
        assert "independent" in results
        assert "centralized" in results
        assert results["independent"]["joint_violations"] >= 0
        assert results["independent"]["useful_work_mwh"] >= 0
