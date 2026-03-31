"""Tests for Paper 3 graceful degradation planner and promoted artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orius.dc3s.graceful import compare_policies, optimized_graceful, plan_graceful_degradation
from scripts.build_graceful_trajectory_figures import build_publication_figure
from scripts.run_paper3_four_policy_benchmark import run_benchmark


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_graceful_degradation_planner_no_fallback():
    state = {"fallback_required": False}
    plan = plan_graceful_degradation(state, {}, {}, "ramp_down", 10)
    assert not plan["actions"]
    assert "not required" in plan["reason"]


def test_graceful_degradation_planner_hard_shutdown():
    state = {"fallback_required": True}
    plan = plan_graceful_degradation(state, {}, {}, "hard_shutdown", 10)
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["charge_mw"] == 0.0
    assert plan["actions"][0]["discharge_mw"] == 0.0


def test_graceful_degradation_planner_optimized_constructive():
    """Optimized mode produces tapered actions over remaining horizon."""
    state = {
        "fallback_required": True,
        "last_action": {"charge_mw": 0.0, "discharge_mw": 0.0},
        "current_soc_mwh": 5000.0,
        "constraints": {
            "min_soc_mwh": 0.0,
            "max_soc_mwh": 10000.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
    }
    plan = plan_graceful_degradation(state, {}, {"useful_work_weight": 1.0}, "optimized", 8)
    assert len(plan["actions"]) == 8
    assert "Optimized" in plan["reason"]


def test_optimized_graceful_consumes_horizon():
    """optimized_graceful returns horizon_steps actions, tapered to zero."""
    constraints = {
        "min_soc_mwh": 0.0,
        "max_soc_mwh": 100.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    acts = optimized_graceful({"charge_mw": 0, "discharge_mw": 20.0}, 5, 50.0, constraints)
    assert len(acts) == 5
    assert acts[0]["discharge_mw"] > acts[-1]["discharge_mw"]


def test_compare_policies_four_policies():
    """compare_policies returns all four policies."""
    constraints = {
        "min_soc_mwh": 0,
        "max_soc_mwh": 100,
        "time_step_hours": 1,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    result = compare_policies({"charge_mw": 0, "discharge_mw": 0.0}, 5, 50.0, constraints)
    assert set(result.keys()) == {
        "blind_persistence",
        "immediate_shutdown",
        "simple_ramp_down",
        "optimized_graceful",
    }


def test_run_benchmark_syncs_promoted_summary(tmp_path: Path):
    paper3_dir = tmp_path / "paper3"
    publication_dir = tmp_path / "publication"
    outputs = run_benchmark(paper3_dir=paper3_dir, publication_dir=publication_dir)

    detail_rows = _read_csv(outputs["policy_compare"])
    summary_rows = _read_csv(outputs["summary"])
    publication_rows = _read_csv(outputs["publication_summary"])

    assert len(detail_rows) == 4 * 5
    assert summary_rows == publication_rows

    by_policy = {row["policy"]: row for row in summary_rows}
    assert float(by_policy["optimized_graceful"]["useful_work_mwh_mean"]) > float(
        by_policy["immediate_shutdown"]["useful_work_mwh_mean"]
    )
    assert float(by_policy["simple_ramp_down"]["useful_work_mwh_mean"]) > float(
        by_policy["immediate_shutdown"]["useful_work_mwh_mean"]
    )
    assert float(by_policy["blind_persistence"]["violation_rate_mean"]) > float(
        by_policy["optimized_graceful"]["violation_rate_mean"]
    )
    assert float(by_policy["blind_persistence"]["severity_mwh_mean"]) > float(
        by_policy["optimized_graceful"]["severity_mwh_mean"]
    )
    assert float(by_policy["optimized_graceful"]["useful_work_mwh_mean"]) >= float(
        by_policy["simple_ramp_down"]["useful_work_mwh_mean"]
    )


def test_build_publication_figure_uses_canonical_surface(tmp_path: Path):
    paper3_dir = tmp_path / "paper3"
    publication_dir = tmp_path / "publication"
    run_benchmark(paper3_dir=paper3_dir, publication_dir=publication_dir)

    outputs = build_publication_figure(
        paper3_dir=paper3_dir,
        publication_dir=publication_dir,
    )

    assert outputs["publication_figure"].exists()
    assert outputs["paper3_figure"].exists()
    assert _read_csv(outputs["publication_summary"]) == _read_csv(
        paper3_dir / "graceful_four_policy_metrics.csv"
    )
