"""Comprehensive tests for CPSBench metrics computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orius.cpsbench_iot.metrics import (
    compute_all_metrics,
    compute_control_metrics,
    compute_forecast_metrics,
    compute_trace_metrics,
    summarize_true_soc_violations,
)


class TestForecastMetrics:
    def test_perfect_forecast(self):
        y = np.array([10.0, 20.0, 30.0])
        lo = np.array([5.0, 15.0, 25.0])
        hi = np.array([15.0, 25.0, 35.0])
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=lo, upper_90=hi)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["picp_90"] == pytest.approx(1.0)

    def test_biased_forecast(self):
        y = np.array([10.0, 20.0, 30.0])
        pred = np.array([15.0, 25.0, 35.0])
        m = compute_forecast_metrics(y_true=y, y_pred=pred, lower_90=y - 2, upper_90=y + 2)
        assert m["mae"] == pytest.approx(5.0)
        assert m["rmse"] == pytest.approx(5.0)

    def test_zero_coverage(self):
        y = np.array([100.0, 200.0])
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=np.array([0.0, 0.0]), upper_90=np.array([1.0, 1.0]))
        assert m["picp_90"] == pytest.approx(0.0)

    def test_mean_interval_width(self):
        y = np.ones(5)
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=np.zeros(5), upper_90=np.full(5, 10.0))
        assert m["mean_interval_width"] == pytest.approx(10.0)

    def test_pinball_loss_fields(self):
        y = np.ones(10)
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=y - 1, upper_90=y + 1)
        assert "pinball_loss_q05" in m
        assert "pinball_loss_q50" in m
        assert "pinball_loss_q95" in m

    def test_winkler_score(self):
        y = np.array([10.0])
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=np.array([5.0]), upper_90=np.array([15.0]))
        assert m["winkler_score_90"] == pytest.approx(10.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="identical length"):
            compute_forecast_metrics(y_true=[1.0], y_pred=[1.0, 2.0], lower_90=[0.0], upper_90=[2.0])

    def test_explicit_95_interval(self):
        y = np.ones(5)
        m = compute_forecast_metrics(y_true=y, y_pred=y, lower_90=y - 1, upper_90=y + 1,
                                      lower_95=y - 2, upper_95=y + 2)
        assert m["picp_95"] == pytest.approx(1.0)


class TestTrueSocViolations:
    def test_no_violations(self):
        s = summarize_true_soc_violations(np.array([50.0, 60.0, 70.0]), 10.0, 90.0)
        assert s["true_soc_violation_rate"] == pytest.approx(0.0)

    def test_below_violation(self):
        s = summarize_true_soc_violations(np.array([5.0, 50.0]), 10.0, 90.0)
        assert s["true_soc_violation_rate"] == pytest.approx(0.5)
        assert s["true_soc_violation_severity_mean_mwh"] == pytest.approx(5.0)

    def test_above_violation(self):
        s = summarize_true_soc_violations(np.array([95.0, 50.0]), 10.0, 90.0)
        assert s["true_soc_violation_rate"] == pytest.approx(0.5)

    def test_all_violations(self):
        s = summarize_true_soc_violations(np.array([5.0, 95.0]), 10.0, 90.0)
        assert s["true_soc_violation_rate"] == pytest.approx(1.0)


class TestControlMetrics:
    def _base(self, n=10, **kw):
        defaults = dict(
            proposed_charge_mw=np.zeros(n),
            proposed_discharge_mw=np.zeros(n),
            safe_charge_mw=np.zeros(n),
            safe_discharge_mw=np.zeros(n),
            soc_mwh=np.full(n, 50.0),
            constraints={"min_soc_mwh": 10.0, "max_soc_mwh": 90.0, "max_power_mw": 50.0,
                          "charge_efficiency": 1.0, "discharge_efficiency": 1.0, "time_step_hours": 1.0,
                          "initial_soc_mwh": 50.0},
        )
        defaults.update(kw)
        return defaults

    def test_no_violations(self):
        m = compute_control_metrics(**self._base())
        assert m["violation_rate"] == pytest.approx(0.0)

    def test_soc_violation(self):
        m = compute_control_metrics(**self._base(soc_mwh=np.full(10, 5.0)))
        assert m["violation_rate"] > 0.0

    def test_intervention_rate(self):
        m = compute_control_metrics(**self._base(
            proposed_charge_mw=np.full(10, 10.0),
            safe_charge_mw=np.full(10, 5.0),
        ))
        assert m["intervention_rate"] == pytest.approx(1.0)

    def test_no_interventions(self):
        m = compute_control_metrics(**self._base())
        assert m["intervention_rate"] == pytest.approx(0.0)

    def test_true_soc_violation(self):
        m = compute_control_metrics(**self._base(true_soc_mwh=np.full(10, 5.0)))
        assert m["true_soc_violation_rate"] > 0.0

    def test_unsafe_command_rate(self):
        m = compute_control_metrics(**self._base(
            proposed_charge_mw=np.full(10, 100.0),
            proposed_discharge_mw=np.full(10, 100.0),
        ))
        assert m["unsafe_command_rate"] > 0.0

    def test_ramp_violation(self):
        safe_dis = np.zeros(10)
        safe_dis[5] = 50.0
        m = compute_control_metrics(**self._base(
            safe_discharge_mw=safe_dis,
            constraints={**self._base()["constraints"], "ramp_mw": 5.0},
        ))
        assert m["violation_rate"] > 0.0

    def test_mismatched_arrays_raises(self):
        with pytest.raises(ValueError, match="identical length"):
            compute_control_metrics(
                proposed_charge_mw=[1.0],
                proposed_discharge_mw=[1.0, 2.0],
                safe_charge_mw=[1.0],
                safe_discharge_mw=[1.0],
                soc_mwh=[50.0],
                constraints={"min_soc_mwh": 10.0, "max_soc_mwh": 90.0},
            )


class TestTraceMetrics:
    def test_all_present(self):
        certs = [{"command_id": "a", "certificate_hash": "h", "proposed_action": {}, "safe_action": {}}]
        m = compute_trace_metrics(certs)
        assert m["certificate_presence_rate"] == pytest.approx(1.0)
        assert m["certificate_missing_fields"] == 0.0

    def test_none_certs(self):
        m = compute_trace_metrics([None, None])
        assert m["certificate_presence_rate"] == pytest.approx(0.0)

    def test_missing_fields(self):
        certs = [{"command_id": "a"}]
        m = compute_trace_metrics(certs)
        assert m["certificate_missing_fields"] > 0

    def test_empty_list(self):
        m = compute_trace_metrics([])
        assert m["certificate_presence_rate"] == pytest.approx(0.0)

    def test_custom_required_fields(self):
        certs = [{"x": 1}]
        m = compute_trace_metrics(certs, required_fields=["x", "y"])
        assert m["certificate_missing_fields"] == 1.0


class TestComputeAllMetrics:
    def test_combines_all_families(self):
        n = 10
        m = compute_all_metrics(
            y_true=np.ones(n) * 100,
            y_pred=np.ones(n) * 100,
            lower_90=np.ones(n) * 90,
            upper_90=np.ones(n) * 110,
            proposed_charge_mw=np.zeros(n),
            proposed_discharge_mw=np.zeros(n),
            safe_charge_mw=np.zeros(n),
            safe_discharge_mw=np.zeros(n),
            soc_mwh=np.full(n, 50.0),
            constraints={"min_soc_mwh": 10.0, "max_soc_mwh": 90.0, "max_power_mw": 50.0,
                          "charge_efficiency": 1.0, "discharge_efficiency": 1.0,
                          "time_step_hours": 1.0, "initial_soc_mwh": 50.0},
            certificates=[{"command_id": f"c{i}", "certificate_hash": "h",
                            "proposed_action": {}, "safe_action": {}} for i in range(n)],
        )
        assert "mae" in m
        assert "violation_rate" in m
        assert "certificate_presence_rate" in m
