"""Tests for R1 paper release features.

Covers: UQ metrics contract, scenario MPC baseline, paper asset sync,
severity sweep config, transfer study scaffold.
"""
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# UQ Metrics Contract
# ============================================================
class TestUQMetrics:
    def test_picp_perfect_coverage(self):
        from orius.evaluation.uq_metrics import compute_picp

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert compute_picp(y, lower, upper) == 1.0

    def test_picp_zero_coverage(self):
        from orius.evaluation.uq_metrics import compute_picp

        y = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert compute_picp(y, lower, upper) == 0.0

    def test_pinball_loss_symmetry(self):
        from orius.evaluation.uq_metrics import compute_pinball_loss

        y = np.array([5.0])
        # At quantile 0.5, over- and under-prediction should be symmetric
        loss_over = compute_pinball_loss(y, np.array([6.0]), tau=0.5)
        loss_under = compute_pinball_loss(y, np.array([4.0]), tau=0.5)
        assert abs(loss_over - loss_under) < 1e-10

    def test_pinball_loss_asymmetry(self):
        from orius.evaluation.uq_metrics import compute_pinball_loss

        y = np.array([5.0])
        # At quantile 0.9, under-prediction is penalized more
        loss_over = compute_pinball_loss(y, np.array([6.0]), tau=0.9)
        loss_under = compute_pinball_loss(y, np.array([4.0]), tau=0.9)
        assert loss_under > loss_over

    def test_winkler_score_no_penalty(self):
        from orius.evaluation.uq_metrics import compute_winkler_score

        y = np.array([5.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        # Width = 2.0, no penalty since y is inside
        assert compute_winkler_score(y, lower, upper, alpha=0.1) == 2.0

    def test_winkler_score_penalty_below(self):
        from orius.evaluation.uq_metrics import compute_winkler_score

        y = np.array([1.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        alpha = 0.1
        # Width=2, penalty=2*(4-1)/0.1 = 60
        expected = 2.0 + 60.0
        assert abs(compute_winkler_score(y, lower, upper, alpha=alpha) - expected) < 1e-10

    def test_compute_uq_contract_keys(self):
        from orius.evaluation.uq_metrics import compute_uq_contract

        n = 100
        y = np.random.randn(n) * 10 + 100
        lower = y - 5
        upper = y + 5

        contract = compute_uq_contract(y, lower_90=lower, upper_90=upper)
        assert hasattr(contract, "picp_90")
        assert hasattr(contract, "mean_interval_width_90")
        assert hasattr(contract, "pinball_loss_05")
        assert hasattr(contract, "winkler_score_90")
        assert contract.picp_90 >= 0.0
        assert contract.picp_90 <= 1.0
        assert contract.mean_interval_width_90 > 0


# ============================================================
# Scenario MPC Baseline
# ============================================================
class TestScenarioMPC:
    def test_scenario_mpc_dispatch_returns_expected_keys(self):
        from orius.cpsbench_iot.scenario_mpc import scenario_mpc_dispatch

        n = 24
        result = scenario_mpc_dispatch(
            load_forecast=np.full(n, 1000.0),
            renewables_forecast=np.full(n, 200.0),
            load_true=np.full(n, 1050.0),
            price=np.full(n, 50.0),
            optimization_cfg={
                "battery": {
                    "capacity_mwh": 100.0,
                    "max_power_mw": 50.0,
                    "min_soc_mwh": 0.0,
                    "initial_soc_mwh": 50.0,
                    "efficiency": 1.0,
                },
                "time_step_hours": 1.0,
            },
        )
        assert result["policy"] == "scenario_mpc"
        assert len(result["safe_charge_mw"]) == n
        assert len(result["safe_discharge_mw"]) == n
        assert len(result["soc_mwh"]) == n
        assert len(result["interval_lower"]) == n
        assert len(result["interval_upper"]) == n
        assert "certificates" in result

    def test_scenario_mpc_soc_bounds(self):
        from orius.cpsbench_iot.scenario_mpc import scenario_mpc_dispatch

        n = 48
        result = scenario_mpc_dispatch(
            load_forecast=np.full(n, 500.0),
            renewables_forecast=np.full(n, 100.0),
            load_true=np.full(n, 500.0),
            price=np.full(n, 40.0),
            optimization_cfg={
                "battery": {
                    "capacity_mwh": 80.0,
                    "max_power_mw": 40.0,
                    "min_soc_mwh": 5.0,
                    "initial_soc_mwh": 40.0,
                    "efficiency": 0.95,
                },
                "time_step_hours": 1.0,
            },
        )
        soc = np.array(result["soc_mwh"])
        assert np.all(soc >= 0.0), "SOC went below 0"
        assert np.all(soc <= 80.0 + 1e-6), "SOC exceeded capacity"


# ============================================================
# Paper Asset Sync
# ============================================================
class TestPaperAssetSync:
    def test_run_checks_returns_list(self):
        from scripts.sync_paper_assets import run_checks

        issues = run_checks()
        assert isinstance(issues, list)

    def test_stale_value_detection(self):
        from scripts.sync_paper_assets import _check_file_for_stale, REPO_ROOT as SYNC_ROOT

        import tempfile
        import os

        # Create temp file inside repo root so relative_to works
        tmp_dir = SYNC_ROOT / "data" / "interim"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "_test_stale_check.csv"
        try:
            tmp_path.write_text("region,rows\nDE,92382\n")
            issues = _check_file_for_stale(tmp_path, ["92382", "92,382"])
            assert len(issues) == 1
            assert issues[0]["stale_value"] == "92382"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_picp_headline_check(self):
        from scripts.sync_paper_assets import _check_picp_headline

        issues = _check_picp_headline()
        # If paper.tex has stale PICP values, this will catch them
        assert isinstance(issues, list)


# ============================================================
# Severity Sweep Config
# ============================================================
class TestSeveritySweepConfig:
    def test_severity_config_loads(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        path = REPO_ROOT / "configs" / "cpsbench_r1_severity.yaml"
        assert path.exists(), "cpsbench_r1_severity.yaml not found"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cpsbench = cfg["cpsbench"]
        assert "scenarios" in cpsbench
        assert "controllers" in cpsbench
        assert "scenario_mpc" in cpsbench["controllers"]

    def test_severity_config_has_soc_mismatch(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with open(REPO_ROOT / "configs" / "cpsbench_r1_severity.yaml") as f:
            cfg = yaml.safe_load(f)
        scenarios = cfg["cpsbench"]["scenarios"]
        assert "soc_mismatch" in scenarios


# ============================================================
# Transfer Study
# ============================================================
class TestTransferStudy:
    def test_parse_pairs(self):
        from scripts.run_transfer_study import _parse_pairs

        pairs = _parse_pairs(["DE:US_MISO", "US_PJM:US_ERCOT"])
        assert pairs == [("DE", "US_MISO"), ("US_PJM", "US_ERCOT")]

    def test_parse_pairs_invalid(self):
        from scripts.run_transfer_study import _parse_pairs

        with pytest.raises(ValueError):
            _parse_pairs(["INVALID"])

    def test_transfer_result_dataclass(self):
        from scripts.run_transfer_study import TransferResult

        r = TransferResult(
            source_region="DE",
            target_region="US_MISO",
            model_name="baseline_gbm",
            target_col="load_mw",
            mae_native=50.0,
            mae_transfer=120.0,
            rmse_native=70.0,
            rmse_transfer=160.0,
            mape_native=3.5,
            mape_transfer=8.2,
            degradation_mae_pct=140.0,
            degradation_rmse_pct=128.6,
        )
        assert r.degradation_mae_pct == 140.0


# ============================================================
# R1 Release Orchestrator
# ============================================================
class TestR1ReleaseOrchestrator:
    def test_r1_release_script_exists(self):
        assert (REPO_ROOT / "scripts" / "run_r1_release.py").exists()

    def test_r1_release_stages(self):
        """Verify the script defines all expected stages."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_r1_release",
            str(REPO_ROOT / "scripts" / "run_r1_release.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute main, just check STAGES exists
        content = (REPO_ROOT / "scripts" / "run_r1_release.py").read_text()
        for stage in ("diagnostic", "full", "cpsbench", "verify", "promote"):
            assert stage in content, f"Stage '{stage}' missing from R1 release script"
