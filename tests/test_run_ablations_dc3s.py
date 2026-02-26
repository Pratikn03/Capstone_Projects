"""Regression tests for DC3S ablation statistics helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_ablations
from scripts.run_ablations import _wilcoxon_safe, run_dc3s_ablation_matrix


def test_wilcoxon_safe_identical_vectors_returns_neutral_result() -> None:
    x = pd.Series([0.0, 0.0, 0.0, 0.0])
    y = pd.Series([0.0, 0.0, 0.0, 0.0])
    stat, p_value = _wilcoxon_safe(x, y)
    assert stat == 0.0
    assert p_value == 1.0


def _write_stub_suite_outputs(out_dir: Path, good_primary_gate: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["dropout", "drift_combo"]
    seeds = list(range(10))
    controllers = ["robust_fixed_interval", "dc3s_wrapped", "deterministic_lp"]

    main_rows: list[dict[str, object]] = []
    for sc in scenarios:
        for seed in seeds:
            for ctrl in controllers:
                if ctrl == "dc3s_wrapped":
                    rate = 0.05 if good_primary_gate else 0.30
                    sev = 0.05 if good_primary_gate else 0.30
                elif ctrl == "robust_fixed_interval":
                    rate = 0.25
                    sev = 0.25
                else:
                    rate = 0.35
                    sev = 0.35
                main_rows.append(
                    {
                        "scenario": sc,
                        "seed": seed,
                        "controller": ctrl,
                        "true_soc_violation_rate": rate,
                        "true_soc_violation_severity_p95_mwh": sev,
                    }
                )
    pd.DataFrame(main_rows).to_csv(out_dir / "dc3s_main_table.csv", index=False)

    sweep_rows: list[dict[str, object]] = []
    for sc in scenarios:
        for seed in seeds:
            for fault_dim in ("dropout", "delay_jitter", "out_of_order", "spikes"):
                for severity in (0.10, 0.20):
                    robust_rate = 0.25
                    robust_sev = 0.25
                    if good_primary_gate:
                        dc3s_rate = 0.05
                        dc3s_sev = 0.05
                    else:
                        dc3s_rate = 0.23
                        dc3s_sev = 0.23
                    sweep_rows.append(
                        {
                            "fault_dimension": fault_dim,
                            "severity": severity,
                            "scenario": sc,
                            "seed": seed,
                            "controller": "robust_fixed_interval",
                            "true_soc_violation_rate": robust_rate,
                            "true_soc_violation_severity_p95_mwh": robust_sev,
                        }
                    )
                    sweep_rows.append(
                        {
                            "fault_dimension": fault_dim,
                            "severity": severity,
                            "scenario": sc,
                            "seed": seed,
                            "controller": "dc3s_wrapped",
                            "true_soc_violation_rate": dc3s_rate,
                            "true_soc_violation_severity_p95_mwh": dc3s_sev,
                        }
                    )
    pd.DataFrame(sweep_rows).to_csv(out_dir / "cpsbench_merged_sweep.csv", index=False)


def _write_cfg(path: Path, enforce: bool = True) -> None:
    path.write_text(
        f"""
bootstrap_n: 1000
scenarios: ["dropout", "drift_combo"]
enforce_primary_gate: {"true" if enforce else "false"}
primary_gate:
  baseline: robust_fixed_interval
  candidate: dc3s_wrapped
  p_threshold: 0.01
  min_relative_reduction: 0.10
  faulted_only: true
  severity_min: 0.0
secondary_diagnostics:
  include_per_fault: true
  baselines: ["robust_fixed_interval", "deterministic_lp"]
""",
        encoding="utf-8",
    )


def test_run_dc3s_ablation_matrix_primary_gate_uses_sweep(monkeypatch, tmp_path: Path) -> None:
    def _stub_run_suite(*, scenarios, seeds, out_dir, horizon):  # noqa: ANN001
        _write_stub_suite_outputs(out_dir, good_primary_gate=True)
        return {"rows": 1, "scenarios": scenarios, "seeds": seeds, "horizon": horizon}

    monkeypatch.setattr(run_ablations, "run_cpsbench_suite", _stub_run_suite)
    cfg_path = tmp_path / "cfg.yaml"
    _write_cfg(cfg_path, enforce=True)

    out_dir = tmp_path / "ablation_out"
    payload = run_dc3s_ablation_matrix(
        output_dir=out_dir,
        seeds=list(range(10)),
        horizon=24,
        cfg_path=cfg_path,
    )
    assert payload["overall_pass"] is True

    stats = json.loads((out_dir / "stats_summary.json").read_text(encoding="utf-8"))
    assert stats["primary_gate"]["scope"] == "aggregate_fault_sweep"
    assert stats["primary_gate"]["passes_all_thresholds"] is True
    assert stats["primary_gate"]["metrics"]["true_soc_violation_rate"]["passes_p01"] is True
    assert stats["primary_gate"]["metrics"]["true_soc_violation_severity_p95_mwh"]["passes_10pct"] is True
    assert stats["primary_gate"]["metrics"]["true_soc_violation_severity_p95_mwh"]["passes_target"] is True

    table = pd.read_csv(out_dir / "table2_ablations.csv")
    assert "analysis_scope" in table.columns
    assert (table["analysis_scope"] == "primary_aggregate_fault_sweep").any()


def test_run_dc3s_ablation_matrix_enforces_primary_gate(monkeypatch, tmp_path: Path) -> None:
    def _stub_run_suite(*, scenarios, seeds, out_dir, horizon):  # noqa: ANN001
        _write_stub_suite_outputs(out_dir, good_primary_gate=False)
        return {"rows": 1, "scenarios": scenarios, "seeds": seeds, "horizon": horizon}

    monkeypatch.setattr(run_ablations, "run_cpsbench_suite", _stub_run_suite)
    cfg_path = tmp_path / "cfg.yaml"
    _write_cfg(cfg_path, enforce=True)

    with pytest.raises(RuntimeError, match="Primary aggregate robust gate failed"):
        run_dc3s_ablation_matrix(
            output_dir=tmp_path / "ablation_out",
            seeds=list(range(10)),
            horizon=24,
            cfg_path=cfg_path,
        )
