"""Tests for dataset-aware research integration in pipeline.run."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import gridpulse.pipeline.run as pr


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _create_minimal_repo(tmp_path: Path, *, with_models: bool = True, n_rows: int = 16) -> Path:
    repo = tmp_path
    (repo / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (repo / "data" / "processed" / "splits").mkdir(parents=True, exist_ok=True)
    (repo / "data" / "processed" / "us_eia930" / "splits").mkdir(parents=True, exist_ok=True)
    (repo / "reports").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "models_eia930").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(parents=True, exist_ok=True)

    (repo / "data" / "raw" / "time_series_60min_singleindex.csv").write_text(
        "timestamp,load_mw\n2024-01-01T00:00:00Z,1.0\n",
        encoding="utf-8",
    )

    pd.DataFrame({"x": [1.0]}).to_parquet(repo / "data" / "processed" / "features.parquet")

    de_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC"),
            "load_mw": np.linspace(100.0, 120.0, n_rows),
            "wind_mw": np.full(n_rows, 25.0),
            "solar_mw": np.full(n_rows, 15.0),
            "price_eur_mwh": np.full(n_rows, 50.0),
        }
    )
    de_df.to_parquet(repo / "data" / "processed" / "splits" / "test.parquet")

    us_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n_rows, freq="h", tz="UTC"),
            "load_mw": np.linspace(200.0, 240.0, n_rows),
            "wind_mw": np.full(n_rows, 30.0),
            "solar_mw": np.full(n_rows, 20.0),
            "price_usd_mwh": np.full(n_rows, 60.0),
        }
    )
    us_df.to_parquet(repo / "data" / "processed" / "us_eia930" / "splits" / "test.parquet")

    _write_yaml(repo / "configs" / "data.yaml", {})
    _write_yaml(repo / "configs" / "train_forecast.yaml", {"task": {"horizon_hours": 4}})
    _write_yaml(
        repo / "configs" / "forecast.yaml",
        {
            "models": {
                "load_mw": "artifacts/models/gbm_lightgbm_load_mw.pkl",
                "wind_mw": "artifacts/models/gbm_lightgbm_wind_mw.pkl",
                "solar_mw": "artifacts/models/gbm_lightgbm_solar_mw.pkl",
            },
            "fallback_order": ["gbm"],
        },
    )
    _write_yaml(
        repo / "configs" / "forecast_eia930.yaml",
        {
            "models": {
                "load_mw": "artifacts/models_eia930/gbm_lightgbm_load_mw.pkl",
                "wind_mw": "artifacts/models_eia930/gbm_lightgbm_wind_mw.pkl",
                "solar_mw": "artifacts/models_eia930/gbm_lightgbm_solar_mw.pkl",
            },
            "fallback_order": ["gbm"],
        },
    )
    _write_yaml(
        repo / "configs" / "uncertainty.yaml",
        {
            "enabled": False,
            "conformal": {"alpha": 0.10, "eps": 1e-6},
            "artifacts_dir": "artifacts/uncertainty",
        },
    )
    _write_yaml(
        repo / "configs" / "optimization.yaml",
        {
            "battery": {
                "capacity_mwh": 100.0,
                "max_power_mw": 50.0,
                "efficiency": 0.95,
                "min_soc_mwh": 0.0,
                "initial_soc_mwh": 50.0,
                "degradation_cost_per_mwh": 10.0,
            },
            "grid": {
                "max_import_mw": 500.0,
                "price_per_mwh": 50.0,
            },
        },
    )

    if with_models:
        for target in ("load_mw", "wind_mw", "solar_mw"):
            (repo / "artifacts" / "models" / f"gbm_lightgbm_{target}.pkl").write_bytes(b"dummy")
            (repo / "artifacts" / "models_eia930" / f"gbm_lightgbm_{target}.pkl").write_bytes(b"dummy")

    return repo


def _patch_main_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr, "register_models", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr, "_snapshot_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr, "_snapshot_configs", lambda *args, **kwargs: None)
    monkeypatch.setattr(pr, "_pip_freeze", lambda *args, **kwargs: None)


def _patch_lightweight_research_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_forecast(context_df, target, horizon, forecast_cfg, models_dir):
        base = {"load_mw": 100.0, "wind_mw": 25.0, "solar_mw": 15.0}[target]
        yhat = np.full(horizon, base, dtype=float)
        return {
            "forecast": yhat.tolist(),
            "quantiles": {
                "0.1": (yhat - 2.0).tolist(),
                "0.9": (yhat + 2.0).tolist(),
            },
        }

    def fake_robust_dispatch(*, load_lower_bound, load_upper_bound, renewables_forecast, price, config, verbose):
        h = len(load_lower_bound)
        return {
            "battery_charge_mw": [0.0] * h,
            "battery_discharge_mw": [0.0] * h,
            "total_cost": 123.0,
            "feasible": True,
            "solver_status": "ok",
        }

    def fake_vss(**kwargs):
        return {
            "vss": 7.0,
            "deterministic_realized_cost": 207.0,
            "robust_realized_cost": 200.0,
            "horizon": len(np.asarray(kwargs["load_true"], dtype=float)),
        }

    def fake_evpi(**kwargs):
        model = kwargs["actual_model"]
        if model == "robust":
            return {
                "evpi": 5.0,
                "actual_realized_cost": 205.0,
                "perfect_info_cost": 200.0,
                "actual_model": "robust",
                "horizon": len(np.asarray(kwargs["load_true"], dtype=float)),
            }
        return {
            "evpi": 9.0,
            "actual_realized_cost": 209.0,
            "perfect_info_cost": 200.0,
            "actual_model": "deterministic",
            "horizon": len(np.asarray(kwargs["load_true"], dtype=float)),
        }

    monkeypatch.setattr(pr, "_forecast_with_gbm", fake_forecast)
    monkeypatch.setattr(pr, "_optimize_robust_dispatch", fake_robust_dispatch)
    monkeypatch.setattr(pr, "_calculate_vss", fake_vss)
    monkeypatch.setattr(pr, "_calculate_evpi", fake_evpi)


def test_run_defaults_include_research_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=16)
    _patch_main_side_effects(monkeypatch)
    monkeypatch.setattr(pr, "_repo_root", lambda: repo)

    calls = {"research": 0}

    def fake_research(**kwargs):
        calls["research"] += 1
        return {
            "rows_written": 2,
            "window_rows": 1,
            "output_csv": str(kwargs["output_csv"]),
        }

    monkeypatch.setattr(pr, "_run_research_step", fake_research)
    monkeypatch.setattr(pr.sys, "argv", ["prog", "--run-id", "run-default"])

    pr.main()

    assert calls["research"] == 2
    manifest = json.loads((repo / "artifacts" / "runs" / "run-default" / "manifest.json").read_text(encoding="utf-8"))
    assert "research" in manifest["steps"]
    assert set(manifest["research_outputs"].keys()) == {"de", "us"}


def test_research_dataset_flag_allows_de_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=16)
    _patch_main_side_effects(monkeypatch)
    monkeypatch.setattr(pr, "_repo_root", lambda: repo)

    outputs = []

    def fake_research(**kwargs):
        outputs.append(Path(kwargs["output_csv"]).name)
        return {"rows_written": 1, "window_rows": 1, "output_csv": str(kwargs["output_csv"])}

    monkeypatch.setattr(pr, "_run_research_step", fake_research)
    monkeypatch.setattr(
        pr.sys,
        "argv",
        ["prog", "--steps", "research", "--research-datasets", "de", "--run-id", "run-de-only"],
    )

    pr.main()
    assert outputs == ["research_metrics_de.csv"]


def test_research_step_requires_gbm_models_and_fails_hard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=False, n_rows=16)
    _patch_main_side_effects(monkeypatch)
    monkeypatch.setattr(pr, "_repo_root", lambda: repo)
    monkeypatch.setattr(
        pr.sys,
        "argv",
        ["prog", "--steps", "research", "--research-datasets", "de", "--run-id", "run-no-models"],
    )

    with pytest.raises(FileNotFoundError):
        pr.main()


def test_faci_online_builder_uses_past_observations_only() -> None:
    base_lower = np.array([90.0, 90.0, 90.0], dtype=float)
    base_upper = np.array([110.0, 110.0, 110.0], dtype=float)

    load_true_a = np.array([300.0, 100.0, 100.0], dtype=float)
    load_true_b = np.array([300.0, -50.0, 100.0], dtype=float)

    dyn_lo_a, dyn_hi_a = pr._apply_faci_online(load_true_a, base_lower, base_upper, 0.10, 0.05, 1e-6)
    dyn_lo_b, dyn_hi_b = pr._apply_faci_online(load_true_b, base_lower, base_upper, 0.10, 0.05, 1e-6)

    assert np.isclose(dyn_lo_a[0], dyn_lo_b[0])
    assert np.isclose(dyn_hi_a[0], dyn_hi_b[0])
    assert np.isclose(dyn_lo_a[1], dyn_lo_b[1])
    assert np.isclose(dyn_hi_a[1], dyn_hi_b[1])
    assert not np.isclose(dyn_hi_a[2] - dyn_lo_a[2], dyn_hi_b[2] - dyn_lo_b[2])


def test_faci_online_respects_learned_scale_bounds() -> None:
    base_lower = np.array([90.0, 90.0, 90.0], dtype=float)
    base_upper = np.array([110.0, 110.0, 110.0], dtype=float)
    load_true = np.array([300.0, 100.0, 100.0], dtype=float)

    dyn_lo_unbounded, dyn_hi_unbounded = pr._apply_faci_online(
        load_true, base_lower, base_upper, 0.10, 0.05, 1e-6
    )
    dyn_lo_bounded, dyn_hi_bounded = pr._apply_faci_online(
        load_true,
        base_lower,
        base_upper,
        0.10,
        0.05,
        1e-6,
        scale_bounds=(0.9, 1.1),
    )

    width_unbounded = float(dyn_hi_unbounded[1] - dyn_lo_unbounded[1])
    width_bounded = float(dyn_hi_bounded[1] - dyn_lo_bounded[1])
    assert width_bounded <= width_unbounded
    assert np.isclose(width_bounded, 22.0, atol=1e-6)


def test_estimate_faci_scale_bounds_from_calibration_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path
    (repo / "artifacts" / "backtests").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "uncertainty").mkdir(parents=True, exist_ok=True)

    y_pred = np.linspace(100.0, 200.0, 200)
    y_true = y_pred + np.linspace(-8.0, 8.0, 200)
    np.savez(
        repo / "artifacts" / "backtests" / "load_mw_calibration.npz",
        y_true=y_true,
        y_pred=y_pred,
    )
    (repo / "artifacts" / "uncertainty" / "load_mw_conformal.json").write_text("{}", encoding="utf-8")

    class _FakeConformal:
        def predict_interval(self, y):
            arr = np.asarray(y, dtype=float)
            return arr - 10.0, arr + 10.0

    monkeypatch.setattr(
        "gridpulse.forecasting.uncertainty.load_conformal",
        lambda path: _FakeConformal(),
    )

    uncertainty_cfg = {
        "faci": {
            "learn_scale_bounds": True,
            "scale_lower_quantile": 0.10,
            "scale_upper_quantile": 0.90,
            "min_scale_samples": 50,
        },
        "calibration_npz": "artifacts/backtests/{target}_calibration.npz",
    }

    bounds = pr._estimate_faci_scale_bounds(
        repo_root=repo,
        uncertainty_cfg=uncertainty_cfg,
        uncertainty_artifacts_dir=repo / "artifacts" / "uncertainty",
        target="load_mw",
        eps=1e-6,
        log=logging.getLogger("test_faci_bounds"),
    )

    assert bounds is not None
    low, high = bounds
    assert 0.0 < low < high
    assert high < 1.0


def test_faci_online_keeps_upper_cap_when_history_bounds_conflict() -> None:
    # Build a long sequence so history-aware clipping is active.
    horizon = 16
    base_lower = np.full(horizon, 90.0, dtype=float)
    base_upper = np.full(horizon, 110.0, dtype=float)

    # Repeated misses create strong pressure to widen intervals.
    load_true = np.full(horizon, 400.0, dtype=float)

    dyn_lo, dyn_hi = pr._apply_faci_online(
        load_true=load_true,
        base_lower=base_lower,
        base_upper=base_upper,
        alpha=0.10,
        gamma=0.05,
        eps=1e-6,
        scale_bounds=(0.9, 1.0),  # explicit no-expansion envelope
        history_quantiles=(0.95, 0.99),  # can conflict with learned bound intersection
        history_warmup=4,
    )

    # With upper cap=1.0, dynamic width should never exceed base width.
    base_width = base_upper - base_lower
    dyn_width = dyn_hi - dyn_lo
    assert np.all(dyn_width <= base_width + 1e-9)


def test_apply_operational_load_stress_expands_bounds_when_enabled() -> None:
    lower = np.array([90.0, 92.0, 91.0], dtype=float)
    upper = np.array([110.0, 112.0, 113.0], dtype=float)
    context_df = pd.DataFrame(
        {
            "load_mw": [100.0, 120.0, 115.0, 140.0, 130.0],
            "wind_mw": [20.0, 18.0, 19.0, 17.0, 16.0],
            "solar_mw": [10.0, 11.0, 12.0, 9.0, 8.0],
        }
    )
    cfg = {
        "stress": {
            "enabled": True,
            "width_quantile": 0.75,
            "volatility_quantile": 0.90,
            "stress_scale": 0.8,
            "lower_relax_fraction": 0.1,
            "step_scale_min": 0.75,
            "step_scale_max": 1.5,
        }
    }

    stressed_lower, stressed_upper, meta = pr._apply_operational_load_stress(
        load_lower_bound=lower,
        load_upper_bound=upper,
        context_df=context_df,
        research_operational_cfg=cfg,
    )

    assert bool(meta["applied"])
    assert np.all(stressed_upper >= upper)
    assert np.mean(stressed_upper - stressed_lower) > np.mean(upper - lower)


def test_build_window_operational_config_applies_dataset_overrides() -> None:
    base_config = {
        "battery": {
            "capacity_mwh": 100.0,
            "max_soc_mwh": 100.0,
            "min_soc_mwh": 5.0,
            "initial_soc_mwh": 50.0,
        },
        "grid": {"max_import_mw": 1000.0},
        "research_operational": {
            "reserve_soc": {"enabled": True, "min_soc_fraction": 0.05},
            "terminal_soc": {
                "enabled": True,
                "target_soc_fraction": 0.10,
                "uncertainty_quantile": 0.90,
                "uncertainty_to_soc_mwh_per_mw": 0.02,
                "enforce_as_min_soc": False,
                "penalty_per_mwh_shortfall": 200.0,
            },
            "grid_cap": {
                "enabled": True,
                "net_load_quantile": 0.90,
                "forecast_feasibility_quantile": 0.90,
                "headroom_mw": 10.0,
                "feasibility_margin_mw": 10.0,
                "min_cap_mw": 50.0,
                "tightness_factor": 1.0,
            },
            "datasets": {
                "us": {
                    "reserve_soc": {"min_soc_fraction": 0.20},
                    "terminal_soc": {"target_soc_fraction": 0.25},
                }
            },
        },
    }
    context_df = pd.DataFrame(
        {
            "load_mw": [100.0, 110.0, 120.0, 130.0],
            "wind_mw": [20.0, 20.0, 20.0, 20.0],
            "solar_mw": [10.0, 10.0, 10.0, 10.0],
        }
    )
    load_lower = np.array([95.0, 100.0, 105.0, 110.0], dtype=float)
    load_upper = np.array([120.0, 125.0, 130.0, 135.0], dtype=float)
    renewables = np.array([30.0, 30.0, 30.0, 30.0], dtype=float)

    cfg = pr._build_window_operational_config(
        base_config=base_config,
        context_df=context_df,
        load_lower_bound=load_lower,
        load_upper_bound=load_upper,
        renewables_forecast=renewables,
        dataset_key="us",
        log=logging.getLogger("test_operational_cfg"),
    )

    assert cfg["battery"]["min_soc_mwh"] >= 20.0
    term_target = cfg["research_operational"]["terminal_soc"]["resolved_target_mwh"]
    assert term_target >= 25.0
    assert cfg["grid"]["max_import_mw"] <= 1000.0


def test_research_step_writes_and_appends_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=16)
    _patch_lightweight_research_dependencies(monkeypatch)

    deterministic_cfg = pr._load_yaml(repo / "configs" / "optimization.yaml")
    forecast_cfg = pr._load_yaml(repo / "configs" / "forecast.yaml")
    uncertainty_cfg = pr._load_yaml(repo / "configs" / "uncertainty.yaml")
    logger = logging.getLogger("test_research_append")

    csv_path = repo / "reports" / "research_metrics_de.csv"
    first = pr._run_research_step(
        repo_root=repo,
        run_id="run-1",
        log=logger,
        research_horizon=4,
        research_window_step=4,
        research_gamma=0.05,
        split_path=repo / "data" / "processed" / "splits" / "test.parquet",
        models_dir=repo / "artifacts" / "models",
        output_csv=csv_path,
        uncertainty_artifacts_dir=repo / "artifacts" / "uncertainty",
        deterministic_config=deterministic_cfg,
        forecast_cfg=forecast_cfg,
        uncertainty_cfg=uncertainty_cfg,
    )
    assert csv_path.exists()
    first_rows = len(pd.read_csv(csv_path))
    assert first_rows == first["rows_written"]

    second = pr._run_research_step(
        repo_root=repo,
        run_id="run-2",
        log=logger,
        research_horizon=4,
        research_window_step=4,
        research_gamma=0.05,
        split_path=repo / "data" / "processed" / "splits" / "test.parquet",
        models_dir=repo / "artifacts" / "models",
        output_csv=csv_path,
        uncertainty_artifacts_dir=repo / "artifacts" / "uncertainty",
        deterministic_config=deterministic_cfg,
        forecast_cfg=forecast_cfg,
        uncertainty_cfg=uncertainty_cfg,
    )
    second_rows = len(pd.read_csv(csv_path))
    assert second_rows == first_rows + second["rows_written"]


def test_research_row_schema_contains_evpi_vss_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=12)
    _patch_lightweight_research_dependencies(monkeypatch)

    deterministic_cfg = pr._load_yaml(repo / "configs" / "optimization.yaml")
    forecast_cfg = pr._load_yaml(repo / "configs" / "forecast.yaml")
    uncertainty_cfg = pr._load_yaml(repo / "configs" / "uncertainty.yaml")

    csv_path = repo / "reports" / "research_metrics_de.csv"
    pr._run_research_step(
        repo_root=repo,
        run_id="schema-run",
        log=logging.getLogger("test_research_schema"),
        research_horizon=4,
        research_window_step=4,
        research_gamma=0.05,
        split_path=repo / "data" / "processed" / "splits" / "test.parquet",
        models_dir=repo / "artifacts" / "models",
        output_csv=csv_path,
        uncertainty_artifacts_dir=repo / "artifacts" / "uncertainty",
        deterministic_config=deterministic_cfg,
        forecast_cfg=forecast_cfg,
        uncertainty_cfg=uncertainty_cfg,
    )
    df = pd.read_csv(csv_path)
    required = {
        "row_type",
        "evpi",
        "evpi_robust",
        "evpi_deterministic",
        "vss",
        "robust_actual_realized_cost",
        "robust_perfect_info_cost",
        "deterministic_actual_realized_cost",
        "deterministic_perfect_info_cost",
        "mean_dynamic_interval_width",
        "mean_base_interval_width",
    }
    assert required.issubset(df.columns)


def test_research_step_calls_regret_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=8)

    counters = {"vss": 0, "evpi": 0}

    def fake_forecast(context_df, target, horizon, forecast_cfg, models_dir):
        yhat = np.full(horizon, {"load_mw": 100.0, "wind_mw": 20.0, "solar_mw": 10.0}[target], dtype=float)
        return {"forecast": yhat.tolist(), "quantiles": {"0.1": (yhat - 1).tolist(), "0.9": (yhat + 1).tolist()}}

    def fake_robust_dispatch(*, load_lower_bound, load_upper_bound, renewables_forecast, price, config, verbose):
        h = len(load_lower_bound)
        return {
            "battery_charge_mw": [0.0] * h,
            "battery_discharge_mw": [0.0] * h,
            "total_cost": 100.0,
            "feasible": True,
            "solver_status": "ok",
        }

    def fake_vss(**kwargs):
        counters["vss"] += 1
        h = len(np.asarray(kwargs["load_true"], dtype=float))
        return {"vss": 1.0, "deterministic_realized_cost": 101.0, "robust_realized_cost": 100.0, "horizon": h}

    def fake_evpi(**kwargs):
        counters["evpi"] += 1
        h = len(np.asarray(kwargs["load_true"], dtype=float))
        val = 2.0 if kwargs.get("actual_model") == "robust" else 3.0
        return {
            "evpi": val,
            "actual_realized_cost": 100.0 + val,
            "perfect_info_cost": 100.0,
            "actual_model": kwargs.get("actual_model"),
            "horizon": h,
        }

    monkeypatch.setattr(pr, "_forecast_with_gbm", fake_forecast)
    monkeypatch.setattr(pr, "_optimize_robust_dispatch", fake_robust_dispatch)
    monkeypatch.setattr(pr, "_calculate_vss", fake_vss)
    monkeypatch.setattr(pr, "_calculate_evpi", fake_evpi)

    deterministic_cfg = pr._load_yaml(repo / "configs" / "optimization.yaml")
    forecast_cfg = pr._load_yaml(repo / "configs" / "forecast.yaml")
    uncertainty_cfg = pr._load_yaml(repo / "configs" / "uncertainty.yaml")

    pr._run_research_step(
        repo_root=repo,
        run_id="calls-run",
        log=logging.getLogger("test_research_calls"),
        research_horizon=4,
        research_window_step=4,
        research_gamma=0.05,
        split_path=repo / "data" / "processed" / "splits" / "test.parquet",
        models_dir=repo / "artifacts" / "models",
        output_csv=repo / "reports" / "research_metrics_de.csv",
        uncertainty_artifacts_dir=repo / "artifacts" / "uncertainty",
        deterministic_config=deterministic_cfg,
        forecast_cfg=forecast_cfg,
        uncertainty_cfg=uncertainty_cfg,
    )

    assert counters["vss"] == 1
    assert counters["evpi"] == 2


def test_research_cache_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _create_minimal_repo(tmp_path, with_models=True, n_rows=16)
    (repo / "reports" / "research_metrics_de.csv").write_text("row_type\nwindow\n", encoding="utf-8")
    (repo / "reports" / "research_metrics_us.csv").write_text("row_type\nwindow\n", encoding="utf-8")
    (repo / ".cache").mkdir(parents=True, exist_ok=True)
    (repo / ".cache" / "pipeline.json").write_text(
        json.dumps({"research_hash_de": "same-hash", "research_hash_us": "same-hash"}, indent=2),
        encoding="utf-8",
    )

    _patch_main_side_effects(monkeypatch)
    monkeypatch.setattr(pr, "_repo_root", lambda: repo)
    monkeypatch.setattr(pr, "_hash_paths", lambda paths, base: "same-hash")

    called = {"research": 0}

    def fake_research(**kwargs):
        called["research"] += 1
        return {"rows_written": 99, "window_rows": 99, "output_csv": str(kwargs["output_csv"])}

    monkeypatch.setattr(pr, "_run_research_step", fake_research)
    monkeypatch.setattr(pr.sys, "argv", ["prog", "--steps", "research", "--run-id", "skip-run"])

    pr.main()
    assert called["research"] == 0


def test_resolve_gbm_model_path_rejects_non_gbm_explicit(tmp_path: Path) -> None:
    repo = tmp_path
    models_dir = repo / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    bad_model = models_dir / "lstm_load_mw.pt"
    bad_model.write_bytes(b"x")

    forecast_cfg = {"models": {"load_mw": str(bad_model)}}
    with pytest.raises(RuntimeError):
        pr._resolve_gbm_model_path("load_mw", forecast_cfg, models_dir)
