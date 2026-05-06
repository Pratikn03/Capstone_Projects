"""Schema-level tests for the publication baseline comparison table."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

scripts_root = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location(
    "build_baseline_comparison_table", scripts_root / "build_baseline_comparison_table.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_model_order_includes_advanced_families() -> None:
    for required in ("prophet", "nbeats_darts", "ngboost", "flaml"):
        assert required in module.MODEL_ORDER, f"{required} missing from MODEL_ORDER"


def test_model_labels_disambiguate_internal_and_oreshkin() -> None:
    assert module.MODEL_LABELS["nbeats"] == "N-BEATS (ours)"
    assert module.MODEL_LABELS["nbeats_darts"] == "N-BEATS (Oreshkin/Darts)"


def test_native_uq_models_declared() -> None:
    assert {"gbm", "prophet", "ngboost"} == module.NATIVE_UQ_MODELS


def _build_metrics(tmp_path: Path) -> Path:
    payload = {
        "targets": {
            "load_mw": {
                "gbm": {
                    "rmse": 100.0,
                    "mae": 70.0,
                    "smape": 0.01,
                    "r2": 0.99,
                    "uncertainty": {"picp_90": 0.91, "mean_interval_width": 220.0},
                },
                "lstm": {
                    "rmse": 200.0,
                    "mae": 150.0,
                    "smape": 0.03,
                    "r2": 0.95,
                    "uncertainty": {"picp_90": 0.86, "mean_interval_width": 320.0},
                },
                "tcn": {
                    "rmse": 180.0,
                    "mae": 130.0,
                    "smape": 0.025,
                    "r2": 0.96,
                    "uncertainty": {"picp_90": 0.88, "mean_interval_width": 305.0},
                },
                "nbeats": {
                    "rmse": 150.0,
                    "mae": 110.0,
                    "smape": 0.022,
                    "r2": 0.97,
                    "uncertainty": {"picp_90": 0.89, "mean_interval_width": 290.0},
                },
                "tft": {
                    "rmse": 140.0,
                    "mae": 105.0,
                    "smape": 0.020,
                    "r2": 0.97,
                    "uncertainty": {"picp_90": 0.89, "mean_interval_width": 285.0},
                },
                "patchtst": {
                    "rmse": 130.0,
                    "mae": 100.0,
                    "smape": 0.019,
                    "r2": 0.975,
                    "uncertainty": {"picp_90": 0.90, "mean_interval_width": 280.0},
                },
                "prophet": {
                    "rmse": 250.0,
                    "mae": 180.0,
                    "smape": 0.04,
                    "r2": 0.92,
                    "uncertainty": {"picp_90": 0.93, "mean_interval_width": 380.0},
                },
                "nbeats_darts": {
                    "rmse": 145.0,
                    "mae": 108.0,
                    "smape": 0.021,
                    "r2": 0.97,
                    "uncertainty": {"picp_90": 0.88, "mean_interval_width": 295.0},
                },
                "ngboost": {
                    "rmse": 160.0,
                    "mae": 115.0,
                    "smape": 0.023,
                    "r2": 0.965,
                    "uncertainty": {"picp_90": 0.92, "mean_interval_width": 310.0},
                },
                "flaml": {
                    "rmse": 110.0,
                    "mae": 80.0,
                    "smape": 0.012,
                    "r2": 0.985,
                    "uncertainty": {"picp_90": 0.90, "mean_interval_width": 240.0},
                },
            }
        }
    }
    p = tmp_path / "week2_metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_extract_rows_emits_all_ten_models(tmp_path: Path) -> None:
    metrics = _build_metrics(tmp_path)
    rows = module.extract_rows("DE", metrics, tmp_path)
    load_rows = [r for r in rows if r["Target"] == "Load"]
    assert len(load_rows) == len(module.MODEL_ORDER)
    models = {r["Model"] for r in load_rows}
    assert "Prophet" in models
    assert "NGBoost" in models
    assert "FLAML" in models
    for r in load_rows:
        if r["Model"] in {"GBM", "Prophet", "NGBoost"}:
            assert r["PICP@90 (%)"] is not None
            assert r["Interval Width (MW)"] is not None


def test_latex_writer_bolds_per_row_best(tmp_path: Path) -> None:
    metrics = _build_metrics(tmp_path)
    rows = module.extract_rows("DE", metrics, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_path = out_dir / "baseline_comparison_de.tex"
    module.write_latex(rows, out_path, "DE")
    text = out_path.read_text(encoding="utf-8")
    assert "GBM" in text
    assert "Prophet" in text
    assert r"\textbf{GBM}" in text
    assert r"\textbf{100.00}" in text
