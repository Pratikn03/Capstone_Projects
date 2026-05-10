#!/usr/bin/env python3
"""Rebuild a conformal artifact and metrics block from saved backtest arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from orius.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval, save_conformal


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalise_horizon_wise(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("horizon-wise must be true or false")


def recalibrate(
    *,
    metrics_path: Path,
    target: str,
    model_key: str,
    calibration_npz: Path,
    test_npz: Path,
    uncertainty_artifact: Path,
    legacy_uncertainty_artifact: Path | None,
    method: str,
    alpha: float,
    horizon_wise: bool,
    q_multiplier: float,
    rolling_window: int,
) -> dict[str, Any]:
    metrics_payload = _load_json(metrics_path)
    target_payload = metrics_payload.get("targets", {}).get(target)
    if not isinstance(target_payload, dict):
        raise KeyError(f"Target {target!r} not found in {metrics_path}")
    model_payload = target_payload.get(model_key)
    if not isinstance(model_payload, dict):
        raise KeyError(f"Model {model_key!r} for target {target!r} not found in {metrics_path}")

    cal = np.load(calibration_npz)
    test = np.load(test_npz)
    cfg = ConformalConfig(
        alpha=alpha,
        method=method,  # type: ignore[arg-type]
        horizon_wise=horizon_wise,
        rolling=True,
        rolling_window=rolling_window,
        q_multiplier=q_multiplier,
    )
    interval = ConformalInterval(cfg)
    if method == "cqr":
        interval.fit_calibration_cqr(cal["y_true"], cal["q_lo"], cal["q_hi"])
        interval_metrics = interval.evaluate_intervals_cqr(
            test["y_true"],
            test["q_lo"],
            test["q_hi"],
            per_horizon=True,
        )
    else:
        interval.fit_calibration(cal["y_true"], cal["y_pred"])
        interval_metrics = interval.evaluate_intervals(
            test["y_true"],
            test["y_pred"],
            per_horizon=True,
        )

    meta = {
        "target": target,
        "model": model_payload.get("model", model_key),
        "artifact_model_key": model_key,
        "method": method,
        "alpha": alpha,
        "q_multiplier": q_multiplier,
        "horizon_wise": horizon_wise,
        "calibration_rows": int(np.asarray(cal["y_true"]).shape[0]),
        "test_rows": int(np.asarray(test["y_true"]).shape[0]),
        "global_coverage": interval_metrics["global_coverage"],
        "global_mean_width": interval_metrics["global_mean_width"],
        "picp_90": interval_metrics.get("picp_90"),
        "mean_interval_width": interval_metrics.get("mean_interval_width"),
        "pinball_loss_q05": interval_metrics.get("pinball_loss_q05"),
        "pinball_loss_q50": interval_metrics.get("pinball_loss_q50"),
        "pinball_loss_q95": interval_metrics.get("pinball_loss_q95"),
        "pinball_loss_mean": interval_metrics.get("pinball_loss_mean"),
        "winkler_score_90": interval_metrics.get("winkler_score_90"),
        "per_horizon_picp": interval_metrics.get("per_horizon_picp", {}),
        "per_horizon_mpiw": interval_metrics.get("per_horizon_mpiw", {}),
        "recalibrated_from_backtests": True,
    }
    save_conformal(uncertainty_artifact, interval, meta=meta)
    if legacy_uncertainty_artifact is not None:
        save_conformal(legacy_uncertainty_artifact, interval, meta=meta)

    model_payload["uncertainty"] = {
        "picp_90": interval_metrics.get("picp_90"),
        "picp_95": interval_metrics.get("picp_95"),
        "mean_interval_width": interval_metrics.get("mean_interval_width"),
        "pinball_loss_q05": interval_metrics.get("pinball_loss_q05"),
        "pinball_loss_q50": interval_metrics.get("pinball_loss_q50"),
        "pinball_loss_q95": interval_metrics.get("pinball_loss_q95"),
        "pinball_loss_mean": interval_metrics.get("pinball_loss_mean"),
        "winkler_score_90": interval_metrics.get("winkler_score_90"),
        "calibration_policy": {
            "method": method,
            "alpha": alpha,
            "horizon_wise": horizon_wise,
            "q_multiplier": q_multiplier,
            "recalibrated_from_backtests": True,
        },
    }
    _write_json(metrics_path, metrics_payload)
    return interval_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-key", default="gbm")
    parser.add_argument("--calibration-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--uncertainty-artifact", type=Path, required=True)
    parser.add_argument("--legacy-uncertainty-artifact", type=Path)
    parser.add_argument("--method", choices=("cqr", "residual"), default="cqr")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--horizon-wise", type=_normalise_horizon_wise, default=True)
    parser.add_argument("--q-multiplier", type=float, default=1.0)
    parser.add_argument("--rolling-window", type=int, default=720)
    args = parser.parse_args()

    metrics = recalibrate(
        metrics_path=args.metrics,
        target=args.target,
        model_key=args.model_key,
        calibration_npz=args.calibration_npz,
        test_npz=args.test_npz,
        uncertainty_artifact=args.uncertainty_artifact,
        legacy_uncertainty_artifact=args.legacy_uncertainty_artifact,
        method=args.method,
        alpha=args.alpha,
        horizon_wise=args.horizon_wise,
        q_multiplier=args.q_multiplier,
        rolling_window=args.rolling_window,
    )
    print(
        "[recalibrate_conformal_from_backtests] "
        f"{args.target}:{args.model_key} PICP90={metrics['picp_90']:.3f} "
        f"width={metrics['mean_interval_width']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
