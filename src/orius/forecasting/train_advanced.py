"""Unified trainer for advanced forecasting baselines on the canonical splits.

Trains Prophet, NGBoost, Darts-N-BEATS, and FLAML on the same train/cal/test
slices, horizon, lookback, and seed list as the GBM/DL pipeline. Emits

    artifacts/runs/{region}/{release_id}/advanced_baselines/{model}_{target}.json
    artifacts/uncertainty/{region}/{model}_{target}_conformal.json

and merges per-model metrics into the regional ``week2_metrics.json`` so the
publication table builder picks them up without schema drift.

Native uncertainty (Prophet's `interval_width`, NGBoost's distributional ppf)
is recorded *and* a conformal sidecar is fitted on the calibration window —
both columns can then be reported side-by-side in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from orius.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval, save_conformal
from orius.forecasting.uncertainty.distributional import (
    NGBoostConfig,
    predict_ngboost_quantiles,
    train_ngboost_distribution,
)
from orius.utils.metrics import mae as _mae
from orius.utils.metrics import r2_score as _r2
from orius.utils.metrics import rmse as _rmse
from orius.utils.metrics import smape as _smape

logger = logging.getLogger(__name__)

CANONICAL_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 2024, 1337, 7777, 9999)
DEFAULT_TARGETS: tuple[str, ...] = ("load_mw", "wind_mw", "solar_mw")
DEFAULT_ALPHA: float = 0.10


@dataclass
class SplitPaths:
    train: Path
    calibration: Path
    test: Path


@dataclass
class AdvancedTrainerConfig:
    region: str
    release_id: str
    splits: SplitPaths
    out_root: Path
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = CANONICAL_SEEDS
    horizon: int = 24
    lookback: int = 168
    alpha: float = DEFAULT_ALPHA
    holiday_country: str | None = None
    timestamp_col: str = "timestamp"
    feature_cols: tuple[str, ...] | None = None
    metrics_json: Path | None = None
    conformal_dir: Path | None = None
    enabled_models: tuple[str, ...] = ("prophet", "ngboost", "nbeats_darts", "flaml")
    flaml_time_budget: int = 600


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_features(df: pd.DataFrame, target: str, hint: tuple[str, ...] | None) -> list[str]:
    if hint:
        missing = [c for c in hint if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing in frame: {missing}")
        return list(hint)
    skip = {target, "timestamp", "utc_timestamp", "datetime", "date", "ds", "y"}
    cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("no numeric feature columns inferred; pass feature_cols explicitly")
    return cols


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    return {
        "rmse": float(_rmse(yt, yp)),
        "mae": float(_mae(yt, yp)),
        "smape": float(_smape(yt, yp)),
        "r2": float(_r2(yt, yp)),
    }


def _interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    n = min(len(y), len(lo), len(hi))
    y, lo, hi = y[:n], lo[:n], hi[:n]
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "picp_90": float(np.mean(covered)) if n else 0.0,
        "mean_interval_width": float(np.mean(width)) if n else 0.0,
    }


def _conformalize(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[ConformalInterval, np.ndarray, np.ndarray, dict[str, float]]:
    cfg = ConformalConfig(alpha=alpha, method="residual", horizon_wise=False, rolling=False)
    interval = ConformalInterval(cfg)
    interval.fit_calibration(np.asarray(y_cal, dtype=float), np.asarray(p_cal, dtype=float))
    lo, hi = interval.predict_interval(np.asarray(p_test, dtype=float).reshape(-1))
    metrics = _interval_metrics(y_test, lo, hi)
    return interval, np.asarray(lo), np.asarray(hi), metrics


def _aggregate_seed_metrics(per_seed: list[dict[str, float]]) -> dict[str, float]:
    if not per_seed:
        return {}
    out: dict[str, float] = {}
    for key in ("rmse", "mae", "smape", "r2"):
        vals = np.asarray([m[key] for m in per_seed if key in m], dtype=float)
        if vals.size:
            out[key] = float(np.median(vals))
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return out


def _resolve_ts_column(df: pd.DataFrame, hint: str) -> str:
    for candidate in (hint, "timestamp", "utc_timestamp", "datetime", "date"):
        if candidate in df.columns:
            return candidate
    raise ValueError("no timestamp column found in dataframe")


def train_prophet(
    *,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    timestamp_col: str,
    seed: int,
    alpha: float,
    holiday_country: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Prophet is not installed. Install the 'baselines' extra: pip install -e .[baselines]"
        ) from exc

    ts_col = _resolve_ts_column(train_df, timestamp_col)
    interval_width = 1.0 - alpha

    def to_pf(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(df[ts_col]).dt.tz_localize(None),
                "y": pd.to_numeric(df[target], errors="coerce"),
            }
        ).dropna()

    fit_df = pd.concat([to_pf(train_df)], ignore_index=True)
    cal_pf = to_pf(cal_df)
    test_pf = to_pf(test_df)

    np.random.seed(int(seed))
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=interval_width,
        uncertainty_samples=500,
    )
    if holiday_country:
        model.add_country_holidays(country_name=holiday_country)
    model.fit(fit_df, iter=300, seed=int(seed))

    cal_fcst = model.predict(cal_pf[["ds"]])
    test_fcst = model.predict(test_pf[["ds"]])
    return (
        cal_pf["y"].to_numpy(dtype=float),
        cal_fcst["yhat"].to_numpy(dtype=float),
        test_pf["y"].to_numpy(dtype=float),
        test_fcst["yhat"].to_numpy(dtype=float),
        test_fcst["yhat_lower"].to_numpy(dtype=float),
        test_fcst["yhat_upper"].to_numpy(dtype=float),
    )


def train_ngboost_target(
    *,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    seed: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = NGBoostConfig(random_state=int(seed))
    fit_frame = pd.concat([train_df, cal_df], axis=0, ignore_index=True)
    x_fit = fit_frame[feature_cols].to_numpy(dtype=float)
    y_fit = fit_frame[target].to_numpy(dtype=float)
    model = train_ngboost_distribution(x_fit, y_fit, cfg=cfg)

    x_cal = cal_df[feature_cols].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_cal = cal_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    cal_q = predict_ngboost_quantiles(model, x_cal, quantiles=(0.5,))[0.5]
    quantiles = (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)
    test_q = predict_ngboost_quantiles(model, x_test, quantiles=quantiles)
    q_lo = test_q[quantiles[0]]
    q_med = test_q[quantiles[1]]
    q_hi = test_q[quantiles[2]]
    return y_cal, cal_q, y_test, q_med, q_lo, q_hi


def train_nbeats_darts(
    *,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    timestamp_col: str,
    horizon: int,
    lookback: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler
        from darts.models import NBEATSModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Darts is not installed. Install the 'baselines' extra: pip install -e .[baselines]"
        ) from exc

    ts_col = _resolve_ts_column(train_df, timestamp_col)

    def to_series(df: pd.DataFrame) -> Any:
        sub = df[[ts_col, target]].copy()
        sub[ts_col] = pd.to_datetime(sub[ts_col]).dt.tz_localize(None)
        sub = sub.dropna().drop_duplicates(subset=[ts_col]).sort_values(ts_col)
        return TimeSeries.from_dataframe(sub, time_col=ts_col, value_cols=[target], freq="h")

    train_series = to_series(train_df)
    full_series = to_series(pd.concat([train_df, cal_df, test_df], ignore_index=True))

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)

    model = NBEATSModel(
        input_chunk_length=int(lookback),
        output_chunk_length=int(horizon),
        n_epochs=80,
        batch_size=64,
        random_state=int(seed),
        generic_architecture=True,
        force_reset=True,
        pl_trainer_kwargs={"accelerator": "auto", "enable_progress_bar": False},
    )
    model.fit(train_scaled, verbose=False)

    n_cal = max(len(to_series(cal_df)), horizon)
    n_test = max(len(to_series(test_df)), horizon)
    full_scaled = scaler.transform(full_series)
    pred_cal_scaled = model.predict(n=n_cal, series=train_scaled)
    pred_cal = scaler.inverse_transform(pred_cal_scaled).values().reshape(-1)
    cal_y = to_series(cal_df).values().reshape(-1)
    cut = min(len(pred_cal), len(cal_y))

    pred_test_scaled = model.predict(
        n=n_test,
        series=full_scaled[: -len(to_series(test_df))],
    )
    pred_test = scaler.inverse_transform(pred_test_scaled).values().reshape(-1)
    test_y = to_series(test_df).values().reshape(-1)
    cut_test = min(len(pred_test), len(test_y))
    return cal_y[:cut], pred_cal[:cut], test_y[:cut_test], pred_test[:cut_test]


def train_flaml_target(
    *,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    seed: int,
    time_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from flaml import AutoML
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FLAML is not installed. Install the 'baselines' extra: pip install -e .[baselines]"
        ) from exc

    fit_frame = pd.concat([train_df, cal_df], axis=0, ignore_index=True)
    x_fit = fit_frame[feature_cols].to_numpy(dtype=float)
    y_fit = fit_frame[target].to_numpy(dtype=float)
    x_cal = cal_df[feature_cols].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_cal = cal_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    automl = AutoML()
    automl.fit(
        x_fit,
        y_fit,
        task="regression",
        metric="rmse",
        time_budget=int(time_budget),
        seed=int(seed),
        verbose=0,
    )
    p_cal = automl.predict(x_cal)
    p_test = automl.predict(x_test)
    meta = {
        "best_estimator": str(automl.best_estimator),
        "best_config": automl.best_config,
        "time_budget_s": int(time_budget),
    }
    return y_cal, p_cal, y_test, p_test, meta


def _save_per_seed(
    *,
    out_dir: Path,
    region: str,
    model_key: str,
    target: str,
    seed: int,
    point: dict[str, float],
    native_uq: dict[str, float] | None,
    conformal_uq: dict[str, float] | None,
    extra: dict[str, Any] | None = None,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "region": region,
        "model": model_key,
        "target": target,
        "seed": int(seed),
        "metrics": point,
    }
    if native_uq:
        payload["uncertainty_native"] = native_uq
    if conformal_uq:
        payload["uncertainty_conformal"] = conformal_uq
    if extra:
        payload["extra"] = extra
    path = out_dir / f"{model_key}_{target}_seed{seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if y_true is not None and y_pred is not None:
        npz_path = out_dir / f"{model_key}_{target}_seed{seed}.npz"
        np.savez(
            npz_path,
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
        )
    return path


def _merge_into_metrics_json(
    *,
    metrics_json: Path,
    target: str,
    model_key: str,
    aggregated: dict[str, float],
    uncertainty: dict[str, float] | None,
    seeds: Iterable[int],
    native_uq: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = (
        json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {"targets": {}}
    )
    targets = payload.setdefault("targets", {})
    target_block = targets.setdefault(target, {})
    model_block: dict[str, Any] = {
        **{k: v for k, v in aggregated.items() if "_" not in k or k.endswith("_mean") or k.endswith("_std")},
        "model": model_key,
        "ensemble_members": len(list(seeds)),
        "seeds": list(seeds),
    }
    if uncertainty:
        model_block["uncertainty"] = {
            "picp_90": uncertainty.get("picp_90"),
            "mean_interval_width": uncertainty.get("mean_interval_width"),
        }
    if native_uq:
        model_block["uncertainty_native"] = native_uq
    target_block[model_key] = model_block
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_advanced_baselines(cfg: AdvancedTrainerConfig) -> dict[str, Any]:
    train_df = _load_table(cfg.splits.train)
    cal_df = _load_table(cfg.splits.calibration)
    test_df = _load_table(cfg.splits.test)
    feature_cols = _resolve_features(train_df, cfg.targets[0], cfg.feature_cols)

    runs_dir = (
        cfg.out_root / "artifacts" / "runs" / cfg.region.lower() / cfg.release_id / "advanced_baselines"
    )
    conformal_dir = cfg.conformal_dir or (cfg.out_root / "artifacts" / "uncertainty" / cfg.region.lower())
    metrics_json = cfg.metrics_json or (cfg.out_root / "reports" / cfg.region.lower() / "week2_metrics.json")

    summary: dict[str, Any] = {
        "region": cfg.region,
        "release_id": cfg.release_id,
        "models": {},
        "metrics_json": str(metrics_json),
        "conformal_dir": str(conformal_dir),
    }

    for target in cfg.targets:
        for model_key in cfg.enabled_models:
            per_seed_point: list[dict[str, float]] = []
            per_seed_conformal: list[dict[str, float]] = []
            native_seed_uq: list[dict[str, float]] = []
            extra_meta: dict[str, Any] = {}
            last_interval: ConformalInterval | None = None
            for seed in cfg.seeds:
                logger.info("training %s/%s seed=%s target=%s", model_key, cfg.region, seed, target)
                try:
                    if model_key == "prophet":
                        y_cal, p_cal, y_test, p_test, lo, hi = train_prophet(
                            train_df=train_df,
                            cal_df=cal_df,
                            test_df=test_df,
                            target=target,
                            timestamp_col=cfg.timestamp_col,
                            seed=seed,
                            alpha=cfg.alpha,
                            holiday_country=cfg.holiday_country,
                        )
                        native_uq = _interval_metrics(y_test, lo, hi)
                    elif model_key == "ngboost":
                        y_cal, p_cal, y_test, p_test, lo, hi = train_ngboost_target(
                            train_df=train_df,
                            cal_df=cal_df,
                            test_df=test_df,
                            target=target,
                            feature_cols=feature_cols,
                            seed=seed,
                            alpha=cfg.alpha,
                        )
                        native_uq = _interval_metrics(y_test, lo, hi)
                    elif model_key == "nbeats_darts":
                        y_cal, p_cal, y_test, p_test = train_nbeats_darts(
                            train_df=train_df,
                            cal_df=cal_df,
                            test_df=test_df,
                            target=target,
                            timestamp_col=cfg.timestamp_col,
                            horizon=cfg.horizon,
                            lookback=cfg.lookback,
                            seed=seed,
                        )
                        native_uq = None
                    elif model_key == "flaml":
                        y_cal, p_cal, y_test, p_test, meta = train_flaml_target(
                            train_df=train_df,
                            cal_df=cal_df,
                            test_df=test_df,
                            target=target,
                            feature_cols=feature_cols,
                            seed=seed,
                            time_budget=cfg.flaml_time_budget,
                        )
                        native_uq = None
                        extra_meta.setdefault("per_seed_meta", {})[str(seed)] = meta
                    else:
                        raise ValueError(f"unknown model key: {model_key}")
                except Exception as exc:
                    logger.warning(
                        "skipping %s/%s seed=%s target=%s: %s", model_key, cfg.region, seed, target, exc
                    )
                    continue

                point = _safe_metrics(y_test, p_test)
                per_seed_point.append(point)
                interval, lo_c, hi_c, conformal_uq = _conformalize(
                    y_cal=y_cal, p_cal=p_cal, y_test=y_test, p_test=p_test, alpha=cfg.alpha
                )
                per_seed_conformal.append(conformal_uq)
                if native_uq is not None:
                    native_seed_uq.append(native_uq)
                last_interval = interval
                _save_per_seed(
                    out_dir=runs_dir,
                    region=cfg.region,
                    model_key=model_key,
                    target=target,
                    seed=seed,
                    point=point,
                    native_uq=native_uq,
                    conformal_uq=conformal_uq,
                    y_true=y_test,
                    y_pred=p_test,
                )

            if not per_seed_point:
                summary["models"].setdefault(model_key, {})[target] = {"status": "skipped"}
                continue

            agg = _aggregate_seed_metrics(per_seed_point)
            agg_uq = {
                "picp_90": float(np.median([m["picp_90"] for m in per_seed_conformal])),
                "mean_interval_width": float(
                    np.median([m["mean_interval_width"] for m in per_seed_conformal])
                ),
            }
            agg_native = (
                {
                    "picp_90": float(np.median([m["picp_90"] for m in native_seed_uq])),
                    "mean_interval_width": float(
                        np.median([m["mean_interval_width"] for m in native_seed_uq])
                    ),
                }
                if native_seed_uq
                else None
            )

            if last_interval is not None:
                conformal_dir.mkdir(parents=True, exist_ok=True)
                save_conformal(
                    conformal_dir / f"{model_key}_{target}_conformal.json",
                    last_interval,
                    meta={
                        "model": model_key,
                        "target": target,
                        "region": cfg.region,
                        "release_id": cfg.release_id,
                        "alpha": cfg.alpha,
                        "global_coverage": agg_uq["picp_90"],
                        "global_mean_width": agg_uq["mean_interval_width"],
                        "picp_90": agg_uq["picp_90"],
                        "mean_interval_width": agg_uq["mean_interval_width"],
                        "seeds": list(cfg.seeds),
                    },
                )

            _merge_into_metrics_json(
                metrics_json=metrics_json,
                target=target,
                model_key=model_key,
                aggregated=agg,
                uncertainty=agg_uq,
                seeds=cfg.seeds,
                native_uq=agg_native,
            )

            summary["models"].setdefault(model_key, {})[target] = {
                "status": "ok",
                "n_seeds": len(per_seed_point),
                "aggregated": agg,
                "uncertainty_conformal": agg_uq,
                "uncertainty_native": agg_native,
            }

    summary_path = runs_dir / "summary.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Prophet/NGBoost/Darts-NBEATS/FLAML on canonical splits")
    p.add_argument("--region", required=True, help="DE or US (matches reports/{region}/week2_metrics.json)")
    p.add_argument("--release-id", required=True)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--out-root", default=Path.cwd(), type=Path)
    p.add_argument("--metrics-json", default=None, type=Path)
    p.add_argument("--conformal-dir", default=None, type=Path)
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(CANONICAL_SEEDS))
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--holiday-country", default=None)
    p.add_argument(
        "--models",
        nargs="+",
        default=["prophet", "ngboost", "nbeats_darts", "flaml"],
        help="Which advanced models to train",
    )
    p.add_argument("--flaml-time-budget", type=int, default=600)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    cfg = AdvancedTrainerConfig(
        region=args.region,
        release_id=args.release_id,
        splits=SplitPaths(train=args.train, calibration=args.calibration, test=args.test),
        out_root=args.out_root,
        targets=tuple(args.targets),
        seeds=tuple(args.seeds),
        horizon=int(args.horizon),
        lookback=int(args.lookback),
        alpha=float(args.alpha),
        holiday_country=args.holiday_country,
        metrics_json=args.metrics_json,
        conformal_dir=args.conformal_dir,
        enabled_models=tuple(args.models),
        flaml_time_budget=int(args.flaml_time_budget),
    )
    summary = run_advanced_baselines(cfg)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    ok = any(
        block.get("status") == "ok"
        for model_blocks in summary.get("models", {}).values()
        for block in model_blocks.values()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
