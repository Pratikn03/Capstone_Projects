"""API router: forecast."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, Query
from pydantic import BaseModel

from gridpulse.forecasting.predict import load_model_bundle, predict_next_24h

router = APIRouter()


def _load_cfg(path: str | Path = "configs/forecast.yaml") -> dict:
    # Key: API endpoint handler
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {
            "data": {"features_path": "data/processed/features.parquet"},
            "models": {},
            "fallback_order": ["lstm", "tcn", "gbm"],
        }
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _resolve_model_path(target: str, cfg: dict) -> Optional[Path]:
    explicit = cfg.get("models", {}).get(target)
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    order = cfg.get("fallback_order", ["lstm", "tcn", "gbm"])
    candidates = []
    for kind in order:
        if kind == "gbm":
            candidates.append(f"gbm_*_{target}.pkl")
        else:
            candidates.append(f"{kind}_{target}.pt")
    for pat in candidates:
        for p in Path("artifacts/models").glob(pat):
            if p.exists():
                return p
    return None


@lru_cache(maxsize=16)
def _cached_bundle(path: str) -> Dict[str, Any]:
    return load_model_bundle(path)


class ForecastResponse(BaseModel):
    generated_at: str
    horizon_hours: int
    forecasts: Dict[str, Any]
    meta: Dict[str, Any] = {}


@router.get("", response_model=ForecastResponse)
def get_forecast(
    targets: Optional[str] = Query(default=None, description="Comma-separated targets"),
    horizon: int = Query(default=24, ge=1, le=168),
):
    cfg = _load_cfg()
    features_path = Path(cfg.get("data", {}).get("features_path", "data/processed/features.parquet"))
    if not features_path.exists():
        return ForecastResponse(
            generated_at=pd.Timestamp.utcnow().isoformat(),
            horizon_hours=horizon,
            forecasts={},
            meta={"note": f"Missing features file: {features_path}. Run the data pipeline first."},
        )

    df = pd.read_parquet(features_path)
    req_targets = [t.strip() for t in targets.split(",")] if targets else ["load_mw", "wind_mw", "solar_mw"]

    results: Dict[str, Any] = {}
    missing = []
    for tgt in req_targets:
        model_path = _resolve_model_path(tgt, cfg)
        if not model_path:
            missing.append(tgt)
            continue
        bundle = _cached_bundle(str(model_path))
        results[tgt] = predict_next_24h(df, bundle, horizon=horizon)

    meta = {"missing_targets": missing} if missing else {}
    if missing and not results:
        meta["note"] = "No trained model bundles found. Train models and update configs/forecast.yaml."
    return ForecastResponse(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        horizon_hours=horizon,
        forecasts=results,
        meta=meta,
    )
