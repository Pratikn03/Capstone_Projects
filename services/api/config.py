"""Serving configuration helpers for the API layer."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


@lru_cache(maxsize=1)
def load_serving_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or os.getenv("GRIDPULSE_SERVING_CONFIG", "configs/serving.yaml"))
    return _load_yaml(cfg_path)


@lru_cache(maxsize=1)
def load_uncertainty_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or os.getenv("GRIDPULSE_UNCERTAINTY_CONFIG", "configs/uncertainty.yaml"))
    return _load_yaml(cfg_path)


def get_conformal_path(target: str, cfg: Optional[dict] = None) -> Path:
    cfg = cfg or load_uncertainty_config()
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts/uncertainty"))
    return artifacts_dir / f"{target}_conformal.json"


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_bms_config(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_serving_config()
    bms_cfg = cfg.get("safety", {}).get("bms", {})
    return {
        "capacity_mwh": _read_float_env("GRIDPULSE_BMS_CAPACITY_MWH", bms_cfg.get("capacity_mwh", 10.0)),
        "max_power_mw": _read_float_env("GRIDPULSE_BMS_MAX_POWER_MW", bms_cfg.get("max_power_mw", 5.0)),
        "min_soc_pct": _read_float_env("GRIDPULSE_BMS_MIN_SOC_PCT", bms_cfg.get("min_soc_pct", 0.05)),
        "max_soc_pct": _read_float_env("GRIDPULSE_BMS_MAX_SOC_PCT", bms_cfg.get("max_soc_pct", 0.95)),
    }


def get_watchdog_timeout(cfg: Optional[dict] = None) -> int:
    cfg = cfg or load_serving_config()
    watchdog_cfg = cfg.get("safety", {}).get("watchdog", {})
    return _read_int_env("GRIDPULSE_WATCHDOG_TIMEOUT_SECONDS", watchdog_cfg.get("timeout_seconds", 30))


def _load_api_keys_from_env() -> Optional[Dict[str, Any]]:
    raw = os.getenv("GRIDPULSE_API_KEYS")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return _load_yaml(path)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=1)
def get_api_keys(cfg: Optional[dict] = None) -> Dict[str, Any]:
    cfg = cfg or load_serving_config()
    env_keys = _load_api_keys_from_env()
    if env_keys is not None:
        return env_keys
    return cfg.get("security", {}).get("api_keys", {})
