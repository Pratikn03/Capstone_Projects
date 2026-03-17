"""DC3S audit health metrics and sustained trigger evaluation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import yaml


DEFAULT_DC3S_HEALTH = {
    "enabled": True,
    "lookback_hours": 24,
    "min_commands": 50,
    "intervention_rate_threshold": 0.30,
    "low_reliability_w_threshold": 0.60,
    "low_reliability_rate_threshold": 0.25,
    "drift_flag_rate_threshold": 0.10,
    "inflation_p95_threshold": 2.0,
    "sustained_windows": 3,
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"1", "true", "yes", "y"}:
            return True
        if lower in {"0", "false", "no", "n"}:
            return False
    return None


def _load_json(payload_json: Any) -> dict[str, Any]:
    if not isinstance(payload_json, str) or not payload_json:
        return {}
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_intervened(payload: dict[str, Any]) -> bool | None:
    top = _safe_bool(payload.get("intervened"))
    if top is not None:
        return top
    shield = payload.get("uncertainty", {}).get("shield_repair", {}) if isinstance(payload.get("uncertainty"), dict) else {}
    if isinstance(shield, dict):
        return _safe_bool(shield.get("repaired"))
    return None


def _extract_intervention_reason(payload: dict[str, Any]) -> str | None:
    reason = payload.get("intervention_reason")
    if isinstance(reason, str) and reason:
        return reason
    shield = payload.get("uncertainty", {}).get("shield_repair", {}) if isinstance(payload.get("uncertainty"), dict) else {}
    if isinstance(shield, dict):
        robust_meta = shield.get("robust_meta")
        if isinstance(robust_meta, dict):
            robust_reason = robust_meta.get("reason")
            if isinstance(robust_reason, str) and robust_reason:
                return robust_reason
        repaired = _safe_bool(shield.get("repaired"))
        if repaired:
            return "projection_clip"
    return None


def _extract_reliability_w(payload: dict[str, Any]) -> float | None:
    top = _safe_float(payload.get("reliability_w"))
    if top is not None:
        return top
    reliability = payload.get("reliability")
    if isinstance(reliability, dict):
        return _safe_float(reliability.get("w_t"))
    return None


def _extract_drift_flag(payload: dict[str, Any]) -> bool | None:
    top = _safe_bool(payload.get("drift_flag"))
    if top is not None:
        return top
    drift = payload.get("drift")
    if isinstance(drift, dict):
        return _safe_bool(drift.get("drift"))
    return None


def _extract_inflation(payload: dict[str, Any]) -> float | None:
    top = _safe_float(payload.get("inflation"))
    if top is not None:
        return top
    uncertainty = payload.get("uncertainty")
    if isinstance(uncertainty, dict):
        meta = uncertainty.get("meta")
        if isinstance(meta, dict):
            return _safe_float(meta.get("inflation"))
    return None


def load_dc3s_audit_config(path: str | Path = "configs/dc3s.yaml") -> dict[str, str]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {
            "duckdb_path": "data/audit/dc3s_audit.duckdb",
            "table_name": "dispatch_certificates",
        }
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    audit = dc3s.get("audit", {}) if isinstance(dc3s, dict) else {}
    return {
        "duckdb_path": str(audit.get("duckdb_path", "data/audit/dc3s_audit.duckdb")),
        "table_name": str(audit.get("table_name", "dispatch_certificates")),
    }


def load_dc3s_health_config(path: str | Path = "configs/monitoring.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return dict(DEFAULT_DC3S_HEALTH)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    dc3s_health = payload.get("dc3s_health", {}) if isinstance(payload, dict) else {}
    merged = dict(DEFAULT_DC3S_HEALTH)
    if isinstance(dc3s_health, dict):
        merged.update(dc3s_health)
    return merged


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_sustained_windows(
    *,
    raw_flags: set[str],
    sustained_windows: int,
    state_path: Path,
    update_state: bool,
) -> tuple[list[str], dict[str, int]]:
    state = _load_state(state_path)
    root = state.setdefault("dc3s_health", {})
    counters = root.get("consecutive_breaches", {})
    if not isinstance(counters, dict):
        counters = {}

    known_flags = ["intervention_rate", "low_reliability_rate", "drift_flag_rate", "inflation_p95"]
    next_counters: dict[str, int] = {}
    triggered: list[str] = []

    for flag in known_flags:
        prev = int(counters.get(flag, 0) or 0)
        if flag in raw_flags:
            cur = prev + 1
        else:
            cur = 0
        next_counters[flag] = cur
        if cur >= sustained_windows and flag in raw_flags:
            triggered.append(flag)

    if update_state:
        root["consecutive_breaches"] = next_counters
        root["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["dc3s_health"] = root
        _save_state(state_path, state)

    return sorted(triggered), next_counters


def compute_dc3s_health(
    *,
    window_hours: int,
    min_commands: int,
    thresholds: dict[str, float],
    duckdb_path: str,
    table_name: str,
    sustained_windows: int = 3,
    state_path: str | Path = "reports/monitoring_state.json",
    update_state: bool = True,
) -> dict[str, Any]:
    output = {
        "window_hours": int(window_hours),
        "commands_total": 0,
        "intervention_rate": 0.0,
        "low_reliability_rate": 0.0,
        "drift_flag_rate": 0.0,
        "inflation_p95": 0.0,
        "triggered_flags": [],
        "triggered": False,
        "insufficient_data": True,
        "sustained_windows": int(sustained_windows),
        "sustained_breach_counts": {},
    }

    db_path = Path(duckdb_path)
    if not db_path.exists():
        return output

    since_ts = datetime.now(timezone.utc) - timedelta(hours=float(window_hours))
    conn = duckdb.connect(str(db_path))
    try:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        if not table_exists or int(table_exists[0]) == 0:
            return output

        table_cols = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        }
        intervened_col = "intervened" if "intervened" in table_cols else "NULL AS intervened"
        intervention_reason_col = (
            "intervention_reason" if "intervention_reason" in table_cols else "NULL AS intervention_reason"
        )
        reliability_w_col = "reliability_w" if "reliability_w" in table_cols else "NULL AS reliability_w"
        drift_flag_col = "drift_flag" if "drift_flag" in table_cols else "NULL AS drift_flag"
        inflation_col = "inflation" if "inflation" in table_cols else "NULL AS inflation"
        created_at_col = "created_at" if "created_at" in table_cols else "NULL AS created_at"

        rows = conn.execute(
            f"""
            SELECT
                command_id,
                {intervened_col},
                {intervention_reason_col},
                {reliability_w_col},
                {drift_flag_col},
                {inflation_col},
                payload_json
            FROM {table_name}
            WHERE TRY_CAST({created_at_col} AS TIMESTAMPTZ) >= ?
               OR TRY_CAST({created_at_col} AS TIMESTAMP) >= ?
            """,
            [since_ts, since_ts.replace(tzinfo=None)],
        ).fetchall()
    finally:
        conn.close()

    commands_total = len(rows)
    output["commands_total"] = commands_total
    if commands_total == 0:
        _, counters = _apply_sustained_windows(
            raw_flags=set(),
            sustained_windows=max(1, int(sustained_windows)),
            state_path=Path(state_path),
            update_state=update_state,
        )
        output["sustained_breach_counts"] = counters
        return output

    intervention_count = 0
    low_rel_count = 0
    drift_count = 0
    infl_values: list[float] = []

    low_rel_threshold = float(thresholds.get("low_reliability_w_threshold", 0.60))
    reasons: list[str] = []

    for row in rows:
        (
            _command_id,
            intervened,
            intervention_reason,
            reliability_w,
            drift_flag,
            inflation,
            payload_json,
        ) = row

        payload = _load_json(payload_json)

        i = _safe_bool(intervened)
        if i is None:
            i = _extract_intervened(payload)
        if i:
            intervention_count += 1

        reason = intervention_reason if isinstance(intervention_reason, str) else _extract_intervention_reason(payload)
        if isinstance(reason, str) and reason:
            reasons.append(reason)

        w = _safe_float(reliability_w)
        if w is None:
            w = _extract_reliability_w(payload)
        if w is not None and w < low_rel_threshold:
            low_rel_count += 1

        d = _safe_bool(drift_flag)
        if d is None:
            d = _extract_drift_flag(payload)
        if d:
            drift_count += 1

        infl = _safe_float(inflation)
        if infl is None:
            infl = _extract_inflation(payload)
        if infl is not None:
            infl_values.append(float(infl))

    intervention_rate = intervention_count / commands_total
    low_reliability_rate = low_rel_count / commands_total
    drift_flag_rate = drift_count / commands_total
    inflation_p95 = float(np.percentile(np.asarray(infl_values, dtype=float), 95)) if infl_values else 0.0

    output.update(
        {
            "intervention_rate": float(intervention_rate),
            "low_reliability_rate": float(low_reliability_rate),
            "drift_flag_rate": float(drift_flag_rate),
            "inflation_p95": float(inflation_p95),
            "top_intervention_reasons": sorted(set(reasons))[:5],
            "insufficient_data": commands_total < int(min_commands),
        }
    )

    if output["insufficient_data"]:
        _, counters = _apply_sustained_windows(
            raw_flags=set(),
            sustained_windows=max(1, int(sustained_windows)),
            state_path=Path(state_path),
            update_state=update_state,
        )
        output["sustained_breach_counts"] = counters
        return output

    raw_flags: set[str] = set()
    if intervention_rate > float(thresholds.get("intervention_rate_threshold", 0.30)):
        raw_flags.add("intervention_rate")
    if low_reliability_rate > float(thresholds.get("low_reliability_rate_threshold", 0.25)):
        raw_flags.add("low_reliability_rate")
    if drift_flag_rate > float(thresholds.get("drift_flag_rate_threshold", 0.10)):
        raw_flags.add("drift_flag_rate")
    if inflation_p95 > float(thresholds.get("inflation_p95_threshold", 2.0)):
        raw_flags.add("inflation_p95")

    triggered_flags, counters = _apply_sustained_windows(
        raw_flags=raw_flags,
        sustained_windows=max(1, int(sustained_windows)),
        state_path=Path(state_path),
        update_state=update_state,
    )
    output["raw_breach_flags"] = sorted(raw_flags)
    output["sustained_breach_counts"] = counters
    output["triggered_flags"] = triggered_flags
    output["triggered"] = bool(triggered_flags)
    return output
