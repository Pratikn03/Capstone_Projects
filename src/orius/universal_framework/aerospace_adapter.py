"""Aerospace domain adapter for ORIUS Universal Framework.

Placeholder for thesis domain 4 (Aerospace). Safety predicates:
- altitude_m above ground
- airspeed_kt in [v_min, v_max]
- bank_angle_deg in [-max_bank, max_bank]
- fuel_remaining_pct >= fuel_min
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.quality import compute_reliability
from orius.dc3s.certificate import make_certificate


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class AerospaceDomainAdapter(DomainAdapter):
    """DomainAdapter for aerospace flight envelope monitoring (placeholder)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        ac = self._cfg.get("aerospace", {})
        self._v_min_kt = _f(ac.get("v_min_kt"), 60.0)
        self._v_max_kt = _f(ac.get("v_max_kt"), 350.0)
        self._max_bank_deg = _f(ac.get("max_bank_deg"), 30.0)
        self._fuel_min_pct = _f(ac.get("fuel_min_pct"), 10.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw flight telemetry into state vector z_t."""
        out = {}
        for k in ("altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("airspeed_kt", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability w_t. Map airspeed_kt -> load_mw for OQE."""
        event = {
            "ts_utc": state.get("ts_utc", ""),
            "load_mw": state.get("airspeed_kt", state.get("load_mw", 0.0)),
            "renewables_mw": state.get("altitude_m", 0.0) / 1000.0,
        }
        last_event = None
        if history:
            prev = history[-1]
            last_event = {
                "ts_utc": prev.get("ts_utc", ""),
                "load_mw": prev.get("airspeed_kt", prev.get("load_mw", 0.0)),
                "renewables_mw": prev.get("altitude_m", 0.0) / 1000.0,
            }
        reliability_cfg = self._cfg.get("reliability", {})
        ftit_cfg = self._cfg.get("ftit", {})
        w_t, flags = compute_reliability(
            event,
            last_event,
            expected_cadence_s=self._expected_cadence_s,
            reliability_cfg=reliability_cfg,
            ftit_cfg=ftit_cfg,
        )
        return float(w_t), {"flags": flags}

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Conformal box over (altitude, airspeed, bank, fuel) inflated by w_t."""
        alt = _f(state.get("altitude_m"), 1000.0)
        v = _f(state.get("airspeed_kt"), 150.0)
        bank = _f(state.get("bank_angle_deg"), 0.0)
        fuel = _f(state.get("fuel_remaining_pct"), 50.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.5, float(quantile) / (w + 1e-9))
        margin = min(margin, 10.0)
        uncertainty = {
            "alt_lower_m": max(0.0, alt - margin * 100),
            "alt_upper_m": alt + margin * 100,
            "v_lower_kt": max(0.0, v - margin),
            "v_upper_kt": v + margin,
            "bank_lower_deg": max(-90.0, bank - margin),
            "bank_upper_deg": min(90.0, bank + margin),
            "fuel_lower_pct": max(0.0, fuel - margin),
            "fuel_upper_pct": min(100.0, fuel + margin),
            "meta": {"inflation": margin, "w_t": w},
        }
        meta = {"inflation": margin, "w_t": w}
        return uncertainty, meta

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Feasible command set from flight envelope limits."""
        return {
            "uncertainty": dict(uncertainty),
            "constraints": dict(constraints),
            "cfg": dict(cfg),
        }

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        tightened_set: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> tuple[Mapping[str, float], Mapping[str, Any]]:
        """Clip command to safe flight envelope. Aerospace: throttle, bank, pitch."""
        throttle = _f(candidate_action.get("throttle", candidate_action.get("throttle_pct", 0.0)), 0.0)
        bank_cmd = _f(candidate_action.get("bank_deg", 0.0), 0.0)
        unc = tightened_set.get("uncertainty", uncertainty)
        v_lo = _f(unc.get("v_lower_kt"), self._v_min_kt)
        fuel_lo = _f(unc.get("fuel_lower_pct"), self._fuel_min_pct)
        throttle_safe = max(0.0, min(1.0, throttle))
        bank_safe = max(-self._max_bank_deg, min(self._max_bank_deg, bank_cmd))
        if v_lo < self._v_min_kt:
            throttle_safe = min(throttle_safe, 0.8)
        if fuel_lo < self._fuel_min_pct:
            throttle_safe = min(throttle_safe, 0.5)
        repaired = (
            abs(throttle_safe - throttle) > 1e-9 or abs(bank_safe - bank_cmd) > 1e-9
        )
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_throttle": throttle,
            "original_bank": bank_cmd,
        }
        if repaired:
            meta["intervention_reason"] = "envelope_clamp"
        return {
            "throttle": float(throttle_safe),
            "bank_deg": float(bank_safe),
        }, meta

    def emit_certificate(
        self,
        *,
        command_id: str,
        device_id: str,
        zone_id: str,
        controller: str,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        reliability: Mapping[str, Any],
        drift: Mapping[str, Any],
        cfg: Mapping[str, Any],
        prev_hash: str | None = None,
        dispatch_plan: Mapping[str, Any] | None = None,
        repair_meta: Mapping[str, Any] | None = None,
        guarantee_meta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Emit certificate with aerospace action fields."""
        model_hash = str(cfg.get("model_hash", ""))
        config_hash = str(cfg.get("config_hash", ""))
        intervened = bool(repair_meta.get("repaired")) if isinstance(repair_meta, Mapping) else None
        intervention_reason = repair_meta.get("intervention_reason") if isinstance(repair_meta, Mapping) else None
        reliability_w = float(reliability.get("w_t", reliability.get("w", 1.0)))
        drift_flag = bool(drift.get("drift", False))
        inflation = float(uncertainty.get("meta", {}).get("inflation", 1.0))
        return make_certificate(
            command_id=command_id,
            device_id=device_id,
            zone_id=zone_id,
            controller=controller,
            proposed_action=dict(proposed_action),
            safe_action=dict(safe_action),
            uncertainty=uncertainty,
            reliability=reliability,
            drift=drift,
            model_hash=model_hash,
            config_hash=config_hash,
            prev_hash=prev_hash,
            dispatch_plan=dispatch_plan,
            intervened=intervened,
            intervention_reason=intervention_reason,
            reliability_w=reliability_w,
            drift_flag=drift_flag,
            inflation=inflation,
        )
