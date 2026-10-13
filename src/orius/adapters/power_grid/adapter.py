"""Power Grid / Microgrid Inverter Domain Adapter for ORIUS/DC3S.

Implements grid-frequency (59.5 - 60.5 Hz) and bus voltage (0.95 - 1.05 pu)
safe operating envelopes under PMU telemetry degradation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from orius.dc3s.domain_adapter import DomainAdapter


class PowerGridInverterAdapter(DomainAdapter):
    """Adapter for grid-connected inverter active/reactive power dispatch."""

    def __init__(
        self,
        f_nom: float = 60.0,
        f_min: float = 59.5,
        f_max: float = 60.5,
        v_nom: float = 1.0,
        v_min: float = 0.95,
        v_max: float = 1.05,
        p_max_mw: float = 10.0,
        q_max_mvar: float = 5.0,
        droop_r: float = 0.05,
    ) -> None:
        self.f_nom = f_nom
        self.f_min = f_min
        self.f_max = f_max
        self.v_nom = v_nom
        self.v_min = v_min
        self.v_max = v_max
        self.p_max_mw = p_max_mw
        self.q_max_mvar = q_max_mvar
        self.droop_r = droop_r

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "frequency_voltage_box",
            "repair_mode": "socp_projection",
            "fallback_mode": "autonomous_droop_standby",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
        }

    def true_constraint_violated(self, state: Mapping[str, Any]) -> bool | None:
        f = float(state.get("frequency_hz", self.f_nom))
        v = float(state.get("voltage_pu", self.v_nom))
        return bool(f < self.f_min - 1e-4 or f > self.f_max + 1e-4 or v < self.v_min - 1e-4 or v > self.v_max + 1e-4)

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        violated = self.true_constraint_violated(observed_state)
        return not violated if violated is not None else None

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        f = float(state.get("frequency_hz", self.f_nom))
        v = float(state.get("voltage_pu", self.v_nom))
        f_margin = min(f - self.f_min, self.f_max - f)
        v_margin = min(v - self.v_min, self.v_max - v)
        return float(min(f_margin, v_margin))

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "frequency_hz": float(raw_packet.get("f_hz", self.f_nom)),
            "voltage_pu": float(raw_packet.get("v_pu", self.v_nom)),
            "p_flow_mw": float(raw_packet.get("p_mw", 0.0)),
            "q_flow_mvar": float(raw_packet.get("q_mvar", 0.0)),
            "is_stale": bool(raw_packet.get("is_stale", False)),
            "packet_delay_ms": float(raw_packet.get("delay_ms", 5.0)),
        }

    def compute_oqe(
        self, state_history: Sequence[Mapping[str, Any]]
    ) -> tuple[float, Mapping[str, Any]]:
        if not state_history:
            return 1.0, {"flags": []}

        latest = state_history[-1]
        flags = []
        score = 1.0

        if latest.get("is_stale"):
            flags.append("telemetry_stale")
            score *= 0.4

        delay = float(latest.get("packet_delay_ms", 5.0))
        if delay > 50.0:
            flags.append("high_latency")
            score *= max(0.2, 1.0 - (delay - 50.0) / 100.0)

        return float(max(0.0, min(1.0, score))), {"flags": flags, "delay_ms": delay}

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        oqe_score: float,
        conformal_quantile: float = 1.96,
    ) -> Mapping[str, Any]:
        # Inflation factor: grows as oqe_score approaches 0
        inflation = 1.0 + (1.0 - oqe_score) * 3.0
        f = float(state.get("frequency_hz", self.f_nom))
        v = float(state.get("voltage_pu", self.v_nom))

        delta_f = 0.02 * conformal_quantile * inflation
        delta_v = 0.005 * conformal_quantile * inflation

        return {
            "f_min": f - delta_f,
            "f_max": f + delta_f,
            "v_min": v - delta_v,
            "v_max": v + delta_v,
            "inflation": inflation,
        }

    def tighten_action_set(
        self,
        uncertainty_set: Mapping[str, Any],
        constraints: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        u_f_max = float(uncertainty_set["f_max"])
        u_f_min = float(uncertainty_set["f_min"])

        # If frequency could breach limits, curtail active power export/import
        p_allow_max = self.p_max_mw
        p_allow_min = -self.p_max_mw

        if u_f_max > self.f_max - 0.1:
            p_allow_max = max(0.0, self.p_max_mw * (self.f_max - u_f_max) / 0.1)
        if u_f_min < self.f_min + 0.1:
            p_allow_min = min(0.0, -self.p_max_mw * (u_f_min - self.f_min) / 0.1)

        return {
            "p_min": p_allow_min,
            "p_max": p_allow_max,
            "q_min": -self.q_max_mvar,
            "q_max": self.q_max_mvar,
            "is_empty": bool(p_allow_min > p_allow_max),
        }

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        admissible_set: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        cand_p = float(candidate_action.get("p_mw", 0.0))
        cand_q = float(candidate_action.get("q_mvar", 0.0))

        if admissible_set.get("is_empty"):
            # Fallback to local droop
            return {"p_mw": 0.0, "q_mvar": 0.0}, {"repaired": True, "fallback": True}

        p_min = float(admissible_set["p_min"])
        p_max = float(admissible_set["p_max"])
        q_min = float(admissible_set["q_min"])
        q_max = float(admissible_set["q_max"])

        safe_p = max(p_min, min(p_max, cand_p))
        safe_q = max(q_min, min(q_max, cand_q))
        intervened = not (math.isclose(safe_p, cand_p, abs_tol=1e-5) and math.isclose(safe_q, cand_q, abs_tol=1e-5))

        return {"p_mw": safe_p, "q_mvar": safe_q}, {"repaired": intervened, "fallback": False}

    def emit_certificate(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "domain": "power_grid_inverter",
            "schema_version": "1.0",
            "certified": True,
            "ttl_steps": 1,
            "payload": dict(record),
        }
