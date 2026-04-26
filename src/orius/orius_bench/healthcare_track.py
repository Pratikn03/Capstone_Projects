"""ORIUS-Bench healthcare track — bounded monitoring and alert release."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class HealthcareTrackAdapter(BenchmarkAdapter):
    """Healthcare monitoring track over the promoted MIMIC runtime row."""

    def __init__(
        self,
        spo2_min_pct: float = 90.0,
        hr_min_bpm: float = 40.0,
        hr_max_bpm: float = 120.0,
        dt: float = 0.25,
        dataset_path: str | Path | None = None,
    ):
        self._spo2_min = spo2_min_pct
        self._hr_min = hr_min_bpm
        self._hr_max = hr_max_bpm
        self._rr_min = 8.0
        self._rr_max = 30.0
        self._dt = dt
        self._spo2 = 97.0
        self._hr = 72.0
        self._rr = 14.0
        self._forecast_spo2 = 97.0
        self._reliability = 1.0
        self._patient_id = "patient-0"
        self._timestamp = ""
        self._is_critical = False
        self._rng: np.random.Generator | None = None
        self._episodes: list[list[dict[str, Any]]] = []
        self._episode_ids: list[str] = []
        self._episode_index_by_id: dict[str, int] = {}
        self._episode: list[dict[str, Any]] = []
        self._episode_idx = 0
        self._last_action: dict[str, Any] = {}
        if dataset_path is not None:
            from orius.orius_bench.real_data_loader import load_healthcare_runtime_rows

            rows = load_healthcare_runtime_rows(Path(dataset_path))
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("patient_id", "patient-0")), []).append(dict(row))
            for patient_id in sorted(grouped):
                episode = grouped[patient_id]
                episode.sort(key=lambda row: int(row.get("step", 0)))
                self._episodes.append(episode)
                self._episode_ids.append(patient_id)
            self._episode_index_by_id = {
                patient_id: index for index, patient_id in enumerate(self._episode_ids)
            }

    def _apply_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        self._spo2 = float(row.get("spo2_pct", 97.0))
        self._hr = float(row.get("hr_bpm", 72.0))
        self._rr = float(row.get("respiratory_rate", 14.0))
        self._forecast_spo2 = float(row.get("forecast_spo2_pct", self._spo2))
        self._reliability = float(row.get("reliability", 1.0))
        self._patient_id = str(row.get("patient_id", "patient-0"))
        self._timestamp = str(row.get("ts_utc", ""))
        self._is_critical = bool(row.get("is_critical", False))
        return self.true_state()

    @property
    def episode_ids(self) -> list[str]:
        return list(self._episode_ids)

    def episode_length(self, patient_id: str) -> int:
        if patient_id not in self._episode_index_by_id:
            raise KeyError(f"Unknown patient_id {patient_id!r}")
        return len(self._episodes[self._episode_index_by_id[patient_id]])

    def load_episode(self, patient_id: str, *, start_step: int = 0) -> Mapping[str, Any]:
        if patient_id not in self._episode_index_by_id:
            raise KeyError(f"Unknown patient_id {patient_id!r}")
        if self._rng is None:
            self._rng = np.random.default_rng(0)
        episode_index = self._episode_index_by_id[patient_id]
        self._episode = self._episodes[episode_index]
        self._episode_idx = max(0, min(int(start_step), len(self._episode) - 1))
        self._last_action = {}
        return self._apply_row(self._episode[self._episode_idx])

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if self._episodes:
            episode_index = int(self._rng.integers(0, len(self._episodes)))
            return self.load_episode(self._episode_ids[episode_index], start_step=0)
        self._spo2 = 92.0
        self._hr = 72.0
        self._rr = 14.0
        self._forecast_spo2 = 92.0
        self._reliability = 1.0
        self._patient_id = "patient-0"
        self._timestamp = ""
        self._is_critical = False
        self._last_action = {}
        return self.true_state()

    @property
    def using_real_data(self) -> bool:
        return bool(self._episodes)

    def true_state(self) -> Mapping[str, Any]:
        return {
            "spo2_pct": float(self._spo2),
            "hr_bpm": float(self._hr),
            "respiratory_rate": float(self._rr),
            "forecast_spo2_pct": float(self._forecast_spo2),
            "reliability": float(self._reliability),
            "patient_id": self._patient_id,
            "ts_utc": self._timestamp,
            "is_critical": bool(self._is_critical),
        }

    def observe(
        self,
        true_state: Mapping[str, Any],
        fault: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        obs = dict(true_state)
        if fault is None:
            return obs
        kind = str(fault.get("kind", ""))
        if kind == "blackout":
            obs["spo2_pct"] = float("nan")
            obs["hr_bpm"] = float("nan")
            obs["respiratory_rate"] = float("nan")
            obs["forecast_spo2_pct"] = float("nan")
            return obs
        if kind == "bias":
            obs["spo2_pct"] = float(obs.get("spo2_pct", 0.0)) + float(fault.get("magnitude", 0.0))
        elif kind == "noise":
            sigma = float(fault.get("sigma", 3.0))
            if self._rng is None:
                raise RuntimeError("HealthcareTrackAdapter.reset() must be called before observe()")
            obs["spo2_pct"] = float(obs.get("spo2_pct", 0.0)) + float(self._rng.normal(0.0, sigma))
        elif kind == "stuck_sensor":
            obs["spo2_pct"] = float(fault.get("frozen_value", 95.0))
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del state, uncertainty
        return {"spo2_min_pct": self._spo2_min}

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        self._last_action = dict(action)
        if self._episodes:
            self._episode_idx = min(self._episode_idx + 1, len(self._episode) - 1)
            return dict(self._apply_row(self._episode[self._episode_idx]))
        alert = max(0.0, min(1.0, float(action.get("alert_level", 0.2))))
        self._spo2 = max(70.0, min(100.0, self._spo2 + (2.0 * alert - 0.1) * self._dt))
        self._hr = max(self._hr_min, min(self._hr_max, 72.0 + 5.0 * (alert - 0.5)))
        return dict(self.true_state())

    def release_contract_status(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            spo2 = float(state.get("spo2_pct"))
            hr = float(state.get("hr_bpm"))
            rr = float(state.get("respiratory_rate"))
        except (TypeError, ValueError):
            return {
                "requires_max_alert": True,
                "max_alert_satisfied": False,
                "violated": True,
                "reason": "unevaluable_patient_state",
            }
        alert = max(0.0, min(1.0, float(state.get("alert_level", state.get("alert_level_executed", 0.0)) or 0.0)))
        certificate_valid = state.get("certificate_valid")
        validity_status = str(state.get("validity_status", "nominal") or "nominal")
        release_requires_max_alert = bool(state.get("release_requires_max_alert", False))
        projected_release_valid = bool(state.get("projected_release_valid", False))
        unsafe_vitals = (
            not math.isfinite(spo2)
            or not math.isfinite(hr)
            or not math.isfinite(rr)
            or spo2 < self._spo2_min
            or hr < self._hr_min
            or hr > self._hr_max
            or rr < self._rr_min
            or rr > self._rr_max
        )
        certificate_degraded = certificate_valid is False or (
            validity_status in {"invalid", "degraded"} and not projected_release_valid
        )
        requires_max_alert = bool(release_requires_max_alert or unsafe_vitals or certificate_degraded)
        max_alert_satisfied = alert >= 0.999
        violated = requires_max_alert and not max_alert_satisfied
        if violated:
            if certificate_degraded:
                reason = "certificate_degraded_without_release"
            elif unsafe_vitals:
                reason = "unsafe_vitals_without_release"
            else:
                reason = "max_alert_required"
        elif requires_max_alert:
            reason = "fail_safe_release"
        else:
            reason = "hold_region"
        return {
            "requires_max_alert": requires_max_alert,
            "max_alert_satisfied": max_alert_satisfied,
            "violated": violated,
            "reason": reason,
        }

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for rec in trajectory:
            if not self.release_contract_status(rec)["violated"]:
                alert = max(0.0, min(1.0, float(rec.get("alert_level", rec.get("alert_level_executed", 0.0)) or 0.0)))
                total += 1.0 - alert
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        status = self.release_contract_status(state)
        return {"violated": bool(status["violated"]), "severity": 1.0 if status["violated"] else 0.0}

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        for key in ("spo2_pct", "hr_bpm", "respiratory_rate"):
            try:
                value = float(observed_state.get(key))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
        return not bool(self.release_contract_status(observed_state)["requires_max_alert"])

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        try:
            spo2 = float(state.get("spo2_pct"))
            hr = float(state.get("hr_bpm"))
            rr = float(state.get("respiratory_rate"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(spo2) or not math.isfinite(hr) or not math.isfinite(rr):
            return None
        contract = self.release_contract_status(state)
        if contract["requires_max_alert"]:
            alert = max(0.0, min(1.0, float(state.get("alert_level", state.get("alert_level_executed", 0.0)) or 0.0)))
            return float(alert - 1.0)
        return min(
            spo2 - self._spo2_min,
            hr - self._hr_min,
            self._hr_max - hr,
            rr - self._rr_min,
            self._rr_max - rr,
        )

    @property
    def domain_name(self) -> str:
        return "healthcare"
