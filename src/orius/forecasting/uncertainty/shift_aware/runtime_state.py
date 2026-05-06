from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aci import update_adaptive_quantile
from .state import AdaptiveQuantileState, ShiftAwareConfig, ShiftValidityState
from .subgroup import SubgroupCoverageTracker


@dataclass
class ShiftAwareRuntimeState:
    tracker: SubgroupCoverageTracker
    adaptive: AdaptiveQuantileState
    last_validity: ShiftValidityState | None = None
    step: int = 0
    validity_trace: list[dict[str, Any]] = field(default_factory=list)
    adaptive_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "last_validity": self.last_validity.to_dict() if self.last_validity else None,
            "step": int(self.step),
            "validity_trace": list(self.validity_trace),
            "adaptive_trace": list(self.adaptive_trace),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, cfg: ShiftAwareConfig) -> ShiftAwareRuntimeState:
        data = dict(payload or {})
        adaptive_raw = dict(data.get("adaptive", {}))
        adaptive = AdaptiveQuantileState(
            mode=str(adaptive_raw.get("mode", cfg.aci_mode)),
            base_alpha=float(adaptive_raw.get("base_alpha", cfg.alpha)),
            effective_alpha=float(adaptive_raw.get("effective_alpha", cfg.alpha)),
            learning_rate=float(adaptive_raw.get("learning_rate", cfg.adaptation_step)),
            alpha_min=float(adaptive_raw.get("alpha_min", cfg.alpha_min)),
            alpha_max=float(adaptive_raw.get("alpha_max", cfg.alpha_max)),
            updates=int(adaptive_raw.get("updates", 0)),
            miss_streak=int(adaptive_raw.get("miss_streak", 0)),
        )
        tracker = SubgroupCoverageTracker.from_dict(data.get("tracker"))
        tracker.set_window_size(int(cfg.coverage_window_size))
        return cls(
            tracker=tracker,
            adaptive=adaptive,
            step=int(data.get("step", 0)),
            validity_trace=list(data.get("validity_trace", [])),
            adaptive_trace=list(data.get("adaptive_trace", [])),
        )


class ShiftAwareRuntimeEngine:
    def __init__(self, cfg: ShiftAwareConfig, state_path: str | None = None):
        self.cfg = cfg
        self.state_path = Path(state_path) if state_path else None
        self.state = self._load()

    def _load(self) -> ShiftAwareRuntimeState:
        if self.state_path is None or not self.state_path.exists():
            return ShiftAwareRuntimeState(
                tracker=SubgroupCoverageTracker(
                    target_coverage=self.cfg.coverage_target, window_size=self.cfg.coverage_window_size
                ),
                adaptive=AdaptiveQuantileState(
                    mode=self.cfg.aci_mode,
                    base_alpha=self.cfg.alpha,
                    effective_alpha=self.cfg.alpha,
                    learning_rate=self.cfg.adaptation_step,
                    alpha_min=self.cfg.alpha_min,
                    alpha_max=self.cfg.alpha_max,
                ),
            )
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return ShiftAwareRuntimeState.from_dict(payload if isinstance(payload, dict) else {}, self.cfg)

    def save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.to_dict(), sort_keys=True), encoding="utf-8")

    def record_adaptive_step(self, miss: bool) -> None:
        update_adaptive_quantile(self.state.adaptive, miss=miss)
        self.state.adaptive_trace.append(
            {
                "t": int(self.state.step),
                "effective_alpha": float(self.state.adaptive.effective_alpha),
                "miss_streak": int(self.state.adaptive.miss_streak),
                "mode": self.state.adaptive.mode,
            }
        )

    def record_validity(self, validity: ShiftValidityState) -> None:
        self.state.last_validity = validity
        self.state.validity_trace.append(
            {
                "t": int(self.state.step),
                **validity.to_dict(),
            }
        )
        self.state.step += 1
