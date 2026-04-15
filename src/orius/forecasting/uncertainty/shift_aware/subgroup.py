from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections import deque
from typing import Any

from .state import GroupCoverageStats


@dataclass
class SubgroupCoverageTracker:
    target_coverage: float = 0.9
    window_size: int = 128

    def __post_init__(self) -> None:
        self._groups: dict[str, GroupCoverageStats] = {}
        self._windows: dict[str, deque[dict[str, float | int]]] = {}

    @staticmethod
    def _bin_idx(value: float, n_bins: int) -> int:
        v = min(max(float(value), 0.0), 0.999999)
        return int(v * max(int(n_bins), 1))

    def build_group_key(
        self,
        *,
        reliability_score: float,
        volatility: float,
        fault_type: str | None,
        ts: str | None,
        custom_key: str | None = None,
        reliability_bins: int = 5,
        volatility_bins: int = 5,
    ) -> str:
        rel_key = f"rel:{self._bin_idx(reliability_score, reliability_bins)}"
        vol_key = f"vol:{self._bin_idx(volatility, volatility_bins)}"
        fault_key = f"fault:{fault_type or 'none'}"
        hour = 0
        if ts:
            try:
                hour = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).hour
            except ValueError:
                hour = 0
        time_key = f"hour:{hour:02d}"
        custom = f"custom:{custom_key}" if custom_key else "custom:none"
        return "|".join([rel_key, vol_key, fault_key, time_key, custom])

    def update(
        self,
        *,
        group_key: str,
        covered: bool,
        interval_width: float,
        abs_residual: float,
    ) -> GroupCoverageStats:
        window = self._windows.get(group_key)
        if window is None:
            window = deque(maxlen=max(1, int(self.window_size)))
            self._windows[group_key] = window
        window.append(
            {
                "covered": int(bool(covered)),
                "miss": int(not bool(covered)),
                "width": float(interval_width),
                "resid": float(abs_residual),
            }
        )

        stats = GroupCoverageStats(group_key=group_key, target_coverage=self.target_coverage)
        stats.count = len(window)
        stats.covered = int(sum(int(x["covered"]) for x in window))
        stats.miss_count = int(sum(int(x["miss"]) for x in window))
        stats.avg_interval_width = float(sum(float(x["width"]) for x in window) / max(stats.count, 1))
        stats.avg_abs_residual = float(sum(float(x["resid"]) for x in window) / max(stats.count, 1))
        self._groups[group_key] = stats
        return stats

    def group_rows(self) -> list[dict[str, Any]]:
        return [g.to_dict() for _, g in sorted(self._groups.items(), key=lambda kv: kv[0])]

    def max_under_coverage_gap(self) -> float:
        return max((g.under_coverage_gap for g in self._groups.values()), default=0.0)

    def set_window_size(self, window_size: int) -> None:
        next_size = max(1, int(window_size))
        if next_size == self.window_size:
            return
        self.window_size = next_size
        resized: dict[str, deque[dict[str, float | int]]] = {}
        for key, old in self._windows.items():
            resized[key] = deque(list(old)[-next_size:], maxlen=next_size)
        self._windows = resized
        for key, dq in self._windows.items():
            stats = GroupCoverageStats(group_key=str(key), target_coverage=self.target_coverage)
            stats.count = len(dq)
            stats.covered = int(sum(int(x["covered"]) for x in dq))
            stats.miss_count = int(sum(int(x["miss"]) for x in dq))
            stats.avg_interval_width = float(sum(float(x["width"]) for x in dq) / max(stats.count, 1))
            stats.avg_abs_residual = float(sum(float(x["resid"]) for x in dq) / max(stats.count, 1))
            self._groups[str(key)] = stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_coverage": float(self.target_coverage),
            "window_size": int(self.window_size),
            "groups": self.group_rows(),
            "windows": {
                key: list(values)
                for key, values in sorted(self._windows.items(), key=lambda kv: kv[0])
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SubgroupCoverageTracker":
        data = dict(payload or {})
        tracker = cls(
            target_coverage=float(data.get("target_coverage", 0.9)),
            window_size=int(data.get("window_size", 128)),
        )
        for key, values in dict(data.get("windows", {})).items():
            dq: deque[dict[str, float | int]] = deque(maxlen=max(1, int(tracker.window_size)))
            for value in values if isinstance(values, list) else []:
                if isinstance(value, dict):
                    dq.append(
                        {
                            "covered": int(value.get("covered", 0)),
                            "miss": int(value.get("miss", 0)),
                            "width": float(value.get("width", 0.0)),
                            "resid": float(value.get("resid", 0.0)),
                        }
                    )
            tracker._windows[str(key)] = dq
            stats = GroupCoverageStats(group_key=str(key), target_coverage=tracker.target_coverage)
            stats.count = len(dq)
            stats.covered = int(sum(int(x["covered"]) for x in dq))
            stats.miss_count = int(sum(int(x["miss"]) for x in dq))
            stats.avg_interval_width = float(sum(float(x["width"]) for x in dq) / max(stats.count, 1))
            stats.avg_abs_residual = float(sum(float(x["resid"]) for x in dq) / max(stats.count, 1))
            tracker._groups[str(key)] = stats
        return tracker
