from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .state import GroupCoverageStats


@dataclass
class SubgroupCoverageTracker:
    target_coverage: float = 0.9

    def __post_init__(self) -> None:
        self._groups: dict[str, GroupCoverageStats] = {}

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
        stats = self._groups.get(group_key)
        if stats is None:
            stats = GroupCoverageStats(group_key=group_key, target_coverage=self.target_coverage)
            self._groups[group_key] = stats

        n_prev = stats.count
        stats.count += 1
        stats.covered += int(covered)
        stats.miss_count += int(not covered)
        stats.avg_interval_width = ((stats.avg_interval_width * n_prev) + float(interval_width)) / float(stats.count)
        stats.avg_abs_residual = ((stats.avg_abs_residual * n_prev) + float(abs_residual)) / float(stats.count)
        return stats

    def group_rows(self) -> list[dict[str, Any]]:
        return [g.to_dict() for _, g in sorted(self._groups.items(), key=lambda kv: kv[0])]

    def max_under_coverage_gap(self) -> float:
        return max((g.under_coverage_gap for g in self._groups.values()), default=0.0)
