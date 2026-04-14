from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .state import GroupCoverageStats, ShiftAwareConfig


@dataclass
class SubgroupCoverageTracker:
    config: ShiftAwareConfig
    target_coverage: float = 0.9
    groups: dict[str, GroupCoverageStats] = field(default_factory=dict)

    def _bucket(self, value: float, n_bins: int) -> str:
        v = min(0.999999, max(0.0, float(value)))
        b = int(v * n_bins)
        return f"b{b}"

    def build_group_key(self, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        rel_key = self._bucket(float(ctx.get("reliability", 1.0)), self.config.reliability_bins)
        vol_key = self._bucket(float(ctx.get("volatility", 0.0)), self.config.volatility_bins)
        fault_key = str(ctx.get("fault_type", "none"))
        hour = int(ctx.get("hour", 0)) % 24
        hour_key = f"h{hour//6}"
        custom = str(ctx.get("custom_group", "default"))
        return f"r:{rel_key}|v:{vol_key}|f:{fault_key}|t:{hour_key}|c:{custom}"

    def update(
        self,
        *,
        covered: bool,
        interval_width: float,
        abs_residual: float,
        context: dict[str, Any] | None = None,
    ) -> GroupCoverageStats:
        key = self.build_group_key(context)
        st = self.groups.get(key)
        if st is None:
            st = GroupCoverageStats(group_key=key, target_coverage=self.target_coverage)
            self.groups[key] = st
        st.count += 1
        if covered:
            st.covered_count += 1
        else:
            st.miss_count += 1
        n = float(st.count)
        st.avg_interval_width = ((n - 1.0) * st.avg_interval_width + float(interval_width)) / n
        st.avg_abs_residual = ((n - 1.0) * st.avg_abs_residual + float(abs_residual)) / n
        return st

    def max_under_coverage_gap(self) -> float:
        if not self.groups:
            return 0.0
        return float(max(v.to_dict()["under_coverage_gap"] for v in self.groups.values()))

    def to_rows(self) -> list[dict[str, Any]]:
        return [self.groups[k].to_dict() for k in sorted(self.groups)]
