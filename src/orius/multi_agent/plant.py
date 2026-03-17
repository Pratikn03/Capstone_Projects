"""Shared feeder plant: multiple batteries with a common transformer limit.

Paper 5 first domain: two or more batteries sharing feeder capacity.
Local DC3S certificates do not auto-compose — joint net power must
respect the shared transformer limit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class SharedFeederPlant:
    """Plant with N batteries sharing a feeder capacity limit.

    Each battery has its own SOC bounds. The shared constraint:
        sum_i (discharge_i - charge_i) <= feeder_capacity_mw
    """

    def __init__(
        self,
        batteries: Sequence[Mapping[str, Any]],
        feeder_capacity_mw: float,
    ):
        self._batteries = list(batteries)
        self._feeder_capacity = float(feeder_capacity_mw)
        self._socs: list[float] = []
        self._last_net: list[float] = []
        self._last_executed: list[Mapping[str, Any]] = []

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        rng = np.random.default_rng(seed)
        self._socs = []
        self._last_net = []
        for b in self._batteries:
            cap = float(b.get("capacity_mwh", 100.0))
            init = float(b.get("initial_soc_frac", 0.5))
            self._socs.append(cap * init)
            self._last_net.append(0.0)
        return self.state()

    def state(self) -> Mapping[str, Any]:
        return {
            "socs_mwh": list(self._socs),
            "feeder_capacity_mw": self._feeder_capacity,
            "n_batteries": len(self._batteries),
        }

    def step(self, actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Advance all batteries. actions[i] = {charge_mw, discharge_mw} for battery i."""
        eta_c = []
        eta_d = []
        soc_min = []
        soc_max = []
        for b in self._batteries:
            eta_c.append(float(b.get("charge_efficiency", 0.95)))
            eta_d.append(float(b.get("discharge_efficiency", 0.95)))
            cap = float(b.get("capacity_mwh", 100.0))
            soc_min.append(float(b.get("min_soc_frac", 0.1)) * cap)
            soc_max.append(float(b.get("max_soc_frac", 0.9)) * cap)

        charges = []
        discharges = []
        for i, a in enumerate(actions):
            c = max(0.0, float(a.get("charge_mw", 0)))
            d = max(0.0, float(a.get("discharge_mw", 0)))
            charges.append(c)
            discharges.append(d)

        # Shared constraint: total net export <= feeder_capacity
        total_net = sum(d - c for c, d in zip(charges, discharges))
        if total_net > self._feeder_capacity + 1e-9:
            scale = self._feeder_capacity / total_net
            charges = [c * scale for c in charges]
            discharges = [d * scale for d in discharges]

        self._last_executed = [
            {"charge_mw": c, "discharge_mw": d}
            for c, d in zip(charges, discharges)
        ]

        # Update SOCs
        for i in range(len(self._batteries)):
            delta = eta_c[i] * charges[i] - discharges[i] / max(eta_d[i], 1e-9)
            self._socs[i] = np.clip(
                self._socs[i] + delta,
                soc_min[i],
                soc_max[i],
            )
            self._last_net[i] = discharges[i] - charges[i]

        return self.state()

    def check_joint_violation(self, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Check if joint action violates shared feeder limit."""
        total_net = 0.0
        for a in actions:
            total_net += float(a.get("discharge_mw", 0)) - float(a.get("charge_mw", 0))
        violated = total_net > self._feeder_capacity + 1e-9
        severity = max(0.0, total_net - self._feeder_capacity)
        return {"violated": violated, "severity": severity, "total_net_mw": total_net}

    def check_local_violations(self) -> list[dict[str, Any]]:
        """Check SOC bounds for each battery."""
        results = []
        for i, b in enumerate(self._batteries):
            cap = float(b.get("capacity_mwh", 100.0))
            lo = float(b.get("min_soc_frac", 0.1)) * cap
            hi = float(b.get("max_soc_frac", 0.9)) * cap
            soc = self._socs[i]
            v = soc < lo - 1e-9 or soc > hi + 1e-9
            sev = max(lo - soc, soc - hi, 0.0)
            results.append({"violated": v, "severity": sev, "soc_mwh": soc})
        return results
