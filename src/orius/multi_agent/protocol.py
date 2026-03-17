"""Shared-constraint protocols for multi-agent safety.

Paper 5: Local certificates do not auto-compose. These protocols
coordinate agents sharing feeder capacity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .margin_allocation import allocate_margins


class SharedConstraintProtocol(ABC):
    """Abstract protocol for multi-agent shared-constraint coordination."""

    @abstractmethod
    def compute_actions(
        self,
        plant_state: Mapping[str, Any],
        local_proposals: Sequence[Mapping[str, Any]],
        feeder_capacity_mw: float,
    ) -> Sequence[Mapping[str, Any]]:
        """Return feasible actions respecting the shared constraint."""


class IndependentLocalProtocol(SharedConstraintProtocol):
    """Each agent acts on its local certificate; no coordination.

    Non-composition: joint action may violate feeder limit.
    """

    def compute_actions(
        self,
        plant_state: Mapping[str, Any],
        local_proposals: Sequence[Mapping[str, Any]],
        feeder_capacity_mw: float,
    ) -> Sequence[Mapping[str, Any]]:
        return list(local_proposals)


class CentralizedCoordinatorProtocol(SharedConstraintProtocol):
    """Central coordinator allocates feeder margin and scales actions."""

    def __init__(self, scheme: str = "equal"):
        self._scheme = scheme

    def compute_actions(
        self,
        plant_state: Mapping[str, Any],
        local_proposals: Sequence[Mapping[str, Any]],
        feeder_capacity_mw: float,
    ) -> Sequence[Mapping[str, Any]]:
        n = len(local_proposals)
        margins = allocate_margins(feeder_capacity_mw, n, scheme=self._scheme)

        total_net = sum(
            float(p.get("discharge_mw", 0)) - float(p.get("charge_mw", 0))
            for p in local_proposals
        )
        if total_net <= feeder_capacity_mw + 1e-9:
            return list(local_proposals)

        scale = feeder_capacity_mw / total_net
        result = []
        for p in local_proposals:
            c = float(p.get("charge_mw", 0)) * scale
            d = float(p.get("discharge_mw", 0)) * scale
            result.append({"charge_mw": c, "discharge_mw": d})
        return result


class DistributedNegotiationProtocol(SharedConstraintProtocol):
    """Agents negotiate margins via iterative best-response.

    Simplified: each agent gets equal share, then scales proportionally.
    """

    def compute_actions(
        self,
        plant_state: Mapping[str, Any],
        local_proposals: Sequence[Mapping[str, Any]],
        feeder_capacity_mw: float,
    ) -> Sequence[Mapping[str, Any]]:
        n = len(local_proposals)
        demands = [
            float(p.get("discharge_mw", 0)) - float(p.get("charge_mw", 0))
            for p in local_proposals
        ]
        margins = allocate_margins(
            feeder_capacity_mw, n, scheme="demand", demands=demands
        )

        result = []
        for i, p in enumerate(local_proposals):
            net = demands[i]
            margin = margins[i]
            if abs(net) <= 1e-9:
                result.append(dict(p))
                continue
            scale = min(1.0, margin / max(abs(net), 1e-9))
            c = float(p.get("charge_mw", 0)) * scale
            d = float(p.get("discharge_mw", 0)) * scale
            result.append({"charge_mw": c, "discharge_mw": d})
        return result
