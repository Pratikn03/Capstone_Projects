"""Non-composition counterexample: transformer-capacity scenario.

Paper 5: Two batteries each with a local certificate; their combined
discharge exceeds the shared feeder limit. Demonstrates that local
certificates do not auto-compose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .plant import SharedFeederPlant
from .protocol import (
    IndependentLocalProtocol,
    CentralizedCoordinatorProtocol,
    DistributedNegotiationProtocol,
)
from .margin_allocation import allocate_margins_fairness


def run_transformer_capacity_scenario(
    feeder_capacity_mw: float = 80.0,
    n_steps: int = 24,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run the non-composition counterexample.

    Two batteries, each proposing 60 MW discharge (total 120 MW).
    Feeder limit 80 MW. Independent protocol violates; centralized/distributed
    scale down to stay within limit.

    Returns
    -------
    dict
        joint_violations, local_violations, useful_work, fairness, margin_quality
    """
    batteries = [
        {
            "capacity_mwh": 100.0,
            "initial_soc_frac": 0.6,
            "min_soc_frac": 0.1,
            "max_soc_frac": 0.9,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        {
            "capacity_mwh": 100.0,
            "initial_soc_frac": 0.6,
            "min_soc_frac": 0.1,
            "max_soc_frac": 0.9,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
    ]

    # Each agent proposes 60 MW discharge (individually safe, jointly exceeds feeder)
    local_proposals = [
        {"charge_mw": 0.0, "discharge_mw": 60.0},
        {"charge_mw": 0.0, "discharge_mw": 60.0},
    ]

    plant = SharedFeederPlant(batteries, feeder_capacity_mw)
    plant.reset(seed)

    protocols = {
        "independent": IndependentLocalProtocol(),
        "centralized": CentralizedCoordinatorProtocol(),
        "distributed": DistributedNegotiationProtocol(),
    }

    results: dict[str, Any] = {}
    for name, protocol in protocols.items():
        joint_violations = 0
        local_violations = 0
        useful_work = 0.0

        p = SharedFeederPlant(batteries, feeder_capacity_mw)
        p.reset(seed)

        for t in range(n_steps):
            state = p.state()
            actions = protocol.compute_actions(
                state, local_proposals, feeder_capacity_mw
            )
            joint = p.check_joint_violation(actions)
            if joint["violated"]:
                joint_violations += 1
            p.step(actions)
            executed = getattr(p, "_last_executed", actions)
            local = p.check_local_violations()
            for lv in local:
                if lv["violated"]:
                    local_violations += 1
            useful_work += sum(
                float(a.get("discharge_mw", 0)) for a in executed
            )

        margins = [40.0, 40.0]  # equal allocation
        demands = [60.0, 60.0]
        fairness = allocate_margins_fairness(margins, demands)

        results[name] = {
            "joint_violations": joint_violations,
            "local_violations": local_violations,
            "useful_work_mwh": useful_work,
            "fairness": fairness,
            "margin_quality": 1.0 - (joint_violations / max(n_steps, 1)),
        }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "multi_agent_transformer_scenario.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["protocol", "joint_violations", "local_violations", "useful_work_mwh", "fairness", "margin_quality"])
            w.writeheader()
            for name, r in results.items():
                row = {"protocol": name, **r}
                w.writerow(row)

    return results
