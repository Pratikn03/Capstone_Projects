"""Non-composition counterexample: transformer-capacity scenario.

Paper 5: Two batteries each with a local certificate; their combined
discharge exceeds the shared feeder limit. Demonstrates that local
certificates do not auto-compose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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
    agent_degradation: list[float] | None = None,
    degradation_per_agent: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Run the non-composition counterexample.

    Two batteries, each proposing 60 MW discharge (total 120 MW).
    Feeder limit 80 MW. Independent protocol violates; centralized/distributed
    scale down to stay within limit.

    Parameters
    ----------
    agent_degradation : list of float or None
        Per-agent degradation factor in [0, 1]. Proposal scaled by (1 - d).
        E.g. [0, 0.2] means agent 1 proposes 60*0.8=48 MW. None = no degradation.
    degradation_per_agent : sequence of float or None
        Per-agent efficiency scale in (0, 1]. Multiplies charge_efficiency and
        discharge_efficiency. E.g. [1.0, 0.85] means agent 1 has 85% efficiency.
        None = no heterogeneous degradation (all 0.95).

    Returns
    -------
    dict
        joint_violations, local_violations, useful_work, fairness, margin_quality
    """
    base_eff = 0.95
    batteries = [
        {
            "capacity_mwh": 100.0,
            "initial_soc_frac": 0.6,
            "min_soc_frac": 0.1,
            "max_soc_frac": 0.9,
            "charge_efficiency": base_eff,
            "discharge_efficiency": base_eff,
        },
        {
            "capacity_mwh": 100.0,
            "initial_soc_frac": 0.6,
            "min_soc_frac": 0.1,
            "max_soc_frac": 0.9,
            "charge_efficiency": base_eff,
            "discharge_efficiency": base_eff,
        },
    ]
    if degradation_per_agent is not None and len(degradation_per_agent) >= len(batteries):
        for i in range(len(batteries)):
            scale = min(1.0, max(1e-6, float(degradation_per_agent[i])))
            batteries[i] = {**batteries[i], "charge_efficiency": base_eff * scale, "discharge_efficiency": base_eff * scale}

    # Each agent proposes 60 MW discharge (individually safe, jointly exceeds feeder)
    base_proposals = [
        {"charge_mw": 0.0, "discharge_mw": 60.0},
        {"charge_mw": 0.0, "discharge_mw": 60.0},
    ]
    # Apply per-agent degradation: scale by (1 - degradation_i)
    if agent_degradation is not None and len(agent_degradation) >= len(base_proposals):
        local_proposals = []
        for i, p in enumerate(base_proposals):
            scale = 1.0 - min(1.0, max(0.0, float(agent_degradation[i])))
            local_proposals.append({
                "charge_mw": float(p.get("charge_mw", 0)) * scale,
                "discharge_mw": float(p.get("discharge_mw", 0)) * scale,
            })
    else:
        local_proposals = [dict(p) for p in base_proposals]

    plant = SharedFeederPlant(batteries, feeder_capacity_mw)
    plant.reset(seed)

    protocols = {
        "independent": IndependentLocalProtocol(),
        "centralized": CentralizedCoordinatorProtocol(),
        "distributed": DistributedNegotiationProtocol(),
    }

    results: dict[str, Any] = {}
    step_log_rows: list[dict[str, Any]] = []

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
            if out_dir is not None:
                socs = state.get("socs_mwh", [])
                step_log_rows.append({
                    "step": t,
                    "protocol": name,
                    "soc_0_mwh": socs[0] if len(socs) > 0 else 0.0,
                    "soc_1_mwh": socs[1] if len(socs) > 1 else 0.0,
                    "joint_violated": 1 if joint["violated"] else 0,
                    "charge_0_mw": executed[0].get("charge_mw", 0) if len(executed) > 0 else 0,
                    "discharge_0_mw": executed[0].get("discharge_mw", 0) if len(executed) > 0 else 0,
                    "charge_1_mw": executed[1].get("charge_mw", 0) if len(executed) > 1 else 0,
                    "discharge_1_mw": executed[1].get("discharge_mw", 0) if len(executed) > 1 else 0,
                })
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

        margin_quality = 1.0 - (joint_violations / max(n_steps, 1))
        results[name] = {
            "joint_violations": joint_violations,
            "local_violations": local_violations,
            "useful_work_mwh": useful_work,
            "fairness": fairness,
            "margin_quality": margin_quality,
            "degradation_allocation_quality": margin_quality,
        }

    if out_dir is not None:
        import csv
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "multi_agent_transformer_scenario.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["protocol", "joint_violations", "local_violations", "useful_work_mwh", "fairness", "margin_quality", "degradation_allocation_quality"])
            w.writeheader()
            for name, r in results.items():
                row = {"protocol": name, **r}
                w.writerow(row)
        step_log_path = out_dir / "multi_agent_step_log.csv"
        with open(step_log_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["step", "protocol", "soc_0_mwh", "soc_1_mwh", "joint_violated", "charge_0_mw", "discharge_0_mw", "charge_1_mw", "discharge_1_mw"],
            )
            w.writeheader()
            w.writerows(step_log_rows)
        protocol_compare_path = out_dir / "protocol_compare.csv"
        with open(protocol_compare_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["protocol", "joint_violations", "local_violations", "useful_work_mwh", "fairness", "margin_quality", "degradation_allocation_quality"])
            w.writeheader()
            for name, r in results.items():
                w.writerow({"protocol": name, **r})
        fairness_metrics_path = out_dir / "fairness_metrics.csv"
        with open(fairness_metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["protocol", "fairness", "margin_quality", "degradation_allocation_quality", "joint_violations", "useful_work_mwh"])
            w.writeheader()
            for name, r in results.items():
                w.writerow({
                    "protocol": name,
                    "fairness": r["fairness"],
                    "margin_quality": r["margin_quality"],
                    "degradation_allocation_quality": r["degradation_allocation_quality"],
                    "joint_violations": r["joint_violations"],
                    "useful_work_mwh": r["useful_work_mwh"],
                })

    return results


def summarize_transformer_capacity_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    feeder_capacity_mw: float,
    proposal_per_agent_mw: float,
    n_steps: int,
) -> dict[str, float | int]:
    """Build a bounded chapter-facing summary from scenario outputs.

    The summary intentionally stays close to what the executable scenario can
    support directly: one two-battery, shared-feeder counterexample with three
    coordination protocols. It does not claim optimality or general N-agent
    composition.
    """

    independent = results.get("independent", {})
    centralized = results.get("centralized", {})
    distributed = results.get("distributed", {})

    independent_joint = int(independent.get("joint_violations", 0))
    centralized_joint = int(centralized.get("joint_violations", 0))
    distributed_joint = int(distributed.get("joint_violations", 0))

    return {
        "n_steps": int(n_steps),
        "feeder_capacity_mw": float(feeder_capacity_mw),
        "proposal_per_agent_mw": float(proposal_per_agent_mw),
        "total_proposed_discharge_mw": float(2.0 * proposal_per_agent_mw),
        "independent_joint_violation_steps": independent_joint,
        "centralized_joint_violation_steps": centralized_joint,
        "distributed_joint_violation_steps": distributed_joint,
        "centralized_violation_reduction_steps": independent_joint - centralized_joint,
        "distributed_violation_reduction_steps": independent_joint - distributed_joint,
        "independent_useful_work_mwh": float(independent.get("useful_work_mwh", 0.0)),
        "centralized_useful_work_mwh": float(centralized.get("useful_work_mwh", 0.0)),
        "distributed_useful_work_mwh": float(distributed.get("useful_work_mwh", 0.0)),
        "centralized_margin_quality": float(centralized.get("margin_quality", 0.0)),
        "distributed_margin_quality": float(distributed.get("margin_quality", 0.0)),
    }
