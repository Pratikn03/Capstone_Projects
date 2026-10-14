#!/usr/bin/env python3
"""Single-Repository Google Colab & Local Runner for ORIUS + K-Bound Epistemic Mind.

Usage in Google Colab (ONLY ONE REPO NEEDED):
    !git clone https://github.com/Pratikn03/Capstone_Projects.git
    %cd Capstone_Projects
    !python3 scripts/run_colab_single_repo.py

This script verifies that 100% of the K-Bound paper ("K-Bound: When Is Label-Free
Adaptation Knowable? Helpful, Harmful, and Unknowable Regimes Under Distribution Shift")
is natively embedded inside the single ORIUS repository (`src/kga/` and `src/orius/kbound/`)
as the Epistemic Mind of ORIUS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import kga  # noqa: E402
from orius.dc3s.pipeline import run_dc3s_step  # noqa: E402
from orius.kbound import (  # noqa: E402
    ABSTAIN,
    ADAPT,
    FREEZE,
    CrashFreePhysicalAIGovernor,
    DomainPhysicsConfig,
    assess_frontier,
    evaluate_actions,
    evaluate_compound_population_mind,
    evaluate_kbound_mind,
    evaluate_population_frontier_mind,
)


def main() -> None:
    print("=" * 84)
    print("ORIUS + K-BOUND EPISTEMIC MIND: SINGLE-REPO COLAB & LOCAL VERIFICATION")
    print("=" * 84)
    print(f"Repository Root : {REPO_ROOT}")
    print(f"kga module path : {Path(kga.__file__).resolve()}")

    # 1. Verify Population Knowability Frontier (Theorems 1, 2, 3 & Corollary 1)
    pop_helpful = evaluate_population_frontier_mind(
        M=0.22, beta=0.08, disagreement_mass=0.35, reliability_w=0.92, alpha=0.05
    )
    pop_harmful = evaluate_population_frontier_mind(
        M=-0.24, beta=0.08, disagreement_mass=0.35, reliability_w=0.55, alpha=0.05
    )
    pop_unknowable = evaluate_population_frontier_mind(
        M=0.03, beta=0.08, disagreement_mass=0.35, reliability_w=0.48, alpha=0.05
    )

    assert pop_helpful["kga_decision"] == ADAPT
    assert pop_helpful["orius_physical_mode"] == "ADAPT_AND_RELEASE"
    assert pop_harmful["kga_decision"] == FREEZE
    assert pop_harmful["orius_physical_mode"] == "REJECT_UPDATE_AND_TIGHTEN"
    assert pop_unknowable["kga_decision"] == ABSTAIN
    assert pop_unknowable["orius_physical_mode"] == "ABSTAIN_AND_INFLATE_TUBE"
    assert pop_unknowable["in_closed_unknowable_band"] is True
    assert abs(pop_unknowable["matched_evidence_abstain_prob_floor"] - 0.90) < 1e-9

    raw_frontier = assess_frontier(M=0.03, beta=0.08)
    assert raw_frontier.action == ABSTAIN
    assert abs(raw_frontier.fibre_audit_floor - 0.08) < 1e-9

    print("\n[1/5] Population Knowability Frontier (Theorems 1-3 & Corollary 1): PASS")
    print(
        f"      Helpful (M=+0.22, beta=0.08)   -> {pop_helpful['kga_decision']:<7} | "
        f"Mode: {pop_helpful['orius_physical_mode']} | "
        f"dDelta=[{pop_helpful['population_benefit_lower']:+.4f}, {pop_helpful['population_benefit_upper']:+.4f}]"
    )
    print(
        f"      Harmful (M=-0.24, beta=0.08)   -> {pop_harmful['kga_decision']:<7} | "
        f"Mode: {pop_harmful['orius_physical_mode']} | "
        f"dDelta=[{pop_harmful['population_benefit_lower']:+.4f}, {pop_harmful['population_benefit_upper']:+.4f}]"
    )
    print(
        f"      Unknowable (M=+0.03, beta=0.08)-> {pop_unknowable['kga_decision']:<7} | "
        f"Mode: {pop_unknowable['orius_physical_mode']} | FibreFloor={raw_frontier.fibre_audit_floor:.3f}, "
        f"P(ABSTAIN)>={pop_unknowable['matched_evidence_abstain_prob_floor']:.2f}"
    )

    # 2. Verify Empirical KGA Benefit-Interval Gate (Proposition 2: ADAPT / FREEZE / ABSTAIN)
    calib_pool_25 = np.linspace(0.005, 0.030, 25)
    emp_adapt = evaluate_kbound_mind(
        reliability_w=0.90, delta_hat=0.085, calib_residuals=calib_pool_25, alpha=0.05
    )
    emp_freeze = evaluate_kbound_mind(
        reliability_w=0.45, delta_hat=-0.095, calib_residuals=calib_pool_25, alpha=0.05
    )
    emp_abstain = evaluate_kbound_mind(
        reliability_w=0.52, delta_hat=0.005, calib_residuals=calib_pool_25, alpha=0.05
    )
    emp_telemetry_loss = evaluate_kbound_mind(
        reliability_w=0.0, delta_hat=0.250, calib_residuals=calib_pool_25, alpha=0.05
    )
    small_n_radius = kga.split_conformal_rank_radius([0.01, 0.02, 0.015, 0.02], alpha=0.05)

    assert emp_adapt.kbound_action == ADAPT
    assert emp_freeze.kbound_action == FREEZE
    assert emp_abstain.kbound_action == ABSTAIN
    assert emp_telemetry_loss.kbound_action == ABSTAIN
    assert emp_telemetry_loss.is_certified_rejection is False
    assert emp_telemetry_loss.is_epistemic_abstention is True
    assert not np.isfinite(small_n_radius)  # k = ceil(5 * 0.95) = 5 > 4 -> +inf

    print("\n[2/5] Empirical KGA Benefit-Interval Gate (Proposition 2): PASS")
    print(
        f"      L > 0  (delta_hat=+0.085) -> {emp_adapt.kbound_action:<7} | "
        f"[{emp_adapt.lower_bound:+.4f}, {emp_adapt.upper_bound:+.4f}] | mult={emp_adapt.inflation_multiplier:.3f}"
    )
    print(
        f"      U < 0  (delta_hat=-0.095) -> {emp_freeze.kbound_action:<7} | "
        f"[{emp_freeze.lower_bound:+.4f}, {emp_freeze.upper_bound:+.4f}] | mult={emp_freeze.inflation_multiplier:.3f}"
    )
    print(
        f"      0 in I (delta_hat=+0.005) -> {emp_abstain.kbound_action:<7} | "
        f"[{emp_abstain.lower_bound:+.4f}, {emp_abstain.upper_bound:+.4f}] | mult={emp_abstain.inflation_multiplier:.3f}"
    )
    print(
        f"      w_t=0  (telemetry loss)   -> {emp_telemetry_loss.kbound_action:<7} | "
        f"Alg.1 fail-closed ABSTAIN (small-n rank eps={small_n_radius})"
    )

    # 3. Verify Compound Cell-to-Population Transfer (Proposition 3)
    compound = evaluate_compound_population_mind(
        delta_hat=0.14,
        epsilon=0.03,
        n_eval=1000,
        delta_sampling=0.025,
        alpha_cell=0.025,
        alpha_population=0.05,
        reliability_w=0.88,
    )
    assert compound["kga_decision"] == ADAPT
    assert compound["population_lower"] > 0.0
    print("\n[3/5] Compound Cell-to-Population Transfer (Proposition 3): PASS")
    print(
        f"      n_eval=1000, eps_total={compound['compound_population_radius']:.4f} -> "
        f"{compound['kga_decision']} [{compound['population_lower']:+.4f}, {compound['population_upper']:+.4f}] "
        f"(Valid >= {1.0 - compound['alpha_spent']:.2f})"
    )

    # 4. Verify K-Bound Action Evaluation Metrics (Cert-False-Adapt / Cert-False-Freeze)
    actions = ["ADAPT", "FREEZE", "ABSTAIN", "ADAPT", "FREEZE", "ABSTAIN"]
    true_deltas = [0.08, -0.09, 0.002, 0.11, -0.07, -0.003]
    eval_report = evaluate_actions(actions, true_deltas)
    s = eval_report.summary
    assert s.false_adapt_rate_all == 0.0
    assert s.false_freeze_rate_all == 0.0
    assert s.directional_accuracy_commitments == 1.0
    print("\n[4/5] K-Bound Action Evaluation Audit (evaluate_actions): PASS")
    print(
        f"      Cert-False-Adapt={s.false_adapt_rate_all:.3f} | "
        f"Cert-False-Freeze={s.false_freeze_rate_all:.3f} | "
        f"DirectionalAcc={s.directional_accuracy_commitments:.3f} | "
        f"AbstainRate={s.abstain_rate:.3f}"
    )

    # 5. Verify End-to-End Coupling inside ORIUS DC3S & CrashFreePhysicalAIGovernor
    from orius.dc3s.battery_adapter import BatteryDomainAdapter

    dc3s_out = run_dc3s_step(
        event={"ts_utc": "2026-10-14T00:00:00Z", "load_mw": 50.0, "renewables_mw": 20.0},
        last_event={"ts_utc": "2026-10-13T23:00:00Z", "load_mw": 49.5, "renewables_mw": 20.2},
        yhat=np.array([50.0]),
        q=np.array([2.5]),
        candidate_action={"charge_mw": 0.0, "discharge_mw": 5.0},
        domain_adapter=BatteryDomainAdapter(),
        state={"current_soc_mwh": 50.0, "capacity_mwh": 100.0, "min_soc_mwh": 10.0, "max_soc_mwh": 90.0},
        cfg={
            "kbound": {
                "delta_hat": -0.095,
                "calib_residuals": calib_pool_25,
                "alpha": 0.05,
                "couple_inflation": True,
            }
        },
    )
    assert dc3s_out["kbound"]["kbound_action"] == FREEZE
    assert dc3s_out["kbound"]["orius_physical_mode"] == "REJECT_UPDATE_AND_TIGHTEN"

    av_cfg = DomainPhysicsConfig(
        domain_name="autonomous_vehicles",
        state_unit="m",
        x_min=0.0,
        x_max=100.0,
        u_min=-8.0,
        u_max=4.0,
        dt=0.05,
        max_brake_decel=8.0,
        process_noise_bound=0.05,
        obs_noise_std=0.1,
    )
    gov = CrashFreePhysicalAIGovernor(config=av_cfg, alpha=0.05)
    u_gov, gov_step = gov.step(
        true_state=82.0,
        true_velocity=15.0,
        state_channels=[82.1, 81.9, 82.0, 82.05, 95.0],
        velocity_channels=[15.0, 15.1, 14.9, 15.0, 22.0],
        u_nominal_ai=4.0,
        dropout=False,
    )
    assert gov_step.true_safe is True

    print("\n[5/5] End-to-End ORIUS DC3S + CrashFreePhysicalAIGovernor Coupling: PASS")
    print(
        f"      DC3S K-Bound Action: {dc3s_out['kbound']['kbound_action']} "
        f"({dc3s_out['kbound']['orius_physical_mode']})"
    )
    print(
        f"      Governor Crash-Free: {gov_step.true_safe} | "
        f"u_nominal={gov_step.u_nominal_ai:+.2f} -> u_governed={u_gov:+.2f} | "
        f"d_stop={gov_step.stopping_distance:.2f} m"
    )

    out_path = REPO_ROOT / "reports" / "publication" / "colab_single_repo_kbound_orius_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "single_repo_ready": True,
        "repo_url": "https://github.com/Pratikn03/Capstone_Projects.git",
        "kga_embedded": True,
        "orius_kbound_embedded": True,
        "population_frontier": {
            "helpful": pop_helpful,
            "harmful": pop_harmful,
            "unknowable": pop_unknowable,
        },
        "empirical_gate": {
            "adapt": emp_adapt.to_dict(),
            "freeze": emp_freeze.to_dict(),
            "abstain": emp_abstain.to_dict(),
        },
        "compound_population": compound,
        "action_audit": {
            "false_adapt_rate_all": s.false_adapt_rate_all,
            "false_freeze_rate_all": s.false_freeze_rate_all,
            "directional_accuracy_commitments": s.directional_accuracy_commitments,
            "abstain_rate": s.abstain_rate,
            "mean_regret": s.mean_regret,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 84)
    print(f"ALL CHECKS PASSED. Saved verification artifact to: {out_path.relative_to(REPO_ROOT)}")
    print("=" * 84)


if __name__ == "__main__":
    main()
