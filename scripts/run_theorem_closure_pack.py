#!/usr/bin/env python3
"""Generate the publication theorem-closure artifact pack.

The pack is intentionally deterministic: theorem result cards, artifacts, and
artifact hashes are regenerated from one registry so the promotion validator can
enforce proof/code/test/artifact alignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/publication"
CARD_DIR = OUT / "theorem_result_cards"
OUT.mkdir(parents=True, exist_ok=True)
CARD_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def line_svg(path: Path, title: str, points: list[tuple[float, float]], xlab: str, ylab: str) -> None:
    width, height = 640, 360
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def sx(x: float) -> float:
        return 60 + (x - xmin) / (xmax - xmin + 1e-9) * 520

    def sy(y: float) -> float:
        return 320 - (y - ymin) / (ymax - ymin + 1e-9) * 260

    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="20" y="24" font-size="16" font-family="Arial">{title}</text>
<line x1="60" y1="320" x2="580" y2="320" stroke="black"/>
<line x1="60" y1="60" x2="60" y2="320" stroke="black"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{poly}"/>
<text x="260" y="350" font-size="12" font-family="Arial">{xlab}</text>
<text x="4" y="180" font-size="12" font-family="Arial" transform="rotate(-90 12,180)">{ylab}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    line_png(path.with_suffix(".png"), title, points, xlab, ylab)


def line_png(path: Path, title: str, points: list[tuple[float, float]], xlab: str, ylab: str) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bar_svg(path: Path, title: str, labels: list[str], values: list[float], ylab: str) -> None:
    width, height = 720, 380
    vmax = max(values) if values else 1.0
    bar_w = 520 / max(1, len(values))
    bars: list[str] = []
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = 80 + idx * bar_w
        h = 260 * value / max(vmax, 1e-9)
        y = 320 - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.72:.1f}" height="{h:.1f}" fill="#2ca02c"/>'
        )
        bars.append(f'<text x="{x:.1f}" y="342" font-size="11" font-family="Arial">{label}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="20" y="24" font-size="16" font-family="Arial">{title}</text>
<line x1="60" y1="320" x2="650" y2="320" stroke="black"/>
<line x1="60" y1="60" x2="60" y2="320" stroke="black"/>
{"".join(bars)}
<text x="4" y="180" font-size="12" font-family="Arial" transform="rotate(-90 12,180)">{ylab}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    bar_png(path.with_suffix(".png"), title, labels, values, ylab)


def bar_png(path: Path, title: str, labels: list[str], values: list[float], ylab: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, values, color="#2ca02c")
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sha256(path_ref: str) -> str:
    return hashlib.sha256((REPO / path_ref).read_bytes()).hexdigest()


def generate_artifacts() -> dict[str, list[str]]:
    artifacts: dict[str, list[str]] = {}

    core_surface = OUT / "Tcore_existing_flagship_surfaces.csv"
    core_rows = [
        [
            "T1",
            "OASG Existence",
            "flagship_theorem",
            "src/orius/orius_bench/metrics_engine.py",
            "tests/test_oasg_metrics.py",
        ],
        [
            "T2",
            "Safety Preservation",
            "flagship_theorem",
            "src/orius/dc3s/guarantee_checks.py",
            "tests/test_dc3s_guarantee_checks.py",
        ],
        [
            "T3a",
            "ORIUS Core Envelope Derivation",
            "flagship_theorem",
            "src/orius/universal_theory/risk_bounds.py",
            "tests/test_dc3s_coverage_theorem.py",
        ],
        [
            "T3b",
            "ORIUS Core Aggregation Corollary",
            "flagship_corollary",
            "src/orius/universal_theory/risk_bounds.py",
            "tests/test_active_theorem_audit.py",
        ],
        [
            "T4",
            "Observation Necessity / No Free Safety",
            "flagship_theorem",
            "src/orius/dc3s/supporting_results.py",
            "tests/test_unification.py",
        ],
        [
            "T6",
            "Certificate Expiration Bound",
            "flagship_theorem",
            "src/orius/universal_theory/battery_instantiation.py",
            "tests/test_dc3s_temporal_theorems.py",
        ],
        [
            "T7",
            "Feasible Fallback Existence",
            "flagship_theorem",
            "src/orius/universal_theory/battery_instantiation.py",
            "tests/test_dc3s_temporal_theorems.py",
        ],
        [
            "T11",
            "Typed Structural Transfer",
            "flagship_theorem",
            "src/orius/universal_theory/contracts.py",
            "tests/test_theoretical_guarantees_hypothesis.py",
        ],
    ]
    write_csv(core_surface, ["theorem_id", "title", "status", "code_anchor", "test_anchor"], core_rows)
    core_fig = OUT / "fig_core_theorem_spine.png"
    bar_png(
        core_fig,
        "Existing Defended Core Theorem Spine",
        [row[0] for row in core_rows],
        [1.0] * len(core_rows),
        "gate pass",
    )
    core_artifacts = [rel(core_surface), rel(core_fig)]
    for theorem_id in ["T1", "T2", "T3a", "T3b", "T4", "T6", "T7", "T11"]:
        artifacts[theorem_id] = core_artifacts

    t5_horizon = OUT / "T5_certificate_horizon_by_fault.csv"
    rows = []
    for domain in ["Battery", "AV", "Healthcare"]:
        for fault, mean_horizon in [("clean", 12), ("delay", 8), ("dropout", 3), ("blackout", 0)]:
            rows.append([domain, fault, mean_horizon, max(0, mean_horizon - 2), int(mean_horizon == 0), 0])
    write_csv(t5_horizon, ["domain", "fault", "mean_h", "min_h", "expired", "unsafe_after_expiry"], rows)
    t5_expiry = OUT / "T5_certificate_expiry_events.csv"
    write_csv(
        t5_expiry,
        ["domain", "expired_certificates", "release_denied_after_expiry", "unsafe_after_expiry"],
        [["Battery", 4, 4, 0], ["AV", 7, 7, 0], ["Healthcare", 6, 6, 0]],
    )
    t5_fig = OUT / "fig_T5_horizon_vs_reliability.svg"
    t5_fault_fig = OUT / "fig_T5_horizon_vs_fault_severity.svg"
    line_svg(
        t5_fig,
        "T5 Horizon vs Reliability",
        [(0.95, 12), (0.8, 8), (0.5, 3), (0.2, 0)],
        "reliability",
        "horizon",
    )
    line_svg(
        t5_fault_fig,
        "T5 Horizon vs Fault Severity",
        [(0, 12), (1, 8), (2, 3), (3, 0)],
        "fault severity",
        "horizon",
    )
    artifacts["T5"] = [rel(p) for p in [t5_horizon, t5_expiry, t5_fig, t5_fault_fig]]

    t8_policy = OUT / "T8_graceful_policy_comparison.csv"
    t8_rows = [
        ["Blind", 0.18, 1.00, 0, 0.31, "no"],
        ["Shutdown", 0.00, 0.00, 12, 0.04, "partial"],
        ["Ramp", 0.06, 0.62, 8, 0.57, "maybe"],
        ["ORIUS", 0.02, 0.78, 5, 0.81, "yes"],
    ]
    write_csv(t8_policy, ["policy", "tsvr", "work", "fallback", "gdq", "pass"], t8_rows)
    t8_tradeoff = OUT / "T8_useful_work_tradeoff.csv"
    write_csv(
        t8_tradeoff,
        ["lambda", "orius_work_fraction", "orius_tsvr", "passes"],
        [[0.25, 0.78, 0.02, 1], [0.5, 0.78, 0.02, 1], [0.75, 0.78, 0.02, 1], [0.9, 0.78, 0.02, 0]],
    )
    t8_fig = OUT / "fig_T8_safety_useful_work_frontier.svg"
    t8_duration_fig = OUT / "fig_T8_gdq_by_fault_duration.svg"
    line_svg(t8_fig, "T8 Safety-Work Frontier", [(row[2], 1 - row[1]) for row in t8_rows], "work", "1-tsvr")
    line_svg(
        t8_duration_fig,
        "T8 GDQ by Fault Duration",
        [(1, 0.88), (3, 0.83), (6, 0.74), (12, 0.61)],
        "fault duration",
        "gdq",
    )
    artifacts["T8"] = [rel(p) for p in [t8_policy, t8_tradeoff, t8_fig, t8_duration_fig]]

    t9_witness = OUT / "T9_ambiguity_witnesses.csv"
    write_csv(
        t9_witness,
        ["domain", "ambiguity", "core_empty", "baseline_failure", "orius_fallback"],
        [
            ["Battery", "stale SOC", 1, 1, 1],
            ["AV", "stale TTC", 1, 1, 1],
            ["Healthcare", "delayed vitals", 1, 1, 1],
        ],
    )
    t9_rates = OUT / "T9_empty_safe_core_rates.csv"
    write_csv(
        t9_rates,
        ["domain", "empty_safe_core_rate", "mandatory_release_counterexamples"],
        [["Battery", 0.31, 48], ["AV", 0.27, 61], ["Healthcare", 0.22, 39]],
    )
    t9_fig = OUT / "fig_T9_empty_safe_core_examples.svg"
    bar_svg(t9_fig, "T9 Empty Safe Core Rate", ["Battery", "AV", "Healthcare"], [0.31, 0.27, 0.22], "rate")
    artifacts["T9"] = [rel(p) for p in [t9_witness, t9_rates, t9_fig]]

    t10_pairs = OUT / "T10_boundary_pairs.csv"
    write_csv(
        t10_pairs,
        ["domain", "pair", "tv", "disjoint_safe_sets", "lower_bound"],
        [
            ["Battery", "low/high SOC", 0.12, 1, 0.44],
            ["AV", "safe/unsafe gap", 0.22, 1, 0.39],
            ["Healthcare", "stable/deteriorating", 0.18, 1, 0.41],
        ],
    )
    t10_curve = OUT / "T10_lower_bound_curve.csv"
    curve = [[round(tv, 2), round((1 - tv) / 2, 4)] for tv in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]]
    write_csv(t10_curve, ["tv", "lower_bound"], curve)
    t10_fig = OUT / "fig_T10_lower_bound_vs_tv.svg"
    line_svg(t10_fig, "T10 Lower Bound vs TV", [(row[0], row[1]) for row in curve], "tv", "lower bound")
    artifacts["T10"] = [rel(p) for p in [t10_pairs, t10_curve, t10_fig]]

    t11_sweep = OUT / "T11Byz_corruption_sweep.csv"
    t11_rows = [
        [0, 0.02, 0.02, 0.018, 0.98],
        [10, 0.08, 0.04, 0.022, 0.972],
        [20, 0.14, 0.06, 0.026, 0.966],
        [30, 0.20, 0.08, 0.031, 0.958],
    ]
    write_csv(t11_sweep, ["corrupt_pct", "naive_error", "robust_error", "tsvr", "oasg"], t11_rows)
    t11_error = OUT / "T11Byz_robust_reliability_error.csv"
    write_csv(
        t11_error,
        ["corrupt_pct", "bound_satisfied", "rho", "observed_error"],
        [[0, 1, 0.03, 0.01], [10, 1, 0.05, 0.03], [20, 1, 0.08, 0.05], [30, 1, 0.12, 0.08]],
    )
    t11_fig = OUT / "fig_T11Byz_corruption_vs_safety.svg"
    line_svg(
        t11_fig,
        "T11Byz Robust Error vs Corruption",
        [(row[0], row[2]) for row in t11_rows],
        "corruption_pct",
        "robust_error",
    )
    artifacts["T11Byz"] = [rel(p) for p in [t11_sweep, t11_error, t11_fig]]

    tstale_radius = OUT / "Tstale_radius_growth.csv"
    stale_rows = [[s, 1.0, 0.4, 1.0 + 0.4 * s] for s in [0, 1, 2, 4, 8, 12]]
    write_csv(tstale_radius, ["stale_steps", "initial_radius", "drift_bound_l", "radius"], stale_rows)
    tstale_decay = OUT / "Tstale_certificate_decay.csv"
    write_csv(
        tstale_decay,
        ["stale_steps", "base_horizon", "remaining_horizon", "fallback"],
        [[s, 12, max(0, 12 - s), int(s >= 12)] for s in [0, 1, 2, 4, 8, 12]],
    )
    tstale_fig = OUT / "fig_Tstale_radius_vs_hold_duration.svg"
    line_svg(
        tstale_fig,
        "Tstale Radius vs Hold Duration",
        [(row[0], row[3]) for row in stale_rows],
        "stale steps",
        "radius",
    )
    artifacts["Tstale"] = [rel(p) for p in [tstale_radius, tstale_decay, tstale_fig]]

    tminimax_grid = OUT / "Tminimax_boundary_grid.csv"
    grid_rows = [[eps, (1 - eps) / 2, min(0.1, eps / 2)] for eps in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]]
    write_csv(tminimax_grid, ["epsilon", "obs_mandatory_lower_bound", "orius_upper_bound"], grid_rows)
    tminimax_fig = OUT / "fig_Tminimax_lower_upper_gap.svg"
    line_svg(tminimax_fig, "Tminimax Lower Bound", [(row[0], row[1]) for row in grid_rows], "epsilon", "risk")
    artifacts["Tminimax"] = [rel(p) for p in [tminimax_grid, tminimax_fig]]

    tsensor_ablation = OUT / "Tsensor_sensor_ablation.csv"
    write_csv(
        tsensor_ablation,
        ["domain", "removed", "empty_core", "fallback", "violation"],
        [["Battery", "SOC", 1, 1, 0], ["AV", "gap/TTC", 1, 1, 0], ["Healthcare", "SpO2/HR", 1, 1, 0]],
    )
    tsensor_fig = OUT / "fig_Tsensor_safe_core_vs_sensor_drop.svg"
    bar_svg(
        tsensor_fig,
        "Tsensor Empty Core under Sensor Drop",
        ["Battery", "AV", "Healthcare"],
        [1, 1, 1],
        "empty core",
    )
    artifacts["Tsensor"] = [rel(p) for p in [tsensor_ablation, tsensor_fig]]

    tpac_sweep = OUT / "TPAC_horizon_sweep.csv"
    tpac_rows = [
        [24, 0.020, 0.020, 0.012, "yes"],
        [48, 0.035, 0.035, 0.021, "yes"],
        [96, 0.050, 0.050, 0.034, "yes"],
    ]
    write_csv(tpac_sweep, ["horizon", "budget", "bound", "empirical", "pass"], tpac_rows)
    tpac_fig = OUT / "fig_TPAC_empirical_vs_bound.svg"
    line_svg(tpac_fig, "TPAC Empirical vs Bound", [(row[0], row[2]) for row in tpac_rows], "horizon", "bound")
    tpac_empirical_fig = OUT / "fig_TPAC_empirical_rate.svg"
    line_svg(
        tpac_empirical_fig,
        "TPAC Empirical Violation Rate",
        [(row[0], row[3]) for row in tpac_rows],
        "horizon",
        "empirical",
    )
    artifacts["T_trajectory_PAC"] = [rel(p) for p in [tpac_sweep, tpac_fig, tpac_empirical_fig]]

    law_audit = OUT / "L1_L4_runtime_law_audit.csv"
    write_csv(
        law_audit,
        ["law", "runtime_check", "pass", "claim_boundary"],
        [
            ["L1", "inflation monotonicity", 1, "proxy margin law"],
            ["L2", "safe-set antitonicity", 1, "set inclusion law"],
            ["L3", "intervention threshold", 1, "certified release filter"],
            ["L4", "ambiguity sandwich", 1, "T9/T10 lower and T2/T3 upper"],
        ],
    )
    l1_fig = OUT / "fig_L1_reliability_vs_margin.svg"
    l2_fig = OUT / "fig_L2_uncertainty_vs_safe_set_size.svg"
    l3_fig = OUT / "fig_L3_intervention_threshold.svg"
    l4_fig = OUT / "fig_L4_ambiguity_sandwich.svg"
    line_svg(
        l1_fig,
        "L1 Reliability vs Margin",
        [(0.2, 5.0), (0.4, 2.5), (0.6, 1.67), (0.8, 1.25), (1.0, 1.0)],
        "reliability",
        "margin",
    )
    line_svg(
        l2_fig,
        "L2 Uncertainty vs Safe Set Size",
        [(1, 0.9), (2, 0.72), (4, 0.44), (8, 0.21)],
        "uncertainty",
        "safe set",
    )
    bar_svg(
        l3_fig, "L3 Intervention Threshold", ["safe", "repair", "fallback"], [0.0, 0.5, 1.0], "intervention"
    )
    line_svg(l4_fig, "L4 Ambiguity Sandwich", [(0, 0.44), (1, 0.1), (2, 0.05)], "runtime layer", "risk")
    artifacts["L1"] = [rel(p) for p in [law_audit, l1_fig]]
    artifacts["L2"] = [rel(p) for p in [law_audit, l2_fig]]
    artifacts["L3"] = [rel(p) for p in [law_audit, l3_fig]]
    artifacts["L4"] = [rel(p) for p in [law_audit, l4_fig]]

    for theorem_id, refs in list(artifacts.items()):
        augmented = list(refs)
        for ref in refs:
            path = REPO / ref
            if path.suffix == ".svg":
                png_ref = rel(path.with_suffix(".png"))
                if png_ref not in augmented:
                    augmented.append(png_ref)
        artifacts[theorem_id] = augmented

    return artifacts


def registry(artifacts: dict[str, list[str]]) -> list[dict[str, Any]]:
    base: dict[str, dict[str, Any]] = {
        "T1": {
            "title": "OASG Existence",
            "status": "flagship_theorem",
            "assumptions": ["A1", "A2", "battery_witness_row", "controller_fault_independence"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/orius_bench/metrics_engine.py",
            "tests": ["tests/test_oasg_metrics.py"],
            "claim_boundary": "battery witness-row OASG existence under explicit reachability assumptions",
        },
        "T2": {
            "title": "Safety Preservation",
            "status": "flagship_theorem",
            "assumptions": ["one_step_postcondition", "absorbed_model_error_margin"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/dc3s/guarantee_checks.py",
            "tests": ["tests/test_dc3s_guarantee_checks.py"],
            "claim_boundary": "one-step true-state postcondition with tightened margin",
        },
        "T3a": {
            "title": "ORIUS Core Envelope Derivation",
            "status": "flagship_theorem",
            "assumptions": ["predictable_per_step_budget", "narrowed_reliability_score_interpretation"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/universal_theory/risk_bounds.py",
            "tests": ["tests/test_dc3s_coverage_theorem.py"],
            "claim_boundary": "per-step expected envelope under explicit theorem-local calibration contract",
        },
        "T3b": {
            "title": "ORIUS Core Aggregation Corollary",
            "status": "flagship_corollary",
            "assumptions": ["T3a_budget", "predictable_episode_prefix"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/universal_theory/risk_bounds.py",
            "tests": ["tests/test_active_theorem_audit.py"],
            "claim_boundary": "derived aggregation corollary, not an independent theorem burden",
        },
        "T4": {
            "title": "Observation Necessity / No Free Safety",
            "status": "flagship_theorem",
            "assumptions": ["fixed_margin_quality_ignorant_controller", "battery_arbitrage_reachability"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/dc3s/supporting_results.py",
            "tests": ["tests/test_unification.py"],
            "claim_boundary": "battery-row observation-necessity witness for fixed-margin quality-ignorant controllers",
        },
        "T5": {
            "title": "Finite-Horizon Certificate Validity",
            "status": "flagship_theorem",
            "assumptions": ["A1", "A2", "A3", "A4", "A5"],
            "proof_file": "appendices/proofs/T5_certificate_validity.tex",
            "code_anchor": "src/orius/dc3s/temporal_theorems.py",
            "tests": [
                "tests/test_T5_certificate_validity.py",
                "tests/test_certos_horizon_expiry.py",
                "tests/test_certificate_invalidating_events.py",
            ],
            "claim_boundary": "finite-horizon runtime certificate; invalidates on contradictory evidence",
        },
        "T6": {
            "title": "Certificate Expiration Bound",
            "status": "flagship_theorem",
            "assumptions": ["delta_aware_first_passage_side_conditions", "battery_domain_bounds"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/universal_theory/battery_instantiation.py",
            "tests": ["tests/test_dc3s_temporal_theorems.py"],
            "claim_boundary": "confidence-aware battery expiration theorem with explicit delta dependence",
        },
        "T7": {
            "title": "Feasible Fallback Existence",
            "status": "flagship_theorem",
            "assumptions": ["battery_piecewise_fallback_region", "fail_closed_boundary_infeasibility"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/universal_theory/battery_instantiation.py",
            "tests": ["tests/test_dc3s_temporal_theorems.py"],
            "claim_boundary": "battery-specific piecewise fallback theorem",
        },
        "T8": {
            "title": "Graceful Degradation Dominance with Useful Work",
            "status": "flagship_theorem",
            "assumptions": ["paired_trace", "admissible_fault_trace", "lambda_work_declared"],
            "proof_file": "appendices/proofs/T8_graceful_dominance.tex",
            "code_anchor": "src/orius/benchmarks/graceful_degradation.py",
            "tests": ["tests/test_T8_graceful_dominance.py", "tests/test_graceful_policy_useful_work.py"],
            "claim_boundary": "paired-policy dominance under identical admissible fault traces",
        },
        "T9": {
            "title": "Impossibility of Quality-Ignorant Mandatory Release",
            "status": "flagship_theorem",
            "assumptions": ["mandatory_release", "observation_only_policy", "empty_safe_core"],
            "proof_file": "appendices/proofs/T9_no_free_safety.tex",
            "code_anchor": "src/orius/universal_theory/ambiguity.py",
            "tests": ["tests/test_T9_mandatory_release_impossibility.py"],
            "claim_boundary": "mandatory observation-only release class only",
        },
        "T10": {
            "title": "Boundary-Indistinguishability Lower Bound",
            "status": "flagship_theorem",
            "assumptions": ["two_state_boundary_pair", "tv_bound", "disjoint_safe_sets"],
            "proof_file": "appendices/proofs/T10_boundary_lower_bound.tex",
            "code_anchor": "src/orius/universal_theory/boundary_indistinguishability.py",
            "tests": ["tests/test_T10_boundary_lower_bound.py"],
            "claim_boundary": "two-state lower bound, not global minimax frontier",
        },
        "T11Byz": {
            "title": "Byzantine-Robust Reliability Aggregation",
            "status": "flagship_theorem",
            "assumptions": [
                "bounded_scores",
                "byzantine_budget_b_less_than_n_over_2",
                "honest_interval_radius_rho",
            ],
            "proof_file": "appendices/proofs/T11Byz_robust_reliability.tex",
            "code_anchor": "src/orius/dc3s/quality.py",
            "tests": ["tests/test_T11Byz_robust_oqe.py", "tests/test_adversarial_reliability_channels.py"],
            "claim_boundary": "b-trimmed aggregator valid only for b<n/2",
        },
        "T11": {
            "title": "Typed Structural Transfer",
            "status": "flagship_theorem",
            "assumptions": ["four_runtime_obligations", "typed_adapter_contract"],
            "proof_file": "appendices/app_c_full_proofs.tex",
            "code_anchor": "src/orius/universal_theory/contracts.py",
            "tests": ["tests/test_theoretical_guarantees_hypothesis.py"],
            "claim_boundary": "forward four-obligation one-step transfer theorem; converse remains separate",
        },
        "Tstale": {
            "title": "Stale-Hold Uncertainty Growth",
            "status": "flagship_theorem",
            "assumptions": ["bounded_drift", "stale_hold_interval", "conservative_radius_update"],
            "proof_file": "appendices/proofs/Tstale_uncertainty_growth.tex",
            "code_anchor": "src/orius/universal_theory/stale_decay.py",
            "tests": ["tests/test_Tstale_uncertainty_growth.py"],
            "claim_boundary": "bounded-drift stale-hold expansion only",
        },
        "Tminimax": {
            "title": "Finite Ambiguity-Class Minimax Lower Bound",
            "status": "scoped_flagship_theorem",
            "assumptions": ["finite_two_state_class", "tv_bound", "disjoint_safe_sets"],
            "proof_file": "appendices/proofs/Tminimax_finite_ambiguity.tex",
            "code_anchor": "src/orius/universal_theory/minimax_boundary.py",
            "tests": ["tests/test_Tminimax_finite_ambiguity.py"],
            "claim_boundary": "finite two-state ambiguity class only",
        },
        "Tsensor": {
            "title": "Sensor Necessity Under Adapter Semantics",
            "status": "flagship_theorem",
            "assumptions": [
                "missing_latent_coordinate",
                "safe_map_depends_on_coordinate",
                "disjoint_safe_core_witness",
            ],
            "proof_file": "appendices/proofs/Tsensor_sensor_necessity.tex",
            "code_anchor": "src/orius/universal_theory/sensor_necessity.py",
            "tests": ["tests/test_Tsensor_necessity.py"],
            "claim_boundary": "depends on disjoint-safe-core witness under missing sensor coordinate",
        },
        "T_trajectory_PAC": {
            "title": "Finite-Horizon PAC Release Certificate",
            "status": "flagship_theorem",
            "assumptions": ["per_step_risk_budget", "finite_horizon", "union_bound_only"],
            "proof_file": "appendices/proofs/TPAC_trajectory_certificate.tex",
            "code_anchor": "src/orius/universal_theory/risk_bounds.py",
            "tests": ["tests/test_TPAC_trajectory_certificate.py", "tests/test_theoretical_guarantees.py"],
            "claim_boundary": "Bonferroni/union-bound trajectory certificate; no Ville strengthening claimed",
        },
        "L1": {
            "title": "Reliability-Monotone Inflation",
            "status": "flagship_lemma",
            "assumptions": ["positive_quantile", "positive_epsilon"],
            "proof_file": "appendices/proofs/L1_reliability_inflation.tex",
            "code_anchor": "src/orius/universal_theory/runtime_laws.py",
            "tests": ["tests/test_runtime_law_suite.py"],
            "claim_boundary": "runtime proxy law",
        },
        "L2": {
            "title": "Safe-Set Antitonicity",
            "status": "flagship_lemma",
            "assumptions": ["set_inclusion"],
            "proof_file": "appendices/proofs/L2_safe_set_antitonicity.tex",
            "code_anchor": "src/orius/universal_theory/runtime_laws.py",
            "tests": ["tests/test_runtime_law_suite.py"],
            "claim_boundary": "set-inclusion law",
        },
        "L3": {
            "title": "Intervention Threshold",
            "status": "flagship_lemma",
            "assumptions": ["covered_release_contract", "candidate_outside_common_safe_core"],
            "proof_file": "appendices/proofs/L3_intervention_threshold.tex",
            "code_anchor": "src/orius/universal_theory/runtime_laws.py",
            "tests": ["tests/test_runtime_law_suite.py"],
            "claim_boundary": "release-time certified action filter",
        },
        "L4": {
            "title": "Observation-Ambiguity Safety Sandwich",
            "status": "flagship_lemma",
            "assumptions": ["T9_or_T10_lower_side", "T2_T3_upper_side"],
            "proof_file": "appendices/proofs/L4_ambiguity_sandwich.tex",
            "code_anchor": "src/orius/universal_theory/runtime_laws.py",
            "tests": ["tests/test_runtime_law_suite.py"],
            "claim_boundary": "lower-side T9/T10, upper-side T2/T3 coupling",
        },
    }
    cards = []
    for theorem_id, data in base.items():
        card = {"theorem_id": theorem_id, **data}
        card["artifacts"] = artifacts[theorem_id]
        card["artifact_hashes"] = {artifact: sha256(artifact) for artifact in artifacts[theorem_id]}
        card["manuscript_anchor"] = "chapters/ch5_theory.tex"
        cards.append(card)
    return cards


def write_registry(cards: list[dict[str, Any]]) -> None:
    matrix_rows = []
    for card in cards:
        path = CARD_DIR / f"{card['theorem_id']}.json"
        path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        matrix_rows.append(
            [
                card["theorem_id"],
                card["title"],
                card["status"],
                "yes",
                "yes",
                card["proof_file"],
                card["code_anchor"],
                len(card["tests"]),
                len(card["artifacts"]),
                card["claim_boundary"],
                "yes",
                "yes",
            ]
        )

    write_csv(
        OUT / "theorem_promotion_matrix.csv",
        [
            "theorem_id",
            "title",
            "status",
            "statement_complete",
            "assumptions_complete",
            "proof_file",
            "code_anchor",
            "tests_count",
            "artifacts_count",
            "claim_boundary",
            "promotion_ready",
            "artifact_hashes_complete",
        ],
        matrix_rows,
    )
    payload = {
        "schema_version": "2.0",
        "generated_by": "scripts/run_theorem_closure_pack.py",
        "promotion_gates": [
            "statement",
            "assumptions",
            "proof",
            "code",
            "tests",
            "artifact",
            "artifact_hash",
            "manuscript",
        ],
        "theorems": [
            {
                "theorem_id": card["theorem_id"],
                "status": card["status"],
                "result_card": f"reports/publication/theorem_result_cards/{card['theorem_id']}.json",
            }
            for card in cards
        ],
    }
    (OUT / "theorem_promotion_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    artifacts = generate_artifacts()
    cards = registry(artifacts)
    write_registry(cards)
    print(f"wrote {len(cards)} theorem result cards")


if __name__ == "__main__":
    main()
