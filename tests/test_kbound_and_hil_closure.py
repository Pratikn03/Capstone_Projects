"""Regression tests for ORIUS + K-Bound Master Gap Register (G1-G14) and HIL Suites A, B, C."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lean4_theorems_fgh_mechanized() -> None:
    """Verify G8 & G11: Theorems F, G, H and LiquidityFloorObligation exist in Lean 4."""
    lean_fgh = REPO_ROOT / "formal" / "Orius" / "TheoremsFGH.lean"
    lean_ae = REPO_ROOT / "formal" / "Orius" / "TheoremsAE.lean"
    assert lean_fgh.exists() and lean_ae.exists()
    content_fgh = lean_fgh.read_text(encoding="utf-8")
    content_ae = lean_ae.read_text(encoding="utf-8")
    assert "sorry" not in content_fgh
    assert "sorry" not in content_ae
    for token in (
        "theorem_f_terminal_safe_stop_invariance",
        "theorem_g_arbitrary_ai_hallucination_immunity",
        "theorem_h_financial_flash_crash_immunity",
        "LiquidityFloorObligation",
        "liquidity_floor_prevents_barrier_singularity",
        "theorem_h_liquidity_floor_safe_liquidation",
    ):
        assert token in content_fgh or token in content_ae, f"Missing Lean 4 declaration: {token}"


def test_distinct_appendix_g_figures_and_ch13_scorecards() -> None:
    """Verify G2 & G4: Chapter 13 figures exist and Appendix G figures are distinct."""
    fig_cw = REPO_ROOT / "reports" / "publication" / "fig_coverage_width.png"
    fig_cwt = REPO_ROOT / "reports" / "publication" / "fig_coverage_width_tradeoff.png"
    assert fig_cw.exists() and fig_cwt.exists()
    assert _sha256(fig_cw) != _sha256(fig_cwt), "G4: fig_coverage_width.png must be distinct from tradeoff plot"

    for fig_name in (
        "SEVEN_DOMAIN_CRASH_FREE_SCORECARD.png",
        "TERMINAL_SAFE_STOP_PHASE_PORTRAIT.png",
        "EPROCESS_AND_BYZANTINE_RESILIENCE.png",
        "HARDWARE_CLOSED_LOOP_MONEY_PLOT.png",
    ):
        for folder in ("figures", "paper/figures", "paper/orius_foundations/figures"):
            path = REPO_ROOT / folder / fig_name
            assert path.exists() and path.stat().st_size > 10_000, f"Missing or empty figure: {path}"


def test_two_regime_tta_active_adaptation_and_freeze() -> None:
    """Verify G12: K-Bound outputs ADAPT under 0ms covariate shift and FREEZE under 100-650ms lag."""
    csv_path = REPO_ROOT / "reports" / "publication" / "tta_multi_baseline_results.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) >= 6

    shift_row = next(r for r in rows if "Active Covariate Shift" in r["regime"])
    assert shift_row["kbound_action"] == "ADAPT"
    assert float(shift_row["kbound_mse"]) < float(shift_row["frozen_mse"])

    lag_rows = [r for r in rows if int(r["lag_ms"]) >= 100]
    assert len(lag_rows) >= 4
    for r in lag_rows:
        assert r["kbound_action"] == "ABSTAIN/FREEZE"
        assert float(r["kbound_mse"]) < float(r["tent_mse"])


def test_closed_loop_robot_suite_a_zero_collisions() -> None:
    """Verify G13 / Test Suite A: 200 closed-loop robot episodes across R-1..R-4 have 0 collisions."""
    summary_path = REPO_ROOT / "reports" / "hil" / "robot_closed_loop" / "robot_obstacle_stopping_summary.csv"
    ledger_path = REPO_ROOT / "reports" / "hil" / "robot_closed_loop" / "robot_certos_ledger.jsonl"
    assert summary_path.exists() and ledger_path.exists()

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    kbound_rows = [r for r in rows if r["controller"] == "ORIUS_KBound"]
    assert len(kbound_rows) == 4
    for r in kbound_rows:
        assert int(r["collisions"]) == 0
        assert float(r["collision_rate_pct"]) == 0.0
        assert float(r["min_clearance_mean_m"]) > 0.5
        assert float(r["certificate_valid_rate_pct"]) == 100.0

    ledger_entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger_entries) > 50
    for entry in ledger_entries:
        assert len(entry["cert_sha256"]) == 64
        assert entry["status"] in {"VALID", "EXPIRED", "FALLBACK_LOCK", "BLACKOUT_BRAKING", "FALLBACK_SAFE_STOP", "TERMINAL_Z_STOP"}


def test_physical_battery_bench_suite_b_zero_violations_and_bus_cap() -> None:
    """Verify G14 / Test Suite B: 4,800-step BMS bench has 0 violations, 96-step CertOS chain, and <=50A bus."""
    summary_path = REPO_ROOT / "reports" / "hil" / "physical_bench" / "battery_bench_summary.csv"
    ledger_path = REPO_ROOT / "reports" / "hil" / "physical_bench" / "battery_bench_certos_96step.jsonl"
    bus_path = REPO_ROOT / "reports" / "hil" / "physical_bench" / "battery_bench_compositional_bus.csv"
    assert summary_path.exists() and ledger_path.exists() and bus_path.exists()

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    kbound = next(r for r in rows if r["controller"] == "ORIUS_KBound_Bench")
    unshielded = next(r for r in rows if r["controller"] == "Unshielded_BMS_Policy")
    assert int(kbound["total_steps"]) == 4800
    assert int(kbound["violation_steps"]) == 0
    assert float(kbound["tsvr_pct"]) == 0.0
    assert float(kbound["min_soc_margin_pct"]) > 1.0
    assert float(kbound["sha256_chain_valid_pct"]) == 100.0
    assert int(unshielded["violation_steps"]) > 1000

    ledger_entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger_entries) == 96
    for entry in ledger_entries:
        assert len(entry["cert_sha256"]) == 64

    bus_rows = list(csv.DictReader(bus_path.open("r", encoding="utf-8")))
    assert len(bus_rows) >= 96
    for r in bus_rows:
        assert abs(float(r["orius_compositional_bus_amp"])) <= float(r["bus_limit_amp"]) + 1e-6
        assert float(r["uncoordinated_bus_amp"]) > float(r["bus_limit_amp"])


def test_suite_c_component_ablations_and_hash_registry() -> None:
    """Verify Test Suite C (C.1 isolated ablations in Table 9.8 & C.2 SHA-256 hash registry in App L/Q)."""
    closure_json = REPO_ROOT / "reports" / "hil" / "hil_and_ablation_closure_summary.json"
    ablation_csv = REPO_ROOT / "reports" / "publication" / "component_ablations_kbound.csv"
    ablation_tex = REPO_ROOT / "reports" / "publication" / "table_component_ablations_kbound.tex"
    app_l_tex = REPO_ROOT / "appendices" / "app_r_locked_artifact_hash_registry.tex"
    app_q_tex = REPO_ROOT / "reports" / "publication" / "reproducibility_appendix_generated.tex"

    assert closure_json.exists() and ablation_csv.exists() and ablation_tex.exists()
    data = json.loads(closure_json.read_text(encoding="utf-8"))
    assert len(data["suite_a_robot_closed_loop"]["subtests"]) == 4
    assert len(data["suite_b_physical_battery_bench"]["subtests"]) == 3
    ablation_rows = data["suite_c_component_ablations"]["rows"]
    assert len(ablation_rows) == 6
    assert ablation_rows[0]["tsvr_pct"] == 0.0
    for row in ablation_rows[1:]:
        assert float(row["tsvr_pct"]) > 5.0

    for tex_file in (app_l_tex, app_q_tex):
        txt = tex_file.read_text(encoding="utf-8").replace(r"\_", "_")
        assert "TheoremsFGH.lean" in txt
        assert "robot_obstacle_stopping_summary.csv" in txt
        assert "battery_bench_summary.csv" in txt


def test_all_compiled_pdfs_zero_unresolved_refs_and_zero_placeholders() -> None:
    """Verify G1 & Phase 4: All compiled PDFs have 0 unresolved references (??/[?]) and 0 placeholder names."""
    from pypdf import PdfReader

    pdfs = [
        "orius_book.pdf",
        "paper.pdf",
        "paper/paper.pdf",
        "paper/orius_book.pdf",
        "paper/orius_foundations/main.pdf",
        "paper/orius_monograph_complete.pdf",
        "paper/ieee/orius_ieee_main.pdf",
        "paper/ieee/orius_ieee_detailed_main.pdf",
        "paper/review/orius_review_dossier.pdf",
        "appendices/app_c_all_theorems.pdf",
    ]
    for rel_path in pdfs:
        pdf_path = REPO_ROOT / rel_path
        assert pdf_path.exists(), f"Missing compiled PDF: {rel_path}"
        reader = PdfReader(str(pdf_path))
        assert len(reader.pages) >= 6
        for idx, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            assert "??" not in txt and "[?]" not in txt, f"Unresolved ref in {rel_path} p.{idx + 1}"
            assert "J. Doe" not in txt and "R. Smith" not in txt, f"Placeholder author in {rel_path} p.{idx + 1}"


def test_kbound_epistemic_mind_full_paper_integration() -> None:
    """Verify 100% of K-Bound (ADAPT, FREEZE, ABSTAIN, Frontier, Compound, Lean 4) is inside ORIUS."""
    import numpy as np

    from orius.kbound import (
        ABSTAIN,
        ADAPT,
        FREEZE,
        assess_frontier,
        evaluate_actions,
        evaluate_compound_population_mind,
        evaluate_kbound_mind,
        evaluate_population_frontier_mind,
    )

    lean_kb = REPO_ROOT / "formal" / "Orius" / "KBoundMind.lean"
    assert lean_kb.exists()
    kb_txt = lean_kb.read_text(encoding="utf-8")
    assert "sorry" not in kb_txt
    for thm in (
        "frontier_positive_sound",
        "frontier_negative_sound",
        "frontier_band_zero_witness",
        "frontier_open_band_opposite_witnesses",
        "matched_evidence_abstention_lower_bound",
        "fibre_audit_floor_ge_beta",
        "cert_false_adapt_sound",
        "cert_false_freeze_sound",
        "kbound_orius_epistemic_physical_coupling",
    ):
        assert thm in kb_txt, f"Missing Lean 4 K-Bound theorem: {thm}"

    # Population Frontier (Theorems 1-3 & Corollary 1)
    pop_adapt = evaluate_population_frontier_mind(M=0.22, beta=0.08, disagreement_mass=0.35, reliability_w=0.92)
    pop_freeze = evaluate_population_frontier_mind(M=-0.24, beta=0.08, disagreement_mass=0.35, reliability_w=0.55)
    pop_abstain = evaluate_population_frontier_mind(M=0.03, beta=0.08, disagreement_mass=0.35, reliability_w=0.48)
    assert pop_adapt["kga_decision"] == ADAPT
    assert pop_freeze["kga_decision"] == FREEZE
    assert pop_abstain["kga_decision"] == ABSTAIN
    assert abs(pop_abstain["fibre_audit_floor_gamma_z"] - 0.08) < 1e-9
    assert abs(pop_abstain["matched_evidence_abstain_prob_floor"] - 0.90) < 1e-9
    assert assess_frontier(M=0.03, beta=0.08).action == ABSTAIN

    # Empirical KGA Gate (Proposition 2)
    calib_pool_25 = np.linspace(0.005, 0.030, 25)
    emp_adapt = evaluate_kbound_mind(reliability_w=0.90, delta_hat=0.085, calib_residuals=calib_pool_25, alpha=0.05)
    emp_freeze = evaluate_kbound_mind(reliability_w=0.45, delta_hat=-0.095, calib_residuals=calib_pool_25, alpha=0.05)
    emp_abstain = evaluate_kbound_mind(reliability_w=0.52, delta_hat=0.005, calib_residuals=calib_pool_25, alpha=0.05)
    assert emp_adapt.kbound_action == ADAPT and emp_adapt.orius_physical_mode == "ADAPT_AND_RELEASE"
    assert emp_freeze.kbound_action == FREEZE and emp_freeze.is_certified_rejection is True
    assert emp_abstain.kbound_action == ABSTAIN and emp_abstain.is_epistemic_abstention is True

    # Compound Cell-to-Population Transfer (Proposition 3)
    comp = evaluate_compound_population_mind(delta_hat=0.14, epsilon=0.03, n_eval=1000)
    assert comp["kga_decision"] == ADAPT and comp["population_lower"] > 0.0

    # Action Evaluation Summary
    rep = evaluate_actions(["ADAPT", "FREEZE", "ABSTAIN"], [0.09, -0.08, 0.002])
    assert rep.summary.false_adapt_rate_all == 0.0
    assert rep.summary.false_freeze_rate_all == 0.0

