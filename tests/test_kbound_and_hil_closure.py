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
