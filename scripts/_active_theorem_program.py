"""Shared builders for the active T1--T11 theorem audit surface."""
from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "publication"
THEOREM_REGISTER_CSV = REPORTS_DIR / "theorem_surface_register.csv"
ASSUMPTION_REGISTER_FILE = REPO_ROOT / "appendices" / "app_b_assumptions.tex"
PROOF_APPENDIX_FILE = REPO_ROOT / "appendices" / "app_c_full_proofs.tex"

AUDIT_JSON = REPORTS_DIR / "active_theorem_audit.json"
AUDIT_CSV = REPORTS_DIR / "active_theorem_audit.csv"
AUDIT_MD = REPORTS_DIR / "active_theorem_audit.md"
REMEDIATION_MD = REPORTS_DIR / "active_theorem_remediation_plan.md"


@dataclass(frozen=True)
class AnchorSpec:
    path: str
    symbol: str | None = None
    note: str = ""


@dataclass(frozen=True)
class TheoremSpec:
    theorem_id: str
    title: str
    register_title: str
    proof_anchor: str
    assumptions_used: tuple[str, ...]
    dependencies: tuple[str, ...]
    weakest_step: str
    rigor_rating: str
    code_correspondence: str
    code_correspondence_detail: str
    severity_if_broken: str
    remediation_class: str
    remediation_detail: str
    code_anchors: tuple[AnchorSpec, ...]
    test_anchors: tuple[AnchorSpec, ...]
    namespace_drift_note: str = ""


@dataclass(frozen=True)
class DriftEntry:
    surface: str
    issue: str
    impact: str
    status: str
    remediation: str


ACTIVE_THEOREM_SPECS: tuple[TheoremSpec, ...] = (
    TheoremSpec(
        theorem_id="T1",
        title="OASG Existence",
        register_title="OASG Existence",
        proof_anchor=r"\section{C.1\quad Battery-Domain OASG Existence}",
        assumptions_used=(
            "A1",
            "A2",
            "A4",
            "Boundary proximity under arbitrage remains non-trivial on the defended battery witness.",
        ),
        dependencies=(
            "Lemma: Observation gap under dropout",
            "Lemma: Boundary proximity under arbitrage",
        ),
        weakest_step=(
            "Lemma boundary-proximity only argues that non-zero price variance pushes the controller near a boundary; "
            "it does not derive the positive boundary-visit fraction from the dispatch model."
        ),
        rigor_rating="plausible-but-informal",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The runtime and metrics code represent observed-vs-true disagreement explicitly, but there is no executable witness "
            "for the proof's boundary-reachability step."
        ),
        severity_if_broken="high",
        remediation_class="add assumption",
        remediation_detail=(
            "Either register a battery-specific boundary-reachability assumption explicitly or prove it from the dispatch model."
        ),
        code_anchors=(
            AnchorSpec("src/orius/orius_bench/metrics_engine.py", "compute_oasg", "Counts observation-action safety gaps."),
            AnchorSpec("src/orius/cpsbench_iot/runner.py", "run_single", "Empirical witness path with true and observed trajectories."),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_orius_bench.py",
                "test_oasg_counts_observation_action_safety_gap",
                "Metric-level witness only; not a proof of existence under optimal arbitrage.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T2",
        title="Safety Preservation",
        register_title="Safety Preservation",
        proof_anchor=r"\section{C.2\quad One-Step Safety Preservation}",
        assumptions_used=(
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A7",
            "The tightening already absorbs the one-step model-error allowance.",
        ),
        dependencies=(
            "Proposition: Inflated set contains the current state",
            "Proposition: Tightened feasibility implies true feasibility",
        ),
        weakest_step=(
            "The proof says the model-error allowance is already absorbed by the tightening, but that absorption is asserted in prose "
            "rather than derived from the actual FTIT construction."
        ),
        rigor_rating="plausible-but-informal",
        code_correspondence="matches",
        code_correspondence_detail=(
            "The main conditional one-step claim matches the guarantee-check and shield surface, even though the earlier general-CPS "
            "statement in the same chapter still contains a broader probabilistic line."
        ),
        severity_if_broken="high",
        remediation_class="strengthen proof",
        remediation_detail=(
            "Make the model-error buffer explicit in the theorem statement or add a lemma tying the implemented tightening to the residual bound."
        ),
        code_anchors=(
            AnchorSpec("src/orius/dc3s/guarantee_checks.py", "check_soc_invariance", "Implements the one-step battery safety check."),
            AnchorSpec("src/orius/dc3s/guarantee_checks.py", "evaluate_guarantee_checks", "Collects the deterministic safety predicates."),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_guarantee_checks.py",
                "test_guarantee_checks_pass_for_safe_action",
                "Checks the implemented one-step safety predicate only.",
            ),
        ),
        namespace_drift_note=(
            "The same chapter still carries a broader general-CPS statement with an unearned probability line; the active audit selects "
            "the main conditional theorem row instead."
        ),
    ),
    TheoremSpec(
        theorem_id="T3",
        title="ORIUS Core Bound",
        register_title="ORIUS Core Bound",
        proof_anchor=r"\section{C.3\quad Battery ORIUS Core Bound}",
        assumptions_used=(
            "A1",
            "A2",
            "A4",
            "A5",
            "A6",
            "A7",
            "Battery risk-envelope contract: P[Z_t=1 | H_t] <= alpha * (1 - w_t).",
            "w_t is a runtime reliability score, not a probability by definition.",
        ),
        dependencies=(
            "Lemma: Aggregation under a predictable risk budget",
            "Battery-specific theorem-local risk budget contract",
        ),
        weakest_step=(
            "The bound is only as rigorous as the unproved battery risk-envelope contract; the proof never derives that contract from conformal "
            "coverage, detector lag, or the OQE construction."
        ),
        rigor_rating="has-a-hole",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The helpers compute the envelope and expose the missing assumptions honestly, but no code path justifies the battery risk budget "
            "from the implemented OQE or calibration stack."
        ),
        severity_if_broken="critical",
        remediation_class="add assumption",
        remediation_detail=(
            "Either promote the battery risk-envelope contract to an explicit defended assumption or add a new battery-specific lemma that "
            "derives it from the implemented calibration logic."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/universal_theory/risk_bounds.py",
                "compute_episode_risk_bound",
                "Implements only the aggregation envelope, not the bridge to the budget contract.",
            ),
            AnchorSpec(
                "src/orius/universal_theory/risk_bounds.py",
                "risk_envelope_assumptions",
                "Makes the missing theorem-local assumptions explicit.",
            ),
            AnchorSpec(
                "src/orius/dc3s/coverage_theorem.py",
                "compute_expected_violation_bound",
                "Backward-compatible wrapper around the narrowed T3 helper.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_coverage_theorem.py",
                "test_compute_expected_violation_bound",
                "Verifies the algebraic envelope, not the bridge from runtime reliability to residual-risk probability.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T4",
        title="No Free Safety",
        register_title="No Free Safety",
        proof_anchor=r"\section{C.4\quad No Free Safety}",
        assumptions_used=(
            "A1",
            "A2",
            "A4",
            "Quality-ignorant controllers admit a fixed effective margin m^pi.",
            "An admissible concentrated fault window can exceed delta_bnd + m^pi.",
        ),
        dependencies=(
            "Definition: Quality-ignorant controller",
            "Lemma: Admissible fault sequence existence",
            "Lemma: No margin compensation for quality-ignorant controllers",
            "Theorem T1 boundary-proximity witness",
        ),
        weakest_step=(
            "The proof reduces every quality-ignorant controller to a fixed scalar margin m^pi without proving that the whole controller class "
            "admits that abstraction."
        ),
        rigor_rating="has-a-hole",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The repo has empirical fault generators and quality-ignorant baseline reductions, but no executable battery witness that "
            "constructs the exact three-phase adversarial sequence used in the proof."
        ),
        severity_if_broken="high",
        remediation_class="strengthen proof",
        remediation_detail=(
            "Define the quality-ignorant controller class more tightly and prove that the fault witness works for that class rather than for "
            "an informal fixed-margin caricature."
        ),
        code_anchors=(
            AnchorSpec("src/orius/cpsbench_iot/scenarios.py", "generate_episode", "Supplies admissible degraded-observation episodes."),
            AnchorSpec("src/orius/cpsbench_iot/runner.py", "run_single", "Runs quality-ignorant baselines against the witness battery plant."),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_unification.py",
                "test_unification_argument_w_t_never_drops_below_one",
                "Supporting quality-ignorant limitation check, not the theorem's adversarial battery witness.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T5",
        title="Certificate validity horizon",
        register_title="Certificate validity horizon",
        proof_anchor=r"\section{C.5\quad Certificate Validity Horizon}",
        assumptions_used=(
            "A1",
            "A2",
            "A4",
            "A5",
            "A6",
            "A7",
            "Future drift remains summarized by the sub-Gaussian sigma_d * sqrt(Delta) tube.",
        ),
        dependencies=(
            "Definition: Safety certificate",
            "Definition: Forward tube",
        ),
        weakest_step=(
            "The chapter proof jumps from issuance-time coverage to future-step safety probabilities without an explicit temporal bridge beyond "
            "the stylized drift tube."
        ),
        rigor_rating="plausible-but-informal",
        code_correspondence="matches",
        code_correspondence_detail=(
            "The deterministic horizon helper matches the stated maximal containment problem, even though the probabilistic interpretation in "
            "the chapter remains compressed."
        ),
        severity_if_broken="medium",
        remediation_class="strengthen proof",
        remediation_detail=(
            "Separate the deterministic tube-containment statement from any future-step probability claim, or prove the temporal concentration step explicitly."
        ),
        code_anchors=(
            AnchorSpec("src/orius/universal_theory/battery_instantiation.py", "forward_tube", "Builds the battery forward tube."),
            AnchorSpec(
                "src/orius/universal_theory/battery_instantiation.py",
                "certificate_validity_horizon",
                "Searches the maximal horizon whose tube stays inside bounds.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_temporal_theorems.py",
                "test_certificate_validity_horizon_matches_expiration_bound_for_zero_action",
                "Checks deterministic containment logic.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T6",
        title="Certificate expiration bound",
        register_title="Certificate expiration bound",
        proof_anchor=r"\section{C.6\quad Certificate Expiration Bound}",
        assumptions_used=(
            "A4",
            "A6",
            "A7",
            "The forward-tube boundary crossing is controlled by the drift scale alone.",
        ),
        dependencies=(
            "Definition: Forward tube",
        ),
        weakest_step=(
            "The proof is scalar and conservative, but it silently assumes the action contribution is already frozen inside the issued certificate."
        ),
        rigor_rating="solid",
        code_correspondence="matches",
        code_correspondence_detail=(
            "The implemented lower bound is exactly the analytical floor(delta_bnd^2 / sigma_d^2) expression defended in the text."
        ),
        severity_if_broken="medium",
        remediation_class="strengthen proof",
        remediation_detail=(
            "Keep the current statement but spell out that the bound is conditioned on the issued safe action and fixed drift model."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/universal_theory/battery_instantiation.py",
                "certificate_expiration_bound",
                "Exact implementation of the analytical lower bound.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_temporal_theorems.py",
                "test_certificate_validity_horizon_matches_expiration_bound_for_zero_action",
                "Checks the bound against the validity-horizon helper on the zero-dispatch witness.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T7",
        title="Feasible fallback existence",
        register_title="Feasible fallback existence",
        proof_anchor=r"\section{C.8\quad Feasible Fallback Existence}",
        assumptions_used=(
            "A1",
            "A3",
            "A4",
            "A8",
            "Current state is interior by at least the one-step model-error margin.",
        ),
        dependencies=(
            "Constructive zero-dispatch battery fallback",
        ),
        weakest_step=(
            "The theorem statement says any currently safe state, but both the appendix proof and the code only support interior states with slack "
            "against model error."
        ),
        rigor_rating="broken",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The battery helper validates zero dispatch only for states that remain safe after adding and subtracting the model-error margin. "
            "That is strictly narrower than the theorem statement."
        ),
        severity_if_broken="high",
        remediation_class="narrow statement",
        remediation_detail=(
            "Replace 'any safe state' with an interior-state condition that matches the implemented fallback validator and the appendix restatement."
        ),
        code_anchors=(
            AnchorSpec("src/orius/universal_theory/battery_instantiation.py", "zero_dispatch_fallback", "Constructive battery fallback."),
            AnchorSpec(
                "src/orius/universal_theory/battery_instantiation.py",
                "validate_battery_fallback",
                "Checks fallback preservation only over the interior slack interval.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_temporal_theorems.py",
                "test_certify_fallback_existence_passes_for_interior_soc",
                "Positive witness explicitly uses an interior SOC state.",
            ),
            AnchorSpec(
                "tests/test_dc3s_temporal_theorems.py",
                "test_certify_fallback_existence_fails_near_boundary",
                "Negative witness exposes the over-broad theorem statement.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T8",
        title="Graceful degradation dominance",
        register_title="Graceful degradation dominance",
        proof_anchor=r"\section{C.9\quad Graceful Degradation Dominance}",
        assumptions_used=(
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A8",
            "The graceful landing regime makes the safe interior absorbing after landing.",
        ),
        dependencies=(
            "Theorem T4 witness for the uncontrolled comparator",
            "Graceful landing absorbs future excursions after the landing phase",
        ),
        weakest_step=(
            "The strict-improvement claim depends on an absorbing safe-landing regime that is asserted by chapter narrative rather than discharged as "
            "a theorem-grade dynamics argument."
        ),
        rigor_rating="has-a-hole",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The helper compares two violation sequences and can detect strict domination, but it does not prove that the implemented graceful "
            "controller always generates the favorable sequence under the same admissible faults."
        ),
        severity_if_broken="medium",
        remediation_class="strengthen proof",
        remediation_detail=(
            "Either prove the absorbing-landing argument formally or narrow the theorem to the comparison helper's sequence-level claim."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/universal_theory/battery_instantiation.py",
                "evaluate_graceful_degradation_dominance",
                "Sequence-level dominance checker.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_dc3s_temporal_theorems.py",
                "test_graceful_degradation_dominance_detects_strict_domination",
                "Checks the comparison helper only.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T9",
        title="Universal Impossibility, T9",
        register_title="Universal Impossibility, T9",
        proof_anchor=r"\section{C.11\quad Universal Impossibility (T9)}",
        assumptions_used=(
            "A4",
            "A6': phi-mixing fault process with geometric decay.",
            "A positive witness constant c > 0 is available from the boundary-reachability argument.",
        ),
        dependencies=(
            "Theorem T4 no-free-safety witness",
            "Azuma-Hoeffding concentration for separated windows",
            "Window buffering argument based on phi-mixing",
        ),
        weakest_step=(
            "The proof imports both the phi-mixing assumption A6' and the witness constant c > 0 without deriving either one on the active theorem surface."
        ),
        rigor_rating="broken",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The helper reproduces the claimed c * d * T scaling algebra once c is supplied externally, but it does not instantiate the witness constant "
            "or the window-separation argument from runtime objects."
        ),
        severity_if_broken="critical",
        remediation_class="downgrade register status",
        remediation_detail=(
            "Treat T9 as a proof sketch or structural conjecture until the missing phi-mixing assumption and witness constant are registered and discharged."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/dc3s/theoretical_guarantees.py",
                "compute_universal_impossibility_bound",
                "Encodes the claimed scaling only after c and effective horizon are handed in.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_theoretical_guarantees.py",
                "test_expected_lower_bound_matches_linear_formula",
                "Algebraic witness test only.",
            ),
            AnchorSpec(
                "tests/test_theoretical_guarantees_hypothesis.py",
                "test_t9_impossibility_bound_matches_linear_scaling",
                "Property test for the helper's formula, not for the proof obligations.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T10",
        title="Boundary-indistinguishability lower bound, T10",
        register_title="Boundary-indistinguishability lower bound, T10",
        proof_anchor=r"\section{C.12\quad Boundary-Indistinguishability Lower Bound (T10)}",
        assumptions_used=(
            "Controller acts only through the degraded observation channel.",
            "TV(P_0,t, P_1,t) <= w_t for the boundary subproblem.",
            "Unsafe-side prior mass satisfies P(H_1,t) >= p_t.",
            "The unsafe-side decision error controls true-state violations.",
        ),
        dependencies=(
            "Le Cam two-point lemma",
            "Boundary-mass sequence p_t",
        ),
        weakest_step=(
            "Le Cam only lower-bounds the maximum of the two hypothesis error rates, but the proof immediately treats that as a lower bound on the unsafe-side "
            "error probability alone."
        ),
        rigor_rating="broken",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The helper and frontier utilities reproduce the stylized lower-bound algebra, but they accept boundary mass as an exogenous input and do not "
            "repair the invalid one-sided Le Cam step."
        ),
        severity_if_broken="critical",
        remediation_class="narrow statement",
        remediation_detail=(
            "Retitle T10 as a stylized lower-bound construction unless a valid unsafe-side error lower bound and a defended boundary-mass model are supplied."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/dc3s/theoretical_guarantees.py",
                "compute_stylized_frontier_lower_bound",
                "Implements the stylized half-sum formula directly.",
            ),
            AnchorSpec(
                "src/orius/universal/reliability_safety_frontier.py",
                "compute_frontier",
                "Chapter-facing plotting proxy for the scoped lower-bound surface.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_theoretical_guarantees.py",
                "test_general_formula_matches_sum_p_one_minus_w",
                "Formula check only.",
            ),
            AnchorSpec(
                "tests/test_theoretical_guarantees_hypothesis.py",
                "test_t10_frontier_lower_bound_matches_sum_formula",
                "Property test for the stylized algebra only.",
            ),
        ),
    ),
    TheoremSpec(
        theorem_id="T11",
        title="Typed structural transfer theorem, T11",
        register_title="Typed structural transfer theorem, T11",
        proof_anchor=r"\section{C.13\quad Typed Structural Transfer and Failure-Mode Converse (T11)}",
        assumptions_used=(
            "Coverage obligation for the observation-consistent state set.",
            "Soundness of the tightened safe-action set.",
            "Repair membership in the tightened safe-action set.",
            "Fallback admissibility when the tightened set is empty.",
        ),
        dependencies=(
            "Definition: Universal adapter contract",
            "Corollary: Episode aggregation under explicit per-step budgets",
            "Proposition: Failure of any transfer obligation breaks the reference proof pattern",
        ),
        weakest_step=(
            "The forward theorem is a clean one-step transfer under four obligations, but the converse is only a proof-pattern failure argument and the "
            "supporting mini-harness still tends to overstate what is actually transferred."
        ),
        rigor_rating="has-a-hole",
        code_correspondence="partial",
        code_correspondence_detail=(
            "The authoritative typed contract surface exists, but the executable checks still mix structural method checks, mini-harness invariants, and "
            "runtime consistency checks instead of discharging the four theorem obligations directly."
        ),
        severity_if_broken="high",
        remediation_class="narrow statement",
        remediation_detail=(
            "Keep T11 as a four-obligation one-step transfer theorem, demote the five-invariant mini-harness to supporting evidence, and avoid implying that "
            "passing the mini-harness automatically yields the episode bound."
        ),
        code_anchors=(
            AnchorSpec(
                "src/orius/universal_theory/contracts.py",
                "ContractVerifier.validate_runtime_step",
                "Authoritative typed runtime contract surface.",
            ),
            AnchorSpec(
                "src/orius/dc3s/theoretical_guarantees.py",
                "evaluate_structural_transfer",
                "Executable four-obligation witness for the narrowed theorem statement.",
            ),
            AnchorSpec(
                "src/orius/universal/contract.py",
                "ContractVerifier.check",
                "Supporting five-invariant reference harness only.",
            ),
        ),
        test_anchors=(
            AnchorSpec(
                "tests/test_theoretical_guarantees_hypothesis.py",
                "test_t11_transfer_requires_all_four_obligations",
                "Covers the active four-obligation theorem surface.",
            ),
            AnchorSpec(
                "tests/test_universal_theory.py",
                "test_contract_verifier_accepts_canonical_domain_instantiations",
                "Checks the typed runtime contract on canonical adapters.",
            ),
            AnchorSpec(
                "tests/test_universal_contract.py",
                "test_passes_all_five_invariants",
                "Supporting mini-harness test, not the full active theorem.",
            ),
        ),
    ),
)


NAMESPACE_DRIFT: tuple[DriftEntry, ...] = (
    DriftEntry(
        surface="src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py",
        issue=(
            "Legacy auxiliary group-coverage and Hoeffding helpers were labeled as Theorem 9 and Theorem 10 even though the active T9/T10 program now "
            "means universal impossibility and the boundary-indistinguishability lower bound."
        ),
        impact="Silent theorem-number reuse makes the code surface contradict the active manuscript namespace.",
        status="resolved",
        remediation="Keep these helpers explicitly tagged as legacy auxiliary coverage analyses and never as active T9/T10 witnesses.",
    ),
    DriftEntry(
        surface="src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py",
        issue=(
            "The five-invariant reference harness was described as if satisfying it automatically discharged the full active T11 theorem and the episode-level bound."
        ),
        impact="That story is wider than the live manuscript, which transfers only the one-step theorem unless a separate per-step risk budget is provided.",
        status="resolved",
        remediation="Describe the five invariants as supporting reference-harness checks and route theorem-facing claims through the typed T11 surface instead.",
    ),
    DriftEntry(
        surface="reports/publication/theorem_surface_register.csv",
        issue=(
            "The register is an inventory, not a one-row-per-theorem program: T1-T4 each appear as both general-CPS and main-theorem rows."
        ),
        impact="Consumers can accidentally pick the broader row and overstate the defended theorem surface.",
        status="open",
        remediation="Use the active T1--T11 audit artifact as the single reconciled theorem map and keep the register as raw inventory only.",
    ),
)


CSV_COLUMNS = (
    "theorem_id",
    "title",
    "statement_location",
    "proof_location",
    "assumptions_used",
    "unresolved_assumptions",
    "dependencies",
    "weakest_step",
    "rigor_rating",
    "code_correspondence",
    "severity_if_broken",
    "remediation_class",
    "code_anchors",
    "test_anchors",
)


def _read_register_rows() -> list[dict[str, str]]:
    with THEOREM_REGISTER_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _register_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_register_rows()
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["register_id"], row["title"])
        lookup[key] = row
    return lookup


def _assumption_lookup() -> dict[str, dict[str, str]]:
    rows = _read_register_rows()
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["group"] == "Assumption register" and row["register_id"]:
            lookup[row["register_id"]] = row
    return lookup


def _find_text_line(path: Path, needle: str) -> int:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return lineno
    raise ValueError(f"Unable to find '{needle}' in {path}")


def _find_py_symbol_line(path: Path, symbol: str) -> int:
    parts = symbol.split(".")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def find_in_nodes(nodes: list[ast.stmt], remaining: list[str]) -> int | None:
        target = remaining[0]
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                if len(remaining) == 1:
                    return int(node.lineno)
                if isinstance(node, ast.ClassDef):
                    nested = find_in_nodes(list(node.body), remaining[1:])
                    if nested is not None:
                        return nested
        return None

    line = find_in_nodes(list(tree.body), parts)
    if line is None and len(parts) == 1:
        target = parts[0]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                line = int(node.lineno)
                break
    if line is None:
        raise ValueError(f"Unable to find symbol '{symbol}' in {path}")
    return line


def _resolve_anchor(anchor: AnchorSpec) -> dict[str, str]:
    path = REPO_ROOT / anchor.path
    if not path.exists():
        raise FileNotFoundError(path)
    line = 1
    if anchor.symbol:
        if path.suffix == ".py":
            line = _find_py_symbol_line(path, anchor.symbol)
        else:
            line = _find_text_line(path, anchor.symbol)
    return {
        "path": anchor.path,
        "symbol": anchor.symbol or "",
        "location": f"{anchor.path}:{line}",
        "note": anchor.note,
    }


def _resolve_assumptions(assumptions: tuple[str, ...], assumption_lookup: dict[str, dict[str, str]]) -> tuple[list[str], list[dict[str, str]], list[str]]:
    locations: list[dict[str, str]] = []
    unresolved: list[str] = []
    for assumption in assumptions:
        key = assumption.strip()
        if key in assumption_lookup:
            row = assumption_lookup[key]
            locations.append(
                {
                    "assumption": key,
                    "location": f"{row['source']}:{row['line']}",
                }
            )
        elif key.startswith("A") and key[:2].isdigit() is False:
            unresolved.append(key)
        elif key.startswith("A") and "'" in key:
            unresolved.append(key)
        elif key.startswith("A") and key not in assumption_lookup:
            unresolved.append(key)
        elif not key.startswith("A"):
            unresolved.append(key)
    return list(assumptions), locations, unresolved


def _build_theorem_row(spec: TheoremSpec, register_lookup: dict[tuple[str, str], dict[str, str]], assumption_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    register_key = (spec.theorem_id, spec.register_title)
    if register_key not in register_lookup:
        raise KeyError(f"Missing theorem register row {register_key!r}")
    row = register_lookup[register_key]
    assumptions_used, assumption_locations, unresolved_assumptions = _resolve_assumptions(
        spec.assumptions_used,
        assumption_lookup,
    )
    return {
        "theorem_id": spec.theorem_id,
        "title": spec.title,
        "statement_location": f"{row['source']}:{row['line']}",
        "proof_location": f"{PROOF_APPENDIX_FILE.relative_to(REPO_ROOT).as_posix()}:{_find_text_line(PROOF_APPENDIX_FILE, spec.proof_anchor)}",
        "assumptions_used": assumptions_used,
        "assumption_locations": assumption_locations,
        "unresolved_assumptions": unresolved_assumptions,
        "dependencies": list(spec.dependencies),
        "weakest_step": spec.weakest_step,
        "rigor_rating": spec.rigor_rating,
        "code_correspondence": spec.code_correspondence,
        "code_correspondence_detail": spec.code_correspondence_detail,
        "severity_if_broken": spec.severity_if_broken,
        "remediation_class": spec.remediation_class,
        "remediation_detail": spec.remediation_detail,
        "namespace_drift_note": spec.namespace_drift_note,
        "code_anchors": [_resolve_anchor(anchor) for anchor in spec.code_anchors],
        "test_anchors": [_resolve_anchor(anchor) for anchor in spec.test_anchors],
    }


def build_active_theorem_audit_payload() -> dict[str, Any]:
    register_lookup = _register_lookup()
    assumption_lookup = _assumption_lookup()
    theorems = [
        _build_theorem_row(spec, register_lookup, assumption_lookup)
        for spec in ACTIVE_THEOREM_SPECS
    ]
    rigor_counts: dict[str, int] = {}
    correspondence_counts: dict[str, int] = {}
    for theorem in theorems:
        rigor = theorem["rigor_rating"]
        rigor_counts[rigor] = rigor_counts.get(rigor, 0) + 1
        correspondence = theorem["code_correspondence"]
        correspondence_counts[correspondence] = correspondence_counts.get(correspondence, 0) + 1
    return {
        "authoritative_surfaces": {
            "statements_t1_t8": "chapters_merged/ch04_theoretical_foundations.tex",
            "statements_t9_t11": "chapters/ch37_universality_completeness.tex",
            "proof_copies": "appendices/app_c_full_proofs.tex",
            "assumption_register": "appendices/app_b_assumptions.tex",
            "inventory_register": "reports/publication/theorem_surface_register.csv",
        },
        "theorems": theorems,
        "namespace_drift": [
            {
                "surface": entry.surface,
                "issue": entry.issue,
                "impact": entry.impact,
                "status": entry.status,
                "remediation": entry.remediation,
            }
            for entry in NAMESPACE_DRIFT
        ],
        "summary": {
            "active_theorem_count": len(theorems),
            "rigor_counts": rigor_counts,
            "code_correspondence_counts": correspondence_counts,
        },
    }


def render_active_theorem_audit_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def render_active_theorem_audit_csv(payload: dict[str, Any]) -> str:
    lines = [",".join(CSV_COLUMNS)]
    for theorem in payload["theorems"]:
        row = {
            "theorem_id": theorem["theorem_id"],
            "title": theorem["title"],
            "statement_location": theorem["statement_location"],
            "proof_location": theorem["proof_location"],
            "assumptions_used": " | ".join(theorem["assumptions_used"]),
            "unresolved_assumptions": " | ".join(theorem["unresolved_assumptions"]),
            "dependencies": " | ".join(theorem["dependencies"]),
            "weakest_step": theorem["weakest_step"],
            "rigor_rating": theorem["rigor_rating"],
            "code_correspondence": theorem["code_correspondence"],
            "severity_if_broken": theorem["severity_if_broken"],
            "remediation_class": theorem["remediation_class"],
            "code_anchors": " | ".join(anchor["location"] for anchor in theorem["code_anchors"]),
            "test_anchors": " | ".join(anchor["location"] for anchor in theorem["test_anchors"]),
        }
        escaped: list[str] = []
        for column in CSV_COLUMNS:
            value = str(row[column]).replace('"', '""')
            if any(ch in value for ch in {",", "\n", '"', "|"}):
                value = f'"{value}"'
            escaped.append(value)
        lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def _format_anchor_block(title: str, anchors: list[dict[str, str]]) -> list[str]:
    lines = [f"- {title}:"]
    for anchor in anchors:
        label = anchor["location"]
        if anchor["symbol"]:
            label += f" (`{anchor['symbol']}`)"
        if anchor["note"]:
            label += f" - {anchor['note']}"
        lines.append(f"  - {label}")
    return lines


def render_active_theorem_audit_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Active T1-T11 Theorem Audit",
        "",
        "This file is generated from `scripts/_active_theorem_program.py` and is the reconciled active audit surface for the defended T1--T11 program.",
        "",
        "## Summary",
        "",
        f"- Active theorem rows: {payload['summary']['active_theorem_count']}",
        f"- Rigor counts: {payload['summary']['rigor_counts']}",
        f"- Code correspondence counts: {payload['summary']['code_correspondence_counts']}",
        "",
        "## Namespace Drift",
        "",
    ]
    for drift in payload["namespace_drift"]:
        lines.extend(
            [
                f"### {drift['surface']}",
                "",
                f"- Issue: {drift['issue']}",
                f"- Impact: {drift['impact']}",
                f"- Status: {drift['status']}",
                f"- Remediation: {drift['remediation']}",
                "",
            ]
        )
    lines.append("## Per-Theorem Audit")
    lines.append("")
    for theorem in payload["theorems"]:
        lines.extend(
            [
                f"### {theorem['theorem_id']}: {theorem['title']}",
                "",
                f"- Statement location: {theorem['statement_location']}",
                f"- Proof location: {theorem['proof_location']}",
                f"- Assumptions used: {theorem['assumptions_used']}",
                f"- Unresolved assumptions: {theorem['unresolved_assumptions'] or '[]'}",
                f"- Dependencies: {theorem['dependencies']}",
                f"- Weakest step: {theorem['weakest_step']}",
                f"- Rigor rating: {theorem['rigor_rating']}",
                f"- Code correspondence: {theorem['code_correspondence']} - {theorem['code_correspondence_detail']}",
                f"- Severity if broken: {theorem['severity_if_broken']}",
                f"- Remediation class: {theorem['remediation_class']} - {theorem['remediation_detail']}",
            ]
        )
        if theorem["namespace_drift_note"]:
            lines.append(f"- Namespace drift note: {theorem['namespace_drift_note']}")
        lines.extend(_format_anchor_block("Code anchors", theorem["code_anchors"]))
        lines.extend(_format_anchor_block("Test anchors", theorem["test_anchors"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_active_theorem_remediation_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Active T1-T11 Remediation Plan",
        "",
        "This file is generated from `scripts/_active_theorem_program.py` and turns the hostile theorem audit into concrete surface-level work.",
        "",
        "## Immediate claim narrowing",
        "",
        "- Narrow T7 to interior states with explicit slack against model error; the current theorem statement is wider than both the appendix proof and the implemented validator.",
        "- Keep T11 as a four-obligation one-step transfer theorem and treat the five-invariant reference harness as supporting evidence only.",
        "- Keep the legacy coverage helpers out of the active T9/T10 namespace.",
        "",
        "## Proof repairs",
        "",
        "- T3: either register the battery risk-envelope budget as an explicit defended assumption or add the missing battery-specific bridge from runtime reliability to residual-risk probability.",
        "- T4: define the quality-ignorant controller class tightly enough that the constructive witness is not just a fixed-margin caricature.",
        "- T5: separate deterministic tube containment from any future-step probability claim unless the temporal concentration argument is written down.",
        "- T8: prove the absorbing safe-landing argument or lower the theorem to a comparison-helper statement.",
        "- T9: register the missing phi-mixing assumption and derive the witness constant c > 0 instead of importing it by fiat.",
        "- T10: repair the one-sided Le Cam step and defend the boundary-mass model, or keep the result explicitly stylized.",
        "- T11: give the converse a real non-vacuous witness family or treat it as a proof-pattern warning instead of a biconditional-style theorem surface.",
        "",
        "## Register and appendix hygiene",
        "",
        "- Treat `reports/publication/theorem_surface_register.csv` as raw inventory only. The reconciled active theorem map lives in `active_theorem_audit.*`.",
        "- Keep appendix proof copies and the active theorem audit synchronized on theorem names, proof anchors, and assumption sets.",
        "- Do not let the general-CPS duplicate rows for T1-T4 silently replace the narrower defended main-theorem rows in downstream tooling.",
        "",
        "## Code and tests",
        "",
        "- T3 helpers must keep surfacing that `w_t` is a bounded reliability score and that the episode envelope needs an explicit per-step risk budget.",
        "- T11 tests must keep distinguishing the active four-obligation theorem helper from the supporting five-invariant mini-harness.",
        "- The theorem validator should fail if the generated active audit artifacts drift from the manuscript/code/test anchors or if legacy theorem numbering leaks back into theorem-facing helpers.",
        "",
        "## Theorem-by-theorem actions",
        "",
    ]
    for theorem in payload["theorems"]:
        lines.append(
            f"- {theorem['theorem_id']} ({theorem['rigor_rating']}, {theorem['severity_if_broken']}): "
            f"{theorem['remediation_class']} - {theorem['remediation_detail']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_active_theorem_audit_outputs() -> None:
    payload = build_active_theorem_audit_payload()
    AUDIT_JSON.write_text(render_active_theorem_audit_json(payload), encoding="utf-8")
    AUDIT_CSV.write_text(render_active_theorem_audit_csv(payload), encoding="utf-8")
    AUDIT_MD.write_text(render_active_theorem_audit_md(payload), encoding="utf-8")
    REMEDIATION_MD.write_text(render_active_theorem_remediation_md(payload), encoding="utf-8")
