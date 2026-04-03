#!/usr/bin/env python3
"""Build support assets for the ORIUS IEEE flagship manuscript family."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_BIB_PATH = REPO_ROOT / "reports" / "publication" / "orius_annotated_bibliography.csv"
BIB_PATH = REPO_ROOT / "paper" / "bibliography" / "orius_monograph.bib"
IEEE_GENERATED_DIR = REPO_ROOT / "paper" / "ieee" / "generated"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
EDITORIAL_DIR = REPO_ROOT / "reports" / "editorial"

BENCHMARK_CSV_PATH = PUBLICATION_DIR / "orius_top_tier_benchmark_corpus.csv"
BENCHMARK_TEX_PATH = IEEE_GENERATED_DIR / "orius_benchmark_corpus_appendix.tex"
BENCHMARK_SUMMARY_TEX_PATH = IEEE_GENERATED_DIR / "orius_benchmark_summary_appendix.tex"
DEPLOYMENT_SCOPE_SOURCE_PATH = PUBLICATION_DIR / "tbl_orius_deployment_validation_scope.tex"
DEPLOYMENT_SCOPE_TEX_PATH = IEEE_GENERATED_DIR / "orius_deployment_validation_scope.tex"
PROFESSOR_PARITY_CSV_PATH = PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv"
PROFESSOR_RUNTIME_CSV_PATH = PUBLICATION_DIR / "orius_runtime_budget_matrix.csv"
PROFESSOR_GOVERNANCE_CSV_PATH = PUBLICATION_DIR / "orius_governance_lifecycle_matrix.csv"
PROFESSOR_BATTERY_CSV_PATH = PUBLICATION_DIR / "table1_main.csv"
PROFESSOR_PARITY_TEX_PATH = IEEE_GENERATED_DIR / "orius_professor_parity_table.tex"
PROFESSOR_RUNTIME_GOV_TEX_PATH = IEEE_GENERATED_DIR / "orius_professor_runtime_governance_table.tex"
PROFESSOR_BATTERY_WITNESS_TEX_PATH = IEEE_GENERATED_DIR / "orius_professor_battery_witness_table.tex"
CLAIM_LEDGER_CSV_PATH = EDITORIAL_DIR / "orius_claim_delta_ledger.csv"
CLAIM_LEDGER_MD_PATH = EDITORIAL_DIR / "orius_claim_delta_ledger.md"
REVISION_LEDGER_CSV_PATH = EDITORIAL_DIR / "orius_flagship_revision_ledger.csv"

TARGET_BENCHMARK_ROWS = 100
ALLOWED_STATUSES = {
    "current_repo_supported",
    "supported_by_primary_literature_only",
    "visionary_requires_closure_work",
}

FAMILY_QUOTAS = {
    "Safety filters and constrained control": 18,
    "General physical AI and systems": 16,
    "Runtime assurance and monitoring": 14,
    "Conformal and calibration": 12,
    "Degradation, anomaly, and CPS security": 10,
    "Energy and storage": 9,
    "Robotics and autonomy": 8,
    "Industrial automation": 5,
    "Healthcare monitoring": 4,
    "Navigation and aerospace": 4,
}

CLAIM_LEDGER_ROWS = [
    (
        "ieee_abstract",
        "ORIUS is presented as a fundamental runtime safety layer for Physical AI under degraded observation.",
        "current_repo_supported",
        "paper/paper.tex; reports/publication/orius_equal_domain_parity_matrix.csv",
        "Keep as the anchor claim in both manuscripts.",
    ),
    (
        "ieee_introduction",
        "ORIUS supplies one reusable runtime grammar for degraded observation, repair, fallback, and certificates across six domains.",
        "current_repo_supported",
        "src/orius/dc3s/domain_adapter.py; reports/publication/orius_domain_closure_matrix.csv",
        "Keep the claim architectural, not equal-domain empirical.",
    ),
    (
        "problem_definition",
        "OASG is the common hidden hazard across batteries, AV, industrial plants, healthcare monitoring, navigation, and aerospace.",
        "current_repo_supported",
        "paper/monograph/ch02_oasg_claim_boundary.tex; paper/monograph/ch08_battery_bridge.tex; paper/monograph/ch09_av_domain.tex",
        "Keep the hazard universal and plant-agnostic.",
    ),
    (
        "related_work",
        "Prior conformal, runtime assurance, safe-control, and CPS-monitoring families are each insufficient because none alone closes degraded observation as an action-semantics problem.",
        "supported_by_primary_literature_only",
        "paper/bibliography/orius_monograph.bib",
        "Trim only if a future venue demands a more conservative related-work posture.",
    ),
    (
        "runtime_architecture",
        "The Detect--Calibrate--Constrain--Shield--Certify kernel is the dominant technical move of ORIUS.",
        "current_repo_supported",
        "paper/monograph/ch04_universal_runtime_layer.tex; paper/monograph/ch05_detect_calibrate_constrain_shield_certify.tex",
        "Retain as the main technical architecture claim.",
    ),
    (
        "benchmark_governance",
        "One universal benchmark schema and one governance surface are sufficient to compare defended rows across domains.",
        "current_repo_supported",
        "docs/UNIVERSAL_BENCHMARK_SPEC.md; docs/UNIVERSAL_GOVERNANCE_SPEC.md; src/orius/certos/runtime.py",
        "Keep the claim bounded to defended rows and audited support surfaces.",
    ),
    (
        "theorem_bridge",
        "The theory bridge establishes that degraded observation can create true-state violations despite observed-state legality, and that repair is structurally necessary.",
        "current_repo_supported",
        "paper/monograph/ch06_theory_bridge.tex; appendices/app_c_full_proofs.tex",
        "Keep theorem statements scoped to the documented assumptions.",
    ),
    (
        "battery_section",
        "Battery provides witness-depth theorem-to-code-to-artifact closure and calibrates the rest of the framework.",
        "current_repo_supported",
        "reports/publication/orius_equal_domain_parity_matrix.csv; chapters/ch11_main_battery_results.tex",
        "Keep battery as witness rather than narrative center.",
    ),
    (
        "av_section",
        "Autonomous vehicles are a defended bounded row under the TTC plus predictive-entry-barrier contract.",
        "current_repo_supported",
        "paper/monograph/ch09_av_domain.tex; reports/publication/orius_equal_domain_parity_matrix.csv",
        "Do not expand this to full-stack or multi-lane autonomy claims.",
    ),
    (
        "industrial_section",
        "Industrial control is a defended bounded row with explicit envelope-preserving repair and fallback semantics.",
        "current_repo_supported",
        "paper/monograph/ch10_industrial_domain.tex; reports/publication/orius_equal_domain_parity_matrix.csv",
        "Keep deployment language bounded to the defended replay surface.",
    ),
    (
        "healthcare_section",
        "Healthcare monitoring is a defended bounded row that treats degraded observation as intervention-suppression risk.",
        "current_repo_supported",
        "paper/monograph/ch11_healthcare_domain.tex; reports/publication/orius_equal_domain_parity_matrix.csv",
        "Keep the row bounded to monitoring/intervention semantics rather than bedside certification.",
    ),
    (
        "navigation_section",
        "Navigation is written as a future defended row in the visionary flagship draft.",
        "visionary_requires_closure_work",
        "reports/publication/orius_equal_domain_parity_matrix.csv; reports/real_data_contract_status.json",
        "Trim to explicit gated language if submission must become artifact-strict.",
    ),
    (
        "aerospace_section",
        "Aerospace is written as a future defended row in the visionary flagship draft.",
        "visionary_requires_closure_work",
        "reports/publication/orius_equal_domain_parity_matrix.csv; reports/real_data_contract_status.json",
        "Trim to explicit experimental-row language if submission must become artifact-strict.",
    ),
    (
        "cross_domain_synthesis",
        "ORIUS is the first universal runtime safety layer for Physical AI with one shared evaluation and governance contract.",
        "supported_by_primary_literature_only",
        "paper/bibliography/orius_monograph.bib; reports/publication/orius_equal_domain_parity_matrix.csv",
        "Keep equal-domain parity as a target, not a present-tense fact, unless blocked rows close.",
    ),
    (
        "limitations_boundary",
        "Equal-domain universality remains a target state until navigation and aerospace clear the governed parity gate.",
        "current_repo_supported",
        "reports/publication/orius_equal_domain_parity_matrix.csv; reports/real_data_contract_status.json",
        "Keep visible in both manuscripts.",
    ),
    (
        "conclusion",
        "ORIUS changes the field by establishing runtime safety layers, rather than nominal controllers, as the right universal object for Physical AI safety.",
        "supported_by_primary_literature_only",
        "paper/bibliography/orius_monograph.bib; paper/monograph/ch16_conclusion_monograph.tex",
        "Retain the field-shaping tone while keeping the evidence gate explicit.",
    ),
]


def _tex_escape(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_bib_notes(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    entries = {}
    for match in re.finditer(r"@misc\{([^,]+),(.*?)\n\}", text, re.S):
        key = match.group(1).strip()
        body = match.group(2)
        note_match = re.search(r"note\s*=\s*\{(.*?)\}\s*$", body, re.S | re.M)
        if note_match:
            note = re.sub(r"\s+", " ", note_match.group(1)).strip()
            entries[key] = note
    return entries


def _family_priority(row: dict[str, str]) -> tuple[int, int, str]:
    source_score = 0 if row["source"] == "curated" else 1
    try:
        year_score = -int(row["year"])
    except ValueError:
        year_score = 0
    return source_score, year_score, row["title"]


def _select_benchmark_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family in by_family:
        by_family[family].sort(key=_family_priority)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for family, quota in FAMILY_QUOTAS.items():
        for row in by_family.get(family, [])[:quota]:
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            selected.append(row)

    remaining = sorted((row for row in rows if row["key"] not in seen), key=_family_priority)
    for row in remaining:
        if len(selected) >= TARGET_BENCHMARK_ROWS:
            break
        seen.add(row["key"])
        selected.append(row)
    return selected[:TARGET_BENCHMARK_ROWS]


def _benchmark_fields(row: dict[str, str], note: str) -> dict[str, str]:
    family = row["family"]
    family_defaults = {
        "Conformal and calibration": (
            "uncertainty calibration under shift",
            "distribution-free or adaptive calibration layer",
            "theory plus benchmarked empirical coverage",
            "high",
            "medium",
            "weak explicit runtime governance",
            "theorem-forward with compact experiments",
            "must connect coverage to repaired actuation and certificates",
        ),
        "Runtime assurance and monitoring": (
            "runtime safety supervision",
            "supervisory monitor or fallback architecture",
            "architecture plus prototype or systems evaluation",
            "medium",
            "medium",
            "strong runtime posture",
            "systems-first with mechanism diagrams",
            "must bind supervision to degraded observation and benchmark semantics",
        ),
        "Safety filters and constrained control": (
            "safe action-set construction",
            "constraint-tightening or repair geometry",
            "theory plus closed-loop evaluation",
            "high",
            "medium",
            "implicit governance via constraint enforcement",
            "control-theoretic with compact proofs",
            "must expose one reusable cross-domain contract instead of plant-local mathematics only",
        ),
        "Degradation, anomaly, and CPS security": (
            "trustworthiness of telemetry",
            "detection or diagnosis of degraded observation",
            "detection benchmark or CPS analysis",
            "low",
            "medium",
            "alarm-centric governance",
            "taxonomy or threat-model driven",
            "must connect alarms to action semantics and repaired releases",
        ),
        "Energy and storage": (
            "energy dispatch under uncertainty",
            "plant-specific uncertainty-aware control",
            "benchmark or operational replay",
            "medium",
            "low",
            "limited explicit runtime governance",
            "application-heavy with cost and safety tables",
            "must generalize beyond the witness plant",
        ),
        "Robotics and autonomy": (
            "autonomy under uncertainty",
            "safety-constrained planning or control",
            "simulation plus selected real-world validation",
            "medium",
            "medium",
            "runtime posture varies by system",
            "figure-driven and benchmark-oriented",
            "must unify autonomy rows with non-robotics CPS rows under one grammar",
        ),
        "Industrial automation": (
            "process safety and continuity",
            "plant monitoring or bounded control",
            "process benchmark or deployment-inspired study",
            "medium",
            "low",
            "procedural safety posture",
            "operational and systems-focused",
            "must expose typed repair, fallback, and certificate semantics",
        ),
        "Healthcare monitoring": (
            "monitoring-driven intervention safety",
            "risk scoring or alarm semantics",
            "clinical dataset evaluation",
            "low",
            "low",
            "compliance-sensitive governance",
            "clinical framing with conservative claims",
            "must connect degraded observation to runtime intervention legality",
        ),
        "Navigation and aerospace": (
            "guidance and flight safety under degraded state estimation",
            "bounded safe envelope or prognostic surface",
            "dataset benchmark or flight-inspired evaluation",
            "medium",
            "medium",
            "mission assurance posture",
            "safety-case oriented with explicit limits",
            "must clear defended real-data replay before equal-domain rhetoric",
        ),
        "General physical AI and systems": (
            "field-level framing for trustworthy AI systems",
            "systems or scientific framing contribution",
            "survey, benchmark, or foundational synthesis",
            "medium",
            "high",
            "governance and methodology heavy",
            "broad positioning and synthesis",
            "must be tied to a concrete runtime contract rather than vision alone",
        ),
    }
    problem, novelty, evidence, theorem_depth, breadth, governance, writing_pattern, gap = family_defaults.get(
        family,
        family_defaults["General physical AI and systems"],
    )
    return {
        "venue_or_note": note or row["source"],
        "lab_tier": "top-tier broad" if row["source"] == "curated" else "legacy carry-forward",
        "problem_framing": problem,
        "novelty_move": novelty,
        "evidence_surface": evidence,
        "theorem_depth": theorem_depth,
        "cross_domain_breadth": breadth,
        "runtime_governance_posture": governance,
        "writing_style_pattern": writing_pattern,
        "orius_gap": gap,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_appendix_tex(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\section{Top-Tier Benchmark Corpus}",
        r"\label{sec:ieee-benchmark-corpus}",
        r"This appendix benchmarks ORIUS against a broad top-tier research corpus spanning conformal inference, runtime assurance, safe control, anomaly and drift detection, cyber-physical systems, and domain-specific safety literature. The corpus is used as a writing and positioning benchmark rather than as a numerical leaderboard.",
        "",
        r"\begin{longtable}{p{0.08\textwidth}p{0.08\textwidth}p{0.17\textwidth}p{0.30\textwidth}p{0.12\textwidth}p{0.17\textwidth}}",
        r"\caption{Broad benchmark corpus used to pressure-test the flagship ORIUS draft.}\\",
        r"\toprule",
        r"\textbf{Year} & \textbf{Tier} & \textbf{Family} & \textbf{Paper} & \textbf{Writing lens} & \textbf{ORIUS gap}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Year} & \textbf{Tier} & \textbf{Family} & \textbf{Paper} & \textbf{Writing lens} & \textbf{ORIUS gap}\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['year'])} & "
            f"{_tex_escape(row['lab_tier'])} & "
            f"{_tex_escape(row['family'])} & "
            f"{_tex_escape(row['title'])} & "
            f"{_tex_escape(row['writing_style_pattern'])} & "
            f"{_tex_escape(row['orius_gap'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    BENCHMARK_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_benchmark_summary_tex(rows: list[dict[str, str]]) -> None:
    family_counts = Counter(row["family"] for row in rows)
    lines = [
        r"\section{Benchmark Corpus Summary}",
        r"\label{sec:ieee-benchmark-summary}",
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{p{0.34\textwidth}r}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Rows}\\",
        r"\midrule",
    ]
    for family, count in family_counts.most_common():
        lines.append(f"{_tex_escape(family)} & {count}\\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Family distribution for the 100-paper benchmark corpus used to pressure-test the ORIUS flagship draft.}",
            r"\label{tab:ieee-benchmark-family-summary}",
            r"\end{table}",
            "",
            r"\paragraph{Writing-gap synthesis.}",
            r"The benchmark corpus pushes the ORIUS flagship draft toward five consistent writing upgrades: lead with the field-level problem instead of artifact logistics; state one dominant technical move early; compress evidence to a few decisive surfaces; keep limitations adjacent to the main claim; and make the conclusion argue for a field object rather than merely restating chapter coverage.",
        ]
    )
    BENCHMARK_SUMMARY_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_SUMMARY_TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_claim_ledgers() -> None:
    rows = [
        {
            "manuscript_surface": surface,
            "claim": claim,
            "support_status": status,
            "current_support_anchor": anchor,
            "later_trim_or_rewrite_action": action,
        }
        for surface, claim, status, anchor, action in CLAIM_LEDGER_ROWS
    ]
    fieldnames = [
        "manuscript_surface",
        "claim",
        "support_status",
        "current_support_anchor",
        "later_trim_or_rewrite_action",
    ]
    _write_csv(CLAIM_LEDGER_CSV_PATH, fieldnames, rows)
    _write_csv(REVISION_LEDGER_CSV_PATH, fieldnames, rows)

    status_counts = Counter(row["support_status"] for row in rows)
    lines = [
        "# ORIUS Claim Delta Ledger",
        "",
        "This ledger is intentionally non-public manuscript support. It tracks where the flagship IEEE draft and the tightened monograph stay within current repo truth, where they rely on primary-literature positioning, and where they intentionally run ahead of current cross-domain closure so later venue-trimming remains explicit.",
        "",
        "## Status Counts",
        "",
    ]
    for status in sorted(ALLOWED_STATUSES):
        lines.append(f"- `{status}`: {status_counts.get(status, 0)}")
    lines.extend(["", "## Section Rows", ""])
    for row in rows:
        lines.append(f"- `{row['manuscript_surface']}`: `{row['support_status']}`")
        lines.append(f"  Claim: {row['claim']}")
        lines.append(f"  Anchor: `{row['current_support_anchor']}`")
        lines.append(f"  Later action: {row['later_trim_or_rewrite_action']}")
    CLAIM_LEDGER_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAIM_LEDGER_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_pct(value: str, scale: float = 1.0) -> str:
    try:
        return f"{float(value) * scale:.1f}"
    except (TypeError, ValueError):
        return _tex_escape(value)


def _title_case_status(value: str) -> str:
    status = value.replace("_", " ").strip()
    mapping = {
        "reference": "Witness row",
        "proof validated": "Defended bounded row",
        "shadow synthetic": "Shadow-synthetic",
        "experimental": "Experimental",
        "reference runtime": "Witness row",
        "witness reference": "Witness row",
        "n/a reference": "Witness row",
        "evaluated": "Evaluated",
        "gated": "Gated",
    }
    return mapping.get(status, status.title())


def _controller_label(value: str) -> str:
    labels = {
        "aci_conformal": "Adaptive conformal + repair",
        "dc3s_ftit": "ORIUS FTIT",
        "dc3s_wrapped": "ORIUS wrapped",
        "deterministic_lp": "Deterministic LP",
        "robust_fixed_interval": "Robust fixed interval",
        "scenario_mpc": "Scenario MPC",
        "scenario_robust": "Scenario robust",
    }
    return labels.get(value, value.replace("_", " "))


def _write_professor_parity_tex() -> None:
    rows = _read_rows(PROFESSOR_PARITY_CSV_PATH)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-domain parity and promotion status for the professor-facing ORIUS draft.}",
        r"\label{tab:prof-cross-domain-status}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.18\textwidth}p{0.18\textwidth}p{0.27\textwidth}p{0.27\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Current row} & \textbf{What is currently closed} & \textbf{Current blocker or boundary} \\",
        r"\midrule",
    ]
    for row in rows:
        closed = []
        if row["adapter_correctness"] == "pass":
            closed.append("adapter")
        if row["replay_status"] == "pass":
            closed.append("replay")
        if row["certos_lifecycle_support"] == "evaluated":
            closed.append("CertOS")
        if row["safe_action_soundness"] in {"pass", "reference_witness"}:
            closed.append("soundness")
        closed_text = ", ".join(closed) if closed else "partial runtime portability"
        lines.append(
            f"{_tex_escape(row['domain'])} & "
            f"{_tex_escape(_title_case_status(row['resulting_tier']))} & "
            f"{_tex_escape(closed_text)} & "
            f"{_tex_escape(row['exact_blocker'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    PROFESSOR_PARITY_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_professor_runtime_governance_tex() -> None:
    runtime_rows = {row["domain"]: row for row in _read_rows(PROFESSOR_RUNTIME_CSV_PATH)}
    governance_rows = {row["domain"]: row for row in _read_rows(PROFESSOR_GOVERNANCE_CSV_PATH)}
    parity_rows = {row["domain"]: row for row in _read_rows(PROFESSOR_PARITY_CSV_PATH)}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Runtime and governance summary for the professor-facing ORIUS draft.}",
        r"\label{tab:prof-runtime-governance}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.18\textwidth}p{0.14\textwidth}p{0.12\textwidth}p{0.11\textwidth}p{0.13\textwidth}p{0.10\textwidth}p{0.15\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Row} & \textbf{P95 latency (ms)} & \textbf{Repair (\%)} & \textbf{Fallback} & \textbf{Audit (\%)} & \textbf{Current limit} \\",
        r"\midrule",
    ]
    for domain in runtime_rows:
        runtime = runtime_rows[domain]
        governance = governance_rows.get(domain, {})
        parity = parity_rows.get(domain, {})
        lines.append(
            f"{_tex_escape(domain)} & "
            f"{_tex_escape(_title_case_status(parity.get('resulting_tier', runtime.get('runtime_budget_depth', ''))))} & "
            f"{_format_pct(runtime['p95_step_latency_ms'])} & "
            f"{_format_pct(runtime['repair_rate_pct'])} & "
            f"{_tex_escape(runtime['fallback_mode'])} & "
            f"{_format_pct(governance.get('governance_completeness_pct', ''))} & "
            f"{_tex_escape(runtime['exact_limit'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    PROFESSOR_RUNTIME_GOV_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_professor_battery_witness_tex() -> None:
    rows = _read_rows(PROFESSOR_BATTERY_CSV_PATH)
    wanted = ["dc3s_ftit", "dc3s_wrapped", "aci_conformal", "deterministic_lp", "robust_fixed_interval"]
    selected = [row for row in rows if row["controller"] in wanted]
    order = {name: idx for idx, name in enumerate(wanted)}
    selected.sort(key=lambda row: order[row["controller"]])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Battery witness summary from the tracked main battery result surface.}",
        r"\label{tab:prof-battery-summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Controller} & \textbf{TSVR} & \textbf{IR (\%)} & \textbf{Cost $\Delta$ (\%)} \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{_tex_escape(_controller_label(row['controller']))} & "
            f"{_format_pct(row['true_soc_violation_rate_mean'])} & "
            f"{_format_pct(row['intervention_rate_mean'], scale=100)} & "
            f"{_format_pct(row['cost_delta_pct_mean'], scale=100)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    PROFESSOR_BATTERY_WITNESS_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def _copy_ieee_support_tables() -> None:
    DEPLOYMENT_SCOPE_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_SCOPE_TEX_PATH.write_text(
        DEPLOYMENT_SCOPE_SOURCE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_professor_parity_tex()
    _write_professor_runtime_governance_tex()
    _write_professor_battery_witness_tex()


def build() -> None:
    rows = _read_rows(ANNOTATED_BIB_PATH)
    note_lookup = _parse_bib_notes(BIB_PATH)
    selected = _select_benchmark_rows(rows)

    benchmark_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        enriched = {
            "rank": str(index),
            "key": row["key"],
            "year": row["year"],
            "family": row["family"],
            "source": row["source"],
            "title": row["title"],
        }
        enriched.update(_benchmark_fields(row, note_lookup.get(row["key"], "")))
        benchmark_rows.append(enriched)

    fieldnames = [
        "rank",
        "key",
        "year",
        "family",
        "source",
        "title",
        "venue_or_note",
        "lab_tier",
        "problem_framing",
        "novelty_move",
        "evidence_surface",
        "theorem_depth",
        "cross_domain_breadth",
        "runtime_governance_posture",
        "writing_style_pattern",
        "orius_gap",
    ]
    _write_csv(BENCHMARK_CSV_PATH, fieldnames, benchmark_rows)
    _write_benchmark_appendix_tex(benchmark_rows)
    _write_benchmark_summary_tex(benchmark_rows)
    _write_claim_ledgers()
    _copy_ieee_support_tables()


if __name__ == "__main__":
    build()
