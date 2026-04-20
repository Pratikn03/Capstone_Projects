#!/usr/bin/env python3
"""Build lightweight IEEE support assets for the active 3-domain ORIUS lane."""
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
IEEE_GENERATED_DIR = REPO_ROOT / "paper" / "ieee" / "generated"
EDITORIAL_DIR = REPO_ROOT / "reports" / "editorial"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def build() -> int:
    """Programmatic entry point for pipeline scripts and tests.

    Delegates directly to ``main()``.  Provided so callers can invoke
    ``ieee_assets_script.build()`` without reaching into ``main()`` directly.
    """
    return main()


def main() -> int:
    closure_rows = _read_rows(PUBLICATION_DIR / "orius_domain_closure_matrix.csv")
    scorecard_rows = _read_rows(PUBLICATION_DIR / "orius_submission_scorecard.csv")
    runtime_rows = _read_rows(PUBLICATION_DIR / "orius_runtime_budget_matrix.csv")

    summary_lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Domain & Tier & Source\\",
        r"\midrule",
    ]
    for row in closure_rows:
        summary_lines.append(f"{row['domain']} & {row['tier']} & {row['source']}\\\\")
    summary_lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_text(IEEE_GENERATED_DIR / "orius_three_domain_summary.tex", "\n".join(summary_lines))

    runtime_lookup = {row["domain"]: row for row in runtime_rows}
    professor_runtime_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Runtime and governance summary for the professor-facing ORIUS draft.}",
        r"\label{tab:prof-runtime-governance}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.24\textwidth}p{0.16\textwidth}p{0.14\textwidth}p{0.14\textwidth}p{0.12\textwidth}p{0.14\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Row} & \textbf{P95 latency (ms)} & \textbf{Fallback (\%)} & \textbf{Audit / lifecycle (\%)} & \textbf{Boundary} \\",
        r"\midrule",
    ]
    professor_runtime_lines.append(
        "Battery Energy Storage & Witness row & "
        f"{runtime_lookup['Battery Energy Storage']['p95_step_ms']} & "
        f"{runtime_lookup['Battery Energy Storage']['fallback_coverage_pct']} & 100.0 & "
        "deepest theorem-to-artifact closure\\\\"
    )
    professor_runtime_lines.append(
        "Autonomous Vehicles & Defended bounded row & "
        f"{runtime_lookup['Autonomous Vehicles']['p95_step_ms']} & "
        f"{runtime_lookup['Autonomous Vehicles']['fallback_coverage_pct']} & 100.0 & "
        "bounded TTC plus predictive-entry-barrier contract\\\\"
    )
    professor_runtime_lines.append(
        "Medical and Healthcare Monitoring & Defended bounded row & "
        f"{runtime_lookup['Medical and Healthcare Monitoring']['p95_step_ms']} & "
        f"{runtime_lookup['Medical and Healthcare Monitoring']['fallback_coverage_pct']} & 100.0 & "
        "bounded monitoring and alert-release semantics\\\\"
    )
    professor_runtime_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    _write_text(IEEE_GENERATED_DIR / "orius_professor_runtime_governance_table.tex", "\n".join(professor_runtime_lines))

    professor_parity_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-domain status for the professor-facing ORIUS draft.}",
        r"\label{tab:prof-cross-domain-status}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.24\textwidth}p{0.18\textwidth}p{0.24\textwidth}p{0.24\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Current row} & \textbf{What is currently closed} & \textbf{Current boundary} \\",
        r"\midrule",
        r"Battery Energy Storage & Witness row & adapter, replay, theorem witness, governance & none\\",
        r"Autonomous Vehicles & Defended bounded row & adapter, replay, governance, bounded runtime contract & TTC plus predictive-entry-barrier only\\",
        r"Medical and Healthcare Monitoring & Defended bounded row & adapter, replay, governance, bounded monitoring contract & certificate-valid release currently uses governance-pass proxy\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ]
    _write_text(IEEE_GENERATED_DIR / "orius_professor_parity_table.tex", "\n".join(professor_parity_lines))

    deployment_scope_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Deployment validation scope for the active three-domain ORIUS draft.}",
        r"\label{tab:orius-deployment-validation-scope}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{2.5cm}p{2.5cm}p{2.2cm}p{2.2cm}p{2.8cm}p{3.0cm}}",
        r"\toprule",
        r"\textbf{Deployment surface} & \textbf{Governing artifact} & \textbf{Scope type} & \textbf{Current status} & \textbf{Bounded manuscript claim} & \textbf{Exact non-claim or gap}\\",
        r"\midrule",
        r"Battery witness runtime & reports/publication/dc3s\_main\_table.csv & witness\_replay & bounded\_reference & Battery supports the deepest runtime and theorem witness surface. & This is still not unrestricted field deployment.\\",
        r"Autonomous-vehicle defended replay & reports/orius\_av/full\_corpus/runtime\_summary.csv & bounded\_replay & defended\_bounded & AV supports bounded replay under the TTC plus predictive-entry-barrier contract. & It does not claim full autonomous-driving closure.\\",
        r"Healthcare defended replay & reports/healthcare/runtime\_governance\_summary.csv & bounded\_replay & defended\_bounded & Healthcare supports bounded monitoring and alert-release claims. & It does not claim regulated clinical deployment and still uses governance-pass proxy wording for certificate-valid release.\\",
        r"OOD and adversarial completeness & chapters/ch34\_outside\_current\_evidence.tex; reports/publication/adversarial\_probing\_robustness\_table.csv & explicit\_non\_claim\_register & bounded\_non\_claim & The monograph can discuss bounded active probing and non-claim discipline. & It does not claim universal adversarial completeness or unrestricted OOD safety.\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    _write_text(IEEE_GENERATED_DIR / "orius_deployment_validation_scope.tex", "\n".join(deployment_scope_lines))

    claim_ledger = [
        "claim_id,support_status,scope,note",
        "IEEE001,current_repo_supported,three-domain,IEEE surfaces now describe only Battery + AV + Healthcare.",
        "IEEE002,current_repo_supported,three-domain,Removed domains are excluded from active submission claims.",
    ]
    _write_text(EDITORIAL_DIR / "orius_claim_delta_ledger.csv", "\n".join(claim_ledger))
    _write_text(
        EDITORIAL_DIR / "orius_claim_delta_ledger.md",
        "\n".join(
            [
                "# ORIUS IEEE Claim Delta Ledger",
                "",
                "- Active program scope: Battery + AV + Healthcare only.",
                "- Removed domains are not part of the current IEEE claim surface.",
                f"- Scorecard rows present: {', '.join(row['target_tier'] for row in scorecard_rows) or 'none'}",
            ]
        ),
    )
    _write_text(
        EDITORIAL_DIR / "orius_flagship_revision_ledger.csv",
        "\n".join(
            [
                "revision_id,summary",
                "R1,Hard-cut IEEE support assets to the three-domain program.",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
