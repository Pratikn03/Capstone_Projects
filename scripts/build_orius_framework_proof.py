#!/usr/bin/env python3
"""Build a witness-first ORIUS framework proof bundle."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    env = dict(os.environ)
    repo_src = str(cwd / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_src}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else repo_src
    )
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _optional_display_path(path: Path, repo_root: Path) -> str:
    return _display_path(path, repo_root) if path.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a witness-first ORIUS framework proof bundle."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds for the universal validation run.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Episode horizon for the validation run.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/orius_framework_proof",
        help="Output directory for the proof bundle.",
    )
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help=(
            "Do not rerun theorem/training/SIL/validation subpipelines; instead, "
            "summarize the already-generated artifacts in the proof bundle directory."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out
    training_dir = out_dir / "training_audit"
    sil_dir = out_dir / "sil_validation"
    validation_dir = out_dir / "universal_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.reuse_existing_artifacts:
        _run([sys.executable, "scripts/verify_integrated_theorem_surface.py"], cwd=repo_root)
        _run(
            [
                sys.executable,
                "scripts/run_universal_training_audit.py",
                "--out",
                str(training_dir),
                "--train-missing",
                "--repair-invalid-splits",
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                "scripts/run_universal_sil_validation.py",
                "--seeds",
                str(args.seeds),
                "--rows",
                str(args.horizon),
                "--out",
                str(sil_dir),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                "scripts/run_universal_orius_validation.py",
                "--seeds",
                str(args.seeds),
                "--horizon",
                str(args.horizon),
                "--out",
                str(validation_dir),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                "scripts/build_domain_closure_matrix.py",
                "--validation-report",
                str(validation_dir / "validation_report.json"),
                "--training-report",
                str(training_dir / "training_audit_report.json"),
                "--out",
                str(validation_dir),
            ],
            cwd=repo_root,
        )

    validation_report_path = validation_dir / "validation_report.json"
    proof_report_path = validation_dir / "proof_domain_report.json"
    training_report_path = training_dir / "training_audit_report.json"
    sil_report_path = sil_dir / "sil_validation_report.json"
    closure_matrix_path = validation_dir / "domain_closure_matrix.csv"
    composition_matrix_path = validation_dir / "paper5_cross_domain_matrix.csv"
    governance_matrix_path = validation_dir / "paper6_cross_domain_matrix.csv"
    theorem_gate_path = repo_root / "reports" / "publication" / "integrated_theorem_gate.json"

    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    proof_report = json.loads(proof_report_path.read_text(encoding="utf-8"))
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    sil_report = json.loads(sil_report_path.read_text(encoding="utf-8"))
    theorem_gate = json.loads(theorem_gate_path.read_text(encoding="utf-8"))

    theorem_gate_pass = int(theorem_gate.get("failed", 0)) == 0
    theorem_gate_summary = (
        f"{theorem_gate.get('passed', 0)}/{theorem_gate.get('total', 0)} theorem rows verified"
    )

    domain_maturity = validation_report.get("domain_maturity", {})
    proof_validated_domains = list(proof_report.get("proof_validated_domains", []))
    if not proof_validated_domains:
        proof_validated_domains = [
            domain
            for domain, maturity in domain_maturity.items()
            if maturity == "proof_validated"
        ]

    proof_candidate_domains = [
        item["domain"]
        for item in proof_report.get("proof_downgraded_domains", [])
        if item.get("domain")
    ]
    if not proof_candidate_domains:
        proof_candidate_domains = [
            domain
            for domain, maturity in domain_maturity.items()
            if maturity == "proof_candidate"
        ]

    shadow_synthetic_domains = [
        domain
        for domain, maturity in domain_maturity.items()
        if maturity in {"shadow", "shadow_synthetic", "portability"}
    ]
    experimental_domains = list(validation_report.get("experimental_domains", []))

    domain_controller_summary: list[dict[str, object]] = []
    for row in validation_report["domain_results"]:
        domain = str(row["domain"])
        maturity = domain_maturity.get(domain, row.get("maturity_label", "unknown"))
        domain_controller_summary.append(
            {
                "domain": domain,
                "data_source": (
                    "locked_artifact"
                    if domain == validation_report.get("reference_domain")
                    else "locked_csv"
                ),
                "evidence_tier": maturity,
                "validation_status": row["validation_status"],
                "baseline_tsvr_mean": f"{float(row['baseline_tsvr_mean']):.4f}",
                "orius_tsvr_mean": f"{float(row['orius_tsvr_mean']):.4f}",
                "orius_reduction_pct": f"{float(row['orius_reduction_pct']):.1f}",
                "harness_status": row["harness_status"],
            }
        )

    summary_csv = out_dir / "domain_controller_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_controller_summary[0].keys()))
        writer.writeheader()
        writer.writerows(domain_controller_summary)

    artifact_rows: list[dict[str, str]] = []
    for row in validation_report["domain_results"]:
        domain = str(row["domain"])
        maturity = domain_maturity.get(domain, row.get("maturity_label", "unknown"))
        artifact_rows.append(
            {
                "domain": domain,
                "data_source": (
                    "locked_artifact"
                    if domain == validation_report.get("reference_domain")
                    else "locked_csv"
                ),
                "evidence_tier": maturity,
                "validation_status": str(row["validation_status"]),
                "harness_status": str(row["harness_status"]),
                "theorem_gate_artifact": _display_path(theorem_gate_path, repo_root),
                "training_audit_artifact": _display_path(training_report_path, repo_root),
                "sil_audit_artifact": _display_path(sil_report_path, repo_root),
                "validation_artifact": _display_path(validation_report_path, repo_root),
                "proof_artifact": _display_path(proof_report_path, repo_root),
                "closure_matrix_artifact": _display_path(closure_matrix_path, repo_root),
                "bounded_composition_artifact": _optional_display_path(
                    composition_matrix_path, repo_root
                ),
                "runtime_governance_artifact": _optional_display_path(
                    governance_matrix_path, repo_root
                ),
            }
        )

    artifact_csv = out_dir / "artifact_register.csv"
    with artifact_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(artifact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(artifact_rows)

    manifest = {
        "reference_domain": validation_report["reference_domain"],
        "validated_domains": validation_report["validated_domains"],
        "proof_validated_domains": proof_validated_domains,
        "proof_candidate_domains": proof_candidate_domains,
        "shadow_synthetic_domains": shadow_synthetic_domains,
        "experimental_domains": experimental_domains,
        "harness_pass": validation_report["harness_pass"],
        "evidence_pass": validation_report["evidence_pass"],
        "integrated_theorem_gate_pass": theorem_gate_pass,
        "integrated_theorem_gate_summary": theorem_gate_summary,
        "training_audit_pass": training_report["all_passed"],
        "sil_audit_pass": sil_report["all_passed"],
        "training_report": _display_path(training_report_path, repo_root),
        "sil_report": _display_path(sil_report_path, repo_root),
        "validation_report": _display_path(validation_report_path, repo_root),
        "proof_report": _display_path(proof_report_path, repo_root),
        "domain_closure_matrix": _display_path(closure_matrix_path, repo_root),
        "bounded_composition_matrix": _optional_display_path(
            composition_matrix_path, repo_root
        ),
        "runtime_governance_matrix": _optional_display_path(
            governance_matrix_path, repo_root
        ),
        "artifact_register": _display_path(artifact_csv, repo_root),
        "domain_controller_summary": _display_path(summary_csv, repo_root),
        "seeds": args.seeds,
        "horizon": args.horizon,
    }
    manifest_path = out_dir / "framework_proof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    proof_md = out_dir / "framework_proof_summary.md"
    proof_md.write_text(
        "\n".join(
            [
                "# ORIUS Framework Proof Summary",
                "",
                "## Locked interpretation",
                f"- Reference domain: `{validation_report['reference_domain']}`",
                f"- Validated domains: `{', '.join(validation_report['validated_domains'])}`",
                f"- Proof-validated domains: `{', '.join(proof_validated_domains) or 'none'}`",
                f"- Proof-candidate domains: `{', '.join(proof_candidate_domains) or 'none'}`",
                f"- Shadow-synthetic domains: `{', '.join(shadow_synthetic_domains) or 'none'}`",
                f"- Experimental domains: `{', '.join(experimental_domains) or 'none'}`",
                f"- Harness pass: `{validation_report['harness_pass']}`",
                f"- Evidence pass: `{validation_report['evidence_pass']}`",
                f"- Integrated theorem gate: `{theorem_gate_summary}`",
                f"- Training audit: `{training_report['all_passed']}`",
                f"- SIL audit: `{sil_report['all_passed']}`",
                "",
                "## Generated artifacts",
                f"- Universal validation report: `{_display_path(validation_report_path, repo_root)}`",
                f"- Proof-domain report: `{_display_path(proof_report_path, repo_root)}`",
                f"- Domain closure matrix: `{_display_path(closure_matrix_path, repo_root)}`",
                f"- Bounded composition matrix: `{_optional_display_path(composition_matrix_path, repo_root) or 'not generated'}`",
                f"- Runtime-governance matrix: `{_optional_display_path(governance_matrix_path, repo_root) or 'not generated'}`",
                f"- Integrated theorem gate: `{_display_path(theorem_gate_path, repo_root)}`",
                f"- Training audit report: `{_display_path(training_report_path, repo_root)}`",
                f"- SIL audit report: `{_display_path(sil_report_path, repo_root)}`",
                f"- Artifact register: `{_display_path(artifact_csv, repo_root)}`",
                f"- Domain/controller summary: `{_display_path(summary_csv, repo_root)}`",
                "",
                "## Proof gate",
                f"- Evaluated proof candidates: `{', '.join(proof_report.get('evaluated_proof_candidates', proof_report.get('promoted_proof_candidates', []))) or 'none'}`",
                f"- Proof-validated domains: `{', '.join(proof_validated_domains) or 'none'}`",
                f"- Locked protocol: `seeds={proof_report['locked_protocol']['seeds']}, horizon={proof_report['locked_protocol']['horizon']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("=== ORIUS Framework Proof Bundle ===")
    print(f"Out dir              -> {out_dir}")
    print(f"Artifact register    -> {artifact_csv}")
    print(f"Controller summary   -> {summary_csv}")
    print(f"Manifest             -> {manifest_path}")
    print(f"Proof summary        -> {proof_md}")


if __name__ == "__main__":
    main()
