#!/usr/bin/env python3
"""Build a battery-anchored ORIUS framework proof bundle.

This orchestration script runs:
  1. the integrated 18-theorem release gate, and
  2. the canonical universal ORIUS validation gate,

then emits a compact bundle that answers:
  - which domain is the validated reference surface,
  - which non-battery domains are proof-validated,
  - which domains remain proof-candidate, shadow, or experimental,
  - where the validation artifacts live, and
  - how the before/after domain metrics summarize in one register.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a battery-anchored ORIUS framework proof bundle.")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds for the universal validation run.")
    parser.add_argument("--horizon", type=int, default=24, help="Episode horizon for the validation run.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/orius_framework_proof",
        help="Output directory for the proof bundle.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out
    validation_dir = out_dir / "universal_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "scripts/verify_integrated_theorem_surface.py",
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

    validation_report_path = validation_dir / "validation_report.json"
    proof_report_path = validation_dir / "proof_domain_report.json"
    theorem_gate_path = repo_root / "reports" / "publication" / "integrated_theorem_gate.json"
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    proof_report = json.loads(proof_report_path.read_text(encoding="utf-8"))
    theorem_gate = json.loads(theorem_gate_path.read_text(encoding="utf-8"))
    theorem_gate_pass = int(theorem_gate.get("failed", 0)) == 0
    theorem_gate_summary = (
        f"{theorem_gate.get('passed', 0)}/{theorem_gate.get('total', 0)} theorem rows verified"
    )

    domain_controller_summary: list[dict[str, object]] = []
    for row in validation_report["domain_results"]:
        domain_controller_summary.append(
            {
                "domain": row["domain"],
                "data_source": row["data_source"],
                "evidence_tier": row["evidence_tier"],
                "validation_status": row["validation_status"],
                "baseline_tsvr_mean": f"{float(row['baseline_tsvr_mean']):.4f}",
                "orius_tsvr_mean": f"{float(row['orius_tsvr_mean']):.4f}",
                "orius_reduction_pct": f"{float(row['orius_reduction_pct']):.1f}",
                "repair_rate_mean": f"{float(row['repair_rate_mean']):.2f}",
                "mean_reliability": f"{float(row['mean_reliability']):.4f}",
                "p95_latency_ms": f"{float(row['p95_latency_ms']):.4f}",
                "proof_gate_reasons": row["proof_gate_reasons"],
            }
        )

    summary_csv = out_dir / "domain_controller_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_controller_summary[0].keys()))
        writer.writeheader()
        writer.writerows(domain_controller_summary)

    artifact_rows: list[dict[str, str]] = []
    for row in validation_report["domain_results"]:
        artifact_rows.append(
            {
                "domain": str(row["domain"]),
                "data_source": str(row["data_source"]),
                "evidence_tier": str(row["evidence_tier"]),
                "validation_status": str(row["validation_status"]),
                "harness_status": str(row["harness_status"]),
                "theorem_gate_artifact": _display_path(theorem_gate_path, repo_root),
                "validation_artifact": _display_path(validation_report_path, repo_root),
                "proof_artifact": _display_path(proof_report_path, repo_root),
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
        "proof_validated_domains": validation_report["proof_validated_domains"],
        "proof_downgraded_domains": validation_report["proof_downgraded_domains"],
        "shadow_synthetic_domains": validation_report["shadow_synthetic_domains"],
        "experimental_domains": validation_report["experimental_domains"],
        "harness_pass": validation_report["harness_pass"],
        "evidence_pass": validation_report["evidence_pass"],
        "integrated_theorem_gate_pass": theorem_gate_pass,
        "integrated_theorem_gate_summary": theorem_gate_summary,
        "validation_report": _display_path(validation_report_path, repo_root),
        "proof_report": _display_path(proof_report_path, repo_root),
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
                f"- Proof-validated domains: `{', '.join(validation_report['proof_validated_domains']) or 'none'}`",
                f"- Proof-downgraded domains: `{', '.join(item['domain'] for item in validation_report['proof_downgraded_domains']) or 'none'}`",
                f"- Shadow-synthetic domains: `{', '.join(validation_report['shadow_synthetic_domains']) or 'none'}`",
                f"- Experimental domains: `{', '.join(validation_report['experimental_domains'])}`",
                f"- Harness pass: `{validation_report['harness_pass']}`",
                f"- Evidence pass: `{validation_report['evidence_pass']}`",
                f"- Integrated theorem gate: `{theorem_gate_summary}`",
                "",
                "## Generated artifacts",
                f"- Universal validation report: `{_display_path(validation_report_path, repo_root)}`",
                f"- Proof-domain report: `{_display_path(proof_report_path, repo_root)}`",
                f"- Integrated theorem gate: `{_display_path(theorem_gate_path, repo_root)}`",
                f"- Artifact register: `{_display_path(artifact_csv, repo_root)}`",
                f"- Domain/controller summary: `{_display_path(summary_csv, repo_root)}`",
                "",
                "## Proof gate",
                f"- Promoted proof candidates: `{', '.join(proof_report.get('promoted_proof_candidates', [])) or 'none'}`",
                f"- Proof-validated domains: `{', '.join(proof_report.get('proof_validated_domains', [])) or 'none'}`",
                f"- Downgraded candidates: `{', '.join(item['domain'] for item in proof_report.get('proof_downgraded_domains', [])) or 'none'}`",
                f"- Locked protocol: `seeds={proof_report['locked_protocol']['seeds']}, horizon={proof_report['locked_protocol']['horizon']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("=== ORIUS Framework Proof Bundle ===")
    print(f"Out dir              → {out_dir}")
    print(f"Artifact register    → {artifact_csv}")
    print(f"Controller summary   → {summary_csv}")
    print(f"Manifest             → {manifest_path}")
    print(f"Proof summary        → {proof_md}")


if __name__ == "__main__":
    main()
