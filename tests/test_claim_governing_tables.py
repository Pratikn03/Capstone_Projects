"""Tests for compact-evidence claim-governing table regeneration."""

from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_target_body(text: str, target: str) -> str:
    lines = text.splitlines()
    capture = False
    body: list[str] = []
    for line in lines:
        if line.startswith(f"{target}:"):
            capture = True
            continue
        if capture and line and not line.startswith(("\t", " ")):
            break
        if capture:
            body.append(line)
    return "\n".join(body)


def test_make_target_regenerates_claim_governing_tables_from_compact_evidence_only() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    body = _make_target_body(makefile, "claim-governing-tables")

    assert "scripts/build_claim_governing_tables.py" in body
    assert "scripts/build_three_domain_ml_artifacts.py" not in body
    assert "scripts/build_top_venue_research_package.py" not in body
    assert "data/" not in body
    assert "artifacts/" not in body


def test_clean_clone_docs_include_claim_governing_table_command() -> None:
    docs = Path("docs/reproducibility.md").read_text(encoding="utf-8")

    assert "make claim-governing-tables" in docs
    assert "reports/publication/three_domain_ml_benchmark.csv" in docs
    assert "reports/publication/three_domain_forecast_calibration_runtime_evidence.csv" in docs


def test_renderer_builds_tables_from_compact_publication_inputs(tmp_path: Path) -> None:
    publication_dir = tmp_path / "reports" / "publication"
    _write_csv(
        publication_dir / "three_domain_ml_benchmark.csv",
        [
            {
                "domain": "Battery Energy Storage",
                "tier": "reference",
                "metric_surface": "locked_publication_nominal",
                "baseline_tsvr_mean": "0.008333",
                "orius_tsvr_mean": "0.000000",
                "relative_delta": "1.000000",
                "intervention_rate": "0.020833",
                "fallback_activation_rate": "0.020833",
                "certificate_valid_release_rate": "0.993056",
                "runtime_witness_pass_rate": "1.000000",
                "strict_runtime_gate": "True",
                "runtime_latency_p95_ms": "0.812",
            },
            {
                "domain": "Autonomous Vehicles",
                "tier": "runtime_contract_closed",
                "metric_surface": "runtime_denominator",
                "baseline_tsvr_mean": "0.289250",
                "orius_tsvr_mean": "0.000163",
                "relative_delta": "0.999438",
                "intervention_rate": "0.500301",
                "fallback_activation_rate": "0.173716",
                "certificate_valid_release_rate": "0.999924",
                "runtime_witness_pass_rate": "0.999837",
                "strict_runtime_gate": "True",
                "runtime_latency_p95_ms": "0.744",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "tier": "runtime_contract_closed",
                "metric_surface": "runtime_denominator",
                "baseline_tsvr_mean": "0.194489",
                "orius_tsvr_mean": "0.000000",
                "relative_delta": "1.000000",
                "intervention_rate": "0.479907",
                "fallback_activation_rate": "0.479907",
                "certificate_valid_release_rate": "1.000000",
                "runtime_witness_pass_rate": "1.000000",
                "strict_runtime_gate": "True",
                "runtime_latency_p95_ms": "0.701",
            },
        ],
    )
    _write_csv(
        publication_dir / "three_domain_forecast_calibration_runtime_evidence.csv",
        [
            {
                "domain": "Battery Energy Storage",
                "active_dataset": "German OPSD battery witness",
                "claim_boundary": "Locked battery witness runtime evidence.",
            },
            {
                "domain": "Autonomous Vehicles",
                "active_dataset": "nuPlan all-zip grouped replay",
                "claim_boundary": "Bounded all-zip grouped nuPlan replay evidence.",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "active_dataset": "MIMIC retrospective monitoring",
                "claim_boundary": "Retrospective MIMIC monitoring evidence.",
            },
        ],
    )

    from scripts.build_claim_governing_tables import build_claim_governing_tables

    written = build_claim_governing_tables(publication_dir=publication_dir)

    rel_written = {path.relative_to(publication_dir).as_posix() for path in written}
    assert rel_written == {
        "claim_governing_three_domain_runtime_evidence.tex",
        "final_runtime_safety_for_paper.csv",
        "tbl_final_runtime_safety.tex",
        "tbl_executive_runtime_evidence.tex",
    }

    claim_tex = (publication_dir / "claim_governing_three_domain_runtime_evidence.tex").read_text(
        encoding="utf-8"
    )
    assert "Battery & witness & 0.008333 & 0.000000" in claim_tex
    assert "OPSD" in claim_tex
    assert "runtime traces" not in claim_tex.lower()
    assert "data/" not in claim_tex
    assert "claim_boundary" not in claim_tex
    assert "road deployment" in claim_tex

    executive_tex = (publication_dir / "tbl_executive_runtime_evidence.tex").read_text(
        encoding="utf-8"
    )
    assert "Executive three-domain ORIUS result summary" in executive_tex
    assert "Healthcare & 0.194489 & 0.000000" in executive_tex

    final_rows = list(csv.DictReader((publication_dir / "final_runtime_safety_for_paper.csv").open()))
    assert final_rows[0]["baseline_tsvr"] == "0.008333"
    assert final_rows[1]["latency_p95_ms"] == "0.744"
