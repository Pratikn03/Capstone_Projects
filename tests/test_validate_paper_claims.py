from __future__ import annotations

from pathlib import Path

from scripts.validate_paper_claims import _check_claim_matrix


def test_check_claim_matrix_accepts_historical_and_inactive_rows(tmp_path: Path) -> None:
    claim_matrix = tmp_path / "claim_matrix.csv"
    claim_matrix.write_text(
        "\n".join(
            [
                "claim_id,status,category,manuscript_locations,claim_text,canonical_value,source_file,source_locator,run_id,timestamp_utc,rounding_rule,notes",
                "C001,Verified,Metric,paper/paper.tex,Active claim,1.0,source.csv,row,,,n/a,ok",
                "C002,Historical,Metric,historical_prelock_draft_md,Old claim,old,source.csv,row,,,n/a,ok",
                "C003,Inactive,Metric,NOT PRESENT,Future claim,none,None,None,,,n/a,ok",
            ]
        ),
        encoding="utf-8",
    )

    findings = []
    _check_claim_matrix(findings, claim_matrix)

    assert findings == []


def test_check_claim_matrix_flags_invalid_historical_mapping(tmp_path: Path) -> None:
    claim_matrix = tmp_path / "claim_matrix.csv"
    claim_matrix.write_text(
        "\n".join(
            [
                "claim_id,status,category,manuscript_locations,claim_text,canonical_value,source_file,source_locator,run_id,timestamp_utc,rounding_rule,notes",
                "C001,Historical,Metric,paper/paper.tex,Old claim,old,source.csv,row,,,n/a,broken",
            ]
        ),
        encoding="utf-8",
    )

    findings = []
    _check_claim_matrix(findings, claim_matrix)

    assert any("Historical claim C001" in finding.detail for finding in findings)
