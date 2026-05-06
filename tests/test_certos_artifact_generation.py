"""CertOS artifact generation test (Paper 6).

Validates that `scripts/build_certos_artifacts.py` produces runtime-derived
artifacts with the expected schema and content, rather than self-declared text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def certos_artifacts(tmp_path: Path) -> Path:
    """Run build_certos_artifacts.py to a temp directory."""
    result = subprocess.run(
        [sys.executable, "scripts/build_certos_artifacts.py", "--out", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"build_certos_artifacts.py failed:\n{result.stderr}"
    return tmp_path


class TestCertOSArtifactGeneration:
    """Verify that CertOS artifacts are generated from runtime truth."""

    def test_summary_json_has_expected_keys(self, certos_artifacts: Path) -> None:
        summary = json.loads((certos_artifacts / "certos_summary.json").read_text())
        expected = {
            "total_steps",
            "valid_steps",
            "degraded_steps",
            "fallback_steps",
            "interventions",
            "audit_entries",
            "hash_chain_ok",
            "validation_events",
            "expire_events",
            "fallback_events",
        }
        missing = expected - set(summary.keys())
        assert not missing, f"certos_summary.json missing keys: {missing}"

    def test_hash_chain_ok(self, certos_artifacts: Path) -> None:
        summary = json.loads((certos_artifacts / "certos_summary.json").read_text())
        assert summary["hash_chain_ok"] is True

    def test_validation_events_positive(self, certos_artifacts: Path) -> None:
        summary = json.loads((certos_artifacts / "certos_summary.json").read_text())
        assert summary["validation_events"] > 0, "No VALIDATE events observed"

    def test_expire_events_positive(self, certos_artifacts: Path) -> None:
        summary = json.loads((certos_artifacts / "certos_summary.json").read_text())
        assert summary["expire_events"] > 0, "No EXPIRE events observed"

    def test_invariant_log_derived_from_runtime(self, certos_artifacts: Path) -> None:
        log_path = certos_artifacts / "invariant_tests.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "INV-1" in content
        assert "INV-2" in content
        assert "INV-3" in content
        # All should pass if runtime is correct
        assert "FAIL" not in content, f"Invariant check failed:\n{content}"

    def test_audit_ops_jsonl_has_validate_entries(self, certos_artifacts: Path) -> None:
        audit_path = certos_artifacts / "audit_ops.jsonl"
        assert audit_path.exists()
        entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(entries) > 0
        ops = {e["op"] for e in entries}
        assert "VALIDATE" in ops, "No VALIDATE entries in audit_ops.jsonl"
        assert "ISSUE" in ops, "No ISSUE entries in audit_ops.jsonl"

    def test_audit_completeness_json(self, certos_artifacts: Path) -> None:
        ac_path = certos_artifacts / "audit_completeness.json"
        assert ac_path.exists()
        ac = json.loads(ac_path.read_text())
        assert ac["completeness_pct"] == 100.0
        assert ac["hash_chain_ok"] is True
        assert ac["validate_events"] > 0

    def test_lifecycle_csv_exists_and_nonempty(self, certos_artifacts: Path) -> None:
        csv_path = certos_artifacts / "certos_lifecycle.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 2, "lifecycle CSV should have header + data rows"
