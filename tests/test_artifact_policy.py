"""Tests for repo artifact policy helpers."""

from __future__ import annotations

from pathlib import Path

import scripts.classify_repo_artifacts as artifact_classifier
from scripts.classify_repo_artifacts import classify_path
from scripts.validate_generated_artifact_policy import TRACKED_ALLOWLIST
from scripts.validate_reproducibility_95 import validate_script_reference_policy


def test_classifies_local_generated_artifacts() -> None:
    assert classify_path("data/orius_av/raw/nuplan.zip") == "local_dataset"
    assert classify_path("artifacts/models/model.pt") == "model_artifact"
    assert classify_path("reports/foo/runtime_traces.csv") == "generated_runtime_artifact"
    assert classify_path("frontend/.next/server/app.js") == "cache_build_output"
    assert classify_path("dashboard-final-smoke.png") == "temporary_ai_codex_artifact"


def test_intentional_bundle_allowlist_is_explicit() -> None:
    assert "reports/orius_bench/benchmark_bundle.tar.gz" in TRACKED_ALLOWLIST


def test_inventory_preserves_tracked_flag_for_untracked_paths(monkeypatch) -> None:
    def fake_git_paths(args: list[str]) -> list[str]:
        if args == ["ls-files"]:
            return ["src/orius/dc3s/certificate.py"]
        if args == ["ls-files", "-o", "--exclude-standard"]:
            return ["reports/run/runtime_traces.csv"]
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(artifact_classifier, "_git_paths", fake_git_paths)

    rows = {str(row["path"]): row for row in artifact_classifier.iter_inventory(include_untracked=True)}

    assert rows["src/orius/dc3s/certificate.py"]["tracked"] is True
    assert rows["reports/run/runtime_traces.csv"]["tracked"] is False


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


def test_default_pdf_targets_do_not_run_mutating_table_repair() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    default_targets = [
        "paper-verify",
        "camera-ready-verify",
        "paper-compile",
        "camera-ready-freeze",
        "paper-freeze",
    ]
    for target in default_targets:
        body = _make_target_body(makefile, target)
        assert "scripts/repair_table_result_integrity.py" not in body, target

    repair_body = _make_target_body(makefile, "table-result-integrity-repair")
    assert "scripts/repair_table_result_integrity.py" in repair_body


def test_paper_compile_is_not_a_table_integrity_scan() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    body = _make_target_body(makefile, "paper-compile")
    assert "scripts/audit_table_result_integrity.py" not in body


def test_script_reference_policy_accepts_referenced_scripts() -> None:
    findings = validate_script_reference_policy(
        script_paths=["scripts/build_scorecard.py"],
        reference_texts={"Makefile": "scorecard:\n\t$(PYTHON) scripts/build_scorecard.py\n"},
        registry={},
    )
    assert findings == []


def test_script_reference_policy_accepts_explicit_registry_entries() -> None:
    findings = validate_script_reference_policy(
        script_paths=["scripts/manual_repair.py"],
        reference_texts={"Makefile": ""},
        registry={
            "scripts/manual_repair.py": {
                "status": "manual",
                "reason": "Used only for explicit artifact repair after audit failures.",
            }
        },
    )
    assert findings == []


def test_script_reference_policy_rejects_orphan_scripts() -> None:
    findings = validate_script_reference_policy(
        script_paths=["scripts/orphan.py"],
        reference_texts={"Makefile": ""},
        registry={},
    )
    assert findings == ["script has no repo reference or registry entry: scripts/orphan.py"]
