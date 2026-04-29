"""Tests for repo artifact policy helpers."""
from __future__ import annotations

from scripts.classify_repo_artifacts import classify_path
from scripts.validate_generated_artifact_policy import TRACKED_ALLOWLIST


def test_classifies_local_generated_artifacts() -> None:
    assert classify_path("data/orius_av/raw/nuplan.zip") == "local_dataset"
    assert classify_path("artifacts/models/model.pt") == "model_artifact"
    assert classify_path("reports/foo/runtime_traces.csv") == "generated_runtime_artifact"
    assert classify_path("frontend/.next/server/app.js") == "cache_build_output"
    assert classify_path("dashboard-final-smoke.png") == "temporary_ai_codex_artifact"


def test_intentional_bundle_allowlist_is_explicit() -> None:
    assert "reports/orius_bench/benchmark_bundle.tar.gz" in TRACKED_ALLOWLIST
