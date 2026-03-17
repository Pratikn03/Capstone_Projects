from __future__ import annotations

from pathlib import Path

from scripts import build_publication_artifact as pub


def test_publication_builder_requires_rac_outputs() -> None:
    required = set(pub.REQUIRED_PUBLICATION)
    assert "rac_cert_summary.json" in required
    assert "fig_rac_sensitivity_vs_width.png" in required
    assert "release_manifest.json" in required


def test_publication_builder_sets_rac_strict_env() -> None:
    source = Path("scripts/build_publication_artifact.py").read_text(encoding="utf-8")
    assert "ORIUS_REQUIRE_RAC_CERT" in source
