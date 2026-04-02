from __future__ import annotations

from services.api.routers import research


def test_research_manifest_uses_tracked_publication_assets() -> None:
    payload = research.research_manifest()
    assert payload["source"] == "reports/publication/release_manifest.json"
    assert "dataset_profiles" in payload


def test_research_region_returns_domain_neutral_shape() -> None:
    payload = research.research_region("DE")
    assert set(payload.keys()) == {
        "stats",
        "timeseries",
        "forecast",
        "dispatch",
        "profiles",
        "metrics",
        "impact",
        "registry",
        "monitoring",
        "anomalies",
        "zscores",
        "battery",
        "pareto",
    }
    assert payload["stats"]["rows"] == 17377


def test_research_reports_exposes_report_inventory() -> None:
    payload = research.research_reports()
    assert payload["meta"]["source"] == "reports"
    assert isinstance(payload["reports"], list)
    assert "DE" in payload["regions"]
    assert "US" in payload["regions"]
