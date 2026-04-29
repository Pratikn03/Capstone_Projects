"""Tracked research and publication artifact endpoints.

These routes intentionally source data from checked-in publication assets and
release manifests rather than local dashboard caches.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Security
from fastapi.responses import FileResponse

from services.api.health import readiness_check
from services.api.security import get_api_key, verify_scope

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "reports"
PUBLICATION_ROOT = REPORTS_ROOT / "publication"
RELEASE_MANIFEST = PUBLICATION_ROOT / "release_manifest.json"
IMPACT_PATHS = {
    "DE": REPORTS_ROOT / "impact_summary.csv",
    "US": REPORTS_ROOT / "eia930" / "impact_summary.csv",
}


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _dataset_stats(region: str) -> dict[str, Any] | None:
    release_manifest = _load_json(RELEASE_MANIFEST)
    profile = (release_manifest.get("dataset_profiles") or {}).get(region.lower())
    if not isinstance(profile, dict):
        return None
    return {
        "region": region,
        "label": "Germany" if region == "DE" else "USA",
        "rows": int(profile.get("rows", 0)),
        "columns": int(profile.get("columns", 0)),
        "column_names": [],
        "date_range": {
            "start": profile.get("date_start_utc"),
            "end": profile.get("date_end_utc"),
        },
        "target_columns": [],
        "weather_features": int(profile.get("weather_feature_count", 0)),
        "lag_features": int(profile.get("lag_feature_count", 0)),
        "calendar_features": int(profile.get("calendar_feature_count", 0)),
        "total_features": int(profile.get("feature_count", 0)),
        "targets_summary": {},
        "targets": {},
        "missing_pct": {},
    }


def _impact_summary(region: str) -> dict[str, Any] | None:
    rows = _read_csv_rows(IMPACT_PATHS[region])
    if not rows:
        return None
    row = rows[0]
    return {
        "region": region,
        "baseline_cost_usd": _to_float(row.get("baseline_cost_usd")),
        "orius_cost_usd": _to_float(row.get("orius_cost_usd")),
        "cost_savings_pct": _to_float(row.get("cost_savings_pct")),
        "baseline_carbon_kg": _to_float(row.get("baseline_carbon_kg")),
        "orius_carbon_kg": _to_float(row.get("orius_carbon_kg")),
        "carbon_reduction_pct": _to_float(row.get("carbon_reduction_pct")),
        "baseline_peak_mw": _to_float(row.get("baseline_peak_mw")),
        "orius_peak_mw": _to_float(row.get("orius_peak_mw")),
        "peak_shaving_pct": _to_float(row.get("peak_shaving_pct")),
    }


def _collect_report_files(root: Path) -> list[dict[str, Any]]:
    type_map = {
        ".md": "Markdown",
        ".json": "JSON",
        ".csv": "CSV",
        ".png": "PNG",
        ".svg": "SVG",
        ".pdf": "PDF",
        ".tex": "TeX",
    }
    reports: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        reports.append(
            {
                "name": path.name,
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "description": "Tracked publication artifact.",
                "type": type_map.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "FILE"),
                "date": path.stat().st_mtime_ns,
                "path": rel,
                "size_bytes": path.stat().st_size,
            }
        )
    reports.sort(key=lambda item: item["date"], reverse=True)
    for item in reports:
        import datetime as _dt

        item["date"] = _dt.datetime.fromtimestamp(item["date"] / 1_000_000_000).date().isoformat()
    return reports


def _reports_response() -> dict[str, Any]:
    release_manifest = _load_json(RELEASE_MANIFEST)
    reports = _collect_report_files(PUBLICATION_ROOT)
    de_impact = _impact_summary("DE")
    us_impact = _impact_summary("US")
    region_reports = {
        "DE": {
            "id": "DE",
            "label": "Germany",
            "reports": reports,
            "metrics": [],
            "metrics_backtest": [],
            "impact": de_impact,
            "robustness": None,
            "training_status": None,
            "meta": {
                "source": "reports",
                "last_updated": release_manifest.get("frozen_at_utc"),
                "metrics_source": "missing",
                "warnings": ["Region metrics are served from tracked publication artifacts only."],
            },
        },
        "US": {
            "id": "US",
            "label": "USA",
            "reports": reports,
            "metrics": [],
            "metrics_backtest": [],
            "impact": us_impact,
            "robustness": None,
            "training_status": None,
            "meta": {
                "source": "reports",
                "last_updated": release_manifest.get("frozen_at_utc"),
                "metrics_source": "missing",
                "warnings": ["Region metrics are served from tracked publication artifacts only."],
            },
        },
    }
    return {
        "reports": reports,
        "metrics": [],
        "metrics_backtest": [],
        "impact": de_impact,
        "robustness": None,
        "regions": region_reports,
        "meta": {
            "source": "reports",
            "last_updated": release_manifest.get("frozen_at_utc"),
            "metrics_source": "missing",
            "warnings": ["Dashboard data is sourced from tracked publication artifacts, not local dashboard caches."],
        },
    }


@router.get("/manifest")
def research_manifest(api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    release_manifest = _load_json(RELEASE_MANIFEST)
    return {
        "source": str(RELEASE_MANIFEST.relative_to(REPO_ROOT)),
        "frozen_at_utc": release_manifest.get("frozen_at_utc"),
        "dataset_profiles": release_manifest.get("dataset_profiles", {}),
        "controllers_present": release_manifest.get("controllers_present", []),
        "artifact_count": len(release_manifest.get("artifact_hashes_sha256", {})),
    }


@router.get("/region/{region}")
def research_region(region: str, api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    region = region.upper()
    if region not in {"DE", "US"}:
        raise HTTPException(status_code=400, detail="Invalid region. Use DE or US.")
    return {
        "stats": _dataset_stats(region),
        "timeseries": [],
        "forecast": {},
        "dispatch": [],
        "profiles": {},
        "metrics": [],
        "impact": _impact_summary(region),
        "registry": [],
        "monitoring": None,
        "anomalies": [],
        "zscores": [],
        "battery": None,
        "pareto": [],
    }


@router.get("/reports")
def research_reports(api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    return _reports_response()


@router.get("/reports/file")
def research_report_file(
    path: str = Query(..., description="Path relative to reports/publication"),
    api_key: str = Security(get_api_key),
) -> FileResponse:
    verify_scope("read", api_key)
    requested = (PUBLICATION_ROOT / path).resolve()
    try:
        requested.relative_to(PUBLICATION_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found.") from None
    if not requested.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(requested, filename=requested.name)


@router.get("/benchmark")
def research_benchmark_summary(api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    return {
        "main_table": _read_csv_rows(PUBLICATION_ROOT / "dc3s_main_table_ci.csv"),
        "latency": _read_csv_rows(PUBLICATION_ROOT / "dc3s_latency_summary.csv"),
        "fault_performance": _read_csv_rows(PUBLICATION_ROOT / "fault_performance_table.csv"),
    }


@router.get("/governance")
def research_governance_summary(api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    metrics_manifest = _load_json(REPO_ROOT / "paper" / "metrics_manifest.json")
    return {
        "readiness": readiness_check(),
        "manuscript_authority": metrics_manifest.get("metric_policy", {}).get("master_manuscript"),
        "claim_family": metrics_manifest.get("claim_family"),
        "release_manifest": str(RELEASE_MANIFEST.relative_to(REPO_ROOT)),
    }
