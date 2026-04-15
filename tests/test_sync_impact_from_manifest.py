from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.sync_impact_from_manifest as sync


def test_sync_impact_from_manifest_preserves_orius_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "paper" / "metrics_manifest.json"
    de_path = tmp_path / "reports" / "impact_summary.csv"
    us_path = tmp_path / "reports" / "eia930" / "impact_summary.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(
            {
                "canonical_metrics": {
                    "de": {
                        "impact": {
                            "baseline_cost_usd": 10.0,
                            "orius_cost_usd": 9.0,
                            "cost_savings_pct_raw": 0.1,
                            "baseline_carbon_kg": 8.0,
                            "orius_carbon_kg": 7.0,
                            "carbon_reduction_pct_raw": 0.125,
                            "baseline_peak_mw": 6.0,
                            "orius_peak_mw": 5.0,
                            "peak_shaving_pct_raw": 1.0 / 6.0,
                        }
                    },
                    "us": {
                        "impact": {
                            "baseline_cost_usd": 20.0,
                            "orius_cost_usd": 19.0,
                            "cost_savings_pct_raw": 0.05,
                            "baseline_carbon_kg": 18.0,
                            "orius_carbon_kg": 17.0,
                            "carbon_reduction_pct_raw": 1.0 / 18.0,
                            "baseline_peak_mw": 16.0,
                            "orius_peak_mw": 15.0,
                            "peak_shaving_pct_raw": 1.0 / 16.0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sync, "MANIFEST", manifest_path)
    monkeypatch.setattr(sync, "DE_IMPACT", de_path)
    monkeypatch.setattr(sync, "US_IMPACT", us_path)

    assert sync.main() == 0

    with de_path.open("r", encoding="utf-8", newline="") as handle:
        de_rows = list(csv.DictReader(handle))
    with us_path.open("r", encoding="utf-8", newline="") as handle:
        us_rows = list(csv.DictReader(handle))

    assert de_rows[0]["orius_cost_usd"] == "9.0"
    assert de_rows[0]["orius_carbon_kg"] == "7.0"
    assert de_rows[0]["orius_peak_mw"] == "5.0"
    assert "gridpulse_cost_usd" not in de_rows[0]
    assert us_rows[0]["orius_cost_usd"] == "19.0"
    assert "gridpulse_cost_usd" not in us_rows[0]
