from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import build_conference_assets as conference_assets


def test_build_dataset_cards_writes_csv_and_figure(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "table1_dataset_summary.csv"
    pd.DataFrame(
        [
            {
                "DatasetKey": "DE",
                "Dataset": "Germany (OPSD)",
                "Country": "DE",
                "Start": "2018-10-07",
                "End": "2020-09-30",
                "Rows": 17377,
                "Signal": "load_mw",
                "Non-Null": 17377,
                "Coverage%": 100.0,
            },
            {
                "DatasetKey": "DE",
                "Dataset": "Germany (OPSD)",
                "Country": "DE",
                "Start": "2018-10-07",
                "End": "2020-09-30",
                "Rows": 17377,
                "Signal": "wind_mw",
                "Non-Null": 17377,
                "Coverage%": 100.0,
            },
        ]
    ).to_csv(summary_path, index=False)
    monkeypatch.setitem(
        conference_assets.DATASET_META,
        "DE",
        {
            "stats_path": tmp_path / "de_stats.json",
            "provenance_path": tmp_path / "dataset_provenance.json",
            "source": "OPSD",
        },
    )
    (tmp_path / "de_stats.json").write_text(json.dumps({"total_features": 94, "columns": 98}), encoding="utf-8")
    (tmp_path / "dataset_provenance.json").write_text(
        json.dumps({"weather_source": "Open-Meteo", "carbon_source": "proxy_generation_mix"}),
        encoding="utf-8",
    )

    payload = conference_assets.build_dataset_cards(summary_path, tmp_path)

    assert payload["rows"] == 1
    assert (tmp_path / "dataset_cards.csv").exists()
    assert (tmp_path / "fig_region_dataset_cards.png").exists()


def test_build_figure_inventory_marks_missing_and_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(conference_assets, "REPO_ROOT", tmp_path)
    out_dir = tmp_path / "reports" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "reports/publication/fig_true_soc_violation_vs_dropout.png",
        "reports/publication/fig_true_soc_severity_p95_vs_dropout.png",
        "reports/publication/fig_cost_safety_pareto.png",
        "reports/publication/fig_transfer_generalization.png",
        "reports/publication/fig_region_dataset_cards.png",
        "reports/publication/fig_calibration_tradeoff.png",
        "reports/publication/figures/fig01_geographic_scope.png",
        "reports/publication/figures/fig11_dispatch_comparison.png",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    payload = conference_assets.build_figure_inventory(out_dir)

    assert payload["summary"]["required_total"] == 8
    assert payload["summary"]["ready_total"] == 8
    assert payload["summary"]["ready"] is True
