from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from orius.data_pipeline import validate_schema as validate_schema_mod


def test_validate_schema_supports_non_de_country(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = 48
    df = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "FR_load_actual_entsoe_transparency": np.linspace(60_000, 61_000, rows),
            "FR_wind_onshore_generation_actual": np.linspace(3_000, 3_500, rows),
            "FR_wind_offshore_generation_actual": np.linspace(500, 750, rows),
            "FR_solar_generation_actual": np.linspace(1_000, 1_200, rows),
            "FR_price_day_ahead": np.linspace(40, 70, rows),
        }
    )
    (raw_dir / "time_series_60min_singleindex.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_schema.py",
            "--in",
            str(raw_dir),
            "--report",
            str(report_path),
            "--country",
            "FR",
        ],
    )
    validate_schema_mod.main()
    report = report_path.read_text(encoding="utf-8")
    assert "Country: **FR**" in report
    assert "Duplicate timestamps: **0**" in report
