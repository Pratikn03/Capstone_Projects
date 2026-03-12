from __future__ import annotations

from scripts.build_dataset_summary_table import build_dataset_summary_rows


def test_build_dataset_summary_rows_uses_dashboard_profile_lock() -> None:
    rows = build_dataset_summary_rows()

    us_rows = [row for row in rows if row["Country"] == "US"]
    de_rows = [row for row in rows if row["Country"] == "DE"]

    assert len(de_rows) == 3
    assert len(us_rows) == 9
    assert {row["Rows"] for row in us_rows} == {13638, 13639, 13453}
    assert {row["Rows"] for row in de_rows} == {17377}
    assert {row["Signal"] for row in us_rows} == {"load_mw", "wind_mw", "solar_mw"}
    assert {row["DatasetKey"] for row in us_rows} == {"US_MISO", "US_PJM", "US_ERCOT"}
    assert {row["DatasetKey"] for row in de_rows} == {"DE"}
