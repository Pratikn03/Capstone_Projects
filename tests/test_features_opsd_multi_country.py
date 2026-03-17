from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from orius.data_pipeline import build_features as build_features_mod
from orius.data_pipeline.build_features import normalize_opsd_country_frame


def _hourly_frame(rows: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({"utc_timestamp": idx})


def test_normalize_opsd_country_frame_supports_de_fr_es() -> None:
    de = _hourly_frame()
    de["DE_load_actual_entsoe_transparency"] = np.linspace(50_000, 55_000, len(de))
    de["DE_wind_generation_actual"] = np.linspace(10_000, 11_000, len(de))
    de["DE_solar_generation_actual"] = np.linspace(2_000, 3_000, len(de))
    de["DE_LU_price_day_ahead"] = np.linspace(40, 60, len(de))
    de_out = normalize_opsd_country_frame(de, country="DE")
    assert list(de_out.columns) == ["timestamp", "load_mw", "wind_mw", "solar_mw", "price_eur_mwh"]

    fr = _hourly_frame()
    fr["FR_load_actual_entsoe_transparency"] = np.linspace(60_000, 61_000, len(fr))
    fr["FR_wind_onshore_generation_actual"] = np.linspace(3_000, 3_500, len(fr))
    fr["FR_wind_offshore_generation_actual"] = np.linspace(1_000, 1_250, len(fr))
    fr["FR_solar_generation_actual"] = np.linspace(1_500, 1_900, len(fr))
    fr["FR_price_day_ahead"] = np.linspace(55, 75, len(fr))
    fr_out = normalize_opsd_country_frame(fr, country="FR")
    assert np.isclose(fr_out.loc[0, "wind_mw"], 4_000.0)
    assert "price_eur_mwh" in fr_out.columns

    es = _hourly_frame()
    es["ES_load_actual_entsoe_power_statistics"] = np.linspace(30_000, 32_000, len(es))
    es["ES_wind_generation_actual"] = np.linspace(5_000, 5_500, len(es))
    es["ES_solar_generation_actual"] = np.linspace(1_000, 1_400, len(es))
    es_out = normalize_opsd_country_frame(es, country="ES")
    assert list(es_out.columns) == ["timestamp", "load_mw", "wind_mw", "solar_mw"]


def test_build_features_cli_country_smoke(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()

    df = _hourly_frame()
    df["ES_load_actual_entsoe_power_statistics"] = np.linspace(30_000, 32_000, len(df))
    df["ES_wind_generation_actual"] = np.linspace(5_000, 5_500, len(df))
    df["ES_solar_generation_actual"] = np.linspace(1_000, 1_400, len(df))
    (raw_dir / "time_series_60min_singleindex.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_features.py",
            "--in",
            str(raw_dir),
            "--out",
            str(out_dir),
            "--country",
            "ES",
            "--holiday-country",
            "ES",
        ],
    )
    build_features_mod.main()

    features = pd.read_parquet(out_dir / "features.parquet")
    assert {"timestamp", "load_mw", "wind_mw", "solar_mw", "price_eur_mwh"}.issubset(features.columns)
    assert not features.empty
