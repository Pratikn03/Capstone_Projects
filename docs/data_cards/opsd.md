# Data Card: Open Power System Data (OPSD Germany)

## 1. Dataset name and version
- Name: Open Power System Data time series, Germany slice
- Version in repo: `opsd-time_series-2020-10-06`

## 2. Source URL and download
- Source URL: [https://open-power-system-data.org/](https://open-power-system-data.org/)
- Repo-local source path: `data/raw/opsd-time_series-2020-10-06/`

## 3. License
- OPSD terms apply; attribution required per source provider.

## 4. Row count
- Locked DE evaluation surface: 17,377 hourly rows.

## 5. Feature schema
- Core signals: `load_mw`, `wind_mw`, `solar_mw`, `price_eur_mwh`
- Calendar features: hour, day-of-week, holidays
- Weather covariates: temperature, irradiance, wind-speed joins
- Engineered feature count in locked publication surface: 94

## 6. Temporal extent
- Locked evaluation period: 2018-10-07 through 2020-09-30 UTC

## 7. Split policy
- Time-ordered split with train, calibration, validation, and test partitions
- Publication defaults come from `data/processed/splits/*.parquet`

## 8. Preprocessing pipeline
- Raw ingestion from OPSD CSV
- Timestamp normalization to UTC
- Weather enrichment
- Lag and rolling-window feature construction
- Export to `data/processed/features.parquet`

## 9. Known issues
- Market and telemetry joins can introduce sparse missingness in weather-side fields
- Real deployment telemetry faults are injected downstream by CPSBench

## 10. Intended ORIUS use
- Battery witness row
- Forecasting, calibration, DC3S replay, and theorem-to-artifact reference surface

## 11. Citation
- See `paper/bibliography/orius_monograph.bib` for the OPSD and supporting power-system references.
