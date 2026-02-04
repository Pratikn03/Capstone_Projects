# Data Guide (Sources, Layout, Licensing)

This repository **does not include raw datasets**. You are responsible for downloading them from the original providers and agreeing to their terms.

## 1) Datasets Used

### A) OPSD Germany (Load / Wind / Solar)
- **Provider:** Open Power System Data (OPSD)
- **Typical signals:** load, wind generation, solar generation (hourly time series)
- **Where to place raw file:** `data/raw/time_series_60min_singleindex.csv`
- **How to prepare:** run the data pipeline in the README

### B) EIA Form 930 (USA; optional)
- **Provider:** U.S. Energy Information Administration (EIA)
- **Typical signals:** balancing‑authority demand + generation
- **Where to place raw files:** `data/raw/us_eia930/` (zipped CSVs)
- **How to prepare:** run:
  ```bash
  python -m gridpulse.data_pipeline.build_features_eia930 --in data/raw/us_eia930 --out data/processed/us_eia930 --ba MISO
  ```

### C) Weather (Optional)
- **Provider:** Open‑Meteo (Berlin hourly)
- **Usage:** optional feature enrichment

### D) Price + Carbon Signals (Recommended)
- **What:** optional time‑aligned price and carbon intensity series for more realistic optimization.
- **Where to place:** any CSV/Parquet file; pass its path to `build_features` via `--signals`.
- **Required columns:** `timestamp` plus any of:
  - `price_eur_mwh` or `price_usd_mwh`
  - `carbon_kg_per_mwh` (or `carbon_gco2_kwh`)
  - `moer_kg_per_mwh` (optional marginal emissions)
- **How to use:**
  ```bash
  python -m gridpulse.data_pipeline.build_features --in data/raw --out data/processed \
    --signals data/raw/price_carbon_signals.csv
  ```
  Signals are merged by timestamp and take precedence over the OPSD price column.

#### Real carbon options
- **Electricity Maps**: download via API using `scripts/download_emaps_carbon.py` (token required) or export a CSV and convert with `scripts/prepare_emaps_carbon.py`.
- **SMARD (Germany)**: use `scripts/download_smard_carbon.py` to compute hourly carbon intensity from the public SMARD generation mix (no token).
- **WattTime MOER**: use `scripts/download_watttime_moer.py` to fetch marginal emissions (MOER) and merge via `scripts/merge_signals.py`.

## 2) What This Repo Stores
- **Processed features:** `data/processed/` (Parquet)
- **Splits:** `data/processed/splits/`
- **Models:** `artifacts/models/` (git‑ignored)
- **Reports/figures:** `reports/` (git‑ignored by default)

## 3) Licensing & Attribution
- Data is owned by the original providers.
- Do **not** redistribute raw data files in this repo.
- Always follow the provider’s license/terms and include attribution when publishing results.

## 4) Reproducible Downloads (Optional)
If you use the built‑in downloader:
```bash
python -m gridpulse.data_pipeline.download_opsd --out data/raw
```

## 5) Notes
- Missing columns (e.g., price) are treated as optional.
- If price/carbon series are not present, optimization cost savings may be near zero.
