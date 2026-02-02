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
