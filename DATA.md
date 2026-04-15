# Data Guide (Sources, Layout, Licensing)

This repository does **not** vendor all raw datasets. ORIUS stores manuscript,
publication, and processed release artifacts in git, while large or restricted
raw datasets remain outside the repository under provider terms.

## 1. Canonical policy

- Raw external datasets stay outside git whenever size, licensing, or provider
  terms make redistribution inappropriate.
- Repo-local raw staging is now the default contract for real-data builders:
  `data/<domain>/raw/<dataset_key>/`.
- `ORIUS_EXTERNAL_DATA_ROOT=/path/to/external/datasets` remains supported only
  as a fallback when repo-local storage is intentionally not used.
- The current source-of-truth surfaces for dataset access are:
  - `data/DATASET_DOWNLOAD_GUIDE.md`
  - `data/av/PLACE_REAL_AV_DATA_HERE.md`
  - `data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md`
  - `data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md`
  - `docs/REAL_DATA_PLUGIN_ACCESS_PLAN.md`
  - `reports/publication/orius_equal_domain_parity_matrix.csv`

## 2. Minimum plugin stack

ORIUS does **not** require many plugins to obtain real datasets. The main
blockers are provider access, storage capacity, and missing domain pipelines.

- Required operational plugin: `GitHub`
- High-value support plugin: `Hugging Face`
- Optional support plugin: `Google Drive`
- Not needed for dataset closure: `Figma`, `Canva`, `Netlify`
- PM-only: `Linear`, `Notion`

Use `docs/REAL_DATA_PLUGIN_ACCESS_PLAN.md` for the reasoning and domain-level
breakdown.

## 3. Canonical domain sources

| Domain | Current canonical source | Current status |
| --- | --- | --- |
| Battery | OPSD + SMARD / EIA-family release surfaces | Reference row |
| Autonomous vehicles | Waymo Open Motion | Real-data contract defined |
| AV companions | Argoverse 2 Motion, Argoverse 2 Sensor | Secondary validation surfaces |
| Industrial | Current defended processed row; raw-source cleanup still mixed | Proof-validated, cleanup still needed |
| Healthcare | PhysioNet BIDMC | Real-data path active |
| Navigation | KITTI Odometry | Real-data contract defined, row still blocked |
| Aerospace | NASA C-MAPSS FD001-FD004 (trainable) plus bounded public ADS-B support lane | Experimental/support-tier only; defended runtime closure pending |

Important boundary:

- Navigation is still blocked by the real-data validation gap.
- Aerospace is still blocked by the missing canonical real-flight runtime row.
- The bounded public ADS-B lane is support-only and does not promote the defended aerospace row.
- MIMIC-III is not staged in this repo and is not the active healthcare runtime source.
- Do not claim equal-domain universality until those rows clear the same gate as
  the defended domains.

## 4. Optional external raw-data fallback

When repo-local raw staging is intentionally not used, stage fallback datasets
under:

```text
$ORIUS_EXTERNAL_DATA_ROOT/
├── waymo_open_motion/
├── argoverse2_motion/
├── argoverse2_sensor/
├── kitti_odometry/
└── aerospace_flight_telemetry/
```

Notes:

- The AV and navigation builders now check repo-local raw storage first and use
  `ORIUS_EXTERNAL_DATA_ROOT` only as a fallback.
- The aerospace external directory remains reserved for the future defended
  runtime telemetry replay surface; the current public ADS-B lane is support-only.
- Full raw payloads should not be copied into git.

## 5. Repo-local preflight

Run this before attempting the full all-domain real-data program:

```bash
python scripts/verify_real_data_preflight.py
```

This check verifies:

- free disk against the repo-local corpus threshold
- required CLIs: `git`, `hf`, `kaggle`
- required Python modules: `pandas`, `pyarrow`, `openpyxl`, `wfdb`, `huggingface_hub`
- presence of the expected raw directories for each domain

## 6. Domain-specific placement manifests

Use the domain placement guides before building processed rows:

- `data/av/PLACE_REAL_AV_DATA_HERE.md`
- `data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md`
- `data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md`

These documents define the processed target file, expected raw layout, and the
minimum contract columns for each domain.

## 7. Energy-source commands still supported

### OPSD Germany

- Provider: Open Power System Data (OPSD)
- Typical signals: load, wind generation, solar generation
- Raw location: `data/raw/time_series_60min_singleindex.csv`

Optional reproducible download:

```bash
python -m orius.data_pipeline.download_opsd --out data/raw
```

### EIA Form 930

- Provider: U.S. Energy Information Administration
- Typical signals: balancing-authority demand and generation
- Raw location: `data/raw/us_eia930/`

Build example:

```bash
python -m orius.data_pipeline.build_features_eia930 \
  --in data/raw/us_eia930 \
  --out data/processed/us_eia930 \
  --ba MISO
```

### Optional weather

```bash
python -m orius.data_pipeline.download_weather \
  --out data/raw \
  --start 2015-01-01 \
  --end 2020-09-30
```

### Optional holidays

```bash
python scripts/generate_holidays.py \
  --country DE \
  --start-year 2015 \
  --end-year 2020 \
  --out data/raw/holidays_de.csv
```

## 8. Licensing and attribution

- Data is owned by the original providers.
- Do not redistribute raw provider data in this repository.
- Always follow the provider license and attribution rules when publishing
  results or release bundles.

## 9. Availability boundary

The final submission claims depend on tracked manuscript and publication
artifacts, not on ignored local caches or private snapshots. The parity matrix
and publication bundle define the current defended evidence boundary.
