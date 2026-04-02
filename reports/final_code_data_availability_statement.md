# Code and Data Availability Statement

The ORIUS final submission is accompanied by the full source repository used to
produce the manuscript, runtime framework, validation logic, and tracked
publication artifacts.

## Code availability

All thesis-facing code is available in this repository, including:

- universal runtime and theorem-facing code under `src/orius/`
- manuscript and appendix sources under `paper/`, `chapters/`, and `appendices/`
- validation and publication scripts under `scripts/`
- tracked release and publication artifacts under `reports/publication/`

## Data availability

The repository contains tracked processed artifacts and publication-facing
summaries needed to support the final submission claims. Raw external datasets
are not fully vendored into the repository when licensing, size, or provider
terms make redistribution inappropriate. In those cases:

- the source and acquisition guidance are documented in `DATA.md` and
  `data/DATASET_DOWNLOAD_GUIDE.md`
- repo-local raw staging under `data/<domain>/raw/<dataset_key>/` is the
  primary contract for real-data builders
- `ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV,
  navigation, and runtime-only telemetry corpora
- domain-specific placement manifests define the canonical raw-data contracts:
  - `data/av/PLACE_REAL_AV_DATA_HERE.md`
  - `data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md`
  - `data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md`
- per-domain provenance manifests record provider, version, inventory, and
  processed-output lineage for real-data builds
- processed publication artifacts are tracked in `reports/publication/`
- dataset profiles and release provenance are locked in `reports/publication/release_manifest.json`
- the current bounded-universal evidence boundary is recorded in
  `reports/publication/orius_equal_domain_parity_matrix.csv`
- the current raw-data closure blockers are recorded in
  `reports/real_data_contract_status.json`
- closure to a fully defended six-domain release additionally requires
  per-domain provenance manifests, processed outputs, split artifacts, model
  bundles, uncertainty/backtest artifacts, replay traces, and governance
  traces under the shared promotion gate

## Availability boundary

The final submission claims depend on tracked manuscript and publication assets,
not on ignored local caches, private dashboard snapshots, or undeclared raw-data
mounts. The active closure program distinguishes:

- current defended evidence: battery witness plus defended bounded AV,
  industrial, and healthcare rows
- current open blockers: navigation real-data replay closure and aerospace
  runtime telemetry closure
