# Data Card: Combined Cycle Power Plant (CCPP)

## 1. Dataset name and version
- Name: UCI Combined Cycle Power Plant dataset
- Repo-local raw path: `data/industrial/raw/CCPP.csv`

## 2. Source URL and download
- UCI Machine Learning Repository CCPP page
- Repo-local provenance manifest: `data/industrial/raw/ccpp_provenance.json`

## 3. License
- UCI repository distribution terms

## 4. Row count
- Determined from the staged raw CSV and summarized in processed outputs

## 5. Feature schema
- Ambient temperature, exhaust vacuum, ambient pressure, relative humidity
- Target and plant-state features used for industrial ORIUS replay
- Processed industrial outputs under `data/industrial/processed/`

## 6. Temporal extent
- Static record corpus rather than long continuous telemetry window

## 7. Split policy
- Deterministic train, calibration, validation, and test partitions
- Bounded to the defended industrial plant family and replay protocol

## 8. Preprocessing pipeline
- Raw CSV normalization
- Feature typing and derived industrial-state construction
- Export to ORIUS industrial processed surfaces and parquet features

## 9. Known issues
- The industrial defended row is bounded to the current plant family
- Additional raw-source cleanup may still be required for companion ZeMA surfaces

## 10. Intended ORIUS use
- Canonical industrial defended bounded row
- Industrial replay, calibration, and runtime-governance validation

## 11. Citation
- See `paper/bibliography/orius_monograph.bib` for CCPP and industrial references.
