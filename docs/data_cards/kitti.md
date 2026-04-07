# Data Card: KITTI Odometry

## 1. Dataset name and version
- Name: KITTI Odometry benchmark
- Version used by repo tooling: poses plus `times.txt` layout

## 2. Source URL and download
- Source URL: [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- Expected raw placement:
  - `data/navigation/raw/kitti_odometry/`
  - or `$ORIUS_EXTERNAL_DATA_ROOT/kitti_odometry/`

## 3. License
- KITTI usage terms; raw payloads must remain out of git history

## 4. Row count
- Sequence-dependent; the builder records per-sequence row counts in
  `data/navigation/raw/kitti_odometry_provenance.json`

## 5. Feature schema
- Processed ORIUS navigation contract:
  - `robot_id`, `step`, `x`, `y`, `vx`, `vy`, `ts_utc`, `source_sequence`
- Derived training features:
  - lagged position and velocity channels
  - hour and minute calendar fields

## 6. Temporal extent
- Sequence-local timestamps reconstructed from KITTI `times.txt`

## 7. Split policy
- Deterministic time-order split into train, calibration, validation, and test
- Processed outputs under `data/navigation/processed/`

## 8. Preprocessing pipeline
- Read per-sequence poses and times
- Convert 3x4 pose matrices into `x`, `y`, `vx`, `vy`
- Export ORIUS navigation CSV
- Build features and time splits via `src/orius/data_pipeline/build_features_navigation.py`

## 9. Known issues
- Canonical raw KITTI corpus is not staged in git
- Navigation defended-row promotion remains blocked until the full replay chain is rebuilt from canonical raw data

## 10. Intended ORIUS use
- Canonical real-data navigation closure source
- Replaces synthetic navigation in the defended path once staged and replayed

## 11. Citation
- See `paper/bibliography/orius_monograph.bib` for KITTI references.
