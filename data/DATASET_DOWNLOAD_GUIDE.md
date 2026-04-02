# ORIUS Real-Dataset Download and Placement Guide

Use this guide with:

- `DATA.md`
- `data/av/PLACE_REAL_AV_DATA_HERE.md`
- `data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md`
- `data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md`

## Current truth

| Domain | Canonical source | Current status | What is still needed |
| --- | --- | --- | --- |
| Battery | OPSD + SMARD / EIA-family release surfaces | Reference row | Keep release packaging clean |
| Autonomous vehicles | Waymo Open Motion | Real-data contract active | Maintain repo-local raw placement and replay artifacts |
| Industrial | CCPP primary + ZeMA companion | Real-data contract active | Keep both raw-source manifests current |
| Healthcare | BIDMC / PhysioNet | Real-data contract active | Keep full raw placement and processed lineage stable |
| Navigation | KITTI Odometry | Blocked real-data gap | Finish full train/validate/replay chain |
| Aerospace | NASA C-MAPSS FD001-FD004 | Trainable real-data row | Validation parity is still pending |

Important boundary:

- Navigation is not a defended real-data row until the full pipeline closes.
- Aerospace now has a real trainable surface, but equal-domain closure is still not claimed.
- Do not use this guide by itself to justify equal-domain universality.

## Repo-local raw-data root

Repo-local placement is now the default contract:

```text
data/
├── av/raw/waymo_open_motion/
├── av/raw/argoverse2_motion/
├── av/raw/argoverse2_sensor/
├── navigation/raw/kitti_odometry/
├── industrial/raw/ccpp/
├── industrial/raw/zema_hydraulic/
├── healthcare/raw/bidmc_csv/
└── aerospace/raw/
```

`ORIUS_EXTERNAL_DATA_ROOT` remains supported only as a fallback for AV and navigation.

## Domain instructions

### 1. Battery

Battery remains the reference row. Use the existing energy feature builders and refresh provenance with:

```bash
python scripts/refresh_real_data_manifests.py
```

### 2. Autonomous vehicles

Canonical source:

- `Waymo Open Motion`

Companion sources:

- `Argoverse 2 Motion`
- `Argoverse 2 Sensor`

Place under:

```text
data/av/raw/waymo_open_motion/
data/av/raw/argoverse2_motion/
data/av/raw/argoverse2_sensor/
```

Build commands:

```bash
python scripts/download_av_datasets.py --source waymo_motion
python scripts/download_av_datasets.py --source argoverse2_motion
python scripts/download_av_datasets.py --source argoverse2_sensor
```

Canonical processed output:

- `data/av/processed/av_trajectories_orius.csv`

Manifests:

- `data/av/raw/waymo_open_motion_provenance.json`
- `data/av/raw/argoverse2_sensor_provenance.json`

### 3. Industrial

Primary trainable source:

- `data/industrial/raw/ccpp/`

Companion dense source:

- `data/industrial/raw/zema_hydraulic/`

Build command:

```bash
python scripts/download_industrial_datasets.py --source ccpp
```

Manifest:

- `data/industrial/raw/ccpp_provenance.json`

### 4. Healthcare

Canonical source:

- BIDMC / PhysioNet

Place under:

```text
data/healthcare/raw/bidmc_csv/
```

Build command:

```bash
python scripts/download_healthcare_datasets.py --source bidmc
```

Manifest:

- `data/healthcare/raw/bidmc_provenance.json`

### 5. Navigation

Canonical source:

- `KITTI Odometry`

Place under:

```text
data/navigation/raw/kitti_odometry/
```

Build commands:

```bash
python scripts/build_navigation_real_dataset.py
PYTHONPATH=src python -m orius.data_pipeline.build_features_navigation
```

Canonical processed output:

- `data/navigation/processed/navigation_orius.csv`

Manifest:

- `data/navigation/raw/kitti_odometry_provenance.json`

Current blocker:

- the row is not defended until train, validation, replay, and artifact generation all pass with no synthetic fallback.

### 6. Aerospace

Canonical trainable source:

- NASA C-MAPSS under `data/aerospace/raw/`

Build command:

```bash
python scripts/download_aerospace_datasets.py
```

Manifest:

- `data/aerospace/raw/cmapss_provenance.json`

Important boundary:

- C-MAPSS now backs the trainable aerospace surface in this repo
- theorem-grade or equal-domain aerospace closure is still not claimed

## Preflight

Run:

```bash
python scripts/verify_real_data_preflight.py
```

This checks:

- repo-local free disk threshold
- required CLIs: `git`, `hf`, `kaggle`
- required Python modules: `pandas`, `pyarrow`, `openpyxl`, `wfdb`, `huggingface_hub`
- expected raw-data directories for each domain
