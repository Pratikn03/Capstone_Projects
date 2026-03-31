# AV Real-Data Placement

Current status:

- Raw HEE dataset is present at `data/av/raw/hee_dataset/`.
- The canonical processed ORIUS file is `data/av/processed/av_trajectories_orius.csv`.
- The canonical real-data source is now `Waymo Open Motion` via an external raw-data root.

External raw-data contract:

- Set `ORIUS_EXTERNAL_DATA_ROOT=/path/to/external/datasets`
- Place full raw or prepared tabular exports under:
  - `$ORIUS_EXTERNAL_DATA_ROOT/waymo_open_motion/`
  - `$ORIUS_EXTERNAL_DATA_ROOT/argoverse2_motion/`
  - `$ORIUS_EXTERNAL_DATA_ROOT/argoverse2_sensor/`

Canonical build commands:

- `python scripts/download_av_datasets.py --source waymo_motion`
- `python scripts/download_av_datasets.py --source argoverse2_motion`
- `python scripts/download_av_datasets.py --source argoverse2_sensor`

Expected canonical processed target:

- `data/av/processed/av_trajectories_orius.csv`

Expected ORIUS longitudinal contract:

- `vehicle_id`
- `step`
- `position_m`
- `speed_mps`
- `speed_limit_mps`
- `lead_position_m`
- `ts_utc`

Notes:

- Full Waymo / Argoverse raw payloads should not be copied into git.
- Synthetic AV trajectories remain available only by explicit request:
  - `python scripts/download_av_datasets.py --source synthetic`
- Legacy CSV conversion remains available for bounded compatibility:
  - `python scripts/download_av_datasets.py --convert path/to/legacy.csv`
