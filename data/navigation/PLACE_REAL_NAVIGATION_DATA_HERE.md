# Navigation Real-Data Placement

Current status:

- The navigation benchmark track still exists for synthetic portability work.
- The canonical real-data source is now `KITTI Odometry` via an external raw-data root.

External raw-data contract:

- Set `ORIUS_EXTERNAL_DATA_ROOT=/path/to/external/datasets`
- Place KITTI Odometry under:
  - `$ORIUS_EXTERNAL_DATA_ROOT/kitti_odometry/`

Canonical build commands:

- `python scripts/build_navigation_real_dataset.py`
- `PYTHONPATH=src python -m orius.data_pipeline.build_features_navigation`

Expected canonical processed target:

- `data/navigation/processed/navigation_orius.csv`

Minimum processed fields should look like:

- `robot_id`
- `step`
- `x`
- `y`
- `vx`
- `vy`
- `ts_utc`

Notes:

- Full KITTI raw payloads should not be copied into git.
- Navigation remains a lower-tier thesis row until the full train and validation path is completed.
- The raw-source metadata contract for this row is tracked in:
  - `data/navigation/raw/external_sources_manifest.json`
