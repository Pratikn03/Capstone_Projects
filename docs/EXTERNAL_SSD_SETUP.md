# ORIUS External SSD Setup on macOS

Use this workflow when the repo and Python environment stay on the MacBook internal SSD but
large raw AV, healthcare, navigation, and aerospace datasets live on an external SSD.

## Target layout

- Repo checkout stays at `<repo-root>`
- `.venv`, code, LaTeX builds, processed outputs, and PDFs stay internal
- Raw external corpora live under one mounted SSD root
- `ORIUS_EXTERNAL_DATA_ROOT` points to that SSD root
- `$HOME/orius_external_data` is a symlink to the same location so the strict
  equal-domain gate remains compatible later

## Buy and format the SSD

Recommended:

- `2 TB` minimum
- USB-C or Thunderbolt SSD
- macOS format: `APFS`
- Partition scheme: `GUID Partition Map`
- Stable volume name such as `ORIUS_SSD`

Format it in `Disk Utility`, then confirm the mount exists:

```bash
ls /Volumes
```

## One-command scaffold from the repo

Default scaffold:

```bash
SSD_VOLUME=<mounted-ssd-root>
make external-ssd-setup SSD_VOLUME="$SSD_VOLUME"
```

This creates:

- `$SSD_VOLUME/orius_external_data/`
- `$SSD_VOLUME/orius_external_data/nuplan/`
- `$SSD_VOLUME/orius_external_data/nuplan_maps/`
- `$SSD_VOLUME/orius_external_data/carla/`
- `$SSD_VOLUME/orius_external_data/waymo_open_motion/`
- `$SSD_VOLUME/orius_external_data/healthcare_private/`
- `$SSD_VOLUME/orius_external_data/argoverse2_motion/`
- `$SSD_VOLUME/orius_external_data/argoverse2_sensor/`
- `$SSD_VOLUME/orius_external_data/kitti_odometry/`
- `$SSD_VOLUME/orius_external_data/aerospace_flight_telemetry/`
- the strict-link path `$HOME/orius_external_data` pointing to the SSD root

If you also want the helper to update `~/.zshrc`:

```bash
make external-ssd-shell SSD_VOLUME="$SSD_VOLUME"
```

That writes a managed block exporting:

```bash
export ORIUS_EXTERNAL_DATA_ROOT="$SSD_VOLUME/orius_external_data"
```

## Verify the SSD layout

Verify only:

```bash
make external-ssd-verify SSD_VOLUME="$SSD_VOLUME"
```

Run the real-data preflight against the SSD root:

```bash
make external-ssd-preflight SSD_VOLUME="$SSD_VOLUME"
```

Or run the script directly:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_real_data_preflight.py \
  --external-root "$SSD_VOLUME/orius_external_data"
```

## What to copy to the SSD

The external SSD root should contain:

```text
$SSD_VOLUME/orius_external_data/
├── nuplan/
├── nuplan_maps/
├── carla/
├── waymo_open_motion/
├── healthcare_private/
├── argoverse2_motion/
├── argoverse2_sensor/
├── kitti_odometry/
└── aerospace_flight_telemetry/
```

Expected data details:

- AV:
  - active evidence lane: completed nuPlan archives under `nuplan/`
  - nuPlan maps under `nuplan_maps/`
  - CARLA assets/runs under `carla/` when simulation closure is performed
  - Waymo remains a legacy compatibility corpus under `waymo_open_motion/`
  - `waymo_open_motion/`
  - `argoverse2_motion/`
  - `argoverse2_sensor/`
  - used through `scripts/download_av_datasets.py`
- Healthcare:
  - private/local row-level healthcare extracts under `healthcare_private/`
  - repo-local artifacts are manifests, summaries, validators, and reproduction commands only
- Navigation:
  - KITTI under `kitti_odometry/`
  - accepted layouts:
    - `dataset/poses/*.txt` and `dataset/sequences/*/times.txt`
    - or `poses/*.txt` and `sequences/*/times.txt`
  - used through `scripts/build_navigation_real_dataset.py`
- Aerospace:
  - provider telemetry under `aerospace_flight_telemetry/**/*.csv` or `**/*.parquet`
  - public ADS-B proxy data is not accepted as the official real-flight closure source
  - used through `scripts/build_aerospace_real_flight_dataset.py`

## First validation flow after copying data

```bash
source ~/.zshrc
echo "$ORIUS_EXTERNAL_DATA_ROOT"

PYTHONPATH=src .venv/bin/python scripts/verify_real_data_preflight.py \
  --external-root "$SSD_VOLUME/orius_external_data"

PYTHONPATH=src .venv/bin/python scripts/refresh_real_data_manifests.py

PYTHONPATH=src .venv/bin/python scripts/build_navigation_real_dataset.py \
  --external-root "$SSD_VOLUME/orius_external_data"

PYTHONPATH=src .venv/bin/python scripts/build_aerospace_real_flight_dataset.py \
  --external-root "$SSD_VOLUME/orius_external_data"
```

For bounded local work after the SSD is set up:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_battery_deep_novelty.py
make thesis-manuscript
```

## Hybrid local + HF Jobs

Use the SSD for raw corpora locally. Use Hugging Face Jobs later only for bounded heavy runs
such as battery deep-learning novelty or diagnostics that do not substitute for the canonical
raw-data obligation.

Hugging Face Jobs do not replace:

- nuPlan, CARLA, Waymo, or Argoverse raw storage
- private/local healthcare raw-row governance
- KITTI real-data closure
- provider-approved aerospace telemetry

Relevant job templates live under:

- `<repo-root>/scripts/hf_jobs/`

## Safety notes

- Do not put the raw corpora into git history.
- Do not use `exFAT` if this is a MacBook-first workflow.
- If `$HOME/orius_external_data` already exists and is not the correct symlink,
  fix it manually before rerunning the setup helper.
