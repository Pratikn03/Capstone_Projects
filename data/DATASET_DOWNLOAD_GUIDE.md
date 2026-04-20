# ORIUS Real-Dataset Download and Placement Guide

This guide reflects the active ORIUS 3-domain submission program only.

Active defended rows:

- Battery
- Autonomous Vehicles
- Healthcare

Current public posture:

- Battery is the witness row.
- AV is a bounded defended row under the TTC plus predictive-entry-barrier contract.
- Healthcare is a bounded defended row under MIMIC monitoring and alert-release semantics.
- Additional domains are future architectural extensions, not current defended rows.

## Repo-local raw-data root

Repo-local placement is the default contract:

```text
data/
├── av/raw/waymo_open_motion/
├── av/raw/argoverse2_motion/
├── av/raw/argoverse2_sensor/
├── healthcare/raw/mimic3/
└── raw/
```

`ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for AV corpora only.

## Domain instructions

### 1. Battery

Battery remains the reference witness row. Refresh provenance with:

```bash
PYTHONPATH=src python scripts/refresh_real_data_manifests.py
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

Manifest:

- `data/av/raw/waymo_open_motion_provenance.json`

### 3. Healthcare

Canonical source:

- `MIMIC-III Waveform Database Matched Subset`

Canonical processed runtime/evaluation surface:

- `data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv`

Canonical manifest:

- `data/healthcare/mimic3/processed/mimic3_manifest.json`

Repo-local raw placement:

```text
data/healthcare/raw/mimic3/
```

Important boundary:

- MIMIC is the only canonical promoted healthcare source on this branch.
- Patient-disjoint split policy is keyed off `patient_id` in the promoted builder and evaluation surfaces.
- BIDMC may remain in-repo as supplemental healthcare support material, but it is not the promoted source of truth.

## Preflight

Run:

```bash
python scripts/verify_real_data_preflight.py
```

This checks:

- repo-local free disk threshold
- required CLIs: `git`, `hf`, `kaggle`
- required Python modules: `pandas`, `pyarrow`, `openpyxl`, `wfdb`, `huggingface_hub`
- expected raw-data directories for the active 3-domain program
