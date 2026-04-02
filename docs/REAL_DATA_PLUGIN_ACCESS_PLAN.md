# ORIUS Real-Data Plugin and Access Plan

This document explains which plugins materially help ORIUS reach real-data,
universal-depth evidence across domains, and which ones do not.

## Summary

You do **not** need many new plugins to get all real datasets. The main
blockers are:

- provider access
- external storage
- missing domain pipelines
- parity-gate completion

The shortest serious answer is:

- Required: `GitHub`
- Strongly recommended: `Hugging Face`
- Optional: `Google Drive`
- Not needed for dataset closure: `Figma`, `Canva`, `Netlify`
- PM-only: `Linear`, `Notion`

## Plugin stack

### Required

#### GitHub

Use GitHub to track:

- domain parity blockers
- raw-source decisions
- manifest status
- closure gates
- remediation work for navigation and aerospace

GitHub is the only plugin that is operationally necessary for turning the
real-data program into a defensible universal-data closure effort.

### High value

#### Hugging Face

Use Hugging Face for:

- cloud compute and preprocessing jobs
- large training or replay workloads
- optional mirrored dataset cards
- benchmark publication support

It is not the official acquisition path for Waymo, KITTI, BIDMC, or future
aerospace telemetry. It becomes valuable **after** raw data is already acquired.

### Optional support

#### Google Drive

Use Google Drive for:

- provider instructions
- access letters or approval packets
- non-git review artifacts
- large documentation bundles

Do not use Drive as the canonical raw-data store.

### Not needed for dataset closure

- `Figma`
- `Canva`
- `Netlify`

These are presentation-layer tools. They help with figures, slides, and public
communication, but not with real-data acquisition or proof-depth closure.

### Project-management only

- `Linear`
- `Notion`

Useful for tracking and documentation, but they do not unlock any raw dataset.

## Domain-by-domain reality

| Domain | Canonical real source | Current truth |
| --- | --- | --- |
| Battery | OPSD + SMARD / EIA-family release surfaces | Real reference row already exists |
| Autonomous vehicles | Waymo Open Motion | Repo-local contract exists; raw corpus still not staged locally |
| AV companions | Argoverse 2 Motion, Argoverse 2 Sensor | Secondary validation and degradation surfaces |
| Industrial | Current defended processed row; raw-source cleanup still mixed | Proof-validated, but raw-source packaging should be cleaned up |
| Healthcare | PhysioNet BIDMC | Real-data path already viable |
| Navigation | KITTI Odometry | Repo-local contract exists, full defended row still blocked |
| Aerospace | NASA C-MAPSS FD001-FD004 plus pending runtime telemetry | Trainable row exists; defended runtime replay surface still missing |

## Repo-local raw-data contract

Repo-local placement is now the primary contract:

```text
data/
├── av/raw/waymo_open_motion/
├── av/raw/argoverse2_motion/
├── av/raw/argoverse2_sensor/
├── navigation/raw/kitti_odometry/
├── healthcare/raw/bidmc_csv/
├── industrial/raw/ccpp/
├── industrial/raw/zema_hydraulic/
└── aerospace/raw/
```

`ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV,
navigation, and runtime-only telemetry corpora.

Per-domain placement guides:

- `data/av/PLACE_REAL_AV_DATA_HERE.md`
- `data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md`
- `data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md`

## What still blocks equal-domain universality

### Navigation

Navigation is still blocked because the real-data row has not yet completed the
full defended chain:

- deterministic processed build
- train/validation completion
- replay completion
- publication artifact emission

### Aerospace

Aerospace is still below equal-domain closure because the trainable C-MAPSS row
does not yet replace the defended runtime replay obligation. For equal-domain
universality, the aerospace target still requires:

- provider-approved multi-flight telemetry
- a real ORIUS safety task
- grouped train/validation/replay pipeline on that runtime surface

`C-MAPSS` now backs the trainable aerospace row in-repo, but it is not enough
by itself for equal-domain aerospace closure.

## Source-of-truth policy

- Per-domain placement manifests define raw-source layout.
- `DATA.md` defines repository-wide data policy.
- `data/DATASET_DOWNLOAD_GUIDE.md` gives operator-facing acquisition guidance.
- `reports/real_data_contract_status.json` records the current per-domain raw-data
  blocker state for the bounded-universal closure program.
- `docs/BOUNDED_UNIVERSAL_CLOSURE_PROGRAM.md` defines the defended-row
  promotion gate and the closure target.
- `reports/publication/orius_equal_domain_parity_matrix.csv` defines the current
  defended equal-domain boundary.

Do not widen manuscript claims based on plugins alone. Only the validation
artifacts and parity matrices can promote a domain tier.
