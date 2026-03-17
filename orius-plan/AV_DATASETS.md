# AV Datasets for ORIUS Vehicles Extension

Best datasets for 1D longitudinal control (position, speed, lead vehicle) used by the ORIUS vehicles prototype.

**See also:** `orius-plan/INDUSTRIAL_DATASETS.md`, `orius-plan/HEALTHCARE_DATASETS.md` for Industrial and Healthcare domains.

---

## Quick Start

```bash
# Generate synthetic trajectories (no download, works immediately)
python scripts/download_av_datasets.py --source synthetic

# Output: data/av/processed/av_trajectories_orius.csv
```

---

## Recommended Datasets

| Dataset | Source | Format | Best for | Access |
|---------|--------|--------|----------|--------|
| **NGSIM** | US DOT / Kaggle | CSV (Vehicle_ID, Local_X, v_Velocity, Spacing) | Highway trajectories, 0.1s resolution | Kaggle: `nigelwilliams/ngsim-vehicle-trajectory-data-us-101` |
| **highD** | RWTH Aachen / levelXdata | CSV | German highway, drone-recorded, precise | Manual request at levelxdata.com |
| **HEE** | Bosch (GitHub) | CSV | ~12k trajectories, 600m highway | `git clone https://github.com/boschresearch/hee_dataset` |
| **INTERACTION** | interaction-dataset.com | CSV + Lanelet2 | Interactive scenarios, multi-country | Request form |
| **Synthetic** | ORIUS script | ORIUS format | Immediate use, no download | `--source synthetic` |

---

## Download Commands

### NGSIM (via Kaggle)

```bash
pip install kaggle
# Configure ~/.kaggle/kaggle.json with your API key
python scripts/download_av_datasets.py --source ngsim
```

### HEE (GitHub)

```bash
python scripts/download_av_datasets.py --source hee
```

### Convert Existing CSV

If you have NGSIM-format CSV (columns: Local_X, v_Velocity, Vehicle_ID, Frame_ID, Spacing, Global_Time):

```bash
python scripts/download_av_datasets.py --convert path/to/trajectories.csv --out data/av/processed/my_av.csv
```

---

## ORIUS Format

Output columns for vehicles prototype:

| Column | Type | Description |
|--------|------|-------------|
| vehicle_id | int | Vehicle identifier |
| step | int | Time step index |
| position_m | float | Longitudinal position (m) |
| speed_mps | float | Speed (m/s) |
| speed_limit_mps | float | Posted speed limit |
| lead_position_m | float \| empty | Lead vehicle position (if following) |
| ts_utc | str | Timestamp (ISO8601) |

---

## Artifact Zip Alignment (Reference)

For the existing agent artifact zip vs repo thesis:

- **Aligned:** Zip T1→T1, Zip T3→T3, Zip T4→T8, Zip T2→T5,T6 (partial)
- **Out-of-scope:** Zip T5–T8 (multi-domain extensions)
- **Gaps:** Canonical T2, T4, T7 not in zip
- **Policy:** Repo thesis is source of edits. Run `make analyze-artifact` to compare.
