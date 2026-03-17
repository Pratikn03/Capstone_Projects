# Healthcare Datasets for ORIUS Multi-Domain Validation

Vital signs datasets (HR, SpO2, respiratory rate) to demonstrate ORIUS beyond energy and AV.

---

## Quick Start

```bash
# Generate synthetic vital signs (no download, works immediately)
python3 scripts/download_healthcare_datasets.py --source synthetic

# Output: data/healthcare/processed/healthcare_orius.csv
```

---

## Recommended Datasets

| Dataset | Source | Format | Best for | Access |
|---------|--------|--------|----------|--------|
| **BIDMC** | PhysioNet | CSV (HR, SpO2, RESP) | Critically-ill patients, 53 recordings × 8 min | Direct: `--source bidmc` |
| **VitalDB** | PhysioNet | CSV | Surgical patients, multi-parameter | Manual: physionet.org/content/vitaldb |
| **Synthetic** | ORIUS script | ORIUS format | Immediate use | `--source synthetic` |

---

## Download Commands

### BIDMC (PhysioNet)

```bash
python3 scripts/download_healthcare_datasets.py --source bidmc
```

PhysioNet BIDMC is open access; the script fetches Numerics CSVs directly.

### Convert Existing CSV

If you have BIDMC-format CSV (columns: Time [s], HR, SpO2, RESP or RR):

```bash
python3 scripts/download_healthcare_datasets.py --convert path/to/bidmc_01_Numerics.csv --out data/healthcare/processed/my_healthcare.csv
```

---

## ORIUS Format

Output columns for healthcare prototype:

| Column | Type | Description |
|--------|------|-------------|
| patient_id | str | Patient/recording identifier |
| step | int | Time step index |
| hr_bpm | float | Heart rate (bpm) |
| spo2_pct | float | Blood oxygen saturation (%) |
| respiratory_rate | float | Respiratory rate (breaths/min) |
| ts_utc | str | Timestamp (ISO8601) |

---

## ORIUS Multi-Domain Coverage

| Domain | Primary use | Datasets |
|--------|-------------|----------|
| **Energy** | Battery, grid, load | EIA-930, OPSD |
| **AV** | Longitudinal control | NGSIM, HEE, synthetic |
| **Industrial** | Process control | CCPP, synthetic |
| **Healthcare** | Vital signs | BIDMC, synthetic |
