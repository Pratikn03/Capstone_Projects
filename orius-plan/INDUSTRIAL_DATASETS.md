# Industrial Datasets for ORIUS Multi-Domain Validation

Process control datasets (temperature, pressure, power) to demonstrate ORIUS beyond energy and AV.

---

## Quick Start

```bash
# Generate synthetic process data (no download, works immediately)
python3 scripts/download_industrial_datasets.py --source synthetic

# Output: data/industrial/processed/industrial_orius.csv
```

---

## Recommended Datasets

| Dataset | Source | Format | Best for | Access |
|---------|--------|--------|----------|--------|
| **CCPP** | UCI ML Repository | xlsx (AT, V, AP, RH, PE) | Combined cycle power plant, 9568 samples | Direct: `--source ccpp` |
| **Hydraulic** | UCI Condition Monitoring | CSV | Pressure, temperature, flow, 2205 cycles | Manual: UCI dataset 447 |
| **Synthetic** | ORIUS script | ORIUS format | Immediate use | `--source synthetic` |

---

## Download Commands

### CCPP (UCI)

```bash
pip install openpyxl  # required for xlsx
python3 scripts/download_industrial_datasets.py --source ccpp
```

### Convert Existing CSV

If you have CCPP-format CSV (columns: AT, V, AP, RH, PE):

```bash
python3 scripts/download_industrial_datasets.py --convert path/to/ccpp.csv --out data/industrial/processed/my_industrial.csv
```

---

## ORIUS Format

Output columns for industrial prototype:

| Column | Type | Description |
|--------|------|-------------|
| sensor_id | int | Sensor/plant identifier |
| step | int | Time step index |
| temp_c | float | Ambient temperature (°C) |
| vacuum_cmhg | float | Exhaust vacuum (cm Hg) |
| pressure_mbar | float | Ambient pressure (mbar) |
| humidity_pct | float | Relative humidity (%) |
| power_mw | float | Net electrical output (MW) |
| ts_utc | str | Timestamp (ISO8601) |

---

## ORIUS Multi-Domain Coverage

| Domain | Primary use | Datasets |
|--------|-------------|----------|
| **Energy** | Battery, grid, load | EIA-930, OPSD |
| **AV** | Longitudinal control | NGSIM, HEE, synthetic |
| **Industrial** | Process control | CCPP, synthetic |
| **Healthcare** | Vital signs | BIDMC, synthetic |
