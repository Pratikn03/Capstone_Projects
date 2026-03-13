#!/usr/bin/env python3
"""Fill tbl08 CSV from Phase 1 baseline_comparison.json results."""
import json
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Read phase1 results
phase1_path = REPO / "reports" / "phase1" / "baseline_comparison.json"
with open(phase1_path) as f:
    phase1 = json.load(f)

target_map = {"Load": "load_mw", "Wind": "wind_mw", "Solar": "solar_mw"}
model_map = {
    "GBM": "gbm", "LSTM": "lstm", "TCN": "tcn",
    "N-BEATS": "nbeats", "TFT": "tft", "PatchTST": "patchtst",
}

csv_path = REPO / "paper" / "assets" / "tables" / "tbl08_forecast_baselines.csv"

# Read existing CSV
rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(dict(row))

# Fill in missing data from Phase 1
filled = 0
for row in rows:
    region = row["Region"]
    target_key = target_map.get(row["Target"])
    model_key = model_map.get(row["Model"])

    if not target_key or not model_key:
        continue

    if region not in phase1 or target_key not in phase1[region]:
        continue
    if model_key not in phase1[region][target_key]:
        continue

    m = phase1[region][target_key][model_key]

    # Fill if RMSE is missing
    if row["RMSE"] == "---":
        row["RMSE"] = f'{m["rmse"]:.2f}'
        row["MAE"] = f'{m["mae"]:.2f}'
        row["sMAPE (%)"] = f'{m["smape"] * 100:.2f}'
        row["R2"] = f'{m["r2"]:.4f}'
        row["PICP@90 (%)"] = "---"
        row["Interval Width (MW)"] = "---"
        filled += 1
        print(f"  Filled: {region} {row['Target']} {row['Model']}")

    # Update GBM with latest Phase 1 numbers
    elif row["Model"] == "GBM":
        old_rmse = row["RMSE"]
        row["RMSE"] = f'{m["rmse"]:.2f}'
        row["MAE"] = f'{m["mae"]:.2f}'
        row["sMAPE (%)"] = f'{m["smape"] * 100:.2f}'
        row["R2"] = f'{m["r2"]:.4f}'
        if old_rmse != row["RMSE"]:
            print(f"  Updated GBM: {region} {row['Target']} RMSE {old_rmse} -> {row['RMSE']}")

# Write updated CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

remaining = sum(1 for r in rows if r["RMSE"] == "---")
print(f"\nFilled {filled} missing rows. CSV updated.")
print(f"Remaining RMSE=--- rows: {remaining}")
print(f"Total rows: {len(rows)}")
