# Adding New Datasets to GridPulse

This guide explains how to add a new dataset to the unified training pipeline.

## Quick Start

1. **Create config file**: Copy the template to create your dataset config
2. **Register dataset**: Add entry to `DATASET_REGISTRY` in `train_dataset.py`
3. **Run training**: `make train-dataset DATASET=YOUR_DATASET`

---

## Step 1: Create Configuration File

Copy the template and customize for your dataset:

```bash
cp configs/train_forecast_template.yaml configs/train_forecast_newdata.yaml
```

Key sections to customize:
- `dataset`: Name and metadata
- `data`: Input/output paths
- `targets`: Variables to forecast
- `features`: Feature engineering settings
- `output`: Where to save results

---

## Step 2: Register the Dataset

Edit `scripts/train_dataset.py` and add to `DATASET_REGISTRY`:

```python
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "DE": DatasetConfig(...),
    "US": DatasetConfig(...),
    
    # Add your new dataset here
    "ERCOT": DatasetConfig(
        name="ERCOT",
        display_name="US EIA-930 (ERCOT)",
        config_file="configs/train_forecast_ercot.yaml",
        features_path="data/processed/ercot/features.parquet",
        splits_path="data/processed/ercot/splits",
        reports_dir="reports/ercot",
        raw_data_path="data/raw/ercot",
        feature_module="gridpulse.data_pipeline.build_features_eia930",
        ba_code="ERCO",  # EIA-930 BA code
    ),
}
```

---

## Step 3: Prepare Raw Data

Place your raw data in the expected location:

```
data/raw/your_dataset/
├── load.csv           # Load data
├── generation.csv     # Generation data (optional)
├── weather.csv        # Weather data (optional)
└── prices.csv         # Price data (optional)
```

---

## Step 4: Run Training

Use the unified training script:

```bash
# Train single dataset
make train-dataset DATASET=ERCOT

# Or with Python directly
python scripts/train_dataset.py --dataset ERCOT

# Train with hyperparameter tuning
python scripts/train_dataset.py --dataset ERCOT --tune

# Train all datasets
python scripts/train_dataset.py --all
```

---

## Step 5: Verify Outputs

Check that these artifacts were created:

```
artifacts/
├── models_ercot/
│   ├── load_mw_gbm.pkl
│   ├── solar_mw_gbm.pkl
│   └── wind_mw_gbm.pkl
└── backtests/
    └── ercot/

reports/ercot/
├── forecast_metrics.csv
├── forecast_intervals.csv
└── evaluation_report.md
```

---

## Current Registered Datasets

| Key | Name | Config File |
|-----|------|-------------|
| DE | German OPSD | `configs/train_forecast.yaml` |
| US | US EIA-930 MISO | `configs/train_forecast_eia930.yaml` |

To see all datasets:
```bash
python scripts/train_dataset.py --list
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 train_dataset.py                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           DATASET_REGISTRY                       │   │
│  │  ┌─────┐  ┌─────┐  ┌───────┐  ┌─────────┐      │   │
│  │  │ DE  │  │ US  │  │ ERCOT │  │ NEW ... │      │   │
│  │  └──┬──┘  └──┬──┘  └───┬───┘  └────┬────┘      │   │
│  └─────┼────────┼─────────┼───────────┼──────────┘   │
└────────┼────────┼─────────┼───────────┼──────────────┘
         │        │         │           │
         ▼        ▼         ▼           ▼
┌─────────────────────────────────────────────────────────┐
│            Same Training Logic for All                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 1. Features  │→ │ 2. Splits    │→ │ 3. Train     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                            ↓                            │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ 5. Reports   │← │ 4. Conformal │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Feature building fails
- Verify raw data exists in expected location
- Check date format matches (ISO 8601)
- Ensure required columns exist

### Training fails
- Check GPU/CPU availability for deep learning
- Verify sufficient memory
- Try reducing `n_trials` in config

### Reports missing conformal coverage
- Ensure `uncertainty.enabled: true` in config
- Check backtest files exist in `artifacts/backtests/`

---

## Contact

Questions? See [ARCHITECTURE.md](../docs/ARCHITECTURE.md) or raise an issue.
