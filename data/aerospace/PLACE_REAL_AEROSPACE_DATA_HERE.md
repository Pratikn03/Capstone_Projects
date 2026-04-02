# Aerospace Real-Data Placement

Current status:

- `data/aerospace/processed/aerospace_orius.csv` is now expected to be built from NASA C-MAPSS.
- The canonical trainable raw source is the repo-local C-MAPSS corpus under `data/aerospace/raw/`.
- The canonical defended runtime source is a separate provider-approved multi-flight telemetry surface.
- Aerospace remains below bounded-universal closure because training parity is not the same as defended validation parity.

## Repo-local raw-data contract

Place the C-MAPSS train corpora here:

```text
data/aerospace/raw/train_FD001.txt
data/aerospace/raw/train_FD002.txt
data/aerospace/raw/train_FD003.txt
data/aerospace/raw/train_FD004.txt
```

Processed target:

- `data/aerospace/processed/aerospace_orius.csv`

Manifest:

- `data/aerospace/raw/cmapss_provenance.json`
- `data/aerospace/raw/multi_flight_runtime_contract.json`

## Runtime replay surface contract

Place the defended runtime replay source here:

```text
data/aerospace/raw/aerospace_flight_telemetry/
```

This surface is intentionally separate from C-MAPSS:

- C-MAPSS is the trainable degradation companion corpus.
- Multi-flight telemetry is the required runtime replay and safety-validation surface.
- A defended aerospace row is not closed until both surfaces exist and the replay path is locked.

## Target processed columns

- `flight_id`
- `step`
- `altitude_m`
- `airspeed_kt`
- `bank_angle_deg`
- `fuel_remaining_pct`
- `ts_utc`

The current converter also carries subset and source-signal metadata to preserve lineage.

## Important boundary

- C-MAPSS now backs the trainable aerospace row in this repo.
- Theorem-grade or equal-domain aerospace closure is still not claimed.
- Bounded-universal aerospace closure is still blocked until the multi-flight runtime replay surface is present.
- Synthetic generation remains available only by explicit `--source synthetic`.

## Related source-of-truth files

- `data/DATASET_DOWNLOAD_GUIDE.md`
- `scripts/download_aerospace_datasets.py`
- `reports/publication/orius_equal_domain_parity_matrix.csv`
