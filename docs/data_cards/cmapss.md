# Data Card: NASA C-MAPSS

## 1. Dataset name and version
- Name: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
- Splits expected by repo: `FD001` through `FD004`

## 2. Source URL and download
- Source URL: NASA Prognostics Data Repository / C-MAPSS release
- Expected raw placement:
  - `data/aerospace/raw/train_FD001.txt`
  - `data/aerospace/raw/train_FD002.txt`
  - `data/aerospace/raw/train_FD003.txt`
  - `data/aerospace/raw/train_FD004.txt`

## 3. License
- NASA public-release dataset terms

## 4. Row count
- Depends on the staged FD corpora; processed summary is written by the builder

## 5. Feature schema
- Engine-health and flight-condition channels such as altitude proxies,
  airspeed proxies, pressure, temperature, rotational speed, and bleed signals
- Processed ORIUS aerospace contract:
  - `flight_id`, `step`, `altitude_m`, `airspeed_kt`, `bank_angle_deg`,
    `fuel_remaining_pct`, `ts_utc`

## 6. Temporal extent
- Simulation-cycle trajectories, not civil-clock telemetry

## 7. Split policy
- Trainable aerospace companion row only
- Runtime defended validation remains a separate multi-flight telemetry surface

## 8. Preprocessing pipeline
- Convert FD text files into ORIUS aerospace rows
- Build processed CSV and parquet features
- Optionally pair with public ADS-B support surfaces

## 9. Known issues
- C-MAPSS is not sufficient for defended aerospace runtime promotion by itself
- Real runtime replay still requires provider-approved multi-flight telemetry

## 10. Intended ORIUS use
- Aerospace trainable degradation companion surface
- Supports model training, calibration, and bounded analysis before runtime telemetry arrives

## 11. Citation
- See `paper/bibliography/orius_monograph.bib` for C-MAPSS references.
