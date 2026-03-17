# Aerospace Datasets for ORIUS (Placeholder)

The Aerospace domain adapter is a **placeholder** for thesis domain 4. No flight-telemetry dataset is yet integrated.

---

## Status

| Item | Status |
|------|--------|
| Adapter | ✓ `AerospaceDomainAdapter` in `universal_framework/aerospace_adapter.py` |
| Dataset | ❌ None |
| Training | ❌ N/A |
| ORIUS-Bench track | ❌ N/A |

---

## Future Datasets (Candidates)

| Dataset | Source | Format | Best for | Access |
|---------|--------|--------|----------|--------|
| **OpenSky** | opensky-network.org | CSV/API | ADS-B flight tracks, altitude, speed | API key |
| **FlightAware** | flightaware.com | CSV/API | Real-time flight data | API key |
| **Synthetic** | ORIUS script | ORIUS format | Immediate use | TBD |

---

## ORIUS Format (Target)

When a dataset is added, output columns should include:

| Column | Type | Description |
|--------|------|-------------|
| flight_id | str | Flight identifier |
| step | int | Time step index |
| altitude_m | float | Altitude (m) |
| airspeed_kt | float | Airspeed (knots) |
| bank_angle_deg | float | Bank angle (degrees) |
| fuel_remaining_pct | float | Fuel remaining (%) |
| ts_utc | str | Timestamp (ISO8601) |

---

## See Also

- `orius-plan/AV_DATASETS.md` — AV trajectories
- `orius-plan/INDUSTRIAL_DATASETS.md` — Process control
- `orius-plan/HEALTHCARE_DATASETS.md` — Vital signs
