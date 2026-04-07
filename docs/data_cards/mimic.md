# Data Card: MIMIC / Credentialed ICU Data Boundary

## 1. Dataset name and version
- Name: MIMIC-III / MIMIC-IV credentialed ICU monitoring surface
- Repo posture: documented target surface; defended accessible row currently uses BIDMC-style PhysioNet CSV corpora

## 2. Source URL and download
- MIMIC portal: [https://physionet.org/content/mimiciii/](https://physionet.org/content/mimiciii/)
- Credentialed access required; not stored in this git repository

## 3. License
- PhysioNet credentialed data-use agreement

## 4. Row count
- Not fixed in-repo; depends on credentialed extraction policy

## 5. Feature schema
- Typical ICU signals include heart rate, SpO2, respiratory rate, blood pressure, and waveform-derived monitoring features
- Current defended repo-compatible healthcare rows live under `data/healthcare/processed/`

## 6. Temporal extent
- ICU stay and bedside-monitoring dependent

## 7. Split policy
- Must be time-aware and patient-disjoint
- Current accessible proxy row uses deterministic split artifacts in `data/healthcare/processed/`

## 8. Preprocessing pipeline
- Credentialed extraction or approved proxy CSV ingestion
- Timestamp normalization
- Monitoring-feature engineering
- ORIUS healthcare contract export

## 9. Known issues
- MIMIC is not staged in-repo because of access controls
- Current defended healthcare row is accessible and governed, but not a public full-MIMIC mirror

## 10. Intended ORIUS use
- Long-term credentialed ICU validation target
- Supports healthcare-domain expansion beyond the current accessible monitoring row

## 11. Citation
- See `paper/bibliography/orius_monograph.bib` for MIMIC and ICU-monitoring references.
