# ORIUS Hugging Face Closure Jobs

These UV job templates are the controlled compute entrypoints for the ORIUS 93+ closure program.
They are designed for Hugging Face Jobs and assume the repo is checked out inside the job workspace.

Shared assumptions:
- `GRIDPULSE_REPO_ROOT` defaults to the current working directory.
- Repo-local raw layout under `data/<domain>/raw/<dataset_key>/` is the primary contract.
- `ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV, navigation, or runtime telemetry corpora.
- Raw datasets stay off-repo and off-Hub unless licensing explicitly permits mirroring.
- For live metric logging, set `TRACKIO_PROJECT` and configure Hugging Face / Trackio credentials in the job secrets.

Suggested job entrypoints:
- `navigation_realdata_closure_job.py`
- `aerospace_flight_closure_job.py`
- `calibration_diagnostics_job.py`
- `runtime_governance_trace_job.py`
