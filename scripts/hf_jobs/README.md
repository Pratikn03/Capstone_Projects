# ORIUS Hugging Face Closure Jobs

These UV job templates are the controlled compute entrypoints for the ORIUS 93+ closure program.
They are designed for Hugging Face Jobs and assume the repo is checked out inside the job workspace.

Shared assumptions:
- `GRIDPULSE_REPO_ROOT` defaults to the current working directory.
- the job templates use the active Python interpreter (`sys.executable`) rather than assuming `python` is on PATH
- Repo-local raw layout under `data/<domain>/raw/<dataset_key>/` is the primary contract.
- `ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV, navigation, or runtime telemetry corpora.
- Raw datasets stay off-repo and off-Hub unless licensing explicitly permits mirroring.
- For live metric logging, set `TRACKIO_PROJECT` and configure Hugging Face / Trackio credentials in the job secrets.

Suggested job entrypoints:
- `canonical_closure_refresh_job.py`
- `navigation_realdata_closure_job.py`
- `aerospace_flight_closure_job.py`
- `aerospace_public_adsb_runtime_job.py`
- `deep_learning_novelty_job.py`
- `calibration_diagnostics_job.py`
- `runtime_governance_trace_job.py`

Execution rule:
- `canonical_closure_refresh_job.py` is the top-level two-lane refresh entrypoint. It refreshes the official canonical artifacts first and keeps any supplemental Hugging Face support evidence visibly separate so it cannot mutate the parity gate by accident.
- `aerospace_public_adsb_runtime_job.py` is the bounded public-flight support entrypoint. It may deepen aerospace runtime evidence, but it must never be treated as a substitute for the official provider-approved multi-flight telemetry lane.
- `deep_learning_novelty_job.py` is the battery witness-row deep-learning novelty entrypoint. It is explicitly bounded to learned reliability and raw-sequence forecasting evidence, not equal-domain promotion.
