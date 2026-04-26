# ORIUS Hugging Face Closure Jobs

These UV job templates are the controlled compute entrypoints for the active ORIUS 3-domain program.
They are designed for Hugging Face Jobs and assume the repo is checked out inside the job workspace.

Shared assumptions:
- `GRIDPULSE_REPO_ROOT` defaults to the current working directory.
- the job templates use the active Python interpreter (`sys.executable`) rather than assuming `python` is on PATH
- Repo-local raw layout under `data/<domain>/raw/<dataset_key>/` is the primary contract.
- `ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV corpora.
- Raw datasets stay off-repo and off-Hub unless licensing explicitly permits mirroring.
- For live metric logging, set `TRACKIO_PROJECT` and configure Hugging Face / Trackio credentials in the job secrets.

Suggested job entrypoints:
- `canonical_closure_refresh_job.py`
- `deep_learning_novelty_job.py`
- `calibration_diagnostics_job.py`
- `runtime_governance_trace_job.py`

Execution rule:
- `canonical_closure_refresh_job.py` is the top-level refresh entrypoint for the active Battery + AV + Healthcare lane.
- `deep_learning_novelty_job.py` is the battery witness-row deep-learning novelty entrypoint. It is explicitly bounded to learned reliability and raw-sequence forecasting evidence, not broader empirical promotion.

Inspection rule:
- Run `python scripts/hf_jobs/deep_learning_novelty_job.py --list-outputs` before launching the deep-learning novelty job when you need a non-mutating inventory of generated files. The wrapper does not download data directly; it trains from repo-local inputs and writes namespaced battery novelty artifacts plus manuscript table/figure copies.
