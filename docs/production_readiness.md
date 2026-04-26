# Production Readiness Summary

Project: ORIUS

## Phase 1 — Production Hardening
- **Config validation**: `scripts/validate_configs.py` validates `configs/*.yaml` with pydantic schemas.  
- **.env.example**: new template for secrets and runtime settings (`.env` is git‑ignored).  
- **Structured logging**: `ORIUS_LOG_FORMAT=json` enables JSON logs across scripts + API.  
- **Health/Readiness probes**: `/health` and `/ready` endpoints for API; compose + k8s probes wire in.  
- **Retries for downloads**: OPSD/SMARD/Open‑Meteo/ElectricityMaps/WattTime use shared retryable HTTP sessions.  
- **Deployment gate**: `scripts/validate_production_readiness.py --strict` fails closed unless API keys, model-hash enforcement, signed certificate provenance, and promoted runtime surfaces are available.
- **Certificate provenance**: DC3S certificates support `HMAC-SHA256` signatures via `ORIUS_CERTIFICATE_SIGNING_KEY`; unsigned certificates remain acceptable only for bounded research/offline validation.
- **Model provenance**: production/staging model loading refuses pickle bundles without a sha256 sidecar or manifest before deserialization.

## Phase 2 — Operations
- **Monitoring + alerting**: `scripts/run_monitoring.py` writes `reports/monitoring_summary.json` and can alert via `ORIUS_ALERT_WEBHOOK`.  
- **Scheduled retraining**: `scripts/retrain_if_needed.py --refresh` retrains only when drift triggers.  
- **Artifact registry**: `scripts/register_models.py` writes `artifacts/registry/models.json`.  
- **Rollback**: set explicit model paths in `configs/forecast.yaml` to pin/rollback to a previous bundle.  
- **Approvals**: deploy workflow uses GitHub `production` environment (supports required reviewers).  

## Phase 3 — Deployment
- **Docker compose**: `docker/docker-compose.yml` now includes healthchecks, env file, and mounted volumes.  
- **systemd**: example units in `deploy/systemd/` for API, dashboard, and retraining timer.  
- **Kubernetes**: manifests in `deploy/k8s/` with readiness/liveness probes.  
- **CI**: GitHub Actions runs lint (syntax), tests, and package build.  
- **Release bundle**: `scripts/build_release_bundle.py` packages reports + run snapshot into `artifacts/submission_bundle_<run_id>`.  
- **AWS ECS Fargate**: templates in `deploy/aws/` for services, scheduled tasks, and observability.  

## Production Target (selected)
- Cloud: AWS ECS Fargate  
- SLO: 99.9% uptime, p95 latency < 500ms  
- Refresh cadence: weekly  
- Retrain cadence: weekly  

## Current Data Scope (recommended)
- Dataset: OPSD Germany (hourly)  
- Signals: OPSD day‑ahead price + **SMARD hourly carbon intensity** (`data/raw/carbon_signals.csv`)  
- Artifacts: `artifacts/runs/<run_id>` with `manifest.json` + `pip_freeze.txt`  

## Notes
- External tokens are not stored; use `.env` or environment variables.  
- Robust dispatch uses quantile heuristics; scenario methods are optional future upgrades.  
- Deployment-ready claims still require external environment controls, key rotation, signed model manifests, device attestation for IoT paths, and domain-specific field/HIL validation beyond the current predeployment evidence.
