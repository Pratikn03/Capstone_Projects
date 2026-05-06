# Production Readiness Summary

Project: ORIUS

## Phase 1 — Production Hardening
- **Config validation**: `scripts/validate_configs.py` validates `configs/*.yaml` with pydantic schemas.  
- **.env.example**: new template for secrets and runtime settings (`.env` is git‑ignored).  
- **Structured logging**: `ORIUS_LOG_FORMAT=json` enables JSON logs across scripts + API.  
- **Health/Readiness probes**: `/health` and `/ready` endpoints for API; compose + k8s probes wire in.  
- **Retries for downloads**: OPSD/SMARD/Open‑Meteo/ElectricityMaps/WattTime use shared retryable HTTP sessions.  
- **Deployment gate**: `scripts/validate_production_readiness.py --strict` fails closed unless API keys, model-hash enforcement, signed certificate provenance, and promoted runtime surfaces are available.
- **Certificate provenance**: DC3S certificates support rotated `HMAC-SHA256` signatures through `ORIUS_CERTIFICATE_KEYS` and append-only certificate events. Unsigned certificates remain acceptable only for bounded research/offline validation.
- **Model provenance**: production/staging model loading refuses pickle bundles without a sha256 sidecar or manifest before deserialization.
- **Device identity**: IoT telemetry, command polling, and ACK paths support per-device HMAC identity with timestamp-skew and nonce-replay checks.

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

## Local Mac / Server Security Profile
Set these variables for a local server-style deployment:

```bash
export ORIUS_ENV=production
export ORIUS_SECRETS_FILE=/absolute/path/outside/git/orius-secrets.yaml
export ORIUS_CERTIFICATE_ACTIVE_KEY_ID=orius-cert-2026-01
export ORIUS_REQUIRE_CERT_SIGNATURE=1
export ORIUS_REQUIRE_DEVICE_SIGNATURE=1
export ORIUS_REQUIRE_ARTIFACT_MANIFEST=1
export ORIUS_REQUIRE_MODEL_HASH=1
```

Use `configs/secrets.example.yaml` only as a template. The real `ORIUS_SECRETS_FILE` must stay outside Git and should define `certificate_keys`, `device_keys`, and operator API keys. For environment-only deployment, `ORIUS_CERTIFICATE_KEYS` and `ORIUS_DEVICE_KEYS` may also be JSON mappings.

Required gates before calling this profile production-ready:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_deployment_security.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_runtime_release_contract.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_api_auth_coverage.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_certificate_schema.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_paper_claims.py
```

This profile is fail-closed for certificate signing and device identity. If keys are missing in `production` or `staging`, release paths should fail instead of emitting unsigned evidence.

## Notes
- External tokens are not stored; use `.env` or environment variables.  
- Robust dispatch uses quantile heuristics; scenario methods are optional future upgrades.  
- Deployment-ready claims still require external environment controls, operational key rotation procedures, cloud/KMS or equivalent secret management, and domain-specific field/HIL validation beyond the current predeployment evidence.
