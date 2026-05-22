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
- **Model provenance**: production/staging model loading refuses pickle/joblib/torch bundles without a sha256 sidecar or manifest before deserialization; strict torch loading refuses unsafe fallback when `weights_only=True` is unsupported.
- **Device identity**: IoT telemetry, command polling, and ACK paths support per-device HMAC identity with timestamp-skew, nonce-replay checks, revocation, and mTLS ingress gating for deployment-grade profiles.

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

Strict profile expectations enforced by `scripts/validate_production_readiness.py --strict`:

- `ORIUS_ENV` must be `production` or `staging`; strict mode is not a dev/test alias.
- `ORIUS_REQUIRE_CERT_SIGNATURE=1`, `ORIUS_REQUIRE_DEVICE_SIGNATURE=1`, `ORIUS_REQUIRE_ARTIFACT_MANIFEST=1`, and `ORIUS_REQUIRE_MODEL_HASH=1` must be explicit, even though some runtime helpers also fail closed when `ORIUS_ENV=production`.
- `ORIUS_CERTIFICATE_ACTIVE_KEY_ID` must identify the current certificate signing key so release witnesses are auditable across rotations.
- `device_keys` must be available from `ORIUS_SECRETS_FILE`, `ORIUS_DEVICE_KEYS`, or the managed secret command; unsigned IoT telemetry, command polling, and ACKs are not acceptable in this profile.
- This strict profile still does not prove field deployment readiness. It is a local/server production-security floor for release hardening and predeployment evidence generation.

## Device Identity and mTLS-Ready Boundary

ORIUS currently authenticates IoT request bodies with per-device `HMAC-SHA256` fields: `device_id`, `device_key_id`, `device_ts_utc`, `device_nonce`, and `device_signature`. The runtime rejects stale timestamps, replayed nonces, missing device keys, revoked credentials, and invalid signatures when device signatures are required.

For mTLS, ORIUS is ingress-ready rather than a TLS terminator. A deployment should terminate client-certificate TLS at a load balancer, reverse proxy, or service mesh that:

- trusts only the provisioned device CA bundle referenced by `ORIUS_DEVICE_CA_BUNDLE`;
- maps the verified client certificate subject/SAN to the same `device_id` inventory used for HMAC keys;
- strips any inbound identity headers before setting trusted internal identity headers;
- blocks requests with missing, expired, revoked, or mismatched client certificates before they reach the API;
- leaves HMAC enabled as an application-layer proof to cover forwarded requests, replay detection, and command ACK integrity.

`ORIUS_REQUIRE_MTLS=1` is therefore a deployment-grade gate that asserts the external ingress layer is mandatory and configured; it is not evidence that this Python process implements TLS termination.

## Key Rotation Posture

Certificate keys and device keys should use overlapping current/next IDs. Keep at least one standby key in the managed secret source before changing the active ID:

1. Add `orius-cert-YYYY-MM-next` and new per-device key IDs to the managed secret payload.
2. Deploy with both old and new keys present; keep `ORIUS_CERTIFICATE_ACTIVE_KEY_ID` on the old key while verifying signatures against both IDs.
3. Rotate `ORIUS_CERTIFICATE_ACTIVE_KEY_ID` to the new key and begin provisioning devices with the new `device_key_id`.
4. Move retired device key IDs to `revoked_device_keys` after the overlap window.
5. Keep old certificate keys available only for historical verification until the audit retention window closes; do not use them for new release witnesses.

Recommended rotation checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_dc3s_certificate_full.py::TestStoreAndGet::test_signature_verification_uses_certificate_key_id_for_rotation -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_iot_device_hmac.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_production_readiness_validator.py -q
```

## Deployment-Grade / Field Claim Profile

Do not use local env-file secrets for field/deployment claims. Start from the strict local/server profile above, then add a managed secret source, device certificate ingress, final release manifests, and completed domain validation gates:

```bash
export ORIUS_SECRET_BACKEND=external_command   # or aws_kms/gcp_kms/azure_key_vault/hsm/vault
export ORIUS_SECRETS_COMMAND="/absolute/path/secret-provider --format json"
export ORIUS_REQUIRE_CERT_SIGNATURE=1
export ORIUS_REQUIRE_DEVICE_SIGNATURE=1
export ORIUS_REQUIRE_ARTIFACT_MANIFEST=1
export ORIUS_REQUIRE_MODEL_HASH=1
export ORIUS_REQUIRE_MTLS=1
export ORIUS_DEVICE_CA_BUNDLE=/absolute/path/device-ca.pem
export ORIUS_REVOKED_DEVICE_KEYS='{"edge-device-001": ["edge-key-2026-01"]}'
export ORIUS_RELEASE_MANIFEST=/absolute/path/predeployment_release_manifest.json
```

The deployment-grade gate is intentionally stricter than local/server research mode:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_production_readiness.py \
  --deployment-grade \
  --release-manifest "$ORIUS_RELEASE_MANIFEST"
```

This gate should fail until an implemented managed secret source, mTLS/device provisioning, strict artifact manifests, the `orius_95_validation_manifest.json`, an explicit current `predeployment_release_manifest.json`, signed append-only release witnesses, and a clean Git tree are all present. Historical freeze manifests are not accepted by glob. That failure is correct if CARLA/nuPlan/HIL, healthcare held-out validation, or physical actuation evidence remains incomplete.

Cloud/KMS/HSM/Vault labels are not enough by themselves in this local repo. Until a cloud provider adapter exists, deployment-grade validation expects `ORIUS_SECRET_BACKEND=external_command` and refuses certificate/device secrets provided directly through env vars or local secret files.

Operational runbooks live in `docs/incident_response.md` and cover SLOs, failure budgets, rollback, certificate key compromise, device revocation, model artifact rollback, and physical actuation stop conditions.

## Notes
- External tokens are not stored; use `.env`, local files outside Git, or managed secret backends.  
- Robust dispatch uses quantile heuristics; scenario methods are optional future upgrades.  
- Deployment-ready claims still require external environment controls, operational key rotation procedures, cloud/KMS/HSM or equivalent secret management, device provisioning/revocation/mTLS, and domain-specific field/HIL validation beyond the current predeployment evidence.
