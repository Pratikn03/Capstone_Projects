# ORIUS Incident Response Runbook

This runbook is for local/server ORIUS deployments and field-deployment preparation.
It does not certify physical actuation, road deployment, clinical deployment, or
battery field operation.

## Operating SLOs

- Safety SLO: no certificate-backed release may proceed after a failed signature,
  stale device credential, invalid fallback, or model-artifact hash mismatch.
- Availability SLO: runtime service availability is secondary to fail-closed
  behavior; emergency hold is preferred over unsafe release.
- Evidence SLO: every release event must be recorded in the append-only certificate
  audit before it is used as deployment evidence.

## Failure Budget

- Any true-state safety violation consumes the full safety failure budget and
  triggers immediate release freeze.
- Any unsigned release in staging or production consumes the full certificate
  failure budget and triggers key/config review.
- Any accepted replayed nonce, revoked device credential, or mismatched model hash
  consumes the full device/artifact integrity budget.

## Alert Triggers

- Certificate verification failure, missing signature, chain break, or append-only
  event conflict.
- Device HMAC failure bursts, stale timestamp bursts, nonce replay, revoked device
  use, or mTLS ingress failure.
- Model artifact hash mismatch, missing manifest in staging/production, or unsafe
  deserialization attempt.
- Runtime release-contract failure: invalid fallback, missing T11 status, bad
  postcondition, or failed append-only marker.
- physical actuation stop condition: CARLA/nuPlan/HIL scenario violation,
  emergency braking failure, or manual operator stop.

## Immediate Containment

1. Put affected domain into fail-closed hold.
2. Disable release for the affected model, device, certificate key, or domain
   adapter.
3. Preserve append-only audit logs, runtime traces, and deployment config snapshots.
4. Revoke impacted device credentials or certificate key IDs before restart.
5. Re-run deployment-security and release-contract validators before restoring
   service.

## Certificate Key Compromise

1. Mark the active certificate key ID compromised in the secrets backend.
2. Rotate `ORIUS_CERTIFICATE_ACTIVE_KEY_ID` to a new managed-secret key.
3. Keep old key material only for historical verification until the audit window
   closes.
4. Rebuild certificate witnesses and verify the append-only event chain.
5. Document affected command IDs, certificate hashes, and rollback decision.

Example containment command set:

```bash
export ORIUS_CERTIFICATE_ACTIVE_KEY_ID=orius-cert-YYYY-MM-emergency
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_deployment_security.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_production_readiness.py --strict
```

Do not delete the compromised key before historical witnesses are copied to the
incident evidence bundle. Disable it for new signing in the managed secret source
and keep verification-only access under the incident retention policy.

## Planned Key Rotation

1. Add the next certificate key and next per-device key IDs to the managed secret
   payload while keeping the current IDs valid.
2. Confirm strict readiness with the current `ORIUS_CERTIFICATE_ACTIVE_KEY_ID`.
3. Rotate `ORIUS_CERTIFICATE_ACTIVE_KEY_ID` to the next certificate key.
4. Provision devices to send the next `device_key_id` and confirm telemetry,
   command polling, and ACK paths verify.
5. Add retired device key IDs to `ORIUS_REVOKED_DEVICE_KEYS` after the overlap
   window and re-run the IoT HMAC tests.

Rotation validation examples:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_dc3s_certificate_full.py::TestStoreAndGet::test_signature_verification_uses_certificate_key_id_for_rotation -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_iot_device_hmac.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_production_readiness_validator.py -q
```

## Device Revocation

1. Add the device ID or device key ID to the managed revocation source used by
   `ORIUS_REVOKED_DEVICE_KEYS`.
2. If mTLS is enabled, revoke the device certificate at the ingress CA/revocation
   layer.
3. Confirm signed telemetry, command polling, and ACKs from the revoked device fail.
4. Provision a replacement credential only after device identity and firmware state
   are verified.

Revocation validation examples:

```bash
export ORIUS_REVOKED_DEVICE_KEYS='{"edge-device-001": ["edge-key-2026-01"]}'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_iot_device_hmac.py::test_revoked_iot_device_credential_fails -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_deployment_security.py
```

## mTLS Ingress Failure

ORIUS expects mTLS enforcement at the ingress layer for deployment-grade claims.
If the ingress layer reports a client-certificate failure:

1. Keep `ORIUS_REQUIRE_MTLS=1`; do not bypass the check to restore availability.
2. Confirm the device CA bundle path in `ORIUS_DEVICE_CA_BUNDLE` matches the
   deployed ingress trust store.
3. Verify the certificate subject/SAN maps to the same `device_id` inventory used
   by HMAC device keys.
4. Revoke the certificate at the ingress CA or service-mesh layer if compromise is
   suspected, then revoke the matching HMAC key ID.
5. Preserve ingress access logs, certificate serials, ORIUS request IDs, and IoT
   nonce records in the incident evidence bundle.

## Model Artifact Rollback

1. Stop loading the suspect model artifact by removing it from the active manifest.
2. Restore the previous model only if its SHA256 manifest and optional signature
   verify.
3. Re-run the domain runtime replay and release-contract witness generation.
4. Record the rollback artifact hash, reason, and validation outputs.

## Runtime Rollback

1. Roll back to the last release manifest with passing deployment-security,
   theorem, paper-claim, and reproducibility gates.
2. Re-run the mutation-guarded fast pytest tier before accepting live traffic.
3. Do not claim field deployment if CARLA/nuPlan/HIL/held-out validation gates are
   incomplete.

## Recovery Criteria

- `scripts/validate_deployment_security.py` passes.
- `scripts/validate_runtime_release_contract.py --strict` passes.
- `scripts/validate_production_readiness.py --deployment-grade` passes only for
  field/deployment claims.
- The git tree and final release manifest are clean and reproducible.

## Incident Record Template

- Incident ID:
- Start/end UTC:
- Affected domain, device IDs, certificate key IDs, and model artifact hashes:
- Triggering alert and first failing validator/test:
- Containment actions and exact commands:
- Evidence bundle paths:
- Rotation/revocation decisions:
- Recovery validators and outputs:
- Remaining deployment-scope exclusions:
