# ORIUS Universal Governance Specification

This document defines the final-project governance contract for ORIUS. It
covers CertOS runtime semantics, manuscript authority, claim locking, and the
tracked artifact boundary used by the thesis package.

## Manuscript authority

- canonical manuscript: `paper/paper.tex`
- narrative companion: `paper/PAPER_DRAFT.md`
- canonical metric lock: `paper/metrics_manifest.json`
- canonical claim register: `paper/claim_matrix.csv`

No thesis-facing claim should depend on ignored local caches such as
`data/dashboard/*`.

## CertOS policy surface

The governance layer is domain-neutral and parameterized by a
`DomainGovernancePolicy`:

| Hook | Purpose |
|---|---|
| `is_actuation(action)` | decides whether a step issues a meaningful plant action |
| `fallback_action(constraints, state)` | provides the safe default action under expiry or revocation |
| `horizon_update(certificate, reliability, drift, constraints, validity_horizon)` | updates or clamps certificate horizon |
| `required_certificate_fields()` | defines minimum fields for audit completeness |
| `integrity_projection(action)` | normalizes action payloads before integrity checks and hashing |

## CertOS invariants

- `INV-1`: no actuation without a valid or fallback certificate path
- `INV-2`: certificate hash chain remains unbroken
- `INV-3`: fallback activates iff runtime validity collapses

These invariants are stated over abstract actuation and certificate semantics,
not battery-only quantities.

## Canonical claim family

The thesis package now uses a `universal_safety` claim family. Canonical claim
classes include:

- benchmark schema version
- manuscript authority
- dataset profile provenance
- benchmark summary claims
- latency budget claims
- artifact provenance

Energy-impact artifacts remain useful supporting evidence, but they are not the
canonical governance surface for the thesis package.

## Backend truth surface

Tracked report and governance data is served from the backend through:

- `/research/manifest`
- `/research/region/{region}`
- `/research/reports`
- `/research/benchmark`
- `/research/governance`

Frontend routes should proxy these endpoints rather than reading local
dashboard caches directly.
