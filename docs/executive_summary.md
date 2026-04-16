# ORIUS — Executive Summary

> One-page submission overview for committee, reviewers, and industry audiences.

## What ORIUS Is

ORIUS (Observation–Reality Integrity for Universal Safety) is a typed runtime safety layer for Physical AI systems operating under degraded observation. When sensors stall, telemetry drops, or calibration drifts, the gap between what a controller *believes* and what the physical state *actually is* can grow unboundedly. ORIUS closes this Observation–Action Safety Gap (OASG) through a five-stage kernel: **Detect → Calibrate → Constrain → Shield → Certify** (DC3S).

## What This Submission Proves

| Domain | Tier | Key Result | Evidence |
|--------|------|------------|----------|
| **Battery (BESS)** | Reference witness | TSVR = 0.0% across 4 controllers, 3 fault families, 288 steps. Zero true-state violations. | 28 SHA256-locked artifacts |
| **Autonomous Vehicles** | Bounded proof-validated | TSVR reduced from 66.0% (baseline) to 62.8% (ORIUS). 9,348 steps, 228 scenarios, 480 OASG cases closed. | 40 SHA256-locked artifacts |

- **11 theorems** proven in full generality over the typed contract algebra (T1–T11), with empirical confirmation on both domains.
- **Same kernel binary** for both domains — domain specifics confined to a 6-method adapter.
- **68 artifacts** total, all SHA256-verified with zero mismatches at freeze time.
- Full CertOS certificate chain valid for both domains (288 + 9,348 certificates).

## What This Submission Does NOT Claim

- Industrial, Healthcare, Navigation, Aerospace domains have **no locked pipeline artifacts**. They appear only as future work.
- AV result is a **bounded proof-of-mechanism** (longitudinal TTC contract), not full autonomous-driving closure.
- Shift-aware guarantees are bounded to observed drift rates; adversarial attacks have not been evaluated.
- Real-time deployment requires ~3x latency reduction (350ms → <100ms).
- No statement claims equal maturity across all domains.

## Evidence Architecture

```
reports/battery_av/overall/
  ├── release_summary.json          ← master 2-domain ledger
  ├── publication_closure_override.json  ← tier assignments
  ├── evidence_freeze_certificate.json   ← SHA256 freeze binding
  └── battery_av_manifest.json

reports/battery_av/battery/          ← 28 battery artifacts
reports/orius_av/full_corpus/        ← 40 AV artifacts
```

## Reproducibility

Single-command pipeline: `make pipeline-full`
Cold-start guide: `ORIUS_REPRODUCIBILITY.md`
Locked dependencies: `requirements.lock.txt`

## Deliverables

| Item | Format | Status |
|------|--------|--------|
| IEEE Paper | 47+ page detailed manuscript | `paper/ieee/` |
| Thesis | 400+ page monograph | `orius_battery_409page_figures_upgraded_main.tex` |
| Claim Ledger | 3-bucket discipline doc | `docs/claim_ledger.md` |
| Evidence Freeze | JSON certificate | `reports/battery_av/overall/evidence_freeze_certificate.json` |
| Artifact Manifests | SHA256-locked JSON | Both domain manifests verified |
