# ORIUS Framework Proof Workflow

This document is the shortest path to a **battery-anchored, all-domain proof
bundle** for ORIUS.

The current lock is:

1. keep **battery** as the validated reference surface,
2. promote **industrial** and **healthcare** only when the canonical proof gate passes,
3. keep **AV** as a proof candidate unless the replay evidence materially improves,
4. keep **navigation** synthetic-shadow unless locked telemetry exists,
5. keep **aerospace** experimental until it earns the same gate.

---

## What “proof” means here

ORIUS proof in this repository is intentionally scoped:

- **battery** = validated reference domain,
- **industrial / healthcare** = proof-validated non-battery domains,
- **AV** = proof candidate,
- **navigation** = shadow synthetic,
- **aerospace** = experimental.

So the goal is **not** to claim every domain is fully proven.
The goal is to generate one artifact bundle showing:

- the canonical universal validation harness passed,
- the 18-theorem surface passed,
- the non-battery training surfaces verified,
- the software-in-loop replay traces passed,
- the per-domain evidence tier is explicit,
- and the artifact locations are recorded in one place.

---

## Battery-like artifacts for all domains

The bundle script writes a common set of review artifacts under:

`reports/orius_framework_proof/`

Key outputs:

- `framework_proof_manifest.json`
- `framework_proof_summary.md`
- `artifact_register.csv`
- `domain_controller_summary.csv`
- `training_audit/training_audit_report.json`
- `sil_validation/sil_validation_report.json`
- `universal_validation/validation_report.json`
- `universal_validation/proof_domain_report.json`
- `universal_validation/domain_validation_summary.csv`
- `universal_validation/tbl_domain_proof_status.tex`

This makes the universal surface easier to review without changing the core
claim boundary that battery remains the deepest reference surface.

Promotion from proof-candidate to proof-validated now requires all of:

- locked non-synthetic telemetry,
- a verified training surface,
- a clean software-in-loop replay pass,
- a nontrivial baseline gap,
- material ORIUS TSVR improvement,
- stable behavior across seeds.

---

## Recommended command

```bash
PYTHONPATH=src python3 scripts/build_orius_framework_proof.py \
  --seeds 1 \
  --horizon 24 \
  --out reports/orius_framework_proof
```

---

## Make target

```bash
make framework-proof
```

Optional overrides:

```bash
make framework-proof PROOF_SEEDS=3 PROOF_HORIZON=48
```

---

## Why this helps thesis work

This bundle is useful when writing the thesis because it gives one compact place
to answer:

- what is actually validated,
- what is only proof-candidate, shadow, or experimental evidence,
- which non-battery domains have training and SIL support,
- how the canonical replay protocol was exercised across domains,
- where the validation and theorem-gate artifacts live,
- whether the ORIUS framework gate passed.
