# ORIUS Framework Proof Workflow

This document is the shortest path to the **witness-first universal proof bundle** for ORIUS.

The current lock is:

1. keep **battery** as the witness row with deepest theorem-to-artifact closure,
2. keep **AV**, **industrial**, and **healthcare** as defended bounded rows only when the canonical gate passes,
3. keep **navigation** shadow-synthetic until a defended real-data row exists,
4. keep **aerospace** experimental until the promotion contract is cleared,
5. keep the universal benchmark and governance semantics identical across all rows.

## What “proof” means here

Proof in this repository is intentionally tiered:

- **battery** = witness row
- **AV** = defended bounded row
- **industrial / healthcare** = defended bounded rows
- **navigation** = shadow-synthetic
- **aerospace** = experimental

So the goal is not to claim equal-domain parity today. The goal is to generate one artifact bundle showing:

- the universal validation harness passed,
- the integrated theorem surface passed,
- the training and software-in-loop audits passed where required,
- the defended-vs-gated posture is explicit,
- and all artifact locations are recorded in one place.

## Bundle outputs

The proof bundle writes review artifacts under:

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
- `universal_validation/domain_closure_matrix.csv`
- `universal_validation/paper5_cross_domain_matrix.csv`
- `universal_validation/paper6_cross_domain_matrix.csv`

This keeps the universal review surface compact without inflating the claim boundary beyond the tracked evidence.

## Recommended command

```bash
PYTHONPATH=src python3 scripts/build_orius_framework_proof.py \
  --seeds 1 \
  --horizon 24 \
  --out reports/orius_framework_proof
```

## Make target

```bash
make framework-proof
```

Optional overrides:

```bash
make framework-proof PROOF_SEEDS=3 PROOF_HORIZON=48
```

## Why this helps the monograph

This bundle gives one compact place to answer:

- what is currently defended,
- what remains bounded or gated,
- how the canonical replay protocol was exercised across domains,
- where the theorem, training, SIL, and validation artifacts live,
- whether the ORIUS framework gate passed.
