# ORIUS Reproducibility Contract

ORIUS supports a large local research workspace, but the public repository must
remain reproducible from tracked source plus declared external artifacts.

## Clean-Clone Contract

1. Install dependencies from pinned lockfiles.
2. Keep raw datasets and model checkpoints outside Git.
3. Generate reports from declared scripts with fixed seeds and input manifests.
4. Validate claim-governing artifacts from compact summaries, not raw traces.
5. Keep every `scripts/*.py` referenced by Makefile/docs/tests/another script, or classify it in `configs/script_registry.yml`.
6. Finish with a clean `git status --short`.

## Required Environment Defaults

```bash
export COPYFILE_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
```

Production or deployment-like runs must additionally set:

```bash
export ORIUS_ENV=production
export ORIUS_API_KEYS='{"<key>":["read","write","admin"]}'
export ORIUS_CERTIFICATE_SIGNING_KEY='<32+ byte secret from secret management>'
export ORIUS_REQUIRE_MODEL_HASH=1
```

The test-only auth bypass is valid only with `ORIUS_ENV=test` or inside pytest.

## Validation Ladder

```bash
.venv/bin/python scripts/validate_generated_artifact_policy.py
.venv/bin/python scripts/validate_no_appledouble.py --exclude-active
.venv/bin/python scripts/validate_api_auth_coverage.py
.venv/bin/python scripts/validate_reproducibility_95.py --allow-dirty
.venv/bin/python scripts/validate_metric_consistency.py
.venv/bin/python scripts/validate_certificate_schema.py
.venv/bin/python scripts/validate_theorem_promotion.py
.venv/bin/python scripts/validate_theorem_surface.py
.venv/bin/python scripts/validate_utility_preserving_safety.py
.venv/bin/python scripts/validate_paper_claims.py
.venv/bin/python scripts/validate_equal_domain_artifact_discipline.py
.venv/bin/python scripts/validate_production_readiness.py
```

## Clean-Clone Claim-Governing Tables

After dependency installation, a clean clone can regenerate the manuscript-facing
claim-governing runtime tables without raw datasets, model checkpoints, or heavy
artifact builders:

```bash
make claim-governing-tables
```

This reads only tracked compact publication evidence:

- `reports/publication/three_domain_ml_benchmark.csv`
- `reports/publication/three_domain_forecast_calibration_runtime_evidence.csv`

It refreshes:

- `reports/publication/claim_governing_three_domain_runtime_evidence.tex`
- `reports/publication/final_runtime_safety_for_paper.csv`
- `reports/publication/tbl_final_runtime_safety.tex`

## Theorem Authority

The current proof-strength authority is the promotion package:

- `reports/publication/theorem_promotion_matrix.json`
- `reports/publication/theorem_result_cards/*.json`
- `scripts/validate_theorem_promotion.py`

The older `active_theorem_audit.*` files are retained as historical traceability
and registry-drift diagnostics. They are not the current promotion authority.

## Utility-Preserving Safety Gate

The final release includes the utility-preserving safety scorecard because it
separates ORIUS from trivial shutdown/always-alert/always-brake baselines. Run:

```bash
make utility-preserving-safety-verify
```

This rebuilds and validates `reports/publication/utility_preserving_safety_scorecard.*`.

## Script Governance

No new Python script should enter the release surface silently. A script is valid
only if it is referenced by Makefile/docs/tests/another tracked script, or if it
has an explicit entry in `configs/script_registry.yml` with a status and reason.

## Deferred Monolith Split Order

Large-module splitting is intentionally deferred until the paper/theorem freeze
is stable. The split order is:

1. Forecasting training.
2. AV runtime.
3. Three-domain artifact builder.
4. Report builder.
5. IoT runner.
6. Pipeline runner.
7. Theorem guarantees.

Before a final release, rerun `validate_reproducibility_95.py` without
`--allow-dirty` and use mutation guards around full pytest:

```bash
git diff --name-only > /tmp/orius_pre_pytest_diff.txt
COPYFILE_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/pytest -q --maxfail=25 \
  | tee reports/audit/full_pytest_low_priority.log
git diff --name-only > /tmp/orius_post_pytest_diff.txt
```
