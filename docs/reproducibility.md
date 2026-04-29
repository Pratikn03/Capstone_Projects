# ORIUS Reproducibility Contract

ORIUS supports a large local research workspace, but the public repository must
remain reproducible from tracked source plus declared external artifacts.

## Clean-Clone Contract

1. Install dependencies from pinned lockfiles.
2. Keep raw datasets and model checkpoints outside Git.
3. Generate reports from declared scripts with fixed seeds and input manifests.
4. Validate claim-governing artifacts from compact summaries, not raw traces.
5. Finish with a clean `git status --short`.

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
.venv/bin/python scripts/validate_theorem_surface.py
.venv/bin/python scripts/validate_paper_claims.py
.venv/bin/python scripts/validate_equal_domain_artifact_discipline.py
.venv/bin/python scripts/validate_production_readiness.py
```

Before a final release, rerun `validate_reproducibility_95.py` without
`--allow-dirty` and use mutation guards around full pytest:

```bash
git diff --name-only > /tmp/orius_pre_pytest_diff.txt
COPYFILE_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/pytest -q --maxfail=25 \
  | tee reports/audit/full_pytest_low_priority.log
git diff --name-only > /tmp/orius_post_pytest_diff.txt
```
