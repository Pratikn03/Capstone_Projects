# ORIUS 95+ Code Quality And Reproducibility Audit

Generated: 2026-04-26

## Executive Status

This hardening pass moves ORIUS closer to a production-quality research
repository by making artifact policy, reproducibility, API auth coverage, model
provenance, certificate signing, and dashboard evidence boundaries executable.

Current status is **improved but not final 95+ release complete** because full
pytest has not been run after all changes, monolith refactors remain future
work, and final clean release validation requires committing or otherwise
settling the current worktree.

## Implemented Gates

- Generated-artifact policy blocks local datasets, model binaries, runtime
  traces, DuckDBs, caches, screenshots, and AppleDouble sidecars from Git.
- Reproducibility gate checks required lockfiles, canonical evidence surfaces,
  pytest marker separation, AppleDouble absence, and artifact policy.
- API auth coverage gate enumerates both FastAPI apps and fails if non-health
  routes lack API-key security.
- DC3S certificate release signs certificates when a signing key is configured
  and can fail closed when signatures are required.
- Forecasting model loading requires hashes in production-like environments and
  uses `weights_only=True` for Torch checkpoints when supported.
- Dashboard live DC3S fallback is marked as degraded local shadow evidence, not
  certificate-backed live evidence.

## Known Remaining Work

- Run mutation-guarded full pytest after committing/stabilizing generated files.
- Continue reducing large monoliths in training, DC3S orchestration, and pipeline
  execution into pure libraries with thin adapters.
- Pin Docker base images by digest and enforce hash-locked Python installs in CI.
- Keep NuPlan/CARLA/healthcare held-out validation as bounded evidence until
  real runtime artifacts and manifests exist.

## Verification Snapshot

- `scripts/validate_generated_artifact_policy.py`: PASS
- `scripts/validate_no_appledouble.py --exclude-active`: PASS
- `scripts/validate_api_auth_coverage.py`: PASS
- `scripts/validate_reproducibility_95.py --allow-dirty`: PASS
- `scripts/validate_metric_consistency.py`: PASS
- `scripts/validate_certificate_schema.py`: PASS
- `scripts/validate_production_readiness.py`: PASS with expected missing-secret warnings
- `scripts/validate_theorem_surface.py`: PASS
- `scripts/validate_paper_claims.py`: PASS
- `scripts/validate_equal_domain_artifact_discipline.py`: PASS
- Targeted pytest for artifact policy, auth coverage, certificate signing,
  runtime auth, interval API, universal API, and ORIUS compatibility API: PASS
