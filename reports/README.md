# Reports Output Contract

This directory contains generated outputs produced by training, evaluation,
publication, and audit workflows. Source code should treat the paths below as
the stable, human-facing contract for where key artifacts land.

## Canonical subdirectories

- `reports/coverage/` — pytest HTML coverage output.
- `reports/publication/` — manifest-locked release-family publication bundle.
- `reports/publish/` — publication audit outputs and delta checks.
- `reports/universal_orius_validation/` — cross-domain ORIUS validation reports.
- `reports/eia930/` — US balancing-authority evaluation outputs.
- `reports/tables/` — generated tables for papers and reports.
- `reports/figures/` — generated figures for papers, docs, and dashboards.

## Usage notes

- Treat this directory as **generated output**, not canonical source.
- Prefer scripts in `scripts/` and Make targets in `Makefile` to regenerate
  artifacts instead of hand-editing files here.
- Publication-facing claims should be verified through:
  - `scripts/validate_paper_claims.py`
  - `scripts/sync_paper_assets.py --check`
  - `scripts/final_publish_audit.py`

## Common producers

- `make test-cov` writes coverage artifacts.
- `make publication-artifact RELEASE_ID=...` writes publication bundles.
- `make publish-audit` writes final audit reports.
- `PYTHONPATH=src python3 scripts/run_universal_orius_validation.py ...`
  writes universal validation outputs.
