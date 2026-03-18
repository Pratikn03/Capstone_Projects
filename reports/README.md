# Reports Output Contract

`reports/` is the canonical home for generated evaluation, publication, and
audit artifacts.

## Stable subdirectories

- `reports/coverage/` — HTML coverage outputs from pytest.
- `reports/publication/` — locked publication bundles, manifests, and release outputs.
- `reports/publish/` — audit outputs used during publish verification.
- `reports/universal_orius_validation/` — cross-domain validation reports and tables.

## Contract rules

1. Generated artifacts should be written under `reports/`, not committed as ad-hoc files elsewhere.
2. Release-family outputs should use a stable, script-owned path and file naming convention.
3. Publication-facing files should be reproducible from canonical scripts in `scripts/`.
4. Temporary scratch folders should be cleaned up or clearly prefixed as temporary.

## Canonical producers

- `scripts/build_publication_artifact.py`
- `scripts/final_publish_audit.py`
- `scripts/run_universal_orius_validation.py`
- `pytest` coverage outputs configured in `pytest.ini`
