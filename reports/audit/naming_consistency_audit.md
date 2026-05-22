# ORIUS Naming Consistency Audit
This audit inventories Python functions/classes/parameters, script/config filenames, report/artifact filenames, and naming risks. It was refreshed after the local cleanup and risk-classification pass.
## Scope
- Python files scanned: 830
- Config files scanned: 35
- Functions/methods scanned: 6205
- Classes scanned: 557
- Parameters scanned: 9375
- True naming issues: 0
- Unclassified naming risks: 0
- Classified naming risks/design debt rows: 23
- Legacy duplicate-function risk rows from previous scan: 14
- Risks requiring later implementation: 0
- Naming/schema risks implemented in this pass: 6
- Risks accepted as protocol/math/API exceptions: 7
- Risks deferred to opportunistic refactors: 10
- Intentional notation/API exceptions documented: 100

## Cleanup Completed
- Renamed private forecasting helpers to feature-oriented snake_case names.
- Renamed five local universal-contract test helper classes to CapWords.
- Removed the stale domain-validation marker and AppleDouble sidecars found during cleanup.
- Added `reports/audit/naming_contract.md` as the canonical naming/schema contract.
- Added `reports/audit/naming_risk_resolution.csv` so every current naming risk has an explicit disposition.

## Result
There are now no remaining true local symbol/file naming issues in the scanned source/report surface, and there are no unclassified naming risks. The high-impact schema/domain/model/CLI/window-unit naming issues are implemented with additive compatibility aliases. Remaining naming work is intentionally deferred private-helper cleanup or accepted protocol/math notation.

## Canonical Contract
- Domains: `battery`, `vehicle`, `healthcare`.
- Datasets: `DE_OPSD`, `AV_NUPLAN_ALLZIP_GROUPED`, `MIMIC3_VITALS`.
- Model identity fields: `model_family`, `estimator`, `role`, `target`, `dataset`, `run_id`, `selected_for_release`.
- Compatibility aliases: `DE`, `AV`, `HEALTHCARE`, `baseline_gbm`, `quantile_gbm`, `gbm_lightgbm`.

## Classified Design Findings
1. **Metrics/report schema names are implemented in the model-quality gate.** `challenger_metrics` rows are surfaced as candidate model rows, and canonical fields include `target`, `model_family`, `estimator`, `role`, `dataset`, `run_id`, and `selected_for_release`.
2. **Domain and dataset names are implemented in the registry and gate output.** Canonical outputs use `domain=battery, dataset=DE_OPSD`; `domain=vehicle, dataset=AV_NUPLAN_ALLZIP_GROUPED`; and `domain=healthcare, dataset=MIMIC3_VITALS`, while `DE`, `AV`, and `HEALTHCARE` remain compatibility aliases.
3. **Model names are implemented as additive fields.** Gate rows preserve legacy `model` keys while adding `model_family`, `estimator`, `role`, `source_model_key`, and `metric_source`.
4. **CLI flags have non-breaking aliases.** Same-line `argparse` definitions using `--out` or `--out-dir` now also accept `--output` or `--output-dir`.
5. **Training window names are implemented in canonical configs.** Battery, vehicle, and healthcare configs now expose `lookback_steps` and `horizon_steps` while preserving physical-time fields.
6. **Private repeated helpers are deferred refactors.** Repeated helpers such as `_write_csv`, `_read_csv`, `_write_json`, `_read_json`, `_load_json`, `_sha256`, `_safe_float`, and `_run` are private/local and should be renamed only when owner files are already being edited.
7. **Mathematical and protocol notation remains documented.** Names such as `H_t`, `H_X`, `T`, `K_factor`, `lipschitz_L`, `observe`, `step`, `to_dict`, and `forward` are accepted exceptions.

## Risk Resolution
- Full classification table: `reports/audit/naming_risk_resolution.csv`
- Canonical naming/schema contract: `reports/audit/naming_contract.md`

## Files Written
- `reports/audit/naming_consistency_issues.csv`
- `reports/audit/naming_intentional_exceptions.csv`
- `reports/audit/naming_inventory_summary.json`
- `reports/audit/naming_refactor_plan.csv`
- `reports/audit/naming_contract.md`
- `reports/audit/naming_risk_resolution.csv`
