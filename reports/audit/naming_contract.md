# ORIUS Naming And Schema Contract

This contract defines the canonical names to use in future ORIUS/GridPulse reports,
validators, configs, and generated audit artifacts. It is audit-only: it documents
the target vocabulary and compatibility aliases without changing scientific
metrics, trained models, paper figures, public script paths, or runtime behavior.

## Canonical Domains

| Canonical domain | Canonical dataset | Compatibility aliases | Notes |
| --- | --- | --- | --- |
| `battery` | `DE_OPSD` | `DE`, `energy`, `OPSD` | Use `battery` for runtime/application domain and `DE_OPSD` for the data source. |
| `vehicle` | `AV_NUPLAN_ALLZIP_GROUPED` | `AV`, `autonomous_vehicle`, `nuPlan`, `Waymo`, `nuplan_allzip_grouped` | Use `vehicle` for runtime/application domain and reserve AV aliases for backward compatibility. |
| `healthcare` | `MIMIC3_VITALS` | `HEALTHCARE`, `MIMIC`, `MIMIC3` | Use `healthcare` for runtime/application domain and `MIMIC3_VITALS` for the data source. |

## Canonical Model Identity Fields

New or regenerated model-quality records should expose model identity as fields,
not only as overloaded names embedded in file paths or JSON keys.

| Field | Meaning | Example |
| --- | --- | --- |
| `domain` | Canonical application/runtime domain. | `battery` |
| `dataset` | Canonical dataset identifier. | `DE_OPSD` |
| `target` | Prediction or control target. | `load_mw` |
| `model_family` | Statistical or neural model family. | `gbm` |
| `estimator` | Concrete estimator or implementation. | `lightgbm` |
| `role` | Release role in the experiment. | `baseline`, `candidate`, `incumbent`, `release` |
| `run_id` | Stable run identifier for generated artifacts. | `de_strict_battery_cpu_20260522T012759Z` |
| `selected_for_release` | Whether this row is the release-selected model for the target. | `true` |

## Compatibility Aliases

These names remain valid for existing artifacts and public CLIs. They should be
resolved to canonical fields by report builders and validators before strict
quality-gate decisions.

| Alias | Canonical interpretation |
| --- | --- |
| `DE` | `domain=battery`, `dataset=DE_OPSD` |
| `AV` | `domain=vehicle`, `dataset=AV_NUPLAN_ALLZIP_GROUPED` |
| `HEALTHCARE` | `domain=healthcare`, `dataset=MIMIC3_VITALS` |
| `baseline_gbm` | `model_family=gbm`, `role=baseline` |
| `quantile_gbm` | `model_family=gbm`, `role=quantile` |
| `gbm_lightgbm` | `model_family=gbm`, `estimator=lightgbm` |

## Metrics Schema Direction

Strict model-quality gates should be able to enumerate every enabled
domain-target-model row without inspecting nested challenger-only fields. Future
schema work should expose retained and challenger models as first-class rows and
record retention metadata separately:

- `target`
- `model_family`
- `estimator`
- `role`
- `dataset`
- `run_id`
- `selected_for_release`
- `retention_decision`
- `retention_reason`
- `source_model_key`

Existing `challenger_metrics` fields remain compatibility inputs until the schema
implementation phase promotes all challenger rows to the canonical row surface.

## CLI Naming Direction

Do not directly rename public flags or scripts. Add non-breaking aliases first:

- file outputs: prefer `--output`
- directory outputs: prefer `--output-dir`
- preserve domain-specific and legacy flags such as `--out`, `--out-dir`,
  `--output-pdf`, `--publication-dir`, `--reports-dir`, and `--metrics-json`
  until callers and tests have migrated.

## Documented Exceptions

The following names are accepted exceptions and should not be mechanically
renamed:

- mathematical notation tied to theorem/paper statements, including `H_t`,
  `H_X`, `T`, `K_factor`, `lipschitz_L`, `X`, and `Y`
- standard protocol or framework methods such as `observe`, `step`, `to_dict`,
  and `forward`
- public compatibility aliases listed above
