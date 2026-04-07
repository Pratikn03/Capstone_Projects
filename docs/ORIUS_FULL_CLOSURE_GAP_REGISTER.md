# ORIUS Full-Closure Gap Register

This register maps the external 18-gap thesis brief onto the actual control
surfaces in this repository.

| Gap | Requested theme | Actual repo surface | Current status | Governing artifact |
| --- | --- | --- | --- | --- |
| 1 | Complete formal proofs | `appendices/app_c_full_proofs.tex`, `chapters/ch17_*`, `chapters/ch18_*`, `chapters/ch37_*` | Partial; proof surface exists, some sections still sketch-like | `reports/publication/integrated_theorem_gate_summary.tex` |
| 2 | Reconcile formal-item count | `scripts/count_formal_items.py`, `chapters/ch15_*`, `chapters/ch36_*`, `chapters_merged/ch08_*` | Implemented by include-tree counting | `orius_battery_409page_figures_upgraded_main.tex` |
| 3 | Notation and preliminaries | `chapters/ch03_problem_formulation.tex`, `chapters/ch15_*`, `appendices/app_a_notation.tex` | In progress; definition surface exists but is distributed | `appendices/app_a_notation.tex` |
| 4 | Marginal-coverage caveat upfront | `frontmatter/abstract.tex`, `chapters/ch01_introduction.tex`, `chapters_merged/ch01_introduction.tex` | Implemented in manuscript text | `frontmatter/abstract.tex` |
| 5 | Quantify A6 | `chapters/ch15_*`, `src/orius/dc3s/quality.py`, `tests/test_detector_lag.py` | Implemented for thesis text and OQE diagnostics | `reports/detector_lag.json` |
| 6 | Full ORIUS-MDP definition | `chapters/ch03_problem_formulation.tex`, `chapters_merged/ch01_introduction.tex` | Implemented in formal spec section | Manuscript chapters |
| 7 | AV TTC fix | `src/orius/vehicles/vehicle_adapter.py`, `tests/test_vehicle_adapter.py` | Core barrier already landed; regression hardening required | `reports/real_data_contract_status.json` |
| 8 | KITTI navigation closure | `scripts/build_navigation_real_dataset.py`, `src/orius/data_pipeline/build_features_navigation.py`, `src/orius/universal_framework/navigation_adapter.py` | Blocked on canonical raw KITTI staging | `reports/real_data_contract_status.json` |
| 9 | Aerospace runtime closure | `scripts/download_aerospace_datasets.py`, `scripts/build_aerospace_public_adsb_runtime.py`, `src/orius/universal_framework/aerospace_adapter.py` | Trainable plus public-support only; defended runtime row blocked | `reports/real_data_contract_status.json` |
| 10 | Confidence intervals and significance | `src/orius/evaluation/stats.py`, publication table builders | Existing stack extended in-place | `reports/publication/release_manifest.json` |
| 11 | Strong baselines | `src/orius/baselines/`, `scripts/run_sota_comparison.py` | Still open; not implemented in this patch | Comparison tables |
| 12 | OQE backend ablation | `scripts/run_ablations.py`, `src/orius/dc3s/quality.py` | Existing ablation path; further backend sweep still open | Publication ablations |
| 13 | Data cards | `docs/data_cards/`, `appendices/app_ad_dataset_cards.tex` | Implemented as source cards plus appendix | Markdown cards |
| 14 | Reproducibility appendix | `reports/publication/release_manifest.json`, `scripts/verify_reproducibility.py`, `appendices/app_ae_reproducibility_appendix.tex` | Implemented | Release manifest |
| 15 | Relationship to prior work | `chapters/ch01_introduction.tex`, `chapters_merged/ch01_introduction.tex` | Implemented | Introduction chapters |
| 16 | Label case studies honestly | `chapters/ch13_case_studies_operational_traces.tex` | Implemented | Case-study chapter |
| 17 | Bibliography depth | `paper/bibliography/orius_monograph.bib` | Already exceeds requested floor | Bibliography |
| 18 | Figure inventory expansion | `scripts/audit_figure_inventory.py`, chapter figure surfaces | Open; audit-first work remains | Figure inventory reports |

Notes:

- `av`, `navigation`, and `aerospace` remain governed by
  `reports/real_data_contract_status.json`.
- No manuscript claim should widen ahead of the corresponding release artifact
  and parity-gate refresh.
