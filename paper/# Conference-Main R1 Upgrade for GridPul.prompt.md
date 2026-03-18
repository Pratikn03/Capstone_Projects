# Conference-Main R1 Upgrade for ORIUS/DC3S

## Summary
The next cycle should be a full R1-style conference-main upgrade, not a light repo-sync pass. The paper should keep the full cost+carbon+safety triangle, frame deployment as mixed IoT+SCADA telemetry, expand the US evidence beyond MISO, and keep the manifest/claim-matrix/validator layer as a core contribution.

The current blockers to fix first are concrete:
- `paper/paper.tex` and `paper/PAPER_DRAFT.md` still headline the `33.8% -> 38.3%` PICP@90 claim, which is not acceptable under standard coverage semantics.
- `paper/claim_matrix.csv` still contains 2 `Unsupported` and 4 `Needs Citation` claims.
- `paper/metrics_manifest.json` and `reports/publication/release_manifest.json` still point to the February 17, 2026 run family.
- The repo now has stronger candidate-run and paper-asset plumbing, but the final R1 evidence pack is not yet regenerated on a clean, multi-region run family.

## Important Interfaces
- Keep `US` as a backward-compatible alias to `US_MISO`.
- Add explicit dataset keys `US_MISO`, `US_PJM`, and `US_ERCOT` to the unified training/report pipeline.
- Keep candidate-run outputs under `artifacts/runs/<dataset>/<run_id>` and `reports/runs/<dataset>/<run_id>`; do not update canonical paper manifests until the full R1 release family passes all gates.
- Make the publication builders consume only the selected run family, never mixed legacy checkpoints from canonical model directories.
- Standardize uncertainty outputs to one metric contract everywhere: `PICP@90`, `PICP@95`, mean interval width, pinball loss, and Winkler score.

## Key Changes

### 1. Rebuild the evidence base around one clean R1 release family
- Create one new release family for all R1 outputs, with one run ID namespace shared across DE, `US_MISO`, `US_PJM`, and `US_ERCOT`.
- Regenerate forecasting, uncertainty, backtest, CPSBench, and publication artifacts only from that release family.
- Keep February 17, 2026 values as historical reference until the new release family fully replaces them; do not partially bump `paper/metrics_manifest.json`.
- Remove manuscript references to the old `33.8% -> 38.3%` calibration headline until the new metrics are recomputed from the clean release family.

### 2. Expand forecasting to a real conference-main comparison set
- Keep GBM/LightGBM as the production tabular baseline.
- Use existing N-BEATS support as a main deep baseline.
- Add TFT and PatchTST as first-class baselines in the training/report pipeline.
- Keep Prophet and AutoML appendix-only; they are useful secondary checks, but they should not crowd the main comparison table.
- Remove LSTM/TCN from headline paper tables unless they are retrained and verified under the same run family; otherwise treat them as legacy experiments.
- Evaluate all forecasting models on DE plus `US_MISO`, `US_PJM`, and `US_ERCOT`, with the same split policy and target set.

### 3. Fix calibration semantics and make the safety results bite
- Recompute coverage with standard conformal definitions and add explicit metric definitions to the manuscript and report builders.
- Add calibration plots by volatility regime, season, and region.
- Keep the controller comparison set fixed as: `deterministic_lp`, `robust_fixed_interval`, `cvar_interval`, `dc3s_wrapped`, `dc3s_ftit`.
- Add one new closed-loop robust control baseline: receding-horizon scenario MPC using the existing scenario-robust dispatch stack, not a brand-new control framework.
- Expand CPSBench fault sweeps to include stronger dropout, delay, stale, spike, and observed-vs-true SOC mismatch cases.
- Evaluate safety on truth SOC plus power bounds, ramp limits, grid-import cap, and solver infeasibility rate.
- Make the benchmark acceptance criterion explicit: at least one hard regime must produce non-trivial truth-SOC failures for non-DC3S baselines, and `dc3s_ftit` must reduce both violation rate and violation severity.

### 4. Expand US external validity and provenance
- Promote the US study from single-BA to multi-BA using `MISO`, `PJM`, and `ERCOT`.
- Generate per-dataset cards with time range, row count, missingness, target coverage, feature count, and source provenance.
- Use weather and carbon inputs with documented provenance and licensing; if any region only has proxy carbon, mark that region as non-headline for carbon claims rather than hiding the limitation.
- Keep cross-region transfer experiments in scope: `DE -> US`, `US -> DE`, and inter-BA transfer among the three US regions.
- Update the paper narrative to describe the deployment target as mixed telemetry rather than pure IoT edge or utility-grade PMU control.

### 5. Rewrite the paper around four defensible contributions
- Contribution 1: telemetry-aware conformal safety layer for dispatch under degraded measurements.
- Contribution 2: stronger closed-loop benchmark evidence under fault severity sweeps and truth-state safety evaluation.
- Contribution 3: multi-region external-validity story across OPSD and three EIA-930 balancing authorities.
- Contribution 4: reproducibility-by-construction governance using the manifest, claim matrix, and validator.
- Keep the final paper asset set fixed:
  - 8 figures: architecture, DC3S runtime flow, region/dataset card figure, calibration tradeoff, fault violation rate sweep, fault severity sweep, cost-carbon-safety Pareto, transfer/generalization figure.
  - 6 tables: dataset summary, forecasting baseline comparison, calibration summary, controller main results, ablations, transfer study.
- Resolve claim-matrix items `C028` to `C033` by either sourcing them reproducibly or removing them from the final manuscript.

## Test Plan
- Fast checks:
  - Unit tests for multi-dataset registry, candidate-run isolation, target-aware verification, Table 13 export, and figure inventory.
  - Static checks for standard metric naming and report-schema consistency.
- Integration checks:
  - GBM-first candidate runs for DE, `US_MISO`, `US_PJM`, and `US_ERCOT`.
  - At least one full deep-baseline smoke run for N-BEATS, TFT, and PatchTST.
  - Report generation must read only from the selected candidate-run directories.
- Release checks:
  - `python3 scripts/validate_paper_claims.py` passes.
  - Figure inventory audit passes with no missing critical assets.
  - Final publish audit passes in an environment with `duckdb` installed.
  - `paper/claim_matrix.csv` has no `Unsupported`, `Needs Citation`, or `Conflicting` rows for claims still present in the manuscript.
  - `paper/metrics_manifest.json`, `reports/publication/release_manifest.json`, and the manuscript all reference the same new run family.
  - Stable-regime calibration is near nominal, and hard-fault regimes show a real safety delta for `dc3s_ftit`.

## Assumptions and Defaults
- Target is a conference-main submission, not a workshop or thesis-first write-up.
- Paper framing keeps cost, carbon, and safety together; safety is not dropped, but the paper is organized around the cost-carbon-safety tradeoff triangle.
- Governance remains a core contribution, not just internal engineering hygiene.
- US expansion uses `MISO`, `PJM`, and `ERCOT`.
- Hardware-in-the-loop remains out of scope for this cycle.
- Required environment additions for the R1 run family include `duckdb` for publication audit and the dependencies needed for N-BEATS, TFT, and PatchTST.

---

## Gap Analysis: What's Still Missing

### P0 — Desk-Rejection Blockers

| Item | Status | Action |
|------|--------|--------|
| `paper.tex` PICP headline `33.8% → 38.3%` | ❌ Still in abstract | Delete or replace with `\PaperMetric{}` token until new run family |
| Standardized UQ metric module | ❌ Not implemented | Create `src/orius/evaluation/uq_metrics.py` with PICP, pinball, Winkler |
| Metric definitions in manuscript | ❌ Not in Methods | Add LaTeX definitions for all 5 UQ metrics |

### P1 — Evidence Infrastructure

| Item | Status | Action |
|------|--------|--------|
| Multi-BA registry + configs (PJM, ERCOT) | ❌ No entries | Add `US_PJM`, `US_ERCOT` to `DATASET_REGISTRY`, create YAML configs |
| `US` backward-compat alias | ❌ Not implemented | `DATASET_REGISTRY["US"] = DATASET_REGISTRY["US_MISO"]` |
| Feature pipeline parameterized by `ba_code` | ❌ Likely hardcodes MISO | Audit `build_features_eia930.py`, accept `ba_code` param |
| CPSBench severity sweep config | ❌ Only basic scenarios | Create `configs/cpsbench_r1_severity.yaml` with sweep grids |
| SOC mismatch injection in CPSBench | ❌ Not implemented | Add observation noise/bias to truth-SOC in simulation |
| Scenario MPC control baseline | ❌ Not implemented | Create `src/orius/optimizer/scenario_mpc.py` wrapping existing CVaR dispatch |
| Claim matrix C028–C033 resolution | ❌ 2 Unsupported + 4 Needs Citation | Source citations or remove claims from manuscript |
| `metrics_manifest.json` frozen status | ❌ Points to Feb 17 run | Add `"status": "frozen_legacy"` field; do not bump until new family passes |

### P2 — Forecasting Baselines & Orchestration

| Item | Status | Action |
|------|--------|--------|
| TFT training wrapper | ❌ Not in pipeline | Model integration + config support |
| PatchTST training wrapper | ❌ Not in pipeline | Model integration + config support |
| N-BEATS verified under candidate-run layout | Partial | Verify it runs with `--run-id` |
| LSTM/TCN demotion from headline tables | ❌ Still listed | Config flag: `headline` vs `appendix` vs `legacy` model tiers |
| R1 release orchestrator (`run_r1_release.py`) | ❌ Not implemented | Stage-gated multi-dataset pipeline under one release family ID |
| Publication builder consumes only run family | ❌ May mix legacy checkpoints | Audit `build_publication_artifact.py` to accept `--run-id` |
| `requirements-r1-baselines.txt` | ❌ Not created | pytorch-forecasting, neuralforecast, etc. |

### P3 — Paper Assets & Transfer Study

| Item | Status | Action |
|------|--------|--------|
| Transfer study scaffold | ❌ Not implemented | Create `scripts/run_transfer_study.py` with cross-region protocol |
| Calibration-by-regime plot builder | ❌ Not implemented | Partition by volatility/season/region |
| Per-dataset card export | ❌ Only preflight exists | Structured dataset card for each region |
| Paper Figure 2: DC3S runtime flow | ❌ | Algorithm diagram |
| Paper Figure 3: Region/dataset card visual | ❌ | 4-region summary |
| Paper Figure 4: Calibration tradeoff | ❌ | PICP vs width by regime |
| Paper Figure 5: Fault violation rate sweep | ❌ | Needs severity sweep data |
| Paper Figure 6: Fault severity sweep | ❌ | Needs severity sweep data |
| Paper Figure 7: Cost-carbon-safety Pareto | ❌ | Needs multi-controller results |
| Paper Figure 8: Transfer/generalization | ❌ | Needs transfer study |
| Paper Table 1: Dataset summary (4 regions) | Partial | Needs PJM/ERCOT rows |
| Paper Table 2: Forecasting baseline comparison | ❌ | Needs TFT/PatchTST results |
| Paper Table 3: Calibration summary | ❌ | Needs UQ contract metrics |
| Paper Table 5: Ablations | ❌ | Needs ablation runs |
| Paper Table 6: Transfer study | ❌ | Needs transfer experiments |

## Execution Order

```
Phase 1 — Unblock (P0 + P1 infra)
  1. Fix paper.tex PICP headline
  2. Create UQ metric module
  3. Add US_PJM and US_ERCOT to registry + configs
  4. Parameterize EIA-930 feature pipeline by ba_code
  5. Resolve claim matrix C028–C033
  6. Freeze metrics_manifest.json as legacy

Phase 2 — Build Evidence (P1 experiments + P2)
  7. Implement scenario MPC baseline
  8. Create CPSBench severity sweep config + SOC mismatch
  9. Add TFT and PatchTST training wrappers
  10. Build R1 release orchestrator
  11. Run diagnostic candidate runs (GBM-first, all 4 regions)

Phase 3 — Full Runs + Paper Assets (P2 + P3)
  12. Run full baseline comparison (all models, all regions)
  13. Run CPSBench severity sweeps
  14. Run transfer study
  15. Generate all 8 figures + 6 tables
  16. Write camera-ready manuscript sections

Phase 4 — Verify + Submit
  17. validate_paper_claims.py passes
  18. Figure inventory audit passes
  19. Claim matrix clean
  20. Manifests aligned to new run family
  21. Promote release family to canonical paths
```
