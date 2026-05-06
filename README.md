# ORIUS — Observation–Reality Integrity for Universal Safety

> ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.
> This repository is the research + reproducibility surface for the ORIUS monograph and its tracked Battery + AV + Healthcare promoted closure artifacts.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Domains](https://img.shields.io/badge/domains-3%20runtime%20rows-orange)
![Monograph](https://img.shields.io/badge/monograph-450%2B%20pages-blueviolet)
![License](https://img.shields.io/badge/license-research-lightgrey)

## Overview

ORIUS addresses a core hazard in cyber-physical and Physical AI systems: a controller can appear safe on the **observed** state while the **true** physical state has already crossed a constraint boundary. ORIUS closes that Observation-Action Safety Gap (OASG) by making observation quality part of runtime state, widening uncertainty under degraded telemetry, tightening the admissible action set, repairing unsafe actions, and emitting auditable runtime certificates.

The canonical runtime flow is:

```text
Detect -> Calibrate -> Constrain -> Shield -> Certify
```

The book-level claim in this repository is:

> ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

The flagship novelty sentence used across the defended ML and review surfaces is:

> ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

That claim is **universal-first** at the architecture level and **artifact-strict** at the evidence level. Project-level status is governed by the committed release and publication artifacts, not by prose alone.

## Current Validated Posture

The current committed posture, as recorded in [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv) and the promoted 3-domain scorecard, is:

- `battery` = `reference` witness row with the deepest theorem-to-artifact closure
- `av` = `runtime_contract_closed` bounded row under the narrowed brake-hold release contract
- `healthcare` = `runtime_contract_closed` bounded row backed by the promoted MIMIC runtime-denominator surface
**Submission scope**: The canonical and only promoted lane is `battery_av_healthcare`. Battery remains the witness row; AV and Healthcare are promoted bounded rows, not equal-depth theorem closures.

## Latest Validated Results

The current promoted 3-domain closure lane is summarized by:

- [`reports/battery_av_healthcare/overall/release_summary.json`](reports/battery_av_healthcare/overall/release_summary.json)
- [`reports/battery_av_healthcare/overall/publication_closure_override.json`](reports/battery_av_healthcare/overall/publication_closure_override.json)
- [`reports/battery_av_healthcare/overall/domain_summary.csv`](reports/battery_av_healthcare/overall/domain_summary.csv)
- [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv)
- [`reports/publication/domain_runtime_contract_summary.json`](reports/publication/domain_runtime_contract_summary.json)
- [`reports/publication/domain_runtime_contract_witnesses.csv`](reports/publication/domain_runtime_contract_witnesses.csv)

T11 remains a forward-only four-obligation transfer theorem. The AV and Healthcare theorem surfaces are supporting runtime lemmas bounded to their emitted domain postconditions: AV brake-hold and Healthcare fail-safe alert release.

| Domain | Current status | Locked result | Source |
| --- | --- | --- | --- |
| Battery Energy Storage | `reference` witness row | TSVR `0.008333 -> 0.000000` on the locked publication-nominal runtime surface | [`reports/publication/three_domain_ml_benchmark.csv`](reports/publication/three_domain_ml_benchmark.csv) |
| Autonomous Vehicles | `runtime_contract_closed` bounded row | TSVR `0.161425 -> 0.000000` on the full 9,348-step runtime denominator | [`reports/publication/three_domain_ml_benchmark.csv`](reports/publication/three_domain_ml_benchmark.csv) |
| Medical and Healthcare Monitoring | `runtime_contract_closed` bounded row | TSVR `0.194489 -> 0.000000` on the full promoted MIMIC runtime denominator | [`reports/publication/three_domain_ml_benchmark.csv`](reports/publication/three_domain_ml_benchmark.csv) |

## Flagship ML / Novelty Surfaces

The defended ML center is grouped calibration plus runtime safety under degraded observation, not forecasting leadership and not a fresh conformal-theory claim.

- [`reports/publication/three_domain_ml_benchmark.csv`](reports/publication/three_domain_ml_benchmark.csv): headline 3-domain safety delta table
- [`reports/publication/three_domain_reliability_calibration.csv`](reports/publication/three_domain_reliability_calibration.csv): grouped calibration package by reliability bucket
- [`reports/publication/three_domain_grouped_coverage.csv`](reports/publication/three_domain_grouped_coverage.csv): grouped coverage with confidence intervals
- [`reports/publication/three_domain_baseline_suite.csv`](reports/publication/three_domain_baseline_suite.csv): cross-domain diagnostic baseline lane
- [`reports/publication/three_domain_ablation_matrix.csv`](reports/publication/three_domain_ablation_matrix.csv): cross-domain diagnostic ablation lane
- [`reports/publication/novelty_separation_matrix.csv`](reports/publication/novelty_separation_matrix.csv): prior-work separation matrix
- [`reports/publication/what_orius_is_not_matrix.csv`](reports/publication/what_orius_is_not_matrix.csv): explicit non-claim matrix

The current committed AV full-corpus release also records:

- `9,348` runtime trace rows in [`reports/orius_av/full_corpus/runtime_traces.csv`](reports/orius_av/full_corpus/runtime_traces.csv)
- a valid CertOS chain for `9,348` certificates in [`reports/orius_av/full_corpus/runtime_governance_summary.csv`](reports/orius_av/full_corpus/runtime_governance_summary.csv) and [`reports/orius_av/full_corpus/certos_verification_summary.json`](reports/orius_av/full_corpus/certos_verification_summary.json)
- `480` locked OASG cases in [`reports/orius_av/full_corpus/oasg_domain_summary.csv`](reports/orius_av/full_corpus/oasg_domain_summary.csv)

Battery remains the witness row even after AV and Healthcare promotion. AV is defended within its bounded longitudinal contract; Healthcare is defended within bounded monitoring-and-alert semantics. Neither row is a claim of full autonomous-driving or regulated clinical deployment closure.

## Canonical Result Surfaces

Project-level promoted closure should be read from these files first:

- [`reports/battery_av_healthcare/overall/release_summary.json`](reports/battery_av_healthcare/overall/release_summary.json): canonical promoted release summary for Battery + AV + Healthcare
- [`reports/battery_av_healthcare/overall/publication_closure_override.json`](reports/battery_av_healthcare/overall/publication_closure_override.json): publication-facing promoted-lane override surface
- [`reports/battery_av_healthcare/overall/domain_summary.csv`](reports/battery_av_healthcare/overall/domain_summary.csv): compact promoted-lane domain status summary
- [`reports/orius_av/full_corpus/summary.json`](reports/orius_av/full_corpus/summary.json): canonical AV full-corpus summary
- [`reports/orius_av/full_corpus/runtime_summary.csv`](reports/orius_av/full_corpus/runtime_summary.csv): AV runtime controller metrics
- [`reports/orius_av/full_corpus/runtime_traces.csv`](reports/orius_av/full_corpus/runtime_traces.csv): AV per-step replay traces
- [`reports/orius_av/full_corpus/runtime_governance_summary.csv`](reports/orius_av/full_corpus/runtime_governance_summary.csv): AV certificate/governance summary
- [`reports/orius_av/full_corpus/oasg_domain_summary.csv`](reports/orius_av/full_corpus/oasg_domain_summary.csv): AV observed-vs-true counterexample summary
- [`data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv`](data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv): canonical promoted healthcare runtime/evaluation surface
- [`data/healthcare/mimic3/processed/mimic3_manifest.json`](data/healthcare/mimic3/processed/mimic3_manifest.json): canonical promoted healthcare manifest
- [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv): defended vs gated closure matrix
- [`reports/publication/orius_maturity_matrix.csv`](reports/publication/orius_maturity_matrix.csv): current maturity posture across the program
- [`reports/publication/orius_submission_scorecard.csv`](reports/publication/orius_submission_scorecard.csv): promoted-lane readiness score, including `three_domain_93_candidate`

The canonical submission monograph controller is now a senior-review
single-flow thesis build. It keeps the active defended claim boundary at
Battery + AV + Healthcare, compiles the polished monograph spine plus curated
appendices, and indexes historical/depth sources instead of compiling duplicate
archive chapters inline.

- [`orius_book.tex`](orius_book.tex)
- [`paper/paper.tex`](paper/paper.tex) as the mirrored internal single-flow controller
- [`orius_battery_409page_figures_upgraded_main.tex`](orius_battery_409page_figures_upgraded_main.tex) as the legacy archival long-form controller retained for internal provenance only
- [`paper/paper.pdf`](paper/paper.pdf)
- [`scripts/build_orius_monograph_assets.py`](scripts/build_orius_monograph_assets.py)

Do not hand-edit generator-owned publication outputs unless the generator change is made in the same patch.

## Artifact Policy

This repository commits **summary artifacts** and **release-facing evidence**, not every heavy runtime byproduct.

- Committed and authoritative AV summaries, tables, and figures live under [`reports/orius_av/full_corpus/`](reports/orius_av/full_corpus/)
- Committed and authoritative promoted closure summaries live under [`reports/battery_av_healthcare/overall/`](reports/battery_av_healthcare/overall/)
- The earlier Battery + AV-only summaries remain readable under [`reports/battery_av/overall/`](reports/battery_av/overall/) as legacy historical artifacts
- The oversized AV audit database, `reports/orius_av/full_corpus/dc3s_av_waymo_dryrun.duckdb`, is intentionally **not** committed in normal git
- Raw AV corpora are not part of the git-tracked repo surface. Local nuPlan archives (`nuplan-v1.1_train_singapore.zip`, `nuplan-maps-v1.0.zip`) are ignored and converted into the ORIUS replay contract locally.
- The intended future path for heavy artifacts is a release asset or Git LFS; this repository currently treats the committed CSV/JSON/PNG surfaces as the canonical lightweight evidence layer

If you need the omitted heavy AV audit DB, regenerate it locally using the commands below.

## How To Verify / Rebuild The Promoted 3-Domain Closure Lane

Build the nuPlan AV replay and feature surface:

```bash
python scripts/build_nuplan_av_surface.py \
  --train-dir . \
  --train-glob "nuplan-v*.zip" \
  --maps-zip nuplan-maps-v1.0.zip \
  --out-dir /tmp/orius_nuplan_smoke \
  --max-dbs 2 \
  --max-scenarios 4 \
  --build-features
```

Omit the bounds for the full local nuPlan Singapore conversion. The legacy
Waymo-named runtime adapter still consumes the source-neutral replay table until
the class names are renamed.

Completed nuPlan zip archives can also be synced to the private Hugging Face
dataset `pratikn03/orius-nuplan-private` with:

```bash
python scripts/sync_nuplan_hf_dataset.py --repo-id pratikn03/orius-nuplan-private --train-dir . --train-glob "nuplan-v*.zip"
```

Run the AV full-corpus runtime:

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
from orius.av_waymo import run_runtime_dry_run

run_runtime_dry_run(
    replay_windows_path=Path("data/orius_av/av/processed_full_corpus/replay_windows.parquet"),
    step_features_path=Path("data/orius_av/av/processed_full_corpus/step_features.parquet"),
    models_dir=Path("artifacts/models_orius_av_full_corpus"),
    out_dir=Path("reports/orius_av/full_corpus"),
    max_scenarios=None,
)
PY
```

Build the AV report surface:

```bash
./.venv/bin/python scripts/build_waymo_av_dry_run_report.py \
  --processed-dir data/orius_av/av/processed_full_corpus \
  --reports-dir reports/orius_av/full_corpus \
  --models-dir artifacts/models_orius_av_full_corpus \
  --uncertainty-dir artifacts/uncertainty/orius_av_full_corpus
```

Refresh the promoted healthcare manifest:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/refresh_real_data_manifests.py
```

Regenerate publication / monograph-facing assets:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/build_orius_monograph_assets.py --submission-scope battery_av_healthcare
```

If you want the combined lane orchestrated from one entrypoint, use [`scripts/run_battery_av_pipeline.py`](scripts/run_battery_av_pipeline.py).

## Runtime Kernel

Implemented runtime boundary:

- `DomainAdapter` is the canonical runtime adapter contract
- [`src/orius/universal_theory/`](src/orius/universal_theory/) holds the domain-neutral theory and contract surface
- [`src/orius/orius_bench/`](src/orius/orius_bench/) defines the replay and metrics schema
- [`src/orius/certos/`](src/orius/certos/) provides runtime governance and audit semantics
- [`src/orius/forecasting/`](src/orius/forecasting/) provides calibration and shift-aware uncertainty support

Canonical benchmark fields:

- `true_constraint_violated`
- `observed_constraint_satisfied`
- `true_margin`
- `observed_margin`
- `intervened`
- `fallback_used`
- `certificate_valid`
- `latency_us`
- `domain_metrics`

## Rate-Distortion Safety Laws (L1-L4)

ORIUS includes four rate-distortion-inspired extension laws. In the live repo
they are treated as \emph{stylized/proxy} surfaces for the extension-law
discussion, not as a fully closed characterization of degraded-observation
safety:

| Law | Statement | Code Witness |
|-----|-----------|-------------|
| **L1** Rate-Distortion Safety Law | Stylized lower-envelope surrogate `D*(C) >= alpha * max(0, 1 - C/H(X))`. | `rate_distortion_safety_law()` |
| **L2** Capacity Bridge | Stylized proxy bridge `w_t <= kappa_d * C / H(X)`. | `capacity_bridge()` |
| **L3** Critical Capacity | Threshold calculator induced by the L2 proxy plus the executable T3 upper envelope. | `critical_capacity()` |
| **L4** Achievability-Converse Sandwich | Stylized proxy lower side plus executable upper side. | `achievability_converse_sandwich()` |

All four laws are implemented in [`src/orius/universal_theory/orius_law.py`](src/orius/universal_theory/orius_law.py) as extension-law helpers. The active audit tracks them as bounded/stylized surfaces rather than fully defended converse theorems.

## Repository Map

```text
.
├── src/orius/                  Runtime package
│   ├── dc3s/                   Safety kernel implementation
│   ├── certos/                 Runtime governance and certificate logic
│   ├── forecasting/            Forecasting, conformal calibration, shift-aware uncertainty
│   ├── orius_bench/            Replay and metrics schema
│   └── universal_theory/       Domain-neutral contracts and theorem bridge
├── services/api/               FastAPI service layer
├── frontend/                   Next.js dashboard
├── scripts/                    Build, validation, training, and publication tooling
├── paper/                      Canonical manuscript and generated monograph assets
├── docs/                       Architecture and workflow documentation
├── reports/battery_av_healthcare/  Canonical three-domain release lane
├── reports/battery_av/         Historical battery + AV release lane
├── reports/orius_av/           AV runtime/report surfaces
├── reports/publication/        Active publication artifacts
└── reports/legacy_archive/     Historical frozen bundles and provenance-only outputs
```

## Dashboard Access

**Local Development Dashboard** (running now):
- **Localhost**: http://localhost:3000
- **Network Address**: http://192.168.4.108:3000
- **Port**: 3000

To start the dashboard:
```bash
cd frontend && npm run dev
```

## Documentation Map

- [`docs/UNIVERSAL_KERNEL_ARCHITECTURE.md`](docs/UNIVERSAL_KERNEL_ARCHITECTURE.md)
- [`docs/UNIVERSAL_BENCHMARK_SPEC.md`](docs/UNIVERSAL_BENCHMARK_SPEC.md)
- [`docs/UNIVERSAL_GOVERNANCE_SPEC.md`](docs/UNIVERSAL_GOVERNANCE_SPEC.md)
- [`docs/ORIUS_THESIS_TERMINOLOGY_GUIDE.md`](docs/ORIUS_THESIS_TERMINOLOGY_GUIDE.md)
- [`docs/ORIUS_FRAMEWORK_PROOF.md`](docs/ORIUS_FRAMEWORK_PROOF.md)
- [`docs/DOC_CONSISTENCY_CHECKLIST.md`](docs/DOC_CONSISTENCY_CHECKLIST.md)
- [`ORIUS_REPRODUCIBILITY.md`](ORIUS_REPRODUCIBILITY.md)

## Historical Material

Historical release bundles and frozen package snapshots are retained under:

- [`reports/legacy_archive/`](reports/legacy_archive/)

They are preserved for provenance only. They are not part of the active three-domain closure authority and should not be used as current narrative truth.

## Citation

If you cite this work, use the monograph-facing ORIUS expansion and the canonical manuscript:

```bibtex
@phdthesis{orius2026,
  title     = {{ORIUS}: Observation--Reality Integrity for Universal Safety
               under Degraded Observation},
  year      = {2026},
  school    = {[University]},
  note      = {Book-length monograph and tracked evidence release.}
}
```

## License

Research code — see [LICENSE](LICENSE) for terms.
