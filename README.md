# ORIUS — Observation–Reality Integrity for Universal Safety

> A universal runtime safety layer for Physical AI under degraded observation.
> This repository is the research + reproducibility surface for the ORIUS monograph and its tracked battery + AV closure artifacts.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Domains](https://img.shields.io/badge/domains-6%20runtime%20rows-orange)
![Monograph](https://img.shields.io/badge/monograph-450%2B%20pages-blueviolet)
![License](https://img.shields.io/badge/license-research-lightgrey)

## Overview

ORIUS addresses a core hazard in cyber-physical and Physical AI systems: a controller can appear safe on the **observed** state while the **true** physical state has already crossed a constraint boundary. ORIUS closes that Observation-Action Safety Gap (OASG) by making observation quality part of runtime state, widening uncertainty under degraded telemetry, tightening the admissible action set, repairing unsafe actions, and emitting auditable runtime certificates.

The canonical runtime flow is:

```text
Detect -> Calibrate -> Constrain -> Shield -> Certify
```

The book-level claim in this repository is:

> ORIUS is a runtime safety layer for Physical AI under degraded observation.

That claim is **universal-first** at the architecture level and **artifact-strict** at the evidence level. Project-level status is governed by the committed release and publication artifacts, not by prose alone.

## Current Validated Posture

The current committed posture, as recorded in [`reports/publication/orius_equal_domain_parity_matrix.csv`](reports/publication/orius_equal_domain_parity_matrix.csv), is:

- `battery` = `reference` witness row with the deepest theorem-to-artifact closure
- `av` = `proof_validated` bounded row under the TTC + predictive-entry-barrier contract
- `industrial` = `architectural_instantiation` only — no locked pipeline artifacts
- `healthcare` = `architectural_instantiation` only — no locked pipeline artifacts
- `navigation` = `shadow_synthetic` row with blocker `navigation_kitti_runtime_missing`
- `aerospace` = `experimental` row with blocker `aerospace_realflight_runtime_missing`

**Submission scope**: Only Battery and AV have locked, SHA256-verified pipeline artifacts. The remaining four domains are future work. No statement in this repository claims equal maturity across all domains.

## Latest Validated Results

The current battery + AV closure lane is summarized by:

- [`reports/battery_av/overall/release_summary.json`](reports/battery_av/overall/release_summary.json)
- [`reports/battery_av/overall/publication_closure_override.json`](reports/battery_av/overall/publication_closure_override.json)
- [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv)

| Domain | Current status | Locked result | Source |
| --- | --- | --- | --- |
| Battery Energy Storage | `reference` witness row | TSVR = 0.0% (all 4 controllers, 288 steps) | [`reports/battery_av/overall/release_summary.json`](reports/battery_av/overall/release_summary.json) |
| Autonomous Vehicles | `proof_validated` bounded row | TSVR 0.660 → 0.628 (9,348 steps, 228 scenarios) | [`reports/battery_av/overall/release_summary.json`](reports/battery_av/overall/release_summary.json) |

The current committed AV full-corpus release also records:

- `9,348` runtime trace rows in [`reports/orius_av/full_corpus/runtime_traces.csv`](reports/orius_av/full_corpus/runtime_traces.csv)
- a valid CertOS chain for `9,348` certificates in [`reports/orius_av/full_corpus/runtime_governance_summary.csv`](reports/orius_av/full_corpus/runtime_governance_summary.csv) and [`reports/orius_av/full_corpus/certos_verification_summary.json`](reports/orius_av/full_corpus/certos_verification_summary.json)
- `480` locked OASG cases in [`reports/orius_av/full_corpus/oasg_domain_summary.csv`](reports/orius_av/full_corpus/oasg_domain_summary.csv)

Battery remains the witness row even after AV promotion. AV is defended and proof-validated within its current bounded longitudinal contract; that is not a claim of full autonomous-driving closure.

## Canonical Result Surfaces

Project-level battery + AV closure should be read from these files first:

- [`reports/battery_av/overall/release_summary.json`](reports/battery_av/overall/release_summary.json): combined release summary for battery + AV
- [`reports/battery_av/overall/publication_closure_override.json`](reports/battery_av/overall/publication_closure_override.json): publication-facing battery + AV override surface
- [`reports/battery_av/overall/domain_summary.csv`](reports/battery_av/overall/domain_summary.csv): compact domain status summary
- [`reports/orius_av/full_corpus/summary.json`](reports/orius_av/full_corpus/summary.json): canonical AV full-corpus summary
- [`reports/orius_av/full_corpus/runtime_summary.csv`](reports/orius_av/full_corpus/runtime_summary.csv): AV runtime controller metrics
- [`reports/orius_av/full_corpus/runtime_traces.csv`](reports/orius_av/full_corpus/runtime_traces.csv): AV per-step replay traces
- [`reports/orius_av/full_corpus/runtime_governance_summary.csv`](reports/orius_av/full_corpus/runtime_governance_summary.csv): AV certificate/governance summary
- [`reports/orius_av/full_corpus/oasg_domain_summary.csv`](reports/orius_av/full_corpus/oasg_domain_summary.csv): AV observed-vs-true counterexample summary
- [`reports/publication/orius_equal_domain_parity_matrix.csv`](reports/publication/orius_equal_domain_parity_matrix.csv): domain posture summary
- [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv): defended vs gated closure matrix
- [`reports/publication/orius_maturity_matrix.csv`](reports/publication/orius_maturity_matrix.csv): current maturity posture across the program

The canonical manuscript controller remains:

- [`paper/paper.tex`](paper/paper.tex)
- [`paper/paper.pdf`](paper/paper.pdf)
- [`scripts/build_orius_monograph_assets.py`](scripts/build_orius_monograph_assets.py)

Do not hand-edit generator-owned publication outputs unless the generator change is made in the same patch.

## Artifact Policy

This repository commits **summary artifacts** and **release-facing evidence**, not every heavy runtime byproduct.

- Committed and authoritative AV summaries, tables, and figures live under [`reports/orius_av/full_corpus/`](reports/orius_av/full_corpus/)
- Committed and authoritative combined battery + AV closure summaries live under [`reports/battery_av/overall/`](reports/battery_av/overall/)
- The oversized AV audit database, `reports/orius_av/full_corpus/dc3s_av_waymo_dryrun.duckdb`, is intentionally **not** committed in normal git
- Raw Waymo shards are not part of the git-tracked repo surface
- The intended future path for heavy artifacts is a release asset or Git LFS; this repository currently treats the committed CSV/JSON/PNG surfaces as the canonical lightweight evidence layer

If you need the omitted heavy AV audit DB, regenerate it locally using the commands below.

## How To Verify / Rebuild The Battery + AV Closure Lane

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

Build the combined battery + AV closure summaries:

```bash
./.venv/bin/python scripts/build_battery_av_closure_artifacts.py \
  --battery-dir reports/battery_av/battery \
  --av-dir reports/orius_av/full_corpus \
  --overall-dir reports/battery_av/overall
```

Regenerate publication / monograph-facing assets:

```bash
./.venv/bin/python scripts/build_orius_monograph_assets.py
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

ORIUS's theoretical foundation is grounded in Shannon rate-distortion theory via four laws that completely characterise the degraded-observation safety problem:

| Law | Statement | Code Witness |
|-----|-----------|-------------|
| **L1** Rate-Distortion Safety Law | D*(C) >= alpha * max(0, 1 - C/H(X)). When channel capacity C < state entropy H(X), positive safety loss is unavoidable. | `rate_distortion_safety_law()` |
| **L2** Capacity Bridge | w_t <= kappa_d * C / H(X). OQE reliability is bounded by normalized channel capacity. | `capacity_bridge()` |
| **L3** Critical Capacity | Below C*_d = H(X)(1 - eps/alpha)/kappa_d, no controller can certify safety. | `critical_capacity()` |
| **L4** Achievability-Converse Sandwich | (alpha/K)(1-w_bar) <= TSVR* <= alpha(1-w_bar). DC3S is within constant factor K=2 of optimal. | `achievability_converse_sandwich()` |

All four laws are implemented in [`src/orius/universal_theory/orius_law.py`](src/orius/universal_theory/orius_law.py) with full proof sketches, and the earlier theorems T9, T10, T_minimax, and T_sensor_converse are derived as corollaries. The empirical phase transition sweep across 6 domains validates L1-L4 in [`reports/publication/orius_law_phase_transition.csv`](reports/publication/orius_law_phase_transition.csv).

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
├── reports/battery_av/         Combined battery + AV release lane
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

They are preserved for provenance only. They are not part of the active battery + AV closure authority and should not be used as current narrative truth.

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
