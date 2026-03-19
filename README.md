# ORIUS

**ORIUS** is a research-to-runtime framework for safe cyber-physical control under degraded observation. Its primary validated instantiation in this repository is **DC3S** (Degradation-Conditioned Conformal Dispatch Safety Shield), a battery-dispatch controller that scores telemetry quality, widens uncertainty when observation degrades, repairs unsafe actions, and emits step-level audit certificates.

This repository is intentionally broader than a single model package. It combines:
- forecasting and uncertainty quantification,
- optimization and safety shielding,
- benchmark and fault-injection harnesses,
- runtime/API/dashboard surfaces,
- paper/thesis artifact locking,
- publication tables, figures, and claim validation.

---

## 1. What this repository proves

### Core validated claim

The battery-domain claim is the main locked result in this repository:

> a dispatch controller can appear safe on **observed telemetry** while violating physical battery limits on **true state**, and DC3S closes that hidden safety gap without blanket conservatism.

### Current evidence boundary

| Scope | Status | Meaning in this repo |
|---|---|---|
| **Battery** | Validated reference domain | Main empirical + theorem-facing evidence surface |
| **Vehicle / AV** | Proof domain | Second domain used by the universal validation gate |
| **Navigation** | Portability-only | Runtime and benchmark support, but not a universal proof claim |
| **Industrial** | Portability-only | Implemented portability surface |
| **Healthcare / Surgical** | Portability-only | Implemented portability surface |
| **Aerospace** | Experimental | Included for architecture completeness, not a locked validation claim |
| **CertOS** | Runtime prototype | Operational certificate/recovery layer, not a deployment claim |

### What is *not* currently claimed

- No field-deployment claim.
- No production-hardware / HIL claim for the main thesis result.
- No claim that every ORIUS domain is equally validated.
- No claim that portability evidence equals universal theorem closure.

---

## 2. Read this first: proof, paper, tables, and figures

If you want the **full proof package** and publication-facing evidence, start here.

### A. Manuscript and claim-lock surface

| Artifact | What it is |
|---|---|
| [`paper/paper.pdf`](paper/paper.pdf) | Current thesis-facing compiled PDF |
| [`paper/paper.tex`](paper/paper.tex) | Main thesis source |
| [`paper/paper_r1.tex`](paper/paper_r1.tex) | Conference / R1 variant |
| [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md) | Long-form paper draft and argument narrative |
| [`paper/README.md`](paper/README.md) | Thesis-writing and paper-production workflow guide |
| [`paper/metrics_manifest.json`](paper/metrics_manifest.json) | Locked metric source of truth |
| [`paper/claim_matrix.csv`](paper/claim_matrix.csv) | Claim-to-evidence mapping |
| [`paper/manifest.yaml`](paper/manifest.yaml) | Paper artifact manifest |
| [`paper/sync_rules.md`](paper/sync_rules.md) | Paper sync / asset rules |

### B. Full proof / theorem locations

| Proof artifact | Purpose |
|---|---|
| [`chapters/ch16_battery_theorem_oasg_existence.tex`](chapters/ch16_battery_theorem_oasg_existence.tex) | Battery theorem block: hidden safety-gap existence |
| [`chapters/ch17_battery_theorem_safety_preservation.tex`](chapters/ch17_battery_theorem_safety_preservation.tex) | Battery theorem block: safety-preservation argument |
| [`chapters/ch18_orius_core_bound_battery.tex`](chapters/ch18_orius_core_bound_battery.tex) | ORIUS core bound discussion for battery domain |
| [`chapters/ch19_no_free_safety_battery.tex`](chapters/ch19_no_free_safety_battery.tex) | Limits / no-free-safety framing |
| [`appendices/app_c_full_proofs.tex`](appendices/app_c_full_proofs.tex) | Main full-proof appendix |
| [`appendices/app_m_verified_theorems_and_gap_audit.tex`](appendices/app_m_verified_theorems_and_gap_audit.tex) | Verified theorem + gap audit appendix |
| [`appendices/app_z_theorem_and_paper_sync_registers.tex`](appendices/app_z_theorem_and_paper_sync_registers.tex) | Theorem/paper synchronization register |
| [`appendices/app_s_claim_evidence_registers.tex`](appendices/app_s_claim_evidence_registers.tex) | Claim/evidence cross-reference appendix |

### C. Key publication tables

| Table artifact | What it summarizes |
|---|---|
| [`reports/publication/dc3s_main_table.csv`](reports/publication/dc3s_main_table.csv) | Main DC3S benchmark summary |
| [`reports/publication/dc3s_fault_breakdown.csv`](reports/publication/dc3s_fault_breakdown.csv) | Per-fault performance breakdown |
| [`reports/publication/fault_performance_table.csv`](reports/publication/fault_performance_table.csv) | Stress/fault benchmark table |
| [`reports/publication/baseline_comparison_all.csv`](reports/publication/baseline_comparison_all.csv) | Forecast model comparison |
| [`reports/publication/claim_evidence_matrix.csv`](reports/publication/claim_evidence_matrix.csv) | Claim-to-artifact matrix |
| [`reports/publication/artifact_traceability_table.csv`](reports/publication/artifact_traceability_table.csv) | Artifact traceability / audit table |
| [`reports/publication/dataset_cards.csv`](reports/publication/dataset_cards.csv) | Dataset inventory for release-family evidence |
| [`reports/publication/cost_safety_pareto.csv`](reports/publication/cost_safety_pareto.csv) | Cost-safety trade-off frontier |

### D. Key publication figures

| Figure artifact | What it shows |
|---|---|
| [`reports/figures/architecture.png`](reports/figures/architecture.png) | System architecture overview |
| [`reports/publication/fig_48h_trace.png`](reports/publication/fig_48h_trace.png) | Main 48h trace figure |
| [`reports/publication/calibration_plot.png`](reports/publication/calibration_plot.png) | Calibration evidence |
| [`reports/publication/blackout_half_life.png`](reports/publication/blackout_half_life.png) | Certificate half-life blackout result |
| [`reports/publication/cross_region_transfer.png`](reports/publication/cross_region_transfer.png) | Cross-region transfer analysis |
| [`reports/publication/ablation_sensitivity.png`](reports/publication/ablation_sensitivity.png) | DC3S ablation sensitivity |
| [`reports/paper1/fig_cost_safety_frontier.png`](reports/paper1/fig_cost_safety_frontier.png) | Cost-safety frontier |
| [`reports/paper2/fig_certificate_shrinkage.png`](reports/paper2/fig_certificate_shrinkage.png) | Paper 2 certificate shrinkage figure |
| [`reports/paper3/fig_degradation_trajectory.png`](reports/paper3/fig_degradation_trajectory.png) | Paper 3 graceful degradation trajectory |

### E. Canonical report/output contracts

- [`reports/README.md`](reports/README.md) — root reports contract.
- [`reports/publication/README.md`](reports/publication/README.md) — publication-specific note.

---

## 3. Repository map

```text
.
├── src/orius/                  Core Python package
│   ├── adapters/               Canonical domain and benchmark entrypoints
│   ├── dc3s/                   Safety logic, RAC-Cert, FTIT, certificates
│   ├── certos/                 Runtime certificate/recovery prototype
│   ├── orius_bench/            Cross-domain benchmark and metrics
│   ├── forecasting/            GBM + deep forecasting models and UQ tooling
│   ├── optimizer/              Deterministic, robust, and CVaR dispatch
│   ├── data_pipeline/          Dataset ingestion, features, and splits
│   ├── monitoring/             Drift and runtime-health checks
│   ├── streaming/              Runtime telemetry ingestion/validation
│   ├── cpsbench_iot/           Truth-vs-observed benchmark harness
│   ├── multi_agent/            Shared-constraint composition layer
│   └── universal_framework/    Domain-agnostic ORIUS runtime pipeline
├── services/api/              FastAPI service layer
├── frontend/                  Next.js operator dashboard
├── scripts/                   Training, release, audit, and publication tooling
├── tests/                     Unit, regression, and workflow tests
├── docs/                      Architecture, runbooks, evaluation, and project docs
├── paper/                     Thesis, paper draft, manifests, and sync rules
├── chapters/                  Thesis chapter sources
├── appendices/                Full proofs, audits, and traceability appendices
├── reports/                   Generated outputs, figures, tables, and publication bundle
├── configs/                   Training / optimization / serving configs
└── deploy/                    Deployment examples and manifests
```

---

## 4. Core system pieces

### Forecasting

The locked forecasting comparison includes:
- GBM,
- LSTM,
- TCN,
- N-BEATS,
- TFT,
- PatchTST.

Canonical datasets:
- **DE / OPSD** for the thesis reference path,
- **US / EIA-930** with MISO as the canonical thesis lock and PJM/ERCOT as supporting release-family evidence.

### DC3S safety loop

The core online runtime loop is:

```text
Detect → Calibrate → Constrain → Shield → Certify
```

In concrete terms:
1. ingest telemetry,
2. score reliability,
3. widen uncertainty when observation degrades,
4. repair unsafe dispatch actions,
5. emit an auditable certificate.

### Benchmarks and cross-domain evidence

- **CPSBench** makes the hidden safety gap measurable by separating `true_state` and `observed_state`.
- **ORIUS-Bench** extends the benchmark idea across battery, navigation, industrial, healthcare, aerospace, and vehicle tracks.
- **Universal validation** treats battery as reference, vehicle as proof domain, and other domains as portability/experimental evidence.

---

## 5. Quick start

### Requirements

- Python **3.11**
- `pip`
- `make`
- Node.js for the frontend
- LaTeX only if you want to build the paper locally

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .
```

### Fast local run

Backend:

```bash
make api
```

Frontend:

```bash
make frontend
```

Then open `http://localhost:3000`.

---

## 6. Training and release workflow

ORIUS uses a **release-family** workflow. A single `RELEASE_ID` is meant to tie together training, validation, publication artifacts, and paper freeze.

### Recommended end-to-end flow

```bash
export RELEASE_ID=FINAL_20260312T120000Z
export PROFILE=standard

make r1-diagnostic RELEASE_ID=$RELEASE_ID
make r1-full RELEASE_ID=$RELEASE_ID PROFILE=$PROFILE
make r1-cpsbench RELEASE_ID=$RELEASE_ID
make r1-verify RELEASE_ID=$RELEASE_ID
make paper-freeze RELEASE_ID=$RELEASE_ID
make r1-promote RELEASE_ID=$RELEASE_ID
```

### Useful entry points

| Command / script | Purpose |
|---|---|
| `make train-dataset DATASET=DE` | Train one dataset profile |
| `make train-all` | Train all registered datasets |
| `make cpsbench` | Run CPSBench / IoT closed-loop validation path |
| `make paper-assets` | Refresh paper-linked tables/figures |
| `make paper-verify` | Validate paper manifest + claim checks |
| `make publish-audit` | Run final publication audit |
| `scripts/run_r1_release.py` | Release-family orchestrator |
| `scripts/build_publication_artifact.py` | Build publication bundle |
| `scripts/post_training_paper_update.py` | Freeze verified outputs into paper-facing assets |

---

## 7. Where to look depending on what you want

### If you want the **paper / thesis argument**
- [`paper/paper.pdf`](paper/paper.pdf)
- [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md)
- [`chapters/`](chapters/)
- [`appendices/`](appendices/)

### If you want the **proofs**
- [`appendices/app_c_full_proofs.tex`](appendices/app_c_full_proofs.tex)
- [`appendices/app_m_verified_theorems_and_gap_audit.tex`](appendices/app_m_verified_theorems_and_gap_audit.tex)
- [`appendices/app_z_theorem_and_paper_sync_registers.tex`](appendices/app_z_theorem_and_paper_sync_registers.tex)

### If you want the **main benchmark and results tables**
- [`reports/publication/`](reports/publication/)
- [`reports/figures/`](reports/figures/)
- [`reports/paper1/`](reports/paper1/)
- [`reports/paper2/`](reports/paper2/)
- [`reports/paper3/`](reports/paper3/)

### If you want the **source code**
- [`src/orius/dc3s/`](src/orius/dc3s/)
- [`src/orius/forecasting/`](src/orius/forecasting/)
- [`src/orius/optimizer/`](src/orius/optimizer/)
- [`src/orius/orius_bench/`](src/orius/orius_bench/)
- [`src/orius/universal_framework/`](src/orius/universal_framework/)

### If you want the **API/dashboard/runtime path**
- [`services/api/`](services/api/)
- [`frontend/`](frontend/)
- [`iot/`](iot/)
- [`deploy/`](deploy/)

### If you want the **governance / reproducibility path**
- [`scripts/validate_paper_claims.py`](scripts/validate_paper_claims.py)
- [`scripts/sync_paper_assets.py`](scripts/sync_paper_assets.py)
- [`scripts/final_publish_audit.py`](scripts/final_publish_audit.py)
- [`scripts/check_paper1_lock_drift.py`](scripts/check_paper1_lock_drift.py)

---

## 8. Documentation map

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture and layer map |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Evaluation methodology |
| [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md) | Training pipeline guide |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Operational runbook |
| [`docs/ASSUMPTIONS_AND_GUARANTEES.md`](docs/ASSUMPTIONS_AND_GUARANTEES.md) | Safety and modeling assumptions |
| [`DATA.md`](DATA.md) | Dataset scope and source details |
| [`PRODUCTION_GUIDE.md`](PRODUCTION_GUIDE.md) | Production/deployment guide |
| [`ORIUS_REPRODUCIBILITY.md`](ORIUS_REPRODUCIBILITY.md) | Reproducibility guidance |
| [`CODEX_IMPLEMENTATION_UPDATE.md`](CODEX_IMPLEMENTATION_UPDATE.md) | Implementation update log |

---

## 9. Recommended reading order

If you are new to the project, use this order:

1. this `README.md`,
2. [`paper/paper.pdf`](paper/paper.pdf),
3. [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md),
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
5. [`reports/publication/`](reports/publication/),
6. [`appendices/app_c_full_proofs.tex`](appendices/app_c_full_proofs.tex),
7. `src/orius/dc3s/` and `src/orius/orius_bench/`.

---

## 10. Final note on interpretation

This repository is designed to be **auditable**. When in doubt, trust the locked artifacts and validation scripts over prose summaries:

- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`
- `reports/publication/release_manifest.json`
- `scripts/validate_paper_claims.py`

If you want, the next step can be either:
1. a **README polish pass** with badges/screenshots/cleaner tables, or
2. a **training implementation pass** where we start wiring the full release workflow end to end.
