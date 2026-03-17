# ORIUS

ORIUS is a research-to-runtime framework for safe battery dispatch under degraded telemetry. It combines forecasting, uncertainty quantification, optimization, safety shielding, closed-loop cyber-physical evaluation, publication-grade artifact generation, and manuscript claim locking in one repository.

The central method is **DC3S**: a telemetry-reliability-weighted conformal safety shield that:
- scores telemetry quality at each control step,
- inflates uncertainty when reliability deteriorates,
- repairs unsafe optimizer actions before execution,
- emits auditable per-step safety certificates.

This repository also includes **CPSBench**, a fault-injection benchmark that separates **true state** from **observed state** so hidden safety failures become measurable rather than silently masked by degraded telemetry.

## What This Repository Contains

- A full forecasting and dispatch stack for Germany (OPSD) and US balancing-authority data derived from EIA-930, forming the reference implementation for the ORIUS framework.
- A DC3S runtime path with guarantee checks, certificate logging, and safety-gated action repair.
- CPSBench scenarios for dropout, delay-jitter, spikes, out-of-order observations, stale-sensor behavior, and drift-style faults.
- A release-family workflow for training, verification, publication artifacts, and paper freeze.
- Locked manuscript infrastructure: metrics manifest, claim matrix, validation scripts, and paper asset sync checks.
- Additive research-hardening tools for multi-country OPSD ingestion, reliability-conditioned conformal analysis, CHIL/pilot bundle tooling, and protocol-ready edge drivers.

## Research Scope

The current thesis and conference drafts make one main scientific claim:

> battery dispatch controllers can appear safe on observed telemetry while violating physical battery limits on true state under degraded measurements, and DC3S closes that hidden gap without blanket conservatism.

The locked paper evidence currently centers on:
- **hidden safety gap**: deterministic dispatch can violate true SOC under degraded telemetry,
- **DC3S safety result**: zero true-SOC violations with low intervention burden in the locked benchmark setup,
- **decision impact**: locked DE and US region-level cost, carbon, and peak outcomes,
- **governance**: publication claims are tied to source artifacts and validator gates.

Hardware validation, PHIL, and field pilots are **not** part of the current submission scope. The repository contains future-pilot infrastructure, but real-device validation remains future work.

**Vehicles prototype**: A `VehicleDomainAdapter` for 1D longitudinal control exists under `src/orius/vehicles/` as an exploratory extension. It is **not** part of the locked thesis or paper claims. All strong safety claims remain battery-only. See `orius-plan/vehicles-extension.md` and `reports/vehicles_prototype/` for prototype artifacts.

## Locked Paper Snapshot

The current paper-facing thesis PDF is [`paper/paper.pdf`](paper/paper.pdf). The conference variant is generated from `paper/paper_r1.tex`.

At a high level, the locked thesis reports:
- deterministic dispatch can violate true SOC under telemetry faults while still appearing safe on observed state,
- DC3S closes that hidden gap with conditional conservatism rather than blanket override,
- DE locked impact: **7.11%** cost savings, **0.30%** carbon reduction, **6.13%** peak shaving,
- positive stochastic value in both DE and US locked runs,
- all publication-facing claims are validator-gated.

For exact numbers, use the manuscript and locked artifact interfaces instead of copying values from this README:
- `paper/metrics_manifest.json`
- `paper/claim_matrix.csv`
- `reports/publication/release_manifest.json`
- `scripts/validate_paper_claims.py`

## Repository Layout

```text
.
├── src/orius/                  Core Python package
│   ├── data_pipeline/          Dataset normalization and feature generation
│   ├── forecasting/            GBM + deep forecasting models and UQ tooling
│   ├── optimizer/              Deterministic, robust, and CVaR dispatch
│   ├── dc3s/                   Shield logic, FTIT, guarantee checks, theory helpers
│   ├── cpsbench_iot/           Truth-vs-observed benchmark harness
│   ├── monitoring/             Drift and health monitoring
│   └── safety/                 BMS-style safety checks
├── services/api/              FastAPI service layer
├── frontend/                  Next.js dashboard
├── iot/                       Closed-loop simulator and edge-agent code
├── scripts/                   Canonical training, release, publication, and paper tooling
├── paper/                     Thesis, conference draft, and paper assets
├── reports/                   Generated evaluation, publication, and run outputs
├── configs/                   Training, DC3S, optimization, IoT, and release configs
├── docs/                      Architecture, evaluation, guarantees, and dataset docs
├── tests/                     Unit and workflow regression coverage
└── deploy/                    Deployment examples and service manifests
```

## Quick Start

### Requirements

- Python **3.11**
- `pip`
- `make`
- Node.js for the dashboard
- LaTeX tools only if you want to compile the papers locally

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .
```

### Fast Local Run

Backend:

```bash
make api
```

Frontend:

```bash
make frontend
```

Then open `http://localhost:3000`.

## Canonical Workflow

ORIUS uses a **release-family** workflow. One `RELEASE_ID` should span candidate training, CPSBench, publication artifacts, and paper freeze.

### 1. Run a release family

```bash
export RELEASE_ID=FINAL_20260312T120000Z
export PROFILE=standard

make r1-full RELEASE_ID=$RELEASE_ID PROFILE=$PROFILE
make r1-cpsbench RELEASE_ID=$RELEASE_ID
make r1-verify RELEASE_ID=$RELEASE_ID
```

### 2. Freeze paper outputs

```bash
make paper-freeze RELEASE_ID=$RELEASE_ID
```

### 3. Promote only after verify passes

```bash
make r1-promote RELEASE_ID=$RELEASE_ID
```

## Primary Entry Points

These are the canonical scripts for research and publication work:

- `scripts/train_dataset.py`
  - Per-dataset training, candidate-run acceptance, and promotion-aware output layout.
- `scripts/run_r1_release.py`
  - Release-family orchestration for `diagnostic`, `full`, `cpsbench`, `verify`, and `promote`.
- `scripts/build_publication_artifact.py`
  - Builds one manifest-locked publication package from a single `release_id`.
- `scripts/post_training_paper_update.py`
  - Final paper-freeze path for the verified release family.
- `scripts/sync_paper_assets.py --check`
  - Verifies that paper assets match locked publication outputs.
- `scripts/validate_paper_claims.py`
  - Hard claim gate across markdown, LaTeX, manifests, and tracked values.
- `scripts/final_publish_audit.py`
  - Final release audit and GO/NO-GO surface.

## Data and Dataset Scope

### Current dataset families

- **DE / OPSD**
  - hourly load, wind, and solar signals for Germany
- **US / EIA-930**
  - canonical thesis lock uses MISO
  - supporting release-family evidence also includes PJM and ERCOT

### Multi-country OPSD support

The OPSD feature builder now supports country-aware normalization:

```bash
python -m orius.data_pipeline.build_features \
  --in data/raw \
  --out data/processed \
  --country DE
```

The same normalized output schema is used downstream:
- `timestamp`
- `load_mw`
- `wind_mw`
- `solar_mw`
- optional `price_eur_mwh`

See:
- `DATA.md`
- `docs/ADDING_DATASETS.md`

## Forecasting, Uncertainty, and Control

### Forecasting layer

The main forecasting surface is a six-model comparison:
- GBM
- LSTM
- TCN
- N-BEATS
- TFT
- PatchTST

The locked thesis headline regions are DE and canonical US/MISO. The shared contract is:
- 24-hour forecast horizon
- 168-hour lookback
- time-aware CV
- 24-hour temporal gap
- shared feature framing
- shared point-metric evaluation contract

### Uncertainty layer

The production paper path uses conformal and adaptive interval tooling already wired into the reporting and publication stack.

The repo also contains an additive research path for **reliability-conditioned conformal analysis**:
- `src/orius/forecasting/uncertainty/reliability_mondrian.py`
- `scripts/compute_reliability_group_coverage.py`

This research path is optional and does **not** replace the locked production publication workflow unless explicitly invoked.

### Control and safety layer

DC3S sits between forecast intervals and command execution. Its responsibilities are:
- telemetry reliability scoring,
- interval inflation,
- drift-aware safety tightening,
- infeasible-action repair,
- auditable certificate emission.

Key implementation surfaces:
- `src/orius/dc3s/`
- `src/orius/safety/`
- `services/api/routers/dc3s.py`
- `docs/ASSUMPTIONS_AND_GUARANTEES.md`

## CPSBench and IoT Surfaces

### CPSBench

CPSBench evaluates controllers against a truth-vs-observed split so hidden safety failures are measurable:
- `src/orius/cpsbench_iot/`
- `make r1-cpsbench`

### Closed-loop IoT simulation

The repo includes a closed-loop simulator and edge-agent path:

```bash
make iot-sim
```

Relevant paths:
- `iot/simulator/run_closed_loop.py`
- `iot/edge_agent/run_agent.py`
- `configs/iot.yaml`

### Edge drivers

The edge agent supports:
- `http_gateway` (default)
- `modbus_tcp` (protocol-ready, implemented for future integration work)

This driver work is present for future-readiness only. It is **not** evidence of completed hardware validation.

## Paper and Publication Workflow

### Canonical manuscript files

- `paper/PAPER_DRAFT.md`
  - canonical thesis source
- `paper/paper.tex`
  - thesis LaTeX twin
- `paper/paper_r1.tex`
  - shorter conference derivative

### Canonical paper checks

```bash
python3 scripts/validate_paper_claims.py
python3 scripts/sync_paper_assets.py --check
```

### Compile papers locally

```bash
cd paper
latexmk -pdf paper.tex
latexmk -pdf paper_r1.tex
```

### Final freeze

```bash
make paper-freeze RELEASE_ID=$RELEASE_ID
```

This freeze path:
- rebuilds paper-facing tables and figures from the chosen release family,
- compiles thesis and conference PDFs,
- renders review images,
- writes immutable frozen copies under `reports/publication/frozen/<RELEASE_ID>/`,
- records PDF hashes in the publication manifest.

## Documentation Guide

Start here if you are orienting yourself in the repo:

- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `docs/ASSUMPTIONS_AND_GUARANTEES.md`
- `docs/ADDING_DATASETS.md`
- `services/api/README.md`
- `frontend/README.md`
- `docker/README.md`
- `deploy/README.md`
- `PRODUCTION_GUIDE.md`

## Research-Hardening Tools

These tools exist to strengthen generality, theory, or future pilot-readiness without changing the current locked submission scope:

- multi-country OPSD normalization
- reliability-binned conformal analysis
- CHIL-style run summarization
- pilot-bundle export and validation
- protocol-ready Modbus TCP driver
- SOC-tube / safety-filter theory helpers

These are additive tools. They are **not** required to reproduce the current thesis lock.

## Boundaries and Non-Goals

This repository does **not** currently claim:
- hardware-validated edge deployment,
- PHIL or field-pilot evidence,
- universal market realism across all tariff and settlement regimes,
- universal transferability across all grids and telemetry stacks,
- deep-model dominance as a general result outside the current data regime.

Those are either future work or separate research tracks.

## Professional Usage Guidance

If you are using this repo for serious research or release work:
- treat `README.md`, `paper/metrics_manifest.json`, and `scripts/README.md` as workflow entry points,
- prefer the release-family commands over ad hoc script chains,
- do not mix artifacts from different `RELEASE_ID`s,
- do not bypass `validate_paper_claims.py` or `sync_paper_assets.py --check`,
- do not treat local PDFs or scratch outputs as canonical evidence unless they are tied to a release manifest.

## Citation

If you reference ORIUS, DC3S, or CPSBench in academic work, cite the manuscript that matches the submission (thesis or conference draft).