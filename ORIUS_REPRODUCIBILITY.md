# ORIUS Full Framework Cold-Start Reproducibility

This document provides step-by-step instructions to reproduce all ORIUS (Papers 2–6) artifacts from a fresh clone.

## Prerequisites

- Python 3.11+
- Git

## Setup

```bash
git clone <repo-url> orius && cd orius
python3 -m venv .venv
. .venv/bin/activate   # or: source .venv/Scripts/activate on Windows
pip install -r requirements.lock.txt
```

## Validation

```bash
python3 scripts/validate_paper_claims.py
python3 scripts/sync_paper_assets.py --check
```

## Track Runs

### Battery track (ORIUS-Bench)

```bash
python scripts/run_orius_bench_release.py --seeds 2 --horizon 24
# Outputs: reports/orius_bench/leaderboard.csv, benchmark_bundle.tar.gz
```

### Blackout study (Paper 2)

```bash
python scripts/run_certificate_half_life_blackout.py --seeds 0 --horizons 0 1 4
# Outputs: reports/publication/certificate_half_life_blackout.csv
```

### Multi-agent scenario (Paper 5)

```bash
python scripts/run_multi_agent_scenario.py
# Outputs: reports/publication/multi_agent_transformer_scenario.csv
```

### Latency micro-benchmark

```bash
python scripts/benchmark_dc3s_steps.py --iterations 5000 --out reports/dc3s_latency_benchmark.json
```

### Fault-performance under packet drop

```bash
python scripts/run_fault_packet_drop_table.py
# Outputs: reports/publication/fault_performance_packet_drop.csv
```

### Graceful degradation figures (Paper 3)

```bash
python scripts/build_graceful_trajectory_figures.py
# Outputs: reports/publication/fig_graceful_four_policies.png
```

## Tests

```bash
pytest tests/test_blackout_expiration.py tests/test_graceful_degradation.py tests/test_certificate_half_life.py tests/test_conformal_reachability.py -v --no-cov
```

## Truth Surfaces

- `paper/metrics_manifest.json` — metrics definitions
- `paper/claim_matrix.csv` — claim-to-artifact mapping
- `paper/orius_program_manifest.json` — phase status and cold-start commands
- `configs/dc3s.yaml`, `configs/optimization.yaml` — locked configs

## Source of Truth

Repo code + manifest + reports remain the source of truth. The agent-artifact zip is for publication asset library only.
