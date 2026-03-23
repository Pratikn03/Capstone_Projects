# ORIUS — Universal CPS Safety Framework

**ORIUS** (Observation–Reality Integrity for Universal Safety) is a
runtime safety framework for cyber-physical systems (CPS) under degraded
telemetry.  Its core component is **DC3S** (Degradation-Conditioned
Conformal Dispatch Safety Shield), a five-stage pipeline that:

1. **Detects** telemetry degradation via an Observation Quality Engine (OQE)
2. **Calibrates** a conformal prediction interval inflated by the reliability score
3. **Constrains** the safe action set under the inflated uncertainty
4. **Shields** the proposed action via joint safe-action repair (SAF projection)
5. **Certifies** each dispatched action with an auditable runtime certificate (CERTos)

The framework is validated across **six CPS domains**: energy management,
autonomous vehicles, industrial process control, medical monitoring (ICU vitals),
aerospace, and navigation.

---

## Quick Start

```bash
pip install -e .
python evaluate_dc3s.py
```

## Reproduce Core Results

```bash
# Single-domain evaluation (fast, ~30 s)
python scripts/run_all_domain_eval.py --rows 100

# Full six-domain validation with evidence gate
python scripts/run_universal_orius_validation.py --seeds 3 --horizon 48

# SOTA baseline comparison (Tube MPC / CBF / Lagrangian vs DC3S)
python scripts/run_sota_comparison.py --seeds 3 --rows 100

# Full test suite
pytest tests/ --no-cov -q
```

## Core Theoretical Result

Under Assumptions A1–A7, DC3S satisfies:

```
E[V] ≤ α(1 − w̄)T
```

where `V` is the number of true-state violations, `α` is the conformal
miscoverage rate, `w̄` is the mean reliability score, and `T` is the episode
length.  Empirical TSVR~= 0 % across all five locked-telemetry proof domains.

## Repository Layout

```
src/orius/
  dc3s/              — Core DC3S pipeline (OQE, RUI, SAF, CERTos)
  adapters/          — Six domain adapters (battery, vehicle, industrial, ...)
  universal_framework/  — Domain-agnostic runner and domain registry
  sota_baselines.py  — Tube MPC / CBF / Lagrangian wrapper adapters
  orius_bench/       — CPSBench evaluation harness and fault engine

scripts/
  run_universal_orius_validation.py  — Full six-domain validation gate
  run_sota_comparison.py             — SOTA strategy comparison
  run_universal_training_audit.py    — Training surface audit
  run_universal_sil_validation.py    — Software-in-loop validation
  generate_hero_figure.py            — OASG hero figure for thesis
  build_orius_framework_proof.py     — Bundle all proof artifacts

tests/                — 950+ unit and integration tests
chapters/             — PhD thesis LaTeX source (360+ pages)
reports/              — Generated tables, figures, and proof artifacts
paper/                — Compiled thesis PDF and publication assets
data/                 — Domain telemetry datasets (locked)
```

## Domain Evidence Status

| Domain | Source | Evidence Tier | TSVR (DC3S) |
|---|---|---|---|
| Energy Management | Locked artifact (ENTSO-E) | Reference | 0.0 % |
| Autonomous Vehicles | Locked CSV | Proof-validated | 0.0 % |
| Industrial Process Control | Locked CSV | Proof-validated | 0.0 % |
| Medical Monitoring (ICU) | Locked CSV | Proof-validated | 0.0 % |
| Aerospace | Locked CSV | Experimental | 9.7 % |
| Navigation | Synthetic simulation | Shadow-synthetic | 0.7 % |

Evidence tier governs the manuscript claim boundary.  Proof-validated rows
require locked non-synthetic telemetry, a verified training surface, a clean
software-in-loop pass, a nontrivial baseline gap, material ORIUS improvement,
and stable seed behavior.

## Theorem Ladder

| Theorem | Claim | Empirical confirmation |
|---|---|---|
| T1 — OASG Existence | TSVR > 0 under quality-ignorant controller | 3.9–26.7 % baseline across domains |
| T2 — Safety Preservation | Pr[φ(a\_t)=1] ≥ 1−α per step | TSVR = 0 % on 5 locked domains |
| T3 — Core Bound | E[V] ≤ α(1−w̄)T | All empirical TSVRs below bound |
| T4 — No-Free-Safety | Any obs-only safe policy has TSVR > 0 | Rule-based baselines confirm |

## Adding a New Domain

1. Subclass `DomainAdapter` in `src/orius/dc3s/domain_adapter.py`
2. Implement the five pipeline stages:
   `ingest_telemetry`, `compute_oqe`, `build_uncertainty_set`,
   `tighten_action_set`, `repair_action`, `emit_certificate`
3. Register with `register_domain("my_domain", MyAdapter)` in
   `src/orius/universal_framework/domain_registry.py`
4. Add a `BenchmarkAdapter` subclass in `src/orius/orius_bench/`
5. Run `pytest tests/ --no-cov -q` to verify no regressions

## Citation

If you use this framework, please cite:

```
@phdthesis{orius2026,
  title  = {ORIUS: Observation--Reality Integrity for Universal Safety},
  year   = {2026},
  note   = {Six-domain CPS safety framework with DC3S runtime shield}
}
```

## License

Research code — see LICENSE for terms.
