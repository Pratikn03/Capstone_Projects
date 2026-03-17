# Assumption Register (A1-A8) with Enforcement Map

This register maps each assumption to explicit code/config enforcement points.

## A1 - Bounded model error

**Meaning**  
Forecast residuals are bounded enough for calibrated intervals to remain meaningful.

**Enforcement**
- `configs/dc3s.yaml`: `alpha0`, `alpha_min`, `infl_max`.
- `src/orius/dc3s/coverage_theorem.py`: empirical coverage checks.
- `src/orius/forecasting/uncertainty/cqr.py`: base conformal/CQR interval generation.

**Runtime signal**
- Coverage artifacts and PICP fields in publication CSVs.

---

## A2 - Bounded telemetry error

**Meaning**  
Observation degradation is tracked and bounded via quality floor and inflation caps.

**Enforcement**
- `configs/dc3s.yaml`: `reliability.min_w: 0.05`, `ambiguity.min_w: 0.05`, `infl_max`.
- `src/orius/dc3s/quality.py`: `compute_reliability()` computes `w_t`.
- `src/orius/dc3s/calibration.py`: uncertainty inflation bounded by config.

**Runtime signal**
- `w_t` and inflation recorded in certificates and benchmark traces.

---

## A3 - Feasible safe repair exists

**Meaning**  
When proposed action is unsafe, projection/repair can return a feasible safe action.

**Enforcement**
- `configs/dc3s.yaml`: `shield.mode: projection`.
- `src/orius/dc3s/shield.py`: `_projection_repair()` and `repair_action()`.
- `src/orius/dc3s/guarantee_checks.py`: post-repair checks.

**Runtime signal**
- Certificate fields `intervened`, `intervention_reason`, `guarantee_checks_passed`.

---

## A4 - Known or identified dynamics

**Meaning**  
Controller and verifier use explicit battery dynamics model.

**Enforcement**
- `src/orius/cpsbench_iot/plant.py`: truth SOC dynamics.
- `src/orius/dc3s/guarantee_checks.py`: `next_soc()` dynamics used for invariance checks.
- `configs/optimization.yaml` and battery constraints from CPSBench runner.

**Runtime signal**
- True-SOC violation computation under plant dynamics.

---

## A5 - Monotone bounded uncertainty inflation

**Meaning**  
Inflation must not decrease below 1 and must remain capped.

**Enforcement**
- `src/orius/dc3s/coverage_theorem.py`: `verify_inflation_geq_one()`.
- `configs/dc3s.yaml`: `infl_max`, `rac_cert.max_q_multiplier`.
- `src/orius/dc3s/calibration.py` and RAC components apply bounded inflation law.

**Runtime signal**
- Certificate/trace fields: `inflation`, `q_multiplier`.

---

## A6 - Bounded detector lag

**Meaning**  
Drift detection delay is controlled by configured detector parameters.

**Enforcement**
- `configs/dc3s.yaml`: `drift.ph_delta`, `drift.ph_lambda`, `warmup_steps`, `cooldown_steps`.
- `src/orius/dc3s/drift.py`: Page-Hinkley detector behavior.

**Runtime signal**
- Drift flags and sensitivity changes in certificate metadata.

---

## A7 - Causal certificate update rule

**Meaning**  
Per-step certificate uses only present/past information, with hash chaining.

**Enforcement**
- `src/orius/dc3s/certificate.py`: `make_certificate()` + `prev_hash` chain.
- `src/orius/cpsbench_iot/runner.py`: certificate emitted each step and linked.
- `configs/dc3s.yaml`: audit store path and table names.

**Runtime signal**
- Certificate hash chain continuity in audit store.

---

## A8 - Admissible fallback policy

**Meaning**  
When uncertainty grows, a fallback safe policy remains available.

**Enforcement**
- `configs/dc3s.yaml`: `shield.mode`, `shield.cvar.*`.
- `src/orius/dc3s/shield.py`: `robust_resolve` / `robust_resolve_cvar` modes plus projection final guard.
- `src/orius/cpsbench_iot/baselines.py`: robust and CVaR baseline adapters.

**Runtime signal**
- Non-zero intervention rates with zero true-state violation in DC3S rows.

---

## Assumption Status Summary

| Assumption | Code anchor present | Config anchor present | Runtime evidence present | Status |
|---|---|---|---|---|
| A1 | yes | yes | yes | enforced |
| A2 | yes | yes | yes | enforced |
| A3 | yes | yes | yes | enforced |
| A4 | yes | yes | yes | enforced |
| A5 | yes | yes | yes | enforced |
| A6 | yes | yes | yes | enforced |
| A7 | yes | yes | yes | enforced |
| A8 | yes | yes | yes | enforced |

