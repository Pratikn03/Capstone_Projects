# DC3S Assumptions and Guarantees (Artifact-Linked)

This is the canonical location for formal guarantees.
If a claim is not listed here, it is not claimed.

## Assumptions Version
`dc3s-assumptions-v1`

This version tag is persisted in DC3S certificates as `assumptions_version`.

## System
Discrete-time battery SOC:

`soc_{t+1} = soc_t + eta_c * charge_t * dt - discharge_t * dt / eta_d`

Constraints:
- `soc_t in [soc_min, soc_max]`
- `charge_t in [0, max_charge]`
- `discharge_t in [0, max_discharge]`
- no simultaneous charging and discharging.

## Assumptions (Required Conditions)
A1. Actuation fidelity: the safe action selected by the controller is applied to the plant, or BMS trips to HOLD.

A2. Bounded disturbance in uncertainty set: net-load disturbance is inside the uncertainty interval used by the robust step.

A3. Model timebase consistency: the same `dt`, `eta_c`, and `eta_d` are used across optimizer and plant.

A4. Observed state may be degraded: controller receives `soc_obs`, while safety is evaluated on `soc_true`.

## Guarantee 1 (Projection Repair Command Safety)
If the repair step enforces power and SOC feasibility for the state it is given, then the command satisfies:
- power bounds
- no simultaneous charge/discharge.

This is a command guarantee, not a plant-state guarantee.

## Guarantee 2 (Robust Feasibility Under Uncertainty Set)
If the robust optimizer returns an action feasible for all disturbances inside `U_t = [L_t, U_t]`, and A1-A3 hold, then one-step SOC constraints are satisfied for any disturbance in `U_t`.

## Corollary (Probabilistic Safety via Coverage)
If uncertainty construction satisfies:

`P(y_t in U_t) >= 1 - alpha`

then Guarantee 2 holds with probability at least `1 - alpha`, conditioned on A1-A3.

## Evidence Links
Truth-vs-observed SOC benchmark implementation:
- `src/gridpulse/cpsbench_iot/plant.py`
- `src/gridpulse/cpsbench_iot/telemetry_soc.py`
- `src/gridpulse/cpsbench_iot/runner.py`

Publication outputs:
- `reports/publication/dc3s_main_table.csv`
- `reports/publication/fig_true_soc_violation_vs_dropout.png`
- `reports/publication/fig_true_soc_severity_p95_vs_dropout.png`

## Certificate Evidence Fields
DC3S certificates persist:
- `guarantee_checks_passed`
- `guarantee_fail_reasons`
- `true_soc_violation_after_apply` (bench/HIL contexts)
- `assumptions_version`
