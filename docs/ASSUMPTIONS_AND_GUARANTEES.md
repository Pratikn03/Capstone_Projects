# DC³S Assumptions and Guarantees (Artifact-Linked)

This document is the **only** place where guarantees are stated.  
If a claim is not here, it is not claimed.

## System
Discrete-time battery SOC:
\[
SOC_{t+1} = SOC_t + \eta_c \cdot P^{chg}_t \Delta t - \frac{1}{\eta_d} \cdot P^{dis}_t \Delta t
\]

Constraints:
- \(SOC_t \in [SOC_{min}, SOC_{max}]\)
- \(P^{chg}_t \in [0, P^{chg}_{max}]\)
- \(P^{dis}_t \in [0, P^{dis}_{max}]\)
- Not both positive simultaneously.

## Assumptions (what must be true)
A1. **Actuation fidelity:** the safe action chosen by the controller is applied to the plant (or BMS trips to HOLD).
A2. **Bounded disturbance in the uncertainty set:** net-load disturbance lies in the uncertainty interval used by the robust step.
A3. **Model timebase consistency:** the same \(\Delta t\), \(\eta_c\), \(\eta_d\) are used across optimizer and plant.
A4. **Observed state may be degraded:** controller observes \(SOC^{obs}\), but safety is evaluated on \(SOC^{true}\).

## Guarantee 1 (Projection Repair Safety)
If the repair step enforces power and SOC feasibility for the **state it is given**, then the command itself satisfies:
- power bounds
- no simultaneous charge/discharge

This is a **command safety guarantee**, not a plant guarantee.

## Guarantee 2 (Robust Feasibility under Uncertainty Set)
If the robust optimizer returns a feasible action for all disturbances inside \(\mathcal{U}_t = [L_t, U_t]\) and A1–A3 hold,
then SOC constraints are satisfied for the next step for any disturbance in \(\mathcal{U}_t\).

## Corollary (Probabilistic Safety via Coverage)
If the uncertainty construction satisfies:
\[
P(y_t \in \mathcal{U}_t) \ge 1-\alpha
\]
then Guarantee 2 holds with probability at least \(1-\alpha\) under A1–A3.

## Evidence Link
The benchmark used to validate violations on \(SOC^{true}\) is:
- `src/gridpulse/cpsbench_iot/plant.py`
- `src/gridpulse/cpsbench_iot/telemetry_soc.py`
- `src/gridpulse/cpsbench_iot/runner.py`

Publication outputs:
- `reports/publication/dc3s_main_table.csv`
- `reports/publication/fig_true_soc_violation_vs_dropout.png`
