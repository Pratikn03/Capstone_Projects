# Hyperparameter Rationale by Domain

This document explains domain-specific hyperparameter choices for thesis reproducibility.

---

## Energy (DE, US regions)

| Param | Value | Rationale |
|-------|-------|------------|
| n_estimators | 1000 (DE), 400 (quantile) | Hourly data, 168h lookback; more trees for long-horizon stability |
| max_depth | 12 | Capture seasonal/weekly patterns |
| num_leaves | 256 | High capacity for load/wind/solar interactions |
| lookback_hours | 168 | 7-day context for daily/weekly cycles |
| horizon_hours | 24 | Standard day-ahead forecast |
| cross_validation | enabled, 10 folds | Robust evaluation on time series |

---

## AV (Autonomous Vehicles)

| Param | Value | Rationale |
|-------|-------|------------|
| n_estimators | 200 | Smaller dataset (~122 rows); avoid overfitting |
| max_depth | 6 | Shallow trees for trajectory smoothness |
| num_leaves | 32 | Moderate capacity |
| lookback_hours | 1.0 | Short-term kinematics |
| horizon_hours | 0.1 | Next-step (clamped to 1 in code for conformal) |
| cross_validation | disabled | Single-vehicle trace; limited folds |

---

## Industrial

| Param | Value | Rationale |
|-------|-------|------------|
| n_estimators | 300 | CCPP/synthetic; moderate capacity |
| max_depth | 8 | Process dynamics |
| lookback_hours | 24 | Daily cycle for power/temp |
| horizon_hours | 1 | Short-horizon setpoint |

---

## Healthcare

| Param | Value | Rationale |
|-------|-------|------------|
| n_estimators | 200 | Vital signs; similar to AV scale |
| lookback_hours | 12 | Recent vital-sign history |
| horizon_hours | 1 | Next-reading prediction |

---

## Aerospace

| Param | Value | Rationale |
|-------|-------|------------|
| n_estimators | 200 | Synthetic flight telemetry |
| lookback_hours | 2 | Flight envelope context |
| horizon_hours | 1 | Next-step airspeed/altitude |

---

## Shared Defaults

| Param | Value | Rationale |
|-------|-------|------------|
| learning_rate | 0.03–0.05 | Standard GBM range |
| quantiles | [0.1, 0.5, 0.9] | CQR lower/point/upper |
| train_ratio | 0.70 | Fixed by thesis requirement |
| calibration_ratio | 0.05 | Dedicated conformal split |
