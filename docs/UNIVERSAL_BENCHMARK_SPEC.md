# ORIUS-Bench Universal Specification

ORIUS-Bench is the thesis-facing replay and metrics contract for cross-domain
evaluation under degraded observation. This document defines the canonical
schema and metrics for the final ORIUS thesis package.

## Scope

ORIUS-Bench standardizes:

- replay inputs and per-step outputs
- the fault model used across domains
- the domain-agnostic benchmark schema
- the canonical metric family used in manuscript and governance locks

## Canonical step schema

Each step record must support the following universal fields:

| Field | Type | Meaning |
|---|---|---|
| `step` | int | replay timestep |
| `true_state` | object | hidden or trusted state view |
| `observed_state` | object | degraded observation used by the controller |
| `action` | object | dispatched or repaired action |
| `true_constraint_violated` | bool or null | whether the true state violates the safety surface |
| `observed_constraint_satisfied` | bool or null | whether the observed state appears safe |
| `true_margin` | float or null | scalar true-state margin to the constraint boundary |
| `observed_margin` | float or null | scalar observed-state margin to the constraint boundary |
| `intervened` | bool | whether repair overrode the candidate action |
| `fallback_used` | bool | whether fallback mode was active |
| `certificate_valid` | bool or null | post-step certificate validity |
| `latency_us` | float or null | end-to-end runtime latency for the step |
| `domain_metrics` | object | optional domain-specific extras that do not redefine the canonical schema |

Legacy SOC-centric fields may be accepted by compatibility code, but they are
not the thesis-facing benchmark contract.

## Canonical metrics

| Metric | Meaning |
|---|---|
| `TSVR` | True-State Violation Rate |
| `OASG` | Observation-Action Safety Gap |
| `CVA` | Certificate Validity Accuracy |
| `GDQ` | Graceful Degradation Quality |
| `IR` | Intervention Rate |
| `AC` | Audit Completeness |
| `RL` | Recovery Latency |

Additional release-facing summaries:

- fallback rate
- latency p50 / p95 / p99
- coverage diagnostics when conformal uncertainty is active

## Fault model

The shared fault protocol is expressed through the following families:

- dropout
- stale readings
- delay / jitter
- out-of-order telemetry
- spikes / corruption
- drift or shift flags when the domain supports them

The protocol is intentionally shared so that transfer behavior can be compared
without re-solving evaluation semantics for each domain.

## Domain responsibilities

Each benchmark domain must expose:

- true-state violation predicate
- observed-state satisfiability predicate
- optional scalar safety margin function
- domain-specific useful-work computation
- any additional domain metrics in a nested payload only

## Artifact authority

Canonical benchmark claims resolve against tracked publication assets such as:

- `reports/publication/dc3s_main_table_ci.csv`
- `reports/publication/dc3s_latency_summary.csv`
- `reports/publication/fault_performance_table.csv`
- `reports/universal_orius_validation/`

Local dashboard caches and ignored generated files are not canonical evidence.
