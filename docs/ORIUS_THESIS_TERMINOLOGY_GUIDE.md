# ORIUS Thesis Terminology Guide

This guide defines the canonical language for the final ORIUS thesis package.
Use these terms in universal chapters, benchmark specs, governance specs, and
artifact-facing documentation. Battery-specific language is still allowed in
energy-domain chapters and evidence sections where the physical meaning matters.

## Canonical universal terms

| Use this | Meaning | Replace when writing universal sections |
|---|---|---|
| `degraded observation` | the sensed channel is incomplete, stale, corrupted, or shifted relative to the plant | `bad telemetry`, `battery data fault` |
| `observation-action safety gap (OASG)` | action appears safe on the observed channel but is unsafe on the true plant state | `SOC-only safety gap`, `dispatch illusion` |
| `true-state violation` | the true plant state violates a constraint after applying an action | `true-SOC violation` outside energy chapters |
| `constraint margin` | scalar or vector distance to the safety boundary | `SOC headroom` outside energy chapters |
| `repair` | transformation from candidate action to action inside the safe set | `dispatch clipping` outside energy chapters |
| `safety certificate` | runtime object recording repaired action, reliability, uncertainty, and assumptions | `battery certificate` outside energy chapters |
| `observation-consistent state set` | conservative true-state set induced by observation and reliability | `SOC tube` outside energy chapters |
| `domain adapter` | domain-specific bridge between the shared ORIUS kernel and a plant/runtime surface | `battery adapter` outside energy chapters |
| `fallback action` | safe default action used when normal certificate validity collapses | `zero dispatch` outside energy chapters |
| `tracked publication artifact` | checked-in report, manifest, table, or figure used as canonical evidence | `dashboard snapshot`, `local cache` |

## When battery language is still correct

Battery-specific language should remain in:

- energy-domain method and results chapters
- battery-only theorem instantiations
- dispatch economics and carbon-impact sections
- physical battery state/action descriptions such as `SOC`, `charge_mw`, and `discharge_mw`

## Writing rules

1. Universal chapters describe the kernel, contracts, governance, and benchmark in domain-neutral language.
2. Battery remains the strongest empirical witness, but not the conceptual center of the thesis.
3. AV, navigation, and aerospace must keep their tier labels explicit; do not flatten tier differences into generic “validated everywhere” language.
4. If a claim is battery-only, say so directly instead of generalizing it.
5. Prefer “true-state violation rate (TSVR)” over any SOC-specific metric name outside energy chapters.

## Tier language

Use the following evidence labels consistently:

- `reference witness`
- `proof-validated`
- `proof-candidate`
- `shadow-synthetic`
- `experimental`

Do not introduce alternate tier labels in manuscript prose or supporting docs.
