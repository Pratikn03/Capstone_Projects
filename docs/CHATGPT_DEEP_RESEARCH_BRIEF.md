# External Research Verification Brief (ORIUS)

## Purpose
Use this brief as the primary context file when asking an external research
assistant to audit ORIUS. It points to the latest canonical artifacts in this
repository and highlights known evaluation caveats.

## Snapshot Date
- Brief created: 2026-03-11 (America/Chicago)

## Canonical Artifact Sources
- Publication release manifest: `reports/publication/release_manifest.json`
  - `generated_at_utc`: `2026-02-26T10:26:49.593877+00:00`
  - `git_commit`: `e2df1e533b3771aaad002198a5942aa61769a878`
  - scenarios: `nominal`, `dropout`
  - seeds: `0..9`
- Metrics policy manifest: `paper/metrics_manifest.json`
  - `generated_at_utc`: `2026-02-19T00:00:00Z`
  - DE run id: `20260217_165756`
  - US run id: `20260217_182305`
- Statistical gate summary: `reports/publication/stats_summary.json`
  - `generated_at`: `2026-02-27T03:04:58.366158`
  - `overall_pass`: `false`
  - primary gate (`aggregate_fault_sweep`) currently fails target relative reduction threshold (p-values pass, effect-size target does not)

## Current Canonical Impact Numbers (from `paper/metrics_manifest.json`)
- Germany (DE):
  - cost savings: `7.11%`
  - carbon reduction: `0.30%`
  - peak shaving: `6.13%`
- USA (US / EIA-930):
  - cost savings: `0.11%`
  - carbon reduction: `0.13%`
  - peak shaving: `0.00%`

## Model + Dataset Profile Anchors
- DE profile: 17,377 rows, 98 columns (`paper/metrics_manifest.json` -> `dataset_profiles.de`)
- US profile: 13,638 rows, 118 columns (`paper/metrics_manifest.json` -> `dataset_profiles.us`)
- Main manuscript: `paper/PAPER_DRAFT.md`
- Runtime system overview: `README.md`, `docs/ARCHITECTURE.md`, `docs/ASSUMPTIONS_AND_GUARANTEES.md`

## What To Ask An External Research Assistant To Verify
- Whether narrative claims in `README.md` and `paper/PAPER_DRAFT.md` are fully consistent with:
  - `paper/metrics_manifest.json`
  - `reports/publication/release_manifest.json`
  - `reports/publication/stats_summary.json`
- Whether any claims imply statistical significance beyond what the gate summary supports (`overall_pass=false`).
- Whether dashboards/API docs align with existing routes and deployment manifests:
  - API service: `services/api/`
  - Frontend app: `frontend/`
  - deploy specs: `deploy/`

## Suggested Prompt Starter
```
You are auditing the ORIUS repository for research-grade consistency.
Treat these files as canonical evidence:
1) reports/publication/release_manifest.json
2) paper/metrics_manifest.json
3) reports/publication/stats_summary.json
4) paper/PAPER_DRAFT.md
5) README.md

Tasks:
- Extract all numerical claims from README and PAPER_DRAFT.
- Verify each claim against canonical artifacts.
- Flag mismatches, missing caveats, and overstatements.
- Propose exact text edits to make claims evidence-accurate.
- Provide a final table: claim | source line | evidence file/field | status | corrected wording.
```
