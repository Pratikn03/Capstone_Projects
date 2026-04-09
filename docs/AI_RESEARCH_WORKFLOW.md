# ORIUS Model-Assisted Research Workflow

This document defines the official assistant-supported workflow for ORIUS. It
is designed for a solo researcher operating with high-sensitivity constraints
across code, theorem writing, manuscript work, and evidence generation.

The goal is role separation:

- the implementation workspace is the primary coding surface
- the synthesis workspace is the primary theorem-framing and manuscript
  restructuring surface
- the literature-grounding workspace is the primary published-source review
  surface
- the adversarial review workspace is the primary hostile reviewer
- local terminal execution remains the only authority for code execution,
  dataset handling, tests, and evidence generation

## 1. Tool roles

### Implementation Workspace

Use the implementation workspace for:

- Python refactors
- dataset pipeline work
- test-debug loops
- local implementation against the checked-out repo

Do not use the implementation workspace as the final authority on research claims. Every promoted
claim must still be checked against repo truth.

### Synthesis Workspace

Use the synthesis workspace for:

- theorem statement tightening
- claim-boundary control
- chapter restructuring
- novelty framing
- reviewer simulation
- synthesis across literature or internal evidence summaries

Do not use the synthesis workspace as the primary code editor for ORIUS.

### Literature-Grounding Workspace

Use the literature-grounding workspace for:

- published paper sets
- public PDFs
- sanitized chapter exports
- public notes and bibliographies

Do not use the literature-grounding workspace as the main coding or repo-analysis surface.

### Adversarial Review Workspace

Use the adversarial review workspace for:

- reject-style manuscript review
- theorem skepticism
- novelty skepticism
- overclaim detection
- hostile prose review after a synthesis draft

Do not use the adversarial review workspace as the only reviewer for theorem text. Final wording must be
checked against code, tests, and artifacts locally.

### Local terminal / repo-grounded agent

Use local terminal execution for:

- all code execution
- all dataset ingestion
- all training and backtests
- all artifact generation
- theorem-to-code trace audits
- commits and validation runs

This is the only execution surface that may be treated as authoritative.

## 2. Artifact handling policy

### Safe to upload to hosted models

- excerpts from `paper/paper.tex`
- excerpts from `chapters/`
- public figures and public tables
- sanitized logs
- aggregate metric summaries
- public-facing artifacts from `reports/publication/`
- pseudocode
- theorem text
- non-sensitive configs
- small synthetic examples

### Keep local-only unless explicitly sanitized

- `data/*/raw/`
- `data/*/processed/`
- `reports/runs/`
- full repo dumps
- full audit ledgers
- private experiment outputs
- anything that reveals raw healthcare, industrial, AV, aerospace, or battery
  telemetry
- anything that reconstructs sensitive data distributions in detail

### Local-only workflows

These must stay local:

- raw data ingestion
- training runs
- backtests
- uncertainty artifact generation
- promotion logic
- runtime safety validation
- integrated theorem-to-code checks

## 3. Default ORIUS operating loop

### Daily build loop

1. Implement one concrete code or manuscript change locally.
2. Run targeted tests or builds locally.
3. Send only the changed excerpt, interface, or summarized failure to the synthesis workspace.
4. Send the same sanitized excerpt to the adversarial review workspace for hostile review.
5. Integrate only what survives both passes.
6. Commit one logical proof, code, or evidence unit.

### Weekly research loop

1. Load 5 to 15 narrow-topic papers into the literature-grounding workspace.
2. Use the synthesis workspace to summarize the field gap relative to ORIUS.
3. Use the adversarial review workspace to attack the novelty and theorem framing.
4. Update exactly one surface locally:
   - theorem surface
   - benchmark surface
   - paper positioning
   - dataset or evidence surface

### Per-feature workflow

For any major ORIUS addition:

1. implement locally
2. add or update tests locally
3. generate artifacts locally
4. draft the manuscript claim in `paper/` or `chapters/`
5. run a synthesis pass on sanitized excerpts
6. run an adversarial review pass on the same sanitized excerpts
7. accept the feature only if:
   - code exists
   - tests exist
   - artifact exists
   - manuscript wording matches evidence

## 4. ORIUS-specific rules

### Code

- Keep implementation and debugging local.
- Use hosted models only on excerpts, interfaces, stack traces, or summarized
  metrics.
- Never ask a hosted model to reason from a raw dataset dump.

### Theorems

- Draft theorem structure with the synthesis workspace.
- Attack proof assumptions with the adversarial review workspace.
- Final theorem wording must be checked against the actual contracts in:
  - `src/orius/universal_theory/`
  - `src/orius/dc3s/`
  - `src/orius/certos/`

### Manuscript

- `paper/paper.tex` remains the canonical paper and thesis manuscript surface.
- `chapters/` remains the longform dissertation and theorem structure.
- `reports/publication/` remains the release-grade evidence surface.
- No hosted model may be treated as permission to widen claim tiers beyond what
  the repo evidence supports.

### Datasets

- Treat all raw domain datasets as controlled material.
- If external model help is needed, share only:
  - schema
  - row counts
  - feature names
  - aggregate distributions
  - summarized failure descriptions

## 5. Acceptance checks

An assistant-supported ORIUS workflow pass is only considered valid if all of the
following hold:

- code changes are implemented and tested locally before a manuscript claim is
  promoted
- theorem wording is reviewed by both a synthesis workspace and a hostile review
  workspace
- every promoted claim maps to:
  - code
  - test
  - artifact
  - manuscript text
- hosted tools saw only sanitized artifacts
- the final paper or thesis change is defensible from repo truth without
  relying on model memory

## 6. Minimal stack recommendation

For this repo, the default stack is:

- one implementation workspace plus local terminal for code and execution
- one synthesis workspace for theorem and manuscript synthesis
- one literature-grounding workspace for paper and PDF review
- one adversarial review workspace for skeptical review

Use fewer tools only if you are dropping a role intentionally. Do not collapse
all four roles into one assistant.
