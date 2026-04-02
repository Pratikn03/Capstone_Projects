# ORIUS World-Class AI Research Workflow

This document defines the official AI-assisted workflow for ORIUS. It is
designed for a solo researcher operating with high sensitivity constraints
across code, theorem writing, manuscript work, and evidence generation.

The goal is role separation:

- `Cursor` is the primary coding workspace.
- `ChatGPT` is the primary synthesis and theorem-framing workspace.
- `NotebookLM` is the primary literature-grounding workspace.
- `Claude` is the primary adversarial reviewer.
- Local terminal execution remains the only authority for code execution,
  dataset handling, tests, and evidence generation.

## 1. Tool roles

### Cursor

Use Cursor for:

- Python refactors
- dataset pipeline work
- test-debug loops
- local implementation against the checked-out repo

Do not use Cursor as the final authority on research claims. Every promoted
claim must still be checked against repo truth.

### ChatGPT

Use ChatGPT for:

- theorem statement tightening
- claim-boundary control
- chapter restructuring
- novelty framing
- reviewer simulation
- synthesis across literature or internal evidence summaries

Do not use ChatGPT as the primary code editor for ORIUS.

### NotebookLM

Use NotebookLM for:

- published paper sets
- public PDFs
- sanitized chapter exports
- public notes and bibliographies

Do not use NotebookLM as the main coding or repo-analysis surface.

### Claude

Use Claude for:

- reject-style manuscript review
- theorem skepticism
- novelty skepticism
- overclaim detection
- hostile prose review after a ChatGPT draft

Do not use Claude as the only reviewer for theorem text. Final wording must be
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
3. Send only the changed excerpt, interface, or summarized failure to ChatGPT.
4. Send the same sanitized excerpt to Claude for hostile review.
5. Integrate only what survives both passes.
6. Commit one logical proof, code, or evidence unit.

### Weekly research loop

1. Load 5 to 15 narrow-topic papers into NotebookLM.
2. Use ChatGPT to synthesize the field gap relative to ORIUS.
3. Use Claude to attack the novelty and theorem framing.
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
5. run a ChatGPT synthesis pass on sanitized excerpts
6. run a Claude adversarial pass on the same sanitized excerpts
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

- Draft theorem structure with ChatGPT.
- Attack proof assumptions with Claude.
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

An AI-assisted ORIUS workflow pass is only considered valid if all of the
following hold:

- code changes are implemented and tested locally before a manuscript claim is
  promoted
- theorem wording is reviewed by both a synthesis model and a hostile reviewer
  model
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

- `Cursor` + local terminal for code and execution
- `ChatGPT` for theorem and manuscript synthesis
- `NotebookLM` for literature grounding
- `Claude` for adversarial review

Use fewer tools only if you are dropping a role intentionally. Do not collapse
all four roles into one assistant.
