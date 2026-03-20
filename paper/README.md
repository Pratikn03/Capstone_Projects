# ORIUS Manuscript Authoring Guide

This folder is the manuscript control room for the ORIUS thesis and paper
variants. Use it as the **authoring guide** for:

- writing the canonical long-form manuscript,
- keeping paper claims synchronized with locked artifacts,
- regenerating tables and figures from trained runs,
- moving from training outputs to a paper-ready PDF.

Canonical policy:

- source of truth: `paper/paper.tex`
- official compiled deliverable: repo-root `paper.pdf`
- shorter derivative: `paper/paper_r1.tex`
- deprecated reference only: `../orius_battery_409page_figures_upgraded_main.*`

---

## 1. Manuscript roles

| File | Role |
|---|---|
| `paper.tex` | Canonical LaTeX manuscript |
| `paper_r1.tex` | Conference / submission variant |
| `PAPER_DRAFT.md` | Narrative draft and argument companion |
| `../paper.pdf` | Official compiled manuscript deliverable |
| `metrics_manifest.json` | Locked metrics source of truth |
| `claim_matrix.csv` | Claim-to-evidence map |
| `manifest.yaml` | Paper artifact manifest |
| `sync_rules.md` | Rules for syncing manuscript text with artifacts |

---

## 2. What “more thesis-like” should mean here

The thesis manuscript should read like a defended research document, not just a
conference paper. That means:

1. **clear claim boundary**  
   - battery is the validated reference domain,  
   - industrial and healthcare are the current proof-validated non-battery
     rows under the canonical replay harness,
   - AV is a proof candidate,
   - navigation is synthetic-shadow evidence,
   - aerospace is experimental.

2. **chapter-like structure**  
   - problem and system context,  
   - method and theorem blocks,  
   - experimental protocol,  
   - results and ablations,  
   - limitations / claim boundary,  
   - reproducibility and audit trail.

3. **artifact-grounded writing**  
   - all numbers should come from locked manifests or generated tables,  
   - figures/tables should be traceable back to `reports/publication/`.

4. **appendix-heavy support**  
   - full proofs, theorem sync, claim evidence registers, and audit appendices
     should stay in `appendices/` and be referenced explicitly.

---

## 3. Manuscript source layout

The canonical long-form story is spread across:

- `paper.tex` for the main compiled manuscript,
- `chapters/` for the chapter-level source blocks,
- `appendices/` for proofs, audits, and traceability,
- `reports/publication/` for locked result artifacts.

Important proof/evidence sources:

- `../chapters/ch16_battery_theorem_oasg_existence.tex`
- `../chapters/ch17_battery_theorem_safety_preservation.tex`
- `../appendices/app_c_full_proofs.tex`
- `../appendices/app_m_verified_theorems_and_gap_audit.tex`
- `../appendices/app_s_claim_evidence_registers.tex`
- `../appendices/app_z_theorem_and_paper_sync_registers.tex`

---

## 4. Tables and figures: source of truth

### Canonical table sources

Use these as the primary paper-facing table inputs:

- `../reports/publication/dc3s_main_table.csv`
- `../reports/publication/dc3s_fault_breakdown.csv`
- `../reports/publication/fault_performance_table.csv`
- `../reports/publication/baseline_comparison_all.csv`
- `../reports/publication/claim_evidence_matrix.csv`
- `../reports/publication/artifact_traceability_table.csv`
- `../reports/publication/dataset_cards.csv`

### Canonical figure sources

Use these as the primary paper-facing figure inputs:

- `../reports/figures/architecture.png`
- `../reports/publication/fig_48h_trace.png`
- `../reports/publication/calibration_plot.png`
- `../reports/publication/blackout_half_life.png`
- `../reports/publication/ablation_sensitivity.png`
- `../reports/publication/cross_region_transfer.png`
- `../reports/paper1/fig_cost_safety_frontier.png`
- `../reports/paper2/fig_certificate_shrinkage.png`
- `../reports/paper3/fig_degradation_trajectory.png`

### Rule

Do **not** hand-edit paper numbers when the corresponding table or figure can be
regenerated from `reports/publication/` or a manifest-backed script.

---

## 5. Training → tables/figures → manuscript workflow

Use the thesis-oriented workflow below when pushing the project toward a more
complete thesis package.

### Step 1 — run training

```bash
export RELEASE_ID=FINAL_20260319T000000Z
export PROFILE=standard

make thesis-train RELEASE_ID=$RELEASE_ID PROFILE=$PROFILE
```

### Step 2 — run the benchmark / validation path

```bash
make thesis-bench RELEASE_ID=$RELEASE_ID
```

### Step 3 — build publication tables and figures

```bash
make thesis-artifacts RELEASE_ID=$RELEASE_ID
```

### Step 4 — verify and compile the canonical manuscript

```bash
make thesis-manuscript
```

### Step 5 — freeze paper-facing outputs

```bash
make thesis-freeze RELEASE_ID=$RELEASE_ID
```

---

## 6. Recommended authoring loop

When writing the thesis, iterate in this order:

1. update manuscript text in `paper.tex` / `PAPER_DRAFT.md`,
2. regenerate affected result artifacts,
3. run paper sync and claim validation,
4. compile the PDF,
5. only then promote the wording into final thesis-facing prose.
5. confirm the canonical output landed at repo root as `paper.pdf`.

---

## 7. Commands you will use most

| Command | Purpose |
|---|---|
| `make thesis-train RELEASE_ID=... PROFILE=...` | Run thesis-facing training stage |
| `make thesis-bench RELEASE_ID=...` | Run benchmark / CPSBench stage |
| `make thesis-artifacts RELEASE_ID=...` | Build publication artifact + paper assets |
| `make thesis-manuscript` | Validate and compile the canonical manuscript to `../paper.pdf` |
| `make thesis-freeze RELEASE_ID=...` | Freeze verified outputs into the paper path |
| `make paper-sync` | Check paper asset synchronization |
| `make publish-audit` | Run final publication audit |

---

## 8. Definition of done for a “thesis-like” manuscript

The manuscript is in a strong thesis state when:

- the main story is chapter-shaped rather than short-paper-shaped,
- every headline result is backed by a locked table/figure/manifest artifact,
- theorem references and appendix proofs are synchronized,
- claim validation passes,
- paper assets compile without placeholder drift,
- the release ID used for training and the paper-facing outputs is recorded.
