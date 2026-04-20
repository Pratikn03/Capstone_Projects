---
title: "Phase 3 -- Flagship Proof Packet at Paper-Rigor (V1)"
author: "ORIUS publication planning surface"
date: "2026-04-19"
---

# Phase 3 -- Flagship Proof Packet at Paper-Rigor (V1)

This book is the standalone implementation plan for the Phase 3 proof rewrite.
It is the planning companion to the execution-facing standalone appendix draft
at `appendices/app_c_flagship_proofs.tex` and
`appendices/app_c_flagship_proofs.pdf`. It defines the proof packet that must
be built, the scaffolding that must exist before proof writing begins, the
week-by-week theorem order, the lint and cross-check gates, and the handoff
package for the external reviewer. It does not replace the canonical theorem
registry, the defended theorem audit, or the current manuscript proof surfaces.

## Normalization Note

The pasted draft that motivated this artifact used older theorem grouping and
labeling. This book keeps the same 6-week structure and proof-writing intent,
but it is normalized to current repo truth:

- The draft says `Flagship-5`, but the current registry-canonical flagship
  defended core is `T1`, `T2`, `T3a`, `T4`, `T11`, and `T_trajectory_PAC`.
- `T3b`, `T6`, and `T8` are supporting defended surfaces, not headline
  flagship rows.
- `T7` is the fallback-existence row. `T8` is graceful-degradation dominance.
- The pasted Week 4 fallback material is therefore tracked here as `T7`
  supporting cleanup, while `T8` remains the supporting graceful-dominance
  theorem.
- This book preserves the pasted structure, technical depth, and sequencing,
  but theorem identities, roles, and labels follow
  `reports/publication/theorem_registry.yml` and
  `reports/publication/defended_theorem_core.*`.

## Entry Gate and Prerequisites

Phase 3 starts only after the following are true:

- Phase 1 is complete: assumptions are locked as `A1` through `A13`, Appendix B
  is canonical, and theorem-facing assumption references fail closed on unknown
  IDs.
- Phase 2 is complete: the six critical gaps are fixed on the active theorem
  surfaces, especially absorbed tightening for `T2`, the canonical delta-aware
  `T6`, the `T3a` split, the `T5` retier, the explicit `A10b` scope on `T9`,
  and the explicit `A13` scope on `T10`.
- `reports/publication/theorem_registry.yml` remains the only hand-edited
  theorem inventory.
- The defended-core split remains fixed during this phase:
  `T1`, `T2`, `T3a`, `T4`, `T11`, `T_trajectory_PAC` headline;
  `T3b`, `T6`, `T8` supporting defended;
  `T7` supporting cleanup only unless separately promoted by a later gate.

If those gates are not closed, Phase 3 only restyles broken proofs.

## Deliverable Definition

The execution-facing theorem artifact for this phase is:

- `appendices/app_c_flagship_proofs.tex`
- `appendices/app_c_flagship_proofs.pdf`

The proof packet is supported by the following scoped scaffolding:

- `appendices/flagship_proofs/notation.tex`
- `appendices/flagship_proofs/template_proof.tex`
- `scripts/proof_lint.py`
- `reports/publication/phase3_flagship_v1_proof_book.md`

Until the monograph is explicitly rewired to consume the new standalone
appendix, `appendices/app_c_full_proofs.tex` remains the current shared
manuscript-facing proof surface.

Success condition:

- every flagship theorem proof step is justified by either a registered
  assumption, a named lemma proved in the same packet, or explicit algebra
- no handwave language survives
- all cross-references resolve
- the proof packet can be handed to an external reviewer without needing verbal
  explanation

## Week 1 -- Infrastructure

Week 1 is non-negotiable. It builds the scaffolding that makes Weeks 2 through
6 linear instead of recursive.

### Task 1.1 -- Notation Table

Create `appendices/flagship_proofs/notation.tex` as the single source of truth
for every symbol used anywhere in the proof packet.

Rule:

- if a symbol appears in a proof and not in the notation table, the proof is
  wrong

Representative structure:

```tex
\begin{tabular}{@{}llp{7cm}@{}}
\toprule
Symbol & Type & Meaning \\
\midrule
$x_t$ & $\mathbb{R}$ & latent true state at step $t$ (battery witness row: SOC) \\
$\hat{x}_t$ & $\mathbb{R}$ & observed / telemetry state at step $t$ \\
$a_t$ & $\mathcal{A}$ & action released at step $t$ \\
$w_t$ & $[0,1]$ & OQE reliability weight at step $t$ \\
$m_t$ & $\mathbb{R}_+$ & conformal half-width at step $t$ \\
$m_t^*$ & $\mathbb{R}_+$ & absorbed tightening margin $m_t + \epsilon_{\mathrm{model}}$ \\
$\mathcal{A}_{\mathrm{set}}(\hat{x}_t)$ & $2^\mathcal{A}$ & shield-permitted action set at $\hat{x}_t$ \\
$\mathcal{C}_{\mathrm{rac}}$ & $2^{\mathbb{R}}$ & reliability-adjusted conformal tube \\
$\phi(x)$ & $\{0,1\}$ & safety predicate $\mathbf{1}[s_{\min} \le x \le s_{\max}]$ \\
$\mathcal{H}_t$ & $\sigma$-algebra & causal history up to step $t$ \\
$V_T$ & $\mathbb{N}$ & violation count $\sum_{t=1}^{T}(1-\phi(x_t))$ \\
$\bar{w}$ & $[0,1]$ & episode-average reliability $T^{-1}\sum_{t=1}^{T} w_t$ \\
\bottomrule
\end{tabular}
```

The final table should include theorem-specific objects for `T11` and
`T_trajectory_PAC`, not just the battery witness-row symbols.

### Task 1.2 -- Proof Template

Create `appendices/flagship_proofs/template_proof.tex` and force every
flagship theorem into the same proof shape.

Template:

```tex
\begin{theorem}[T<N>: <Name>]\label{thm:t<n>}
\textbf{Assumptions.} A<i>, A<j>, ...
\textbf{Statement.} <formal predicate with explicit quantifiers and
conditioning, no prose>
\end{theorem}

\begin{proof}
\textbf{Step 1} (<reason: assumption / lemma / algebra>).
  \[ <formula> \]
  Justification: <one sentence tying this to A<i> or Lemma <k>>.

\textbf{Step 2} (<reason>). ...

\textbf{Step N} (conclusion).
  \[ <formula matching the statement> \]
\end{proof}
```

Template rules:

- every step is labeled with its justification type
- every step produces displayed math, not narrative filler
- every justification cites `A<i>` or `Lemma <k>`, not "by construction"
- theorem statements carry explicit quantifiers and conditioning

### Task 1.3 -- Lemma Inventory

Before writing any full proof, list the theorem-local lemmas the packet will
need. The inventory is one line per lemma and is numbered before prose is
filled in.

Registry-normalized inventory:

- `T1` needs:
  `Lemma 1.1` boundary proximity under arbitrage reachability;
  `Lemma 1.2` positive observation-gap probability under admissible dropout;
  `Lemma 1.3` joint positive probability of near-boundary and dropout events.
- `T2` needs:
  `Lemma 2.1` absorbed tightened safe-set soundness;
  `Lemma 2.2` telemetry-tube membership from `A2` and `A5`.
- `T3a` needs:
  `Lemma 3.1` conformal coverage at level `1-\alpha`;
  `Lemma 3.2` reliability inflation monotonicity / calibration bridge;
  `Lemma 3.3` zero violation on the event `x_t \in \mathcal{C}_{rac}` via `T2`.
- `T4` needs:
  `Lemma 4.1` fixed quality-ignorant margin cannot absorb the full witness gap;
  `Lemma 4.2` admissible fault-window construction.
- `T11` needs:
  `Lemma 11.1` coverage implies latent membership in the observation-consistent
  set with probability at least `1-\alpha`;
  `Lemma 11.2` soundness plus repair-membership preserves safety on that event.
- `T_trajectory_PAC` needs:
  `Lemma P.1` finite-sample conformal correction over the horizon;
  `Lemma P.2` tail control under the active concentration assumptions;
  `Lemma P.3` Bonferroni / union-bound aggregation over epochs.
- `T7` supporting cleanup may add:
  `Lemma 7.1` interior-margin stability under bounded one-step dynamics.
- `T8` supporting defended may add:
  `Lemma 8.1` shared-trace sequence comparison for graceful versus uncontrolled
  release policies.

Target count:

- approximately 12 to 14 unique lemmas after shared lemmas are deduplicated

### Task 1.4 -- Gap-Check Script

Create `scripts/proof_lint.py` and make it fail closed on proof-writing debt.

Minimum blocked-language list:

```python
BLOCKED = [
    "clearly",
    "obviously",
    "by construction",
    "plausible",
    "trivially",
    "it follows that",
    "straightforward",
]
```

Minimum registry-consistency check:

```python
REGISTRY = load_yaml("reports/publication/theorem_registry.yml")
for step in proof.steps:
    for cite in step.citations:
        assert cite in REGISTRY.assumptions or cite in REGISTRY.lemmas
```

The final script should also check:

- every theorem declares assumptions explicitly
- every cited lemma exists in the packet
- no theorem cites stale `T3` instead of `T3a` or `T3b`
- `T7` and `T8` labels are not swapped

### Week 1 Exit Criteria

- notation table exists and covers all symbols used in the packet
- proof template exists and is adopted
- lemma inventory is listed and numbered
- `proof_lint.py` runs and passes on the current Phase 2-corrected packet draft

## Week 2 -- T1 and T2

### T1 -- OASG Existence

Scope:

- headline flagship defended
- battery witness-row theorem
- explicit assumptions include `A11` arbitrage reachability and `A12`
  controller-fault independence

Proof shape:

1. `P[E_{close}(t)] \ge p_c > 0` for infinitely many `t`.
   Justification: `Lemma 1.1` boundary proximity under arbitrage reachability.
2. `P[E_{drop}(t)] \ge d > 0` for all `t`.
   Justification: admissible degraded-observation event plus `A12`.
3. `P[E_{close}(t) \cap E_{drop}(t)] \ge \min(p_c,d)/2 > 0`.
   Justification: `Lemma 1.3`.
4. On `E_{close}(t) \cap E_{drop}(t)`, construct an observation-action pair
   such that observed release is safe while true-state postcondition fails.
   Justification: `Lemma 1.1` plus the admissible observation-gap witness.

Week 2 lemma write-up for `T1`:

- `Lemma 1.1` formalizes periodic boundary approach for the battery witness row
  under explicit arbitrage reachability
- `Lemma 1.2` lower-bounds observation gap under admissible dropout
- `Lemma 1.3` combines the two events under the active independence structure

### T2 -- One-Step Safety Preservation

Scope:

- headline flagship defended
- absorbed tightening theorem
- live code witnesses are
  `src/orius/dc3s/safety_filter_theory.py`,
  `src/orius/dc3s/guarantee_checks.py`, and
  `src/orius/certos/runtime.py`

Proof shape:

1. `x_t \in \mathcal{C}_{rac} \Rightarrow |x_t - \hat{x}_t| \le m_t`.
   Justification: `A2`, `A5`, and the definition of the active tube.
2. `a_t \in \mathcal{A}_{set}(\hat{x}_t) \Rightarrow
   s_{min} + m_t^* \le \hat{x}_t + \Delta(a_t) \le s_{max} - m_t^*`.
   Justification: `Lemma 2.1`.
3. `x_{t+1} = \hat{x}_t + \Delta(a_t) + (x_t-\hat{x}_t) + \epsilon_t`,
   with `|\epsilon_t| \le \epsilon_{model}` and `|x_t-\hat{x}_t| \le m_t`.
   Justification: `A1` and `A4`.
4. Algebra closes the lower and upper bounds using
   `m_t^* = m_t + \epsilon_{model}`.

Week 2 lemma write-up for `T2`:

- `Lemma 2.1` uses the active absorbed tightening implementation, not the old
  generic `shield.py` phrasing
- `Lemma 2.2` discharges tube membership under `A2` and `A5`

### Week 2 Exit Criteria

- `T1` proof and three local lemmas drafted in packet form
- `T2` proof and two local lemmas drafted in packet form
- `proof_lint.py` passes on those theorem sections

## Week 3 -- T3a and T4

### T3a -- Per-Step Envelope Derivation

This is the hardest flagship proof in the packet. It is the theorem that the
old overloaded `T3` used to blur.

Proof shape:

1. `P[x_t \in \mathcal{C}_{conformal}(\alpha)\mid \mathcal{H}_t] \ge 1-\alpha`.
   Justification: `Lemma 3.1`.
2. `\mathcal{C}_{rac}(w_t) = \mathcal{C}_{conformal}(\alpha) \oplus B(0,m_t)`,
   where `m_t` changes monotonically with reliability.
   Justification: `A5` and `Lemma 3.2`.
3. Convert the active calibration bridge into the repo-scoped reliability-aware
   coverage statement without reintroducing global weighted-exchangeability
   claims.
4. On `x_t \in \mathcal{C}_{rac}`, invoke `Lemma 3.3` plus `T2` to show the
   repaired release cannot violate the next-step safety predicate.
5. Use conditional decomposition to obtain the per-step envelope.

Week 3 lemma write-up for `T3a`:

- `Lemma 3.1` cites the relevant conformal coverage results instead of
  reproving them
- `Lemma 3.2` states the active reliability inflation / calibration bridge in
  repo-scoped language
- `Lemma 3.3` is the deterministic reduction from active tube membership to `T2`

### T4 -- No Free Safety

Scope:

- headline flagship defended
- fixed-margin quality-ignorant controller class only
- do not widen the controller class inside the proof

Proof shape:

1. Define the quality-ignorant controller class with fixed margin `m^\pi`.
2. Use the explicit reachability assumption to show `m^\pi` cannot absorb the
   full admissible witness gap.
3. Construct an admissible fault window with gap
   `\delta \in (m^\pi, g_{max}]`.
4. Build the unsafe observed-safe release witness at that step.

Week 3 lemma write-up for `T4`:

- `Lemma 4.1` keeps the fixed-margin scope explicit
- `Lemma 4.2` constructs the admissible fault window

### Week 3 Exit Criteria

- `T3a` proof and its three local lemmas drafted
- `T4` proof and its two local lemmas drafted
- `proof_lint.py` passes on the expanded packet

## Week 4 -- T11, T7/T8 Cleanup, and T_trajectory_PAC

### T11 -- Typed Structural Transfer

Scope:

- headline flagship defended
- forward-only four-obligation theorem
- no silent promotion of the supporting five-invariant mini-harness

Required polish:

- add explicit quantification over step and admissible disturbance
- cite the four obligations by name:
  `T11.Cov`, `T11.Snd`, `T11.Rep`, `T11.FB`
- state explicitly that `T11` does not prove those obligations for each domain;
  domain discharge remains Phase 4 work

### T7 -- Fallback-Existence Supporting Cleanup

The pasted draft folded fallback existence into `T8`. In current repo truth,
that material belongs under `T7`, and `T7` is not part of the headline
flagship defended core.

Supporting cleanup proof shape:

1. interior safe state with slack at least `\epsilon_{model}`
2. bounded one-step dynamics under the fallback action
3. algebra closing the safety interval after one step

Use this only as supporting cleanup unless the theorem is separately promoted by
another gate.

### T8 -- Graceful-Degradation Dominance

Current scope:

- supporting defended
- shared fault trace
- sequence-level comparison only

The proof must stay helper-scoped:

- graceful policy and uncontrolled policy are evaluated on the same admissible
  fault trace
- comparison is about violation-sequence dominance, not a universal
  absorbing-landing dynamics theorem

### T_trajectory_PAC

Scope:

- headline flagship defended in current repo truth
- keep the theorem narrowed to the implemented Bonferroni / union-bound
  certificate

Proof shape:

1. finite-sample conformal correction over the horizon
2. tail control under the active concentration assumptions
3. union-bound / epoch aggregation with explicit effective-sample accounting

Do not silently strengthen this theorem into a martingale result during Phase 3.

### Week 4 Exit Criteria

- `T11` proof polished into template form
- `T7` fallback material disentangled from `T8`
- `T8` restated and proved as supporting sequence-level dominance
- `T_trajectory_PAC` proof and three local lemmas drafted

## Week 5 -- Lemma Consolidation and Cross-Checks

Week 5 is a packet-integrity week, not a new-theorem week.

### Task 5.1 -- Lemma Audit

For every lemma in the inventory:

- statement typeset
- proof written in the same packet
- references resolved
- theorem citations consistent with the packet numbering

Expected unique lemma count after deduplication:

- approximately 12 to 14

### Task 5.2 -- Cross-Reference Verification

Reference check:

```bash
pdflatex -interaction=nonstopmode appendices/app_c_flagship_proofs.tex 2>&1 | \
  grep -E "undefined|multiply"
```

Expected output:

- zero lines

### Task 5.3 -- Assumption Citation Audit

Assumption declarations must equal proof citations for each flagship theorem.

Representative check:

```python
from proof_lint import extract_assumption_citations

for thm in flagship_theorems:
    declared = thm.assumptions
    cited = extract_assumption_citations(thm.proof_file)
    assert set(declared) == set(cited), f"{thm.id}: mismatch"
```

### Week 5 Exit Criteria

- every used lemma is proved in the same packet
- no undefined references remain
- declared assumptions match cited assumptions on each flagship theorem

## Week 6 -- Integration and Red-Team Prep

### Task 6.1 -- Self-Read as Adversary

Read the full packet cold and annotate it like a skeptical reviewer.

For each theorem:

- mark every step that causes a pause
- write a reviewer-style objection
- fix the top ten paused points before the packet leaves the repo

Expected fix count:

- roughly 5 to 15 real corrections

### Task 6.2 -- Boundary-Case Sweep

Each theorem gets boundary cases written down explicitly. Representative cases:

- `T3a`: if `w_t = 0` for all `t`, the bound degenerates but must remain valid
- `T2`: if `\epsilon_{model} = 0`, the absorbed-tightening algebra should reduce
  cleanly
- `T1`: if degraded-observation rate collapses to zero, the existence witness
  must not overclaim
- `T_trajectory_PAC`: if calibration sample size is small, the certificate must
  fail conservatively rather than silently become vacuous

Any broken edge case means a missing hypothesis or a mislabeled scope boundary.

### Task 6.3 -- External Red-Team Package

Produce the packet that Phase 6 will actually hand to the reviewer:

- `appendices/app_c_flagship_proofs.pdf`
- `reports/publication/theorem_registry.yml`
- Appendix B PDF / assumption register export
- `reports/publication/lemma_index.md`
- `reports/publication/proof_lint_pass.log`

### Week 6 Exit Criteria

- full packet compiles cleanly
- no handwave language survives
- every proof step is justified
- reviewer handoff bundle is complete

## Parallel Phase 4 Note

Phase 4 can begin in parallel once Week 3 stabilizes. The proof packet and the
runtime-facing theorem linkage should advance together, not in separate worlds.

Recommended parallel rule:

- one theorem-facing property-based test file per active theorem surface as the
  proof text stabilizes

By the end of Week 6, the proof packet should have matching runtime or test
surfaces for:

- `T2`
- `T3a`
- `T6`
- `T11`
- `T_trajectory_PAC`

`T7` and `T8` stay explicitly supporting unless separately promoted later.

## Common Pitfalls

- Starting proof writing before Phase 1 or Phase 2 is actually closed.
- Letting lemma proofs grow into hidden theorems.
- Copying the structure of `appendices/app_c_full_proofs.tex` instead of
  rewriting from the template.
- Re-deriving known conformal or concentration results instead of citing the
  literature and narrowing the local bridge step.
- Reintroducing stale `T3` language after the `T3a` / `T3b` split.
- Swapping `T7` fallback existence and `T8` graceful dominance.
- Quietly widening the scope of `T4`, `T8`, or `T11`.
- Adding new theorem families during the packet rewrite.

## Exit Gate

Phase 3 may be closed only when all of the following are true:

- every flagship theorem uses the packet template
- every flagship proof step is justified by a registered assumption, a named
  packet-local lemma, or explicit algebra
- every lemma used is proved in the same packet
- `proof_lint.py` passes with zero blocked phrases
- the notation table covers every symbol used
- all references resolve
- the adversarial self-read has been completed and the fixes applied

If any one of those gates fails, Phase 3 is not closed.

## Time Budget

| Week | Output | Hours |
| --- | --- | ---: |
| 1 | Infrastructure scaffold | 20 |
| 2 | `T1`, `T2`, five lemmas | 25 |
| 3 | `T3a`, `T4`, five lemmas | 30 |
| 4 | `T11`, `T7/T8` cleanup, `T_trajectory_PAC`, four lemmas | 25 |
| 5 | Lemma consolidation and cross-checks | 20 |
| 6 | Integration and red-team prep | 15 |
| Total | Submission-ready proof packet | 135 |

Calendar interpretation:

- `15` to `20` hours per week implies roughly `7` to `9` weeks elapsed time
- `25` hours per week implies roughly `5` to `6` weeks elapsed time

The calendar should be blocked explicitly. This phase is where the project
stops looking like an ambitious bachelor's project and starts looking like a
reviewable theorem packet.

## Appendix -- Theorem Mapping and Artifact Map

### Current theorem-role map

| Theorem | Current role | Phase 3 handling |
| --- | --- | --- |
| `T1` | Flagship defended | Full packet proof rewrite |
| `T2` | Flagship defended | Full packet proof rewrite with absorbed tightening |
| `T3a` | Flagship defended | Full packet proof rewrite |
| `T3b` | Supporting defended | Mention only as aggregation corollary |
| `T4` | Flagship defended | Full packet proof rewrite |
| `T6` | Supporting defended | Referenced as auxiliary runtime-linked theorem |
| `T7` | Draft / non-defended fallback surface | Supporting cleanup only |
| `T8` | Supporting defended | Supporting graceful-dominance rewrite |
| `T11` | Flagship defended | Full packet proof polish |
| `T_trajectory_PAC` | Flagship defended | Full packet proof rewrite |

### Artifact map

| Artifact | Role |
| --- | --- |
| `appendices/app_c_flagship_proofs.tex` | Final submission-facing proof packet |
| `appendices/flagship_proofs/notation.tex` | Symbol source of truth |
| `appendices/flagship_proofs/template_proof.tex` | Forced theorem template |
| `scripts/proof_lint.py` | Proof-writing lint and citation checker |
| `reports/publication/theorem_registry.yml` | Canonical theorem inventory |
| `reports/publication/defended_theorem_core.md` | Current defended-core classification |
| `reports/publication/lemma_index.md` | Reviewer-facing lemma index |
| `reports/publication/proof_lint_pass.log` | Zero-violation lint evidence |

### Core references to cite rather than rederive

- Vovk et al. on conformal validity
- Barber et al. on modern conformal framing
- Shalev-Shwartz and related concentration references for finite-sample and
  sub-Gaussian tails

The packet should cite known results where appropriate and reserve local proof
effort for the ORIUS-specific bridge steps, scope boundaries, and code-facing
assumption alignment.
