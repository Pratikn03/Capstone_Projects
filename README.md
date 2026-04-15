# ORIUS — Observation–Reality Integrity for Universal Safety

> A universal runtime safety layer for Physical AI under degraded observation.
> The canonical manuscript is a book-length monograph built from tracked evidence and generator-backed assets.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Domains](https://img.shields.io/badge/domains-6%20runtime%20rows-orange)
![Monograph](https://img.shields.io/badge/monograph-450%2B%20pages-blueviolet)
![Bibliography](https://img.shields.io/badge/references-150%2B-green)
![License](https://img.shields.io/badge/license-research-lightgrey)

## Overview

**ORIUS** addresses a core hazard in cyber-physical and Physical AI systems: a controller can evaluate safety against the observed state while the true physical state has already crossed a constraint boundary. ORIUS closes that Observation-Action Safety Gap (OASG) by making observation quality part of the runtime state, inflating uncertainty when telemetry degrades, tightening the admissible action set, repairing unsafe actions, and emitting auditable runtime certificates.

The book-level claim in this repository is:

> ORIUS is a fundamental runtime safety layer for Physical AI under degraded observation.

The current monograph is **universal-first**, but it is also **artifact-strict** about evidence depth:

- `battery` = witness row with deepest theorem-to-artifact closure
- `av` = defended bounded row under the TTC plus predictive-entry-barrier contract
- `industrial` = defended bounded row
- `healthcare` = defended bounded row
- `navigation` = shadow/support-tier row until the KITTI-built replay surface is staged
- `aerospace` = experimental/support-tier row; the bounded public ADS-B lane does not promote the defended row

Equal-domain universality remains a program target, not a present-tense repository claim. Support-tier execution for blocked rows is explicit opt-in.

The bounded-universal closure target for the next evidence cycle is stricter:

- `battery` remains the witness row
- `av`, `industrial`, `healthcare`, `navigation`, and `aerospace` must all clear the same defended bounded-row promotion gate
- strict validation fails loudly if the KITTI-backed navigation replay row is not staged
- strict validation fails loudly if the canonical aerospace real-flight runtime row is not staged
- the public ADS-B aerospace lane remains support-only and cannot clear the defended promotion gate by itself

## Canonical surfaces

The active control surfaces for the monograph are:

- [`paper/paper.tex`](paper/paper.tex): canonical manuscript controller
- [`paper/paper.pdf`](paper/paper.pdf): canonical compiled book
- [`paper/review/orius_review_dossier.tex`](paper/review/orius_review_dossier.tex): reviewer dossier source
- [`paper/metrics_manifest.json`](paper/metrics_manifest.json): manuscript metric authority
- [`paper/claim_matrix.csv`](paper/claim_matrix.csv): claim-to-evidence map
- [`reports/publication/orius_equal_domain_parity_matrix.csv`](reports/publication/orius_equal_domain_parity_matrix.csv): domain posture summary
- [`reports/publication/orius_domain_closure_matrix.csv`](reports/publication/orius_domain_closure_matrix.csv): defended vs gated closure matrix
- [`reports/publication/orius_universal_claim_matrix.csv`](reports/publication/orius_universal_claim_matrix.csv): universal claim register

Generator-owned monograph assets are produced by:

- [`scripts/build_orius_monograph_assets.py`](scripts/build_orius_monograph_assets.py)

That generator owns:

- `paper/monograph/`
- `paper/review/`
- `paper/bibliography/orius_monograph.bib`
- parity and closure matrices
- reviewer appendix assets
- artifact-index and chapter-map tables

Do not hand-edit generator-owned outputs unless the generator change is made in the same patch.

For the bounded-universal closure program, the active code/data closure helpers are:

- [`scripts/refresh_real_data_manifests.py`](scripts/refresh_real_data_manifests.py)
- [`scripts/verify_real_data_preflight.py`](scripts/verify_real_data_preflight.py)
- [`scripts/run_universal_training_audit.py`](scripts/run_universal_training_audit.py)
- [`scripts/run_universal_orius_validation.py`](scripts/run_universal_orius_validation.py)
- [`scripts/build_domain_closure_matrix.py`](scripts/build_domain_closure_matrix.py)

## Runtime kernel

The ORIUS runtime kernel is:

```text
Detect → Calibrate → Constrain → Shield → Certify
```

Implemented runtime boundary:

- `DomainAdapter` is the sole canonical runtime adapter contract.
- `src/orius/universal_theory/` is the domain-neutral theory and contract surface.
- `src/orius/orius_bench/` defines the domain-neutral replay and metrics schema.
- `src/orius/certos/` provides runtime governance and audit semantics.
- backend-served tracked artifacts are the only canonical dashboard/report truth path.

The canonical benchmark fields are:

- `true_constraint_violated`
- `observed_constraint_satisfied`
- `true_margin`
- `observed_margin`
- `intervened`
- `fallback_used`
- `certificate_valid`
- `latency_us`
- `domain_metrics`

## Evidence posture

ORIUS is universal at the architecture level today. Evidence parity remains governed by tracked artifacts.

Promotion to a defended row requires:

- valid tracked data manifest
- reproducible replay or train/cal/test protocol
- compatibility with the universal benchmark contract
- compatibility with certificate and governance surfaces
- bounded theorem alignment where applicable
- material safety improvement under the locked protocol
- latency and runtime audit pass

Battery remains the deepest witness row. Navigation and aerospace remain lower-tier until their defended runtime surfaces are staged and revalidated; support-tier outputs do not upgrade those rows by default.

## Build and verification

Regenerate monograph assets:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_orius_monograph_assets.py
```

Build the book:

```bash
make orius-book
```

Build the reviewer dossier:

```bash
make orius-review-pack
```

Run the claim validator:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_paper_claims.py
```

Run the publication/submission audit:

```bash
make publish-audit
```

## Repository map

```text
.
├── src/orius/                  Runtime package
│   ├── adapters/               Domain packages and helpers
│   ├── dc3s/                   Safety kernel implementation
│   ├── certos/                 Runtime governance and audit chain
│   ├── orius_bench/            Universal replay and metrics
│   ├── universal_theory/       Domain-neutral contracts and theorem bridge
│   ├── multi_agent/            Shared-constraint composition
│   └── forecasting/            Forecasting and calibration support
├── services/api/              FastAPI service layer
├── frontend/                  Next.js dashboard
├── scripts/                   Build, validation, and publication tooling
├── paper/                     Canonical manuscript, monograph assets, review dossier
├── chapters/                  Static thesis-depth chapter sources
├── docs/                      Architecture and workflow documentation
├── reports/publication/       Active publication artifacts
└── reports/legacy_archive/    Quarantined historical frozen bundles
```

## Documentation map

- [`docs/UNIVERSAL_KERNEL_ARCHITECTURE.md`](docs/UNIVERSAL_KERNEL_ARCHITECTURE.md)
- [`docs/UNIVERSAL_BENCHMARK_SPEC.md`](docs/UNIVERSAL_BENCHMARK_SPEC.md)
- [`docs/UNIVERSAL_GOVERNANCE_SPEC.md`](docs/UNIVERSAL_GOVERNANCE_SPEC.md)
- [`docs/ORIUS_THESIS_TERMINOLOGY_GUIDE.md`](docs/ORIUS_THESIS_TERMINOLOGY_GUIDE.md)
- [`docs/ORIUS_FRAMEWORK_PROOF.md`](docs/ORIUS_FRAMEWORK_PROOF.md)
- [`docs/DOC_CONSISTENCY_CHECKLIST.md`](docs/DOC_CONSISTENCY_CHECKLIST.md)

## Historical material

Historical release bundles and frozen package snapshots are retained under:

- [`reports/legacy_archive/`](reports/legacy_archive/)

They are preserved for provenance only. They are not part of the active monograph control path and should not be used as current narrative authority.

## Citation

If you cite this work, use the monograph-facing ORIUS expansion and the canonical manuscript:

```bibtex
@phdthesis{orius2026,
  title     = {{ORIUS}: Observation--Reality Integrity for Universal Safety
               under Degraded Observation},
  year      = {2026},
  school    = {[University]},
  note      = {Book-length monograph and tracked evidence release.}
}
```

## License

Research code — see [LICENSE](LICENSE) for terms.
