# Top-Venue Research Package: ORIUS / DC3S / GridPulse

Generated: `2026-05-05T02:22:01Z`

## Positioning Verdict

ORIUS provides a reliability-aware runtime safety layer for physical AI under degraded observation, enforcing certificate-backed action release through uncertainty coverage, repair, and fallback.

The strongest defensible position is **top-venue systems/safety research under bounded predeployment validation**. The current repo supports a serious three-domain claim, but it should not be framed as road deployment, live clinical deployment, or unrestricted field certification.

## 1. What Is The New Problem Formulation?

ORIUS formalizes degraded observation as a physical-AI release hazard: an action can be legal on the observed state while unsafe on the true state. The scientific object is the gap between observation-conditioned release and true-state safety under reliability loss.

Claim authority: `docs/claim_ledger.md`.

## Source-Backed Novelty Gap

| Lane | Source Anchor | ORIUS Gap Closed | Boundary |
|---|---|---|---|
| runtime_assurance | [NASA Formal Verification Framework for Runtime Assurance](https://ntrs.nasa.gov/citations/20240006522) | ORIUS adds observation-reliability, certificate-backed release, and cross-domain artifact discipline. | Positioning source only; not evidence of ORIUS deployment certification. |
| uncertainty_coverage | [Conformal Prediction: A Gentle Introduction](https://www.emerald.com/ftmal/article/16/4/494/1332423/Conformal-Prediction-A-Gentle-Introduction) | ORIUS connects coverage sets to runtime action release and fail-closed certificates. | Does not create a new universal conformal theorem by itself. |
| safety_critical_control | [Control Barrier Functions: Theory and Applications](https://www.coogan.ece.gatech.edu/papers/pdf/amesecc19.pdf) | ORIUS focuses on observation ambiguity and reliability-aware release rather than only nominal-state safety. | ORIUS is not claimed to replace all barrier-function controllers. |
| clinical_dataset | [MIMIC-IV PhysioNet](https://www.physionet.org/content/mimiciv/2.1/) | ORIUS evaluates runtime alert/fallback legality and calibration rather than only point prediction. | Retrospective evidence only; not live clinical deployment. |
| clinical_dataset | [eICU Collaborative Research Database PhysioNet](https://www.physionet.org/content/eicu-crd/2.0/) | Adds a path to external source validation beyond single-source retrospective replay. | Access-controlled dataset; future validation tier unless artifacts exist. |
| clinical_reporting | [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) | Forces clear calibration, validation, population, and utility reporting for healthcare ORIUS. | Reporting standard, not proof of clinical effectiveness. |
| clinical_bias | [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505) | Defines reviewer-safe subgroup, source-holdout, and applicability claims. | Does not authorize live clinical use. |
| clinical_trial_reporting | [CONSORT-AI](https://www.nature.com/articles/s41591-020-1034-x) | Clarifies that ORIUS healthcare remains retrospective and is not prospective trial evidence. | No prospective clinical trial is claimed. |
| clinical_live_evaluation | [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) | Supplies the boundary between retrospective monitoring evidence and live decision-support evidence. | ORIUS does not claim DECIDE-AI live evaluation completion. |
| clinical_diagnostic_reporting | [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8) | Helps prevent healthcare ORIUS from overstating diagnostic or clinical-decision status. | Healthcare row is monitoring/runtime evidence, not diagnostic approval. |
| av_closed_loop | [nuPlan closed-loop planning benchmark](https://arxiv.org/abs/2106.11810) | ORIUS uses nuPlan replay/surrogate evidence now and treats full closed-loop validation as a next tier. | Current completed evidence is bounded replay/surrogate unless closed-loop artifacts exist. |
| av_stress_simulation | [CARLA simulator](https://arxiv.org/abs/1711.03938) | CARLA is the best next tier for repeatable degraded-observation closed-loop stress. | CARLA completion cannot be claimed until a real CARLA runtime loop produces artifacts. |
| av_dataset | [Waymo Motion Dataset](https://www.waymo.jp/intl/jp/open/data/motion/) | Supports AV runtime replay evidence but not closed-loop road deployment. | Dataset replay is not road deployment. |

## 2. What Is Theoretically Proven?

Observation-only mandatory-release controllers face a lower bound under safety-relevant ambiguity; ORIUS achieves an alpha-bounded upper guarantee under covered uncertainty sets.

Active theorem audit rows: `29`.
Defense tiers: `{'flagship_defended': 8, 'supporting_defended': 14, 'draft_non_defended': 7}`.
- `T4`: `flagship_defended`, `paper_rigorous`, code `partial`
- `T10_T11_ObservationAmbiguitySandwich`: `supporting_defended`, `proof_runtime_linked`, code `matches`
- `T11`: `flagship_defended`, `paper_rigorous`, code `matches`
- `T11_AV_BrakeHold`: `supporting_defended`, `proof_runtime_linked`, code `matches`
- `T11_HC_FailSafeRelease`: `supporting_defended`, `proof_runtime_linked`, code `matches`

The theorem wording must keep the lower-bound and upper-bound pieces separate: observation-only controllers face a Bayes ambiguity lower bound, while ORIUS gets an alpha-bounded upper guarantee only under covered uncertainty sets.

Publication-safe optimality language: ORIUS is **safety-optimal under covered observation ambiguity**, not globally optimal for every physical-AI system.

## 3. What Is Implemented At Runtime?

The implemented runtime contract constructs domain-specific safety predicates, emits certificate evidence, applies repair or fallback before release, and records witness fields for audit. The promoted empirical surface is the runtime denominator, not the diagnostic validation harness.

## 4. What Is Validated In Each Domain?

| Domain | Evidence Tier | ORIUS TSVR | Fallback / Intervention | Witness | External Boundary |
|---|---:|---:|---:|---:|---|
| Battery Energy Storage | reference | 0.000000 | 0.020833 | 1.000000 | HIL/simulator rehearsal, not unrestricted field deployment. |
| Autonomous Vehicles | runtime_contract_closed | 0.000163 | 0.173716 | 0.999837 | Bounded kinematic nuPlan closed-loop planner evaluation: ego actions update future ego speed and gap, but this is not CARLA, road deployment, or full autonomous-driving field closure. |
| Medical and Healthcare Monitoring | runtime_contract_closed | 0.000000 | 0.479907 | 1.000000 | Retrospective source-holdout/time-forward split validation, not prospective trial evidence and not live clinical deployment. |

Equal artifact discipline passes for: Battery Energy Storage, Autonomous Vehicles, Medical and Healthcare Monitoring.

## 5. What Is Explicitly Not Claimed?

- AV is not full autonomous-driving field closure.
- Healthcare is not live clinical deployment.
- Healthcare is not clinical decision support approval.
- Healthcare is not prospective trial evidence.
- Battery is not yet physical HIL or field deployment.
- Universality means shared runtime contract and adapter discipline, not equal real-world maturity in every domain.

## Healthcare / Biomedical Evidence

Healthcare is framed as **retrospective source-holdout and time-forward monitoring validation**. It is not prospective trial evidence, not live clinical deployment, and not regulatory clinical decision support approval.

Current healthcare training evidence: primary target `hr_bpm`, RMSE `1.353953089494559`, PICP90 `0.90297659622813`. The calibration repair is explicitly reported as `healthcare_calibration_repaired:scale=1.154,picp_90=0.902`.

Healthcare comparator framing should use NEWS2/MEWS-style thresholding, conformal alert-only, predictor-only no-runtime, and fixed conservative alert baselines.

## AV Evidence

AV is promoted as all-zip grouped nuPlan replay/surrogate runtime-contract evidence. CARLA completion, road deployment, and full autonomous-driving field closure remain outside the completed evidence.

## Battery Evidence

Battery remains the deepest witness row. Software HIL/simulator rehearsal is valid predeployment evidence, while physical HIL or field deployment remains the next stronger tier.

## Freeze Status

Current freeze status: `freeze_complete_manifest_present`.

Final frozen-release claims are not allowed until `predeployment_release_manifest.json` and frozen hash locks exist.

## 95+ Readiness Gate

| Dimension | Baseline | Target | Status | Remaining Blocker |
|---|---:|---:|---|---|
| core_idea_novelty | 82 | 95 | pass | Need final paper text to keep ORIUS framed as a unified observation-reliability release contract, not a predictor/filter/fallback-only system. |
| theory | 78 | 95 | pass | Keep T9/T10 assumption-qualified and prevent any global-optimality or all-ambiguity-implies-violation overclaim beyond the discharged Battery/AV/Healthcare boundary-law artifacts. |
| three_domain_runtime_evidence | 86 | 95 | pass | Keep AV and Healthcare utility-safety frontiers explicit in the paper so conservative fallback is reported as a measured cost, not hidden utility success. |
| external_validation_depth | 72 | 95 | blocked | Complete prospective/live healthcare validation and physical or high-fidelity battery HIL; AV remains bounded nuPlan closed-loop rather than road deployment or CARLA. |
| reproducibility_and_freeze | 80 | 95 | pass | Finish max freeze, remove AppleDouble/git hygiene noise, run clean full pytest with mutation guard, and publish frozen hashes. |
| claim_quality | 88 | 95 | pass | Keep manuscript, README, claim ledger, and package synchronized after every new validation run. |
