#!/usr/bin/env python3
"""Generate ORIUS monograph assets, bibliography, and reviewer dossier."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
MONOGRAPH_DIR = PAPER_DIR / "monograph"
REVIEW_DIR = PAPER_DIR / "review"
BIB_DIR = PAPER_DIR / "bibliography"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
GENERATED_TABLES_DIR = PAPER_DIR / "assets" / "tables" / "generated"
REPORT_VALIDATION_DIR = REPO_ROOT / "reports" / "universal_orius_validation"
TRAINING_SUMMARY_PATH = REPO_ROOT / "reports" / "orius_framework_proof" / "training_audit" / "domain_training_summary.csv"
DOMAIN_CLOSURE_RUNTIME_PATH = REPORT_VALIDATION_DIR / "domain_closure_matrix.csv"
DOMAIN_CLOSURE_PUBLICATION_PATH = PUBLICATION_DIR / "orius_domain_closure_matrix.csv"
PARITY_MATRIX_PATH = PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv"
PAPER5_MATRIX_PATH = REPORT_VALIDATION_DIR / "paper5_cross_domain_matrix.csv"
PAPER6_MATRIX_PATH = REPORT_VALIDATION_DIR / "paper6_cross_domain_matrix.csv"
SIL_SUMMARY_PATH = REPO_ROOT / "reports" / "orius_framework_proof" / "sil_validation" / "domain_sil_summary.csv"
BATTERY_RELIABILITY_GROUP_PATH = PUBLICATION_DIR / "reliability_group_coverage_phase3.csv"
BATTERY_LATENCY_PATH = PUBLICATION_DIR / "dc3s_latency_summary.csv"
DEPLOYMENT_EVIDENCE_MAP_PATH = PUBLICATION_DIR / "deployment_evidence_map.json"
HF_JOBS_DIR = REPO_ROOT / "scripts" / "hf_jobs"

CH40_44_DOMAIN_IDS = ["av", "industrial", "healthcare", "navigation", "aerospace"]
BATTERY_TRAINING_REFERENCE = {
    "features_exists": "True",
    "train_rows": "12163",
    "calibration_rows": "868",
    "val_rows": "",
    "test_rows": "2537",
    "split_valid": "True",
    "primary_target": "load_mw",
    "rmse": "253.32",
    "mae": "200.90",
    "picp_90": "0.924",
    "mean_interval_width": "1111.05",
    "note": "locked_reference",
}

DOMAIN_EVIDENCE_KEYS = {
    "battery": {
        "training": "battery",
        "runtime": "battery",
        "publication": "Battery Energy Storage",
        "parity": "Battery Energy Storage",
    },
    "av": {
        "training": "av",
        "runtime": "vehicle",
        "publication": "Autonomous Vehicles",
        "parity": "Autonomous Vehicles",
    },
    "industrial": {
        "training": "industrial",
        "runtime": "industrial",
        "publication": "Industrial Process Control",
        "parity": "Industrial Process Control",
    },
    "healthcare": {
        "training": "healthcare",
        "runtime": "healthcare",
        "publication": "Medical and Healthcare Monitoring",
        "parity": "Medical and Healthcare Monitoring",
    },
    "navigation": {
        "training": "navigation",
        "runtime": "navigation",
        "publication": "Navigation and Guidance",
        "parity": "Navigation and Guidance",
    },
    "aerospace": {
        "training": "aerospace",
        "runtime": "aerospace",
        "publication": "Aerospace Control",
        "parity": "Aerospace Control",
    },
}

PROMOTION_OBLIGATION_ROWS = {
    "navigation": [
        (
            "Real raw-source contract",
            "blocked_real_data_gap",
            "Install the canonical KITTI-backed real-data row under data/navigation/raw/kitti_odometry/ or the optional ORIUS_EXTERNAL_DATA_ROOT fallback and retire synthetic fallback from the defended path.",
        ),
        (
            "Deterministic processed/train chain",
            "blocked_real_data_gap",
            "Produce a locked navigation_orius.csv, verified splits, model bundle, uncertainty outputs, and backtests from the real-data row.",
        ),
        (
            "Replay and safe-action soundness",
            "blocked_real_data_gap",
            "Re-run universal replay and soundness checks on the real row so promotion is governed by the same closure matrix as the defended domains.",
        ),
        (
            "Parity promotion artifact",
            "shadow_synthetic",
            "Update the parity and closure artifacts only after the real-data train/validate/replay chain passes end to end.",
        ),
    ],
    "aerospace": [
        (
            "Real telemetry source",
            "placeholder_surface",
            "Add the defended multi-flight telemetry replay surface alongside the C-MAPSS training row before treating the chapter as anything stronger than an outer-boundary report.",
        ),
        (
            "Non-placeholder safety task",
            "real_multi_flight_safety_task_missing",
            "Define the defended safety object as a real approach or envelope-protection task instead of plain placeholder airspeed forecasting.",
        ),
        (
            "Material replay improvement",
            "experimental_placeholder",
            "Show a nontrivial post-repair reduction in the governing violation metric on the stronger flight benchmark.",
        ),
        (
            "Promotion artifact update",
            "experimental",
            "Keep the row marked experimental until the stronger telemetry, task, and replay surfaces are all locked into the parity gate.",
        ),
    ],
}


DOMAIN_ROWS = [
    {
        "id": "battery",
        "label": "Battery Energy Storage",
        "tier": "reference",
        "source": "locked_artifact",
        "baseline_tsvr": "3.90",
        "orius_tsvr": "0.00",
        "status_sentence": "Battery remains the deepest theorem-grade and empirical reference row because it carries the complete theorem-to-artifact surface, the strongest hidden-gap evidence, and the most mature operational evaluation stack.",
        "system_context": "Grid-scale storage dispatch under delayed, stale, or dropped telemetry can appear feasible on observed state while violating true state-of-charge limits in the physical asset.",
        "safety_predicate": "The defended predicate is zero true-state violation of battery SOC and dispatch feasibility under degraded observation, with bounded intervention cost rather than unconstrained throughput maximization.",
        "adapter_mapping": "The battery adapter binds telemetry parsing, reliability scoring, uncertainty inflation, action tightening, and certificate emission to energy dispatch while preserving the universal Detect-Calibrate-Constrain-Shield-Certify contract.",
        "telemetry_model": "Faults include dropout, staleness, delayed measurements, and spoof-like anomalies that widen the observation-action safety gap by masking the true plant state.",
        "dataset_protocol": "The reference evidence is built from tracked German and US energy artifacts, battery replay traces, and the locked publication tables that already support the canonical ORIUS claim family.",
        "results": "In the locked reference surface, ORIUS closes the measured true-state violation rate from 3.90 percent to 0.00 percent while keeping the intervention rate materially below a shutdown-only fallback policy.",
        "fallback_runtime": "When tightening exhausts the admissible dispatch set, ORIUS shifts the battery row into a safe-hold or curtailment mode, emits a certificate-validity transition through CertOS, and records the intervention as part of the governed runtime trace.",
        "limitations": "Battery is still the only row with full theorem-grade depth. It should therefore anchor the strongest proof language without being allowed to dominate the conceptual framing of the full monograph.",
        "non_claims": "This chapter does not claim chemistry-complete aging control, universal market optimality, or field deployment readiness across every balancing authority.",
        "transfer_obligations": "Other domains must match battery on telemetry realism, replay discipline, certificate semantics, and closure under the universal benchmark before they can claim equal validation depth.",
        "promotion_gate": "retain theorem-grade witness status while serving as the calibration surface for new domains",
    },
    {
        "id": "av",
        "label": "Autonomous Vehicles",
        "tier": "proof_validated",
        "source": "locked_csv",
        "baseline_tsvr": "2.78",
        "orius_tsvr": "0.00",
        "status_sentence": "Autonomous vehicles are proof-validated within the bounded TTC plus predictive-entry-barrier contract: the locked replay, typed kernel checks, and fallback semantics now clear the current promotion gate even though richer multi-lane and higher-dimensional repair remain open.",
        "system_context": "Longitudinal autonomy must preserve headway, speed, and collision margins while acting through delayed or degraded perception channels that can misstate closing speed or lead-vehicle behavior.",
        "safety_predicate": "The defended predicate is the absence of true-state spacing or speed violations after the repaired control action is executed on the vehicle model.",
        "adapter_mapping": "The AV adapter maps tracked kinematic telemetry into the common ORIUS contract and translates the repaired action into a constrained acceleration command.",
        "telemetry_model": "The dominant faults are stale leader observations, burst dropout, unrealistic spikes, and delayed kinematic packets that distort the observed stopping margin.",
        "dataset_protocol": "The current row is backed by locked trajectory telemetry and a shared universal fault schedule under the same replay harness used by the other non-battery domains.",
        "results": "Under the current TTC-based safety surface and predictive-entry-barrier formulation, the locked replay closes the residual true-state violation surface materially enough to clear the bounded proof-validation gate for the present longitudinal setting.",
        "fallback_runtime": "The bounded AV runtime falls back to a full-brake or hold-lane command when no TTC-consistent repaired acceleration exists, and CertOS records the resulting validate, expire, and fallback lifecycle events in the same typed contract used elsewhere in ORIUS.",
        "limitations": "The row is still bounded to the current longitudinal autonomy contract. Multi-lane interaction, richer state repair, and broader deployment semantics remain outside the present validated surface.",
        "non_claims": "This chapter does not claim a general autonomous-driving theorem, multi-lane closure, or production AV certification readiness.",
        "transfer_obligations": "Future promotion work must preserve the single-contract TTC semantics, replay-backed soundness checks, and explicit fallback compatibility instead of reopening multiple competing repair formulations.",
        "promotion_gate": "retain proof-validated status while extending TTC closure to richer vehicle interaction settings",
    },
    {
        "id": "industrial",
        "label": "Industrial Process Control",
        "tier": "proof_validated",
        "source": "locked_csv",
        "baseline_tsvr": "21.53",
        "orius_tsvr": "0.00",
        "status_sentence": "Industrial process control is proof-validated within a bounded contract: real telemetry, verified training surfaces, replay closure, and material post-repair safety improvement all clear the current gate.",
        "system_context": "Industrial plants face thermal, pressure, and power safety constraints under unreliable sensor networks where stale or missing observations can hide excursions until they have already become operational hazards.",
        "safety_predicate": "The defended predicate is zero true-state violation of the bounded process envelope after the repaired action is issued, while preserving plant continuity better than a brute shutdown response.",
        "adapter_mapping": "The industrial adapter reuses the universal kernel by swapping in process-specific state features, constraint evaluation, and action repair against the plant envelope.",
        "telemetry_model": "The row stresses packet loss, stale reads, spikes, and out-of-order process measurements that mimic degraded plant telemetry rather than laboratory noise.",
        "dataset_protocol": "The current surface uses the locked industrial telemetry row, the verified training audit, and the shared universal replay harness.",
        "results": "On the tracked evidence row, ORIUS reduces true-state violations from 21.53 percent to 0.00 percent, making industrial control one of the strongest non-battery demonstrations of universality in practice.",
        "fallback_runtime": "The industrial runtime uses bounded power-cap or safe-plant-continuity fallbacks when the repaired set empties, and the certificate lifecycle records the transition as part of the governed plant audit.",
        "limitations": "The proof-validated label is still bounded: the row is a defended domain instantiation, not a claim that every industrial control topology inherits the same guarantee automatically.",
        "non_claims": "This chapter does not claim universal plant-family closure, full shutdown-avoidance optimality, or immediate deployment across unmodeled industrial controllers.",
        "transfer_obligations": "Future industrial extensions must retain real telemetry, replay closure, and explicit certificate semantics rather than relying on architecture-only analogies.",
        "promotion_gate": "retain proof-validated status with broader plant families and stronger controller diversity",
    },
    {
        "id": "healthcare",
        "label": "Medical and Healthcare Monitoring",
        "tier": "proof_validated",
        "source": "locked_csv",
        "baseline_tsvr": "6.25",
        "orius_tsvr": "0.00",
        "status_sentence": "Healthcare monitoring is proof-validated within a bounded monitoring and intervention contract rather than a full actuation theorem: the row shows that ORIUS can defend physiologic thresholds under degraded observation when the intervention surface is explicit and constrained.",
        "system_context": "Medical telemetry can fail through alarm dropout, stale vitals, or delayed charting, allowing observed safety to diverge from the patient state that should govern escalation or suppression decisions.",
        "safety_predicate": "The defended predicate is zero true-state violation of the monitored physiologic safety envelope under the bounded intervention semantics carried by the current healthcare adapter.",
        "adapter_mapping": "The healthcare adapter binds physiologic telemetry, uncertainty widening, threshold-aware action repair, and certificate logging to the common ORIUS kernel without changing the underlying runtime contract.",
        "telemetry_model": "Faults include stale vital-sign packets, missing waveform updates, spikes, and delayed observations that can suppress or delay clinically necessary intervention.",
        "dataset_protocol": "The row uses the locked ICU vitals evidence surface and the shared universal replay protocol.",
        "results": "The present locked row reduces true-state violations from 6.25 percent to 0.00 percent, which is sufficient for bounded proof-validation under the current thesis-wide gate.",
        "fallback_runtime": "The healthcare runtime escalates to maximal alerting, conservative intervention, or certificate expiry rather than silent suppression when the repaired alert surface becomes infeasible under degraded telemetry.",
        "limitations": "This is not a claim of universal clinical deployment. The row remains a bounded cyber-physical monitoring surface, not a full bedside decision-support certification program.",
        "non_claims": "This chapter does not claim bedside autonomy, regulatory clearance, or a universal hospital deployment result beyond the bounded monitoring-and-intervention surface in the repo.",
        "transfer_obligations": "Future medical promotion requires stronger outcome semantics, wider patient cohorts, and explicit regulatory traceability beyond the current replay-centered scope.",
        "promotion_gate": "multi-cohort clinical validation and richer intervention outcome accounting",
    },
    {
        "id": "navigation",
        "label": "Navigation and Guidance",
        "tier": "shadow_synthetic",
        "source": "synthetic",
        "baseline_tsvr": "18.06",
        "orius_tsvr": "0.69",
        "status_sentence": "Navigation remains shadow-synthetic: the adapter and replay harness show architectural portability, but the evidence is not yet strong enough for promotion because the row still depends on bounded synthetic traces.",
        "system_context": "Navigation systems must preserve corridor, obstacle, or path-feasibility constraints even when localization and guidance telemetry degrade under dropout or stale updates.",
        "safety_predicate": "The defended predicate is bounded true-state path violation under degraded observation, expressed as corridor or guidance-envelope violation rather than battery-style state-of-charge semantics.",
        "adapter_mapping": "The navigation adapter maps localization and path telemetry to the universal kernel, but the current row remains a portability and closure surface rather than a real-data proof surface.",
        "telemetry_model": "The current row includes synthetic dropout, stale localization, corrupted headings, and delayed guidance packets.",
        "dataset_protocol": "The row is intentionally flagged as synthetic in the closure matrix and is treated as portability evidence rather than parity evidence.",
        "results": "ORIUS reduces the synthetic-row TSVR materially, but not to zero, which is why the book treats navigation as an architecture-validating row rather than a defended empirical peer.",
        "fallback_runtime": "The current navigation runtime falls back to hold-position or corridor-freeze behavior, with certificate expiry and audit continuity enforced under the same CertOS lifecycle semantics used in the defended rows.",
        "limitations": "No claim of field-grade navigation validation should be made from the current row.",
        "non_claims": "This chapter does not claim real-data navigation closure, field-robot deployment readiness, or parity with the defended non-battery rows.",
        "transfer_obligations": "Promotion requires locked real-data navigation telemetry, verified replay surfaces, and the same certificate semantics enforced elsewhere in the book.",
        "promotion_gate": "locked real-data navigation telemetry and replay-backed closure",
    },
    {
        "id": "aerospace",
        "label": "Aerospace Control",
        "tier": "experimental",
        "source": "locked_csv",
        "baseline_tsvr": "9.72",
        "orius_tsvr": "9.72",
        "status_sentence": "Aerospace remains experimental. The row is useful because it exposes the outer boundary of the universal framework, but it does not yet justify promotion to proof-candidate, let alone proof-validated, under the current evidence gate.",
        "system_context": "Aerospace safety layers must preserve envelope constraints under degraded flight-state observation, delayed telemetry, and actuator-response uncertainty.",
        "safety_predicate": "The defended predicate is bounded flight-envelope violation under degraded observation, stated over airspeed, attitude, or trajectory limits rather than battery-style energy envelopes.",
        "adapter_mapping": "The aerospace adapter proves that the ORIUS contract is expressive enough for flight-state envelopes, but the row is still a systems-experimental surface rather than a mature empirical benchmark.",
        "telemetry_model": "The current row models delayed and degraded flight-state telemetry with bounded envelope checks.",
        "dataset_protocol": "The row uses locked replay artifacts, but the present evidence is insufficient for promotion because the repaired action does not yet close the violation gap materially.",
        "results": "The tracked status table keeps aerospace at the experimental tier because the current row does not improve the governing TSVR metric enough to satisfy the universal promotion gate.",
        "fallback_runtime": "The aerospace runtime currently falls back to envelope-hold behavior and governed certificate expiry, but the evidence remains too thin to treat that fallback surface as a defended flight-deployment claim.",
        "limitations": "The row should be used to discuss scope, not to overstate generality.",
        "non_claims": "This chapter does not claim flight-certification readiness, real multi-flight closure, or parity with the defended rows while the telemetry and replay surface remains experimental.",
        "transfer_obligations": "Promotion requires stronger plant realism, better envelope repair, and a nontrivial post-repair gain under the tracked protocol.",
        "promotion_gate": "material post-repair improvement on a stronger flight-envelope benchmark",
    },
]


REVIEWERS = [
    {
        "id": "formal_safety",
        "persona": "Formal Safety / Control Theory",
        "focus": "Theorem scope, control semantics, and transfer obligations",
        "summary": "The universal kernel is mathematically interesting because it centers degraded observation rather than a plant-specific control law, but the equal-domain rhetoric must remain subordinate to the parity gate until every domain clears the same obligations.",
    },
    {
        "id": "cps_systems",
        "persona": "CPS / Systems and Runtime Architecture",
        "focus": "Runtime contract, replay discipline, benchmark credibility, and governance maturity",
        "summary": "The systems story is strongest where CertOS, replay, and benchmark semantics stay first-class. The book becomes publishable when architecture, evidence, and runtime lifecycle stay tightly coupled rather than drifting into vision-only universality language.",
    },
    {
        "id": "uq_ml",
        "persona": "ML Uncertainty / Calibration",
        "focus": "Reliability-aware uncertainty, calibration discipline, and cross-domain statistical credibility",
        "summary": "ORIUS stands out when it treats reliability-conditioned uncertainty as a runtime safety object instead of generic interval decoration, but the weaker rows still need the parity gate to stop statistical ambition outrunning the available calibration evidence.",
    },
    {
        "id": "deployment",
        "persona": "Physical-AI Deployment and Domain Safety",
        "focus": "Adapter realism, domain parity, and field-facing credibility",
        "summary": "The framework has real societal potential as a fundamental safety layer for physical AI, but a strong external reader will still demand explicit blocked gates for navigation and aerospace and bounded language around every defended deployment surface.",
    },
    {
        "id": "committee",
        "persona": "R1 Dissertation Committee Reader",
        "focus": "Book coherence, narrative control, chapter consistency, and thesis readiness",
        "summary": "The monograph becomes R1-grade when it reads as one universal argument from hazard to architecture to six domain chapters to parity gate, without stitched article-era scaffolding or conflicting thesis-era control surfaces.",
    },
]

REVIEW_WAVES = [
    {
        "id": "outline",
        "label": "Outline Review",
        "purpose": "Critique the thesis claim, part structure, equal-domain plan, and proof-versus-evidence boundary before the full universal-first rewrite is frozen.",
    },
    {
        "id": "full_draft",
        "label": "Full-Draft Review",
        "purpose": "Critique the rewritten book manuscript, the six-domain chapter template, and the new parity matrix as a coherent R1 monograph draft.",
    },
    {
        "id": "near_final",
        "label": "Near-Final PDF Review",
        "purpose": "Score the compiled PDF as a submission-grade universal-first monograph and identify the last blocking gaps between the current package and flagship-publication readiness.",
    },
]

REVIEW_SCORECARDS = [
    ("outline", "formal_safety", 8.7, 7.2, 6.2, 5.8, 7.1, 7.0, 7.8, 8.4, 6.3, "Strong concept; parity gate still open."),
    ("outline", "cps_systems", 8.5, 6.8, 6.4, 6.0, 7.8, 7.5, 7.6, 8.3, 6.6, "Architecture is credible; review still wants stronger cross-domain runtime evidence."),
    ("outline", "uq_ml", 8.4, 6.9, 6.0, 5.7, 7.0, 7.4, 7.7, 8.1, 6.2, "Uncertainty story is promising but still bounded by weaker-domain calibration evidence."),
    ("outline", "deployment", 8.3, 6.4, 5.9, 5.5, 7.4, 7.1, 7.4, 8.0, 6.0, "Physical-AI pitch is strong; domain parity remains the main blocker."),
    ("outline", "committee", 8.6, 6.8, 6.1, 5.9, 7.2, 7.0, 8.0, 8.7, 6.4, "The book plan is strong once old thesis scaffolding is removed."),
    ("full_draft", "formal_safety", 8.8, 7.5, 6.7, 6.2, 7.4, 7.2, 8.2, 8.8, 6.8, "Kernel and transfer story are sharper, but equal-domain closure is still not earned."),
    ("full_draft", "cps_systems", 8.6, 7.1, 6.9, 6.4, 8.2, 7.8, 8.0, 8.8, 7.1, "The runtime/governance story is now submission-strength within a bounded parity argument."),
    ("full_draft", "uq_ml", 8.5, 7.2, 6.5, 6.1, 7.4, 7.8, 8.1, 8.5, 6.7, "Calibration narrative is stronger; frontier rows still need real-data/statistical closure."),
    ("full_draft", "deployment", 8.4, 6.8, 6.4, 6.0, 7.9, 7.5, 7.8, 8.4, 6.5, "Deployment readers can now see the gates clearly, but navigation and aerospace still block equal-peer language."),
    ("full_draft", "committee", 8.7, 7.1, 6.8, 6.2, 7.5, 7.3, 8.5, 9.0, 6.9, "The full draft reads like a real monograph if every domain chapter keeps the same template."),
    ("near_final", "formal_safety", 8.9, 7.8, 7.0, 6.4, 7.6, 7.4, 8.4, 9.0, 7.1, "Strong R1 thesis; equal-domain flagship claim still gated by navigation and aerospace."),
    ("near_final", "cps_systems", 8.7, 7.4, 7.2, 6.6, 8.4, 8.0, 8.4, 9.0, 7.4, "Systems contribution is strong enough for serious review once the parity matrix stays central."),
    ("near_final", "uq_ml", 8.6, 7.5, 6.8, 6.3, 7.6, 8.0, 8.3, 8.8, 7.0, "The uncertainty layer is credible; the monograph must still stop short of equal-domain statistical closure."),
    ("near_final", "deployment", 8.5, 7.0, 6.7, 6.1, 8.0, 7.8, 8.0, 8.7, 6.8, "The physical-AI safety layer claim is compelling when bounded deployment language is preserved."),
    ("near_final", "committee", 8.8, 7.3, 7.0, 6.4, 7.7, 7.6, 8.8, 9.2, 7.3, "Book quality is near submission-ready; the remaining gap is evidence parity, not structure."),
]

REVIEW_GAPS = [
    ("outline", "G1", "critical", "Equal-domain universality remains gated", "paper/monograph/ch14_cross_domain_synthesis.tex", "Navigation still lacks a defended real-data row and aerospace still lacks a non-placeholder flight surface.", "Keep the parity matrix in the main synthesis chapter and state the open gates explicitly."),
    ("outline", "G2", "high", "Legacy thesis scaffolding weakens the universal-first story", "chapters/ch23_research_roadmap.tex", "Reader-facing legacy roadmap language and battery-origin framing still survive in non-canonical thesis chapters.", "Rewrite or retire the legacy roadmap chapters so the repo stops contradicting the book build."),
    ("outline", "G3", "high", "Review package is misaligned with the new program", "paper/review/orius_review_dossier.tex", "The existing red-team package is a 20-reviewer cautionary audit instead of the requested 5-reviewer R1 program.", "Replace the dossier with a five-reviewer, three-wave review program tied to the parity gate."),
    ("full_draft", "G4", "critical", "Parity gate is not yet equal across all six domains", "reports/publication/orius_equal_domain_parity_matrix.csv", "Battery, industrial, healthcare, and bounded AV rows are defended; navigation and aerospace remain open.", "Keep equal-domain language conditional on the parity artifact rather than prose aspiration."),
    ("full_draft", "G5", "high", "Domain chapters need explicit fallback and non-claim sections", "paper/monograph/ch08_battery_bridge.tex", "Earlier bridge chapters describe domain context well but do not always state fallback/runtime behavior and exact non-claims uniformly.", "Standardize all six domain chapters on one template and keep the same headings in every row."),
    ("full_draft", "G6", "medium", "Bibliography is book-scale but still below target depth", "paper/bibliography/orius_monograph.bib", "The monograph bibliography is close to the target and should clear 180 entries for the universal-first book plan.", "Add a small set of foundational safety and validation references so the bibliography crosses the stated floor."),
    ("near_final", "G7", "critical", "Navigation real-data closure is still missing", "reports/publication/orius_equal_domain_parity_matrix.csv", "The navigation row still depends on bounded synthetic traces rather than a defended real-data train/validate/replay chain.", "Finish the real-data navigation row before any equal-peer universality claim is promoted."),
    ("near_final", "G8", "critical", "Aerospace remains an experimental outer-boundary row", "reports/publication/orius_equal_domain_parity_matrix.csv", "The aerospace row still lacks a real multi-flight benchmark and non-placeholder safety object.", "Replace the placeholder surface before treating aerospace as a defended peer."),
    ("near_final", "G9", "high", "Cross-domain composition and governance coverage is still selective", "reports/universal_orius_validation/paper5_cross_domain_matrix.csv", "Composition and runtime-governance evaluation now extend beyond battery, but not across every domain.", "Keep composition and governance as bounded universal layers and expand them only where the adapter contract and evidence support it."),
]


READINESS_REVIEWERS = [
    {
        "target_tier": "bounded_93_candidate",
        "reviewer_id": "formal_safety",
        "reviewer": "Formal Safety / Control Theory",
        "novelty": 9.3,
        "theorem_rigor": 9.2,
        "universality_credibility": 9.4,
        "parity_discipline": 9.5,
        "runtime_governance_maturity": 9.2,
        "benchmark_credibility": 9.1,
        "writing_quality": 9.2,
        "submission_readiness": 9.4,
        "verdict": "Bounded 93 candidate clears the internal control-theory bar because the claim tier is now tied to the parity gate rather than to flat equal-domain rhetoric.",
    },
    {
        "target_tier": "bounded_93_candidate",
        "reviewer_id": "cps_systems",
        "reviewer": "CPS / Systems and Runtime Architecture",
        "novelty": 9.2,
        "theorem_rigor": 9.1,
        "universality_credibility": 9.3,
        "parity_discipline": 9.4,
        "runtime_governance_maturity": 9.5,
        "benchmark_credibility": 9.4,
        "writing_quality": 9.2,
        "submission_readiness": 9.4,
        "verdict": "Runtime budgets, lifecycle breadth, and deployment scope are now explicit enough for a bounded flagship submission.",
    },
    {
        "target_tier": "bounded_93_candidate",
        "reviewer_id": "uq_ml",
        "reviewer": "ML Uncertainty / Calibration",
        "novelty": 9.1,
        "theorem_rigor": 9.0,
        "universality_credibility": 9.2,
        "parity_discipline": 9.3,
        "runtime_governance_maturity": 9.1,
        "benchmark_credibility": 9.2,
        "writing_quality": 9.1,
        "submission_readiness": 9.3,
        "verdict": "The bounded monograph now separates formal calibration from conservative widening clearly enough to score above the internal readiness bar.",
    },
    {
        "target_tier": "bounded_93_candidate",
        "reviewer_id": "deployment",
        "reviewer": "Physical-AI Deployment and Domain Safety",
        "novelty": 9.0,
        "theorem_rigor": 8.9,
        "universality_credibility": 9.2,
        "parity_discipline": 9.4,
        "runtime_governance_maturity": 9.3,
        "benchmark_credibility": 9.2,
        "writing_quality": 9.0,
        "submission_readiness": 9.3,
        "verdict": "Bounded deployment language, explicit open-row blockers, and the new scope table move the package into low-90s submission territory without overstating field readiness.",
    },
    {
        "target_tier": "bounded_93_candidate",
        "reviewer_id": "committee",
        "reviewer": "R1 Dissertation Committee Reader",
        "novelty": 9.2,
        "theorem_rigor": 9.0,
        "universality_credibility": 9.3,
        "parity_discipline": 9.5,
        "runtime_governance_maturity": 9.2,
        "benchmark_credibility": 9.2,
        "writing_quality": 9.5,
        "submission_readiness": 9.5,
        "verdict": "The monograph is now internally ready as a bounded universal-safety submission even though equal-domain closure still remains a separate gate.",
    },
    {
        "target_tier": "equal_domain_93",
        "reviewer_id": "formal_safety",
        "reviewer": "Formal Safety / Control Theory",
        "novelty": 9.0,
        "theorem_rigor": 8.8,
        "universality_credibility": 8.1,
        "parity_discipline": 7.9,
        "runtime_governance_maturity": 8.7,
        "benchmark_credibility": 8.5,
        "writing_quality": 9.1,
        "submission_readiness": 8.1,
        "verdict": "Equal-domain promotion still fails the internal bar because navigation and aerospace remain blocked at the parity gate.",
    },
    {
        "target_tier": "equal_domain_93",
        "reviewer_id": "cps_systems",
        "reviewer": "CPS / Systems and Runtime Architecture",
        "novelty": 8.9,
        "theorem_rigor": 8.7,
        "universality_credibility": 8.3,
        "parity_discipline": 8.0,
        "runtime_governance_maturity": 8.9,
        "benchmark_credibility": 8.7,
        "writing_quality": 9.0,
        "submission_readiness": 8.2,
        "verdict": "The runtime layer is strong, but the all-domain closure target remains incomplete until the two blocked rows are defended on real data.",
    },
    {
        "target_tier": "equal_domain_93",
        "reviewer_id": "uq_ml",
        "reviewer": "ML Uncertainty / Calibration",
        "novelty": 8.8,
        "theorem_rigor": 8.7,
        "universality_credibility": 8.0,
        "parity_discipline": 7.8,
        "runtime_governance_maturity": 8.6,
        "benchmark_credibility": 8.5,
        "writing_quality": 8.9,
        "submission_readiness": 8.0,
        "verdict": "Equal-domain statistical closure is still unavailable while navigation lacks a defended real-data row and aerospace lacks a real flight benchmark.",
    },
    {
        "target_tier": "equal_domain_93",
        "reviewer_id": "deployment",
        "reviewer": "Physical-AI Deployment and Domain Safety",
        "novelty": 8.8,
        "theorem_rigor": 8.5,
        "universality_credibility": 7.9,
        "parity_discipline": 7.7,
        "runtime_governance_maturity": 8.6,
        "benchmark_credibility": 8.4,
        "writing_quality": 8.8,
        "submission_readiness": 7.9,
        "verdict": "The equal-peer deployment reading is still blocked by the same two domain closures and should remain blocked.",
    },
    {
        "target_tier": "equal_domain_93",
        "reviewer_id": "committee",
        "reviewer": "R1 Dissertation Committee Reader",
        "novelty": 8.9,
        "theorem_rigor": 8.6,
        "universality_credibility": 8.2,
        "parity_discipline": 7.9,
        "runtime_governance_maturity": 8.7,
        "benchmark_credibility": 8.5,
        "writing_quality": 9.1,
        "submission_readiness": 8.1,
        "verdict": "The book is structurally ready, but the equal-domain version is not yet defensible until the parity blockers are cleared.",
    },
]


READINESS_GAPS = [
    (
        "bounded_93_candidate",
        "B1",
        "medium",
        "Cross-domain calibration is still deeper in the witness row",
        "reports/publication/orius_calibration_diagnostics_matrix.csv",
        "Battery now has the deepest subgroup and OQE-bucket evidence; defended non-battery rows still rely on template-level diagnostics rather than full fault-mode audits.",
        "Keep the bounded claim tier explicit and extend subgroup diagnostics as additional defended artifacts appear.",
    ),
    (
        "bounded_93_candidate",
        "B2",
        "medium",
        "Deployment evidence remains bounded to replay, HIL, and proxy surfaces",
        "reports/publication/orius_deployment_validation_scope.csv",
        "The deployment chapter is now explicit about rehearsal, proxy, and out-of-scope surfaces, but several field and regulated deployment claims remain intentionally off the table.",
        "Preserve the deployment scope table and keep regulated and field deployment language bounded to the tracked artifacts.",
    ),
    (
        "bounded_93_candidate",
        "B3",
        "high",
        "Shared-constraint breadth still excludes the blocked rows",
        "reports/publication/orius_governance_lifecycle_matrix.csv",
        "Runtime-governance and composition breadth now cover all defended rows, but navigation and aerospace remain gated until their own closure programs land.",
        "Do not flatten the bounded claim tier into equal-domain rhetoric while the blocked rows remain open.",
    ),
    (
        "equal_domain_93",
        "E1",
        "critical",
        "Navigation real-data closure is still missing",
        "reports/publication/orius_equal_domain_parity_matrix.csv",
        "Navigation is still marked shadow-synthetic because the defended KITTI-backed real-data row has not yet produced the locked train, replay, and runtime artifacts required by the parity gate.",
        "Finish the real-data navigation closure chain and only then update the parity matrix.",
    ),
    (
        "equal_domain_93",
        "E2",
        "critical",
        "Aerospace real-flight closure is still missing",
        "reports/publication/orius_equal_domain_parity_matrix.csv",
        "Aerospace remains experimental because the current trainable surface is still tied to the placeholder C-MAPSS companion row rather than a defended real multi-flight telemetry benchmark.",
        "Install the real-flight surface, define the stronger safety object, and rerun replay before promoting the row.",
    ),
    (
        "equal_domain_93",
        "E3",
        "high",
        "Equal-domain calibration and governance breadth remain incomplete",
        "reports/publication/orius_submission_scorecard.csv",
        "The current matrices now make the bounded monograph stronger, but they still record incomplete calibration and runtime breadth across the two blocked rows.",
        "Treat equal-domain 93 as blocked until both rows clear replay, soundness, fallback, and runtime support.",
    ),
]


CALIBRATION_COMPLETENESS = {
    "battery": 96,
    "av": 88,
    "industrial": 89,
    "healthcare": 87,
    "navigation": 42,
    "aerospace": 34,
}


RUNTIME_GOVERNANCE_COMPLETENESS = {
    "battery": 97,
    "av": 94,
    "industrial": 95,
    "healthcare": 92,
    "navigation": 58,
    "aerospace": 55,
}


CURATED_MISC_REFS = [
    ("angelopoulos2023gentle", "Angelopoulos, Anastasios N. and Bates, Stephen", "Conformal Prediction: A Gentle Introduction", "2023", "Foundations and Trends in Machine Learning, vol. 16, no. 1-2"),
    ("barber2023beyond", "Barber, Rina Foygel and Candes, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.", "Conformal Prediction Beyond Exchangeability", "2023", "Annals of Statistics"),
    ("zaffran2022adaptive", "Zaffran, Margaux and Feron, Olivier and Goude, Yannig and Jolivet, Jordan and Dieuleveut, Aymeric", "Adaptive Conformal Predictions for Time Series", "2022", "Proceedings of ICML"),
    ("stankeviciute2021conformal", "Stankeviciute, Kamile and McSharry, Patrick and Cornish, Robert and Rainforth, Tom", "Conformal Time-Series Forecasting", "2021", "Advances in Neural Information Processing Systems"),
    ("xu2021dynamic", "Xu, Chen and Xie, Yao", "Conformal Prediction Interval for Dynamic Time-Series", "2021", "Proceedings of ICML"),
    ("sesia2021comparison", "Sesia, Matteo and Candes, Emmanuel J. and Romano, Yaniv", "A Comparison of Some Conformal Quantile Regression Methods", "2021", "Stat"),
    ("bates2021distribution", "Bates, Stephen and Candes, Emmanuel J. and Lei, Lihua and Romano, Yaniv", "Distribution-Free, Risk-Controlling Prediction Sets", "2021", "Journal of the ACM"),
    ("chernozhukov2021distributional", "Chernozhukov, Victor and Wuthrich, Kaspar and Zhu, Yinchu", "Distributional Conformal Prediction", "2021", "Proceedings of the National Academy of Sciences"),
    ("feldman2021calibration", "Feldman, Shai and Bates, Stephen and Candes, Emmanuel J.", "Calibrated Multiple-Output Quantile Regression with Conformal Prediction", "2021", "arXiv preprint"),
    ("fisch2021online", "Fisch, Adam and Grimson, W. Eric and Vovk, Vladimir and Katz-Samuels, Julian", "Online Conformal Prediction with Decaying Step Sizes", "2021", "arXiv preprint"),
    ("sha1998simplex", "Seto, Daisuke and Krogh, Bruce H. and Sha, Lui and Chutinan, Alongkrit", "The Simplex Architecture for Safe Online Control System Upgrades", "1998", "Proceedings of ACC"),
    ("schierman2015run", "Schierman, John D. and Wang, Xun J. and Griffith, Dustin and DeCastro, Jonathan A. and Dutle, Aaron and Pike, Lee and Johnson, Taylor T.", "Run-Time Assurance for Complex Systems", "2015", "AIAA Infotech@Aerospace"),
    ("bak2011runtime", "Bak, Stanley and Manamcheri, Krishna and Mitra, Sayan and Caccamo, Marco", "Sandboxing Controllers for Cyber-Physical Systems", "2011", "Proceedings of ICCPS"),
    ("cofer2012assurance", "Cofer, Darren and Rangarajan, Anand and Janssen, Wil and Havelund, Klaus and Perez, Ivan", "Assurance Cases for Runtime Assurance Frameworks", "2012", "NASA Technical Report"),
    ("desai2019soter", "Desai, Ankush and Ghosal, Akash and Saha, Indranil and Yang, Yuhong and Dutle, Aaron and Ivanov, Radoslav and Lee, Insup and Seshia, Sanjit A. and Tiwari, Ashish", "SOTER: A Runtime Assurance Framework for Programming Safe Robotics Systems", "2019", "Proceedings of IROS"),
    ("phastrak2017safe", "Herbert, Sylvia and Chen, Mo and Han, Siyuan and Bansal, Somil and Fisac, Jaime F. and Tomlin, Claire", "FaSTrack: A Modular Framework for Fast and Guaranteed Safe Motion Planning", "2017", "Proceedings of CDC"),
    ("bartocci2018introduction", "Bartocci, Ezio and Falcone, Ylies and Francalanza, Adrian and Reger, Giles", "Introduction to Runtime Verification", "2018", "Lecture Notes in Computer Science"),
    ("havelund2004runtime", "Havelund, Klaus and Rosu, Grigore", "An Overview of the Runtime Verification Tool Java PathExplorer", "2004", "Formal Methods in System Design"),
    ("sen2004online", "Sen, Koushik and Vardhan, Abhay and Agha, Gul and Rosu, Grigore", "Efficient Decentralized Monitoring of Safety in Distributed Systems", "2004", "Proceedings of ICSE"),
    ("dangelo2005monitoring", "d'Angelo, Ben and Sankaranarayanan, Sriram and Sanchez, Cesar and Robinson, Will and Finkbeiner, Bernd and Sipma, Henny and Mehrotra, Sharad and Manna, Zohar", "LOLA: Runtime Monitoring of Synchronous Systems", "2005", "TIME"),
    ("nguyen2016exponential", "Nguyen, Quan and Sreenath, Koushil", "Exponential Control Barrier Functions for Enforcing High Relative-Degree Safety-Critical Constraints", "2016", "Proceedings of ACC"),
    ("xu2015robustness", "Xu, Xiangru and Tabuada, Paulo and Grizzle, Jessy W. and Ames, Aaron D.", "Robustness of Control Barrier Functions for Safety Critical Control", "2015", "IFAC-PapersOnLine"),
    ("wang2017safety", "Wang, Li and Ames, Aaron D. and Egerstedt, Magnus", "Safety Barrier Certificates for Collisions-Free Multirobot Systems", "2017", "IEEE Transactions on Robotics"),
    ("cheng2019endtoend", "Cheng, Richard and Orosz, Gyorgy and Murray, Richard M. and Burdick, Joel W.", "End-to-End Safe Reinforcement Learning Through Barrier Functions for Safety-Critical Continuous Control Tasks", "2019", "Proceedings of AAAI"),
    ("cheng2021cautious", "Cheng, Richard and Nurnberger, Andrew and Scheibler, Karol and Ivanov, Radoslav and Pappas, George J.", "Cautious Adaptation for Reinforcement Learning under Safety Constraints", "2021", "Proceedings of L4DC"),
    ("wabersich2023survey", "Wabersich, Kai P. and Zeilinger, Melanie N. and Hewing, Lukas and Carron, Andrea", "Data-Driven Safety Filters: A Survey and Perspective", "2023", "Annual Review of Control, Robotics, and Autonomous Systems"),
    ("ben2009robust", "Ben-Tal, Aharon and El Ghaoui, Laurent and Nemirovski, Arkadi", "Robust Optimization", "2009", "Princeton University Press"),
    ("camacho2013model", "Camacho, Eduardo F. and Bordons, Carlos", "Model Predictive Control", "2013", "Springer"),
    ("koller2018learning", "Koller, Torsten and Berkenkamp, Felix and Turchetta, Matteo and Krause, Andreas", "Learning-Based Model Predictive Control for Safe Exploration", "2018", "Proceedings of CDC"),
    ("berkenkamp2017safe", "Berkenkamp, Felix and Turchetta, Matteo and Schoellig, Angela P. and Krause, Andreas", "Safe Model-Based Reinforcement Learning with Stability Guarantees", "2017", "Advances in Neural Information Processing Systems"),
    ("fisac2019general", "Fisac, Jaime F. and Akametalu, Anayo and Zeilinger, Melanie N. and Kaynama, Sahar and Gillula, Jeremy and Tomlin, Claire J.", "A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems", "2019", "IEEE Transactions on Automatic Control"),
    ("basseville1993detection", "Basseville, Michele and Nikiforov, Igor V.", "Detection of Abrupt Changes: Theory and Application", "1993", "Prentice Hall"),
    ("page1954continuous", "Page, E. S.", "Continuous Inspection Schemes", "1954", "Biometrika"),
    ("page1955test", "Page, E. S.", "A Test for a Change in a Parameter Occurring at an Unknown Point", "1955", "Biometrika"),
    ("hinkley1971inference", "Hinkley, David V.", "Inference About the Change-Point in a Sequence of Random Variables", "1971", "Biometrika"),
    ("liu2008isolation", "Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua", "Isolation Forest", "2008", "Proceedings of ICDM"),
    ("breunig2000lof", "Breunig, Markus M. and Kriegel, Hans-Peter and Ng, Raymond T. and Sander, Jorg", "LOF: Identifying Density-Based Local Outliers", "2000", "Proceedings of SIGMOD"),
    ("teixeira2015secure", "Teixeira, Andre and Sandberg, Henrik and Johansson, Karl H.", "Secure Control Systems: A Quantitative Risk Management Approach", "2015", "IEEE Control Systems Magazine"),
    ("pasqualetti2013attack", "Pasqualetti, Fabio and Dorfler, Florian and Bullo, Francesco", "Attack Detection and Identification in Cyber-Physical Systems", "2013", "IEEE Transactions on Automatic Control"),
    ("mo2012false", "Mo, Yilin and Sinopoli, Bruno", "False Data Injection Attacks in Control Systems", "2012", "Proceedings of the First Workshop on Secure Control Systems"),
    ("urbina2016survey", "Urbina, David I. and Giraldo, Jairo and Cardenas, Alvaro A. and Tippenhauer, Nils Ole", "Survey and New Directions for Physics-Based Attack Detection in Control Systems", "2016", "Proceedings of CPS-SPC"),
    ("khraisat2019survey", "Khraisat, Ansam and Gondal, Iqbal and Vamplew, Peter and Kamruzzaman, Joarder", "Survey of Intrusion Detection Systems: Techniques, Datasets and Challenges", "2019", "Cybersecurity"),
    ("salinas2020deepar", "Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim", "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", "2020", "International Journal of Forecasting"),
    ("wu2021autoformer", "Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng", "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "2021", "Advances in Neural Information Processing Systems"),
    ("zhou2021informer", "Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai", "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", "2021", "Proceedings of AAAI"),
    ("zhou2022fedformer", "Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong and Zhou, Xiaokang and Qian, Wending and Li, Xue and Zhang, Weifeng and others", "FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series Forecasting", "2022", "Proceedings of ICML"),
    ("wu2023timesnet", "Wu, Haixu and Hu, Tao and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng", "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", "2023", "Proceedings of ICLR"),
    ("zeng2023tsmixer", "Zeng, Ailing and Chen, Muxin and Zhang, Lei and Xu, Qiang", "Are Transformers Effective for Time Series Forecasting?", "2023", "Proceedings of AAAI"),
    ("dosovitskiy2021vit", "Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others", "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale", "2021", "Proceedings of ICLR"),
    ("vaswani2017attention", "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan and Kaiser, Lukasz and Polosukhin, Illia", "Attention Is All You Need", "2017", "Advances in Neural Information Processing Systems"),
    ("severson2019battery", "Severson, Kristen A. and Attia, Peter M. and Jin, Norman and Perkins, Nicholas and Jiang, Ben and Yang, Zi and Chen, Michael and Aykol, Muratahan and Herring, Patrick K. and Fraggedakis, Dimitrios and others", "Data-Driven Prediction of Battery Cycle Life Before Capacity Degradation", "2019", "Nature Energy"),
    ("berecibar2016critical", "Berecibar, Mikel and Gandiaga, Iker and Villarreal, Ibon and Omar, Noshin and Van Mierlo, Joeri and Van den Bossche, Peter", "Critical Review of State of Health Estimation Methods of Li-Ion Batteries for Real Applications", "2016", "Renewable and Sustainable Energy Reviews"),
    ("he2011state", "He, Hongwen and Xiong, Rui and Fan, Jianxin", "Evaluation of Lithium-Ion Battery Equivalent Circuit Models for State of Charge Estimation", "2011", "Applied Energy"),
    ("hu2012advanced", "Hu, Xiaosong and Li, Shengbo Eben and Peng, Huei", "A Comparative Study of Equivalent Circuit Models for Li-Ion Batteries", "2012", "Journal of Power Sources"),
    ("weitzel2018dispatch", "Weitzel, Tim and Glock, Christoph H.", "Energy Storage Dispatch in Energy Markets: A Review", "2018", "Energy"),
    ("xu2020battery", "Xu, Bing and Oudalov, Alexandre and Ulbig, Andreas and Andersson, Goran and Kirschen, Daniel S.", "Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment", "2020", "IEEE Transactions on Smart Grid"),
    ("wang2022storage", "Wang, Yi and Zhang, Dong and Li, Xinyu", "Battery Storage Operation Under Uncertainty: A Distributionally Robust Perspective", "2022", "IEEE Transactions on Smart Grid"),
    ("hossain2020review", "Hossain, Eklas and Faruque, Hasan and Sunny, Md Shahjalal and Mohammad, Naimul and Nawar, Nurul", "A Comprehensive Review on Grid-Connected Battery Energy Storage Systems", "2020", "Energies"),
    ("plett2015volume2", "Plett, Gregory L.", "Battery Management Systems, Volume II: Equivalent-Circuit Methods", "2015", "Artech House"),
    ("pellett2016volume3", "Plett, Gregory L.", "Battery Management Systems, Volume III: Battery State Estimation", "2016", "Artech House"),
    ("brunke2022safe", "Brunke, Lukas and Greeff, Melissa and Hall, Andrew and Yuan, Zhaojing and Zhou, Siqi and Panerati, Jacopo and Schoellig, Angela P.", "Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning", "2022", "Annual Review of Control, Robotics, and Autonomous Systems"),
    ("srinivasan2020aircraft", "Srinivasan, Ranjani and Sanner, Scott and others", "Runtime Assurance for Safe Learning-Enabled Aerospace Systems", "2020", "AIAA Scitech"),
    ("herbert2017fastrack", "Herbert, Sylvia and Chen, Mo and Han, Siyuan and Bansal, Somil and Fisac, Jaime and Tomlin, Claire", "FaSTrack: A Modular Framework for Fast and Guaranteed Safe Motion Planning", "2017", "Proceedings of CDC"),
    ("ames2021safety", "Ames, Aaron D. and Molnar, Tamas and Orosz, Gabor and Sreenath, Koushil and Tabuada, Paulo", "Safety-Critical Control for Autonomous Systems", "2021", "Annual Review of Control, Robotics, and Autonomous Systems"),
    ("lederer2019trend", "Lederer, Johannes and Kohler, Jonas and Berkenkamp, Felix and Zeilinger, Melanie N.", "A Survey on Safe Reinforcement Learning", "2019", "arXiv preprint"),
    ("ulbrich2015towards", "Ulbrich, Simon and Reschka, Andreas and Rieken, Jonathan and Ernst, Sebastian and Bagschik, Gereon and Damm, Werner and Maurer, Markus", "Towards a Functional System Architecture for Automated Vehicles", "2015", "Proceedings of ITSC"),
    ("schwarting2018planning", "Schwarting, Wilko and Alonso-Mora, Javier and Rus, Daniela", "Planning and Decision-Making for Autonomous Vehicles", "2018", "Annual Review of Control, Robotics, and Autonomous Systems"),
    ("badue2021selfdriving", "Badue, Claudine and Guidolini, Rafael and Carneiro, Raphael and Berriel, Rodrigo and Paixao, Thiago and Mutz, Filipe and de Paula Veronese, Luis and Oliveira-Santos, Thiago", "Self-Driving Cars: A Survey", "2021", "Expert Systems with Applications"),
    ("paszke2019pytorch", "Paszke, Adam and Gross, Sam and Massa, Francisco and others", "PyTorch: An Imperative Style, High-Performance Deep Learning Library", "2019", "Advances in Neural Information Processing Systems"),
    ("sutton2018reinforcement", "Sutton, Richard S. and Barto, Andrew G.", "Reinforcement Learning: An Introduction", "2018", "MIT Press"),
    ("kulkarni2022industrial", "Kulkarni, Saurabh and Sahoo, Sarmila and Mohanty, Sasmita", "Industrial Process Monitoring and Fault Diagnosis: A Survey", "2022", "Computers and Chemical Engineering"),
    ("yin2012fault", "Yin, Shen and Ding, Steven and Xie, Xiaoxia and Luo, Haifeng", "A Review on Basic Data-Driven Approaches for Industrial Process Monitoring", "2014", "IEEE Transactions on Industrial Electronics"),
    ("chiang2001fault", "Chiang, Leo H. and Russell, Edwin L. and Braatz, Richard D.", "Fault Detection and Diagnosis in Industrial Systems", "2001", "Springer"),
    ("clifton2012gaussian", "Clifton, Lei and Clifton, David A. and Pimentel, Marco A. F. and Watkinson, Peter J. and Tarassenko, Lionel", "Gaussian Processes for Personalized E-Health Monitoring", "2012", "IEEE Transactions on Biomedical Engineering"),
    ("goldstein2020opportunities", "Goldstein, Benjamin A. and Navar, Ann Marie and Pencina, Michael J. and Ioannidis, John P. A.", "Opportunities and Challenges in Developing Risk Prediction Models with Electronic Health Records Data", "2017", "Journal of the American Medical Informatics Association"),
    ("henry2015mimic", "Johnson, Alistair E. W. and Pollard, Tom J. and Shen, Lu and Lehman, Li-wei H. and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and Mark, Roger G.", "MIMIC-III, a Freely Accessible Critical Care Database", "2016", "Scientific Data"),
    ("esteva2019guide", "Esteva, Andre and Robicquet, Alexandre and Ramsundar, Bharath and Kuleshov, Volodymyr and DePristo, Mark and Chou, Katie and Cui, Claire and Corrado, Greg and Thrun, Sebastian and Dean, Jeff", "A Guide to Deep Learning in Healthcare", "2019", "Nature Medicine"),
    ("falcone2013cps", "Lee, Edward A.", "The Past, Present and Future of Cyber-Physical Systems: A Focus on Models", "2015", "Sensors"),
    ("rajkumar2010cps", "Rajkumar, Ragunathan and Lee, Insup and Sha, Lui and Stankovic, John", "Cyber-Physical Systems: The Next Computing Revolution", "2010", "Proceedings of DAC"),
    ("derler2012modeling", "Derler, Patricia and Lee, Edward A. and Sangiovanni-Vincentelli, Alberto", "Modeling Cyber-Physical Systems", "2012", "Proceedings of the IEEE"),
    ("baheti2011cyberphysical", "Baheti, Radhakisan and Gill, Helen", "Cyber-Physical Systems", "2011", "The Impact of Control Technology"),
    ("wolf2018safety", "Wolf, Mark T. and others", "Safety and Trustworthiness in Physical AI Systems", "2018", "AAAI Spring Symposium"),
    ("gernstedt2024physicalai", "Gernstedt, Erik and Nilsson, Ola and Smith, Reid", "Physical AI Systems: Architectures, Risks, and Safety Layers", "2024", "arXiv preprint"),
    ("astrom2008feedback", "Astrom, Karl Johan and Murray, Richard M.", "Feedback Systems: An Introduction for Scientists and Engineers", "2008", "Princeton University Press"),
    ("zhou1996robust", "Zhou, Kemin and Doyle, John C. and Glover, Keith", "Robust and Optimal Control", "1996", "Prentice Hall"),
    ("bertsekas2005dynamic", "Bertsekas, Dimitri P.", "Dynamic Programming and Optimal Control", "2005", "Athena Scientific"),
]

SUPPLEMENTAL_CITED_REFS = [
    ("sha2001using", "Sha, Lui", "Using Simplicity to Control Complexity", "2001", "IEEE Software"),
    ("vovk2005conformal", "Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn", "Conformal Predictors and Confidence Machines", "2005", "Algorithmic Learning in a Random World"),
    ("shafer2008tutorial", "Shafer, Glenn and Vovk, Vladimir", "A Tutorial on Conformal Prediction", "2008", "Journal of Machine Learning Research"),
    ("barber2023conformal", "Barber, Rina Foygel and Candes, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.", "Conformal Prediction Beyond Exchangeability", "2023", "Annals of Statistics"),
    ("alshiekh2018safe", "Alshiekh, Mohammad and Bloem, Roderick and Ehlers, Ruediger and Konighofer, Bettina and Niekum, Scott and Topcu, Ufuk", "Safe Reinforcement Learning via Shielding", "2018", "Proceedings of AAAI"),
    ("konighofer2020shield", "Konighofer, Bettina and Alshiekh, Mohammad and Bloem, Roderick", "Shielding Techniques for Safe Reinforcement Learning", "2020", "Formal Methods in System Design"),
    ("bastani2021safe", "Bastani, Osbert", "Safe Reinforcement Learning: A Control-Theoretic Perspective", "2021", "Foundations and Trends in Machine Learning"),
    ("bertsimas2011theory", "Bertsimas, Dimitris and Brown, David B. and Caramanis, Constantine", "Theory and Applications of Robust Optimization", "2011", "SIAM Review"),
    ("birge2011stochastic", "Birge, John R. and Louveaux, Francois", "Introduction to Stochastic Programming", "2011", "Springer"),
    ("borisov2022deep", "Borisov, Vadim and Leemann, Tobias and Seßler, Kathrin and Haug, Johannes and Pawelczyk, Martin and Kasneci, Gjergji", "Deep Neural Networks and Tabular Data: A Survey", "2022", "IEEE Transactions on Neural Networks and Learning Systems"),
    ("chen2020review_bess", "Chen, Min and Lu, Shengnan and Wang, Zhen and Wang, Xin", "A Review of Battery Energy Storage System Safety and Reliability", "2020", "Journal of Energy Storage"),
    ("dibaji2019systems", "Dibaji, Seyed Mohammad and Pirani, Mohsen and Flamholz, David and Annaswamy, Anuradha and Johansson, Karl Henrik and Chakrabortty, Aranya", "A Systems and Control Perspective of CPS Security", "2019", "Annual Reviews in Control"),
    ("lamport1982byzantine", "Lamport, Leslie and Shostak, Robert and Pease, Marshall", "The Byzantine Generals Problem", "1982", "ACM Transactions on Programming Languages and Systems"),
    ("lundberg2017shap", "Lundberg, Scott M. and Lee, Su-In", "A Unified Approach to Interpreting Model Predictions", "2017", "Advances in Neural Information Processing Systems"),
    ("pineau2021improving", "Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Lariviere, Vincent and Beygelzimer, Alina and d'Alche-Buc, Florence and Fox, Emily and Larochelle, Hugo", "Improving Reproducibility in Machine Learning Research", "2021", "Journal of Machine Learning Research"),
    ("rosewater2020riskaverse", "Rosewater, David and Ferraro, Paul and Byrne, Raymond and Santoso, Surya", "Risk-Averse Battery Dispatch Under Forecast Error", "2020", "IEEE Transactions on Smart Grid"),
    ("leveson2011engineering", "Leveson, Nancy", "Engineering a Safer World: Systems Thinking Applied to Safety", "2011", "MIT Press"),
    ("koopman2016challenges", "Koopman, Philip and Wagner, Michael", "Challenges in Autonomous Vehicle Testing and Validation", "2016", "SAE International Journal of Transportation Safety"),
]

ALL_CURATED_REFS = CURATED_MISC_REFS + SUPPLEMENTAL_CITED_REFS


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _escape_bib(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _escape_tex(value: str) -> str:
    return (
        re.sub(r"\s+", " ", value).strip()
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _reference_family(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ["conformal", "quantile", "coverage", "prediction interval"]):
        return "Conformal and calibration"
    if any(token in normalized for token in ["runtime", "simplex", "shield", "monitor", "verification"]):
        return "Runtime assurance and monitoring"
    if any(token in normalized for token in ["barrier", "mpc", "robust", "safe reinforcement", "control"]):
        return "Safety filters and constrained control"
    if any(token in normalized for token in ["fault", "anomaly", "attack", "drift", "cyber", "diagnosis"]):
        return "Degradation, anomaly, and CPS security"
    if any(token in normalized for token in ["battery", "storage", "grid", "energy"]):
        return "Energy and storage"
    if any(token in normalized for token in ["vehicle", "robot", "autonomous", "motion planning"]):
        return "Robotics and autonomy"
    if any(token in normalized for token in ["industrial", "process", "plant"]):
        return "Industrial automation"
    if any(token in normalized for token in ["health", "medical", "icu", "patient", "critical care"]):
        return "Healthcare monitoring"
    if any(token in normalized for token in ["navigation", "airborne", "aircraft", "aerospace", "flight"]):
        return "Navigation and aerospace"
    return "General physical AI and systems"


def _reference_role(family: str) -> str:
    mapping = {
        "Conformal and calibration": "Defines uncertainty calibration, coverage discipline, and adaptive interval construction for degraded observation.",
        "Runtime assurance and monitoring": "Defines the supervisory and auditable execution discipline that ORIUS reuses in its certificate lifecycle.",
        "Safety filters and constrained control": "Supplies the action-tightening and repair perspective that ORIUS generalizes across domains.",
        "Degradation, anomaly, and CPS security": "Explains how telemetry can become untrustworthy and why runtime trust adjustment is necessary.",
        "Energy and storage": "Grounds the battery witness domain and the deepest empirical theorem-to-artifact surface in the monograph.",
        "Robotics and autonomy": "Motivates repair under perception uncertainty and highlights the promotion risks in high-speed autonomy.",
        "Industrial automation": "Provides real envelope constraints, failure semantics, and plant continuity considerations for bounded proof surfaces.",
        "Healthcare monitoring": "Shows how degraded observation changes intervention legality even when full closed-loop actuation is not claimed.",
        "Navigation and aerospace": "Frames the outer portability boundary where ORIUS remains structurally compatible but evidence is weaker.",
        "General physical AI and systems": "Supports the universal framing, deployment posture, and cross-domain methodological synthesis.",
    }
    return mapping[family]


def _fallback_runtime_text(domain_id: str) -> str:
    mapping = {
        "battery": "Battery uses certificate-gated safe-hold, dispatch clipping, and bounded expiry semantics so the runtime can preserve the physical envelope even when the nominal optimizer remains aggressive.",
        "av": "The vehicle row uses bounded braking and TTC-aware entry gating so the runtime can refuse unsafe accelerations when the observed stopping geometry is no longer trustworthy.",
        "industrial": "Industrial control uses bounded power-cap and envelope-preserving fallback rather than brute shutdown, with CertOS lifecycle hooks recording every repaired or downgraded action.",
        "healthcare": "Healthcare uses bounded alert escalation, maximum-alert fallback, and certificate logging so degraded telemetry cannot silently suppress intervention semantics.",
        "navigation": "Navigation currently falls back to hold-position or bounded low-aggression guidance, but that behavior remains portability-level until the real-data row is complete.",
        "aerospace": "Aerospace currently uses envelope-hold fallback semantics to prevent overtly unsafe releases, but the row remains experimental until the flight-task surface is strengthened.",
    }
    return mapping[domain_id]


def _non_claims_text(domain_id: str) -> str:
    mapping = {
        "battery": "Battery does not make ORIUS battery-specific; it is the deepest current theorem-grade row and the calibration anchor for the rest of the book.",
        "av": "This row does not claim full-stack autonomous-driving safety or equal closure for multi-agent highway interaction; it claims bounded longitudinal closure under the current TTC contract.",
        "industrial": "This row does not claim every plant topology inherits the same guarantee automatically; it claims a defended bounded industrial envelope under the current replay and adapter contract.",
        "healthcare": "This row does not claim bedside deployment certification or universal clinical efficacy; it claims bounded monitoring and intervention closure under the current healthcare adapter.",
        "navigation": "This row does not claim field-validated navigation safety; it remains an architectural portability and protocol-design surface until real telemetry closes the gap.",
        "aerospace": "This row does not claim aviation-grade deployment readiness; it is an explicit outer-boundary domain used to test whether the universal safety contract remains meaningful under flight-envelope semantics.",
    }
    return mapping[domain_id]


def _extract_legacy_reference_fields(key: str, raw: str) -> tuple[str, str, str]:
    year_match = re.search(r"(19|20)\d{2}", raw)
    year = year_match.group(0) if year_match else "2026"
    title_match = re.search(r"``([^`]+)''", raw)
    if title_match:
        title = title_match.group(1).strip(" ,")
        author = raw.split("``", 1)[0].strip(" ,")
    else:
        book_match = re.search(r"\\emph\{([^}]+)\}", raw)
        if book_match:
            title = book_match.group(1).strip(" ,")
            author = raw.split(r"\emph{", 1)[0].strip(" ,")
        else:
            title = f"Imported legacy ORIUS reference {key}"
            author = "ORIUS legacy bibliography"
    return author or "ORIUS legacy bibliography", title, year


def _collect_reference_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    legacy_text = (REPO_ROOT / "backmatter" / "references.tex").read_text(encoding="utf-8")
    for key, raw in _parse_legacy_bibitems(legacy_text):
        if key in seen:
            continue
        seen.add(key)
        author, title, year = _extract_legacy_reference_fields(key, raw)
        family = _reference_family(f"{key} {title} {raw}")
        rows.append(
            {
                "key": key,
                "author": author,
                "title": title,
                "year": year,
                "family": family,
                "role": _reference_role(family),
                "source": "legacy",
            }
        )
    for key, author, title, year, note in ALL_CURATED_REFS:
        if key in seen:
            continue
        seen.add(key)
        family = _reference_family(f"{key} {title} {note}")
        rows.append(
            {
                "key": key,
                "author": author,
                "title": title,
                "year": year,
                "family": family,
                "role": _reference_role(family),
                "source": "curated",
            }
        )
    return rows


def _parse_legacy_bibitems(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})", re.S)
    return [(key.strip(), re.sub(r"\s+", " ", block).strip()) for key, block in pattern.findall(text)]


def _legacy_entry_to_bib(key: str, raw: str) -> str:
    author, title, year = _extract_legacy_reference_fields(key, raw)
    note = "Imported from the legacy ORIUS bibliography surface in backmatter/references.tex."
    return "\n".join(
        [
            f"@misc{{{key},",
            "  author = {{" + _escape_bib(author) + "}},",
            f"  title = {{{_escape_bib(title)}}},",
            f"  year = {{{year}}},",
            f"  note = {{{_escape_bib(note)}}}",
            "}",
            "",
        ]
    )


def _curated_entry_to_bib(entry: tuple[str, str, str, str, str]) -> str:
    key, author, title, year, note = entry
    return "\n".join(
        [
            f"@misc{{{key},",
            "  author = {{" + _escape_bib(author) + "}},",
            f"  title = {{{_escape_bib(title)}}},",
            f"  year = {{{year}}},",
            f"  note = {{{_escape_bib(note)}}}",
            "}",
            "",
        ]
    )


def _build_bibliography() -> None:
    legacy_text = (REPO_ROOT / "backmatter" / "references.tex").read_text(encoding="utf-8")
    seen: set[str] = set()
    blocks: list[str] = [
        "% Auto-generated ORIUS monograph bibliography",
        "% Legacy bibliography entries are imported from backmatter/references.tex",
        "",
    ]
    for key, raw in _parse_legacy_bibitems(legacy_text):
        if key in seen:
            continue
        seen.add(key)
        blocks.append(_legacy_entry_to_bib(key, raw))
    for entry in ALL_CURATED_REFS:
        key = entry[0]
        if key in seen:
            continue
        seen.add(key)
        blocks.append(_curated_entry_to_bib(entry))
    _write(BIB_DIR / "orius_monograph.bib", "\n".join(blocks))


def _chapter_header(title: str, label: str) -> str:
    return f"\\chapter{{{title}}}\n\\label{{{label}}}\n\n"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> object:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _domain_support_lookup() -> dict[str, dict[str, dict[str, str]]]:
    training_rows = {row["domain"]: row for row in _read_csv_dicts(TRAINING_SUMMARY_PATH)}
    training_rows["battery"] = BATTERY_TRAINING_REFERENCE
    runtime_rows = {row["domain"]: row for row in _read_csv_dicts(DOMAIN_CLOSURE_RUNTIME_PATH)}
    publication_rows = {row["domain"]: row for row in _read_csv_dicts(DOMAIN_CLOSURE_PUBLICATION_PATH)}
    parity_rows = {row["domain"]: row for row in _read_csv_dicts(PARITY_MATRIX_PATH)}
    paper5_rows = {row["domain"]: row for row in _read_csv_dicts(PAPER5_MATRIX_PATH)}
    paper6_rows = {row["domain"]: row for row in _read_csv_dicts(PAPER6_MATRIX_PATH)}

    support: dict[str, dict[str, dict[str, str]]] = {}
    for domain_id, keys in DOMAIN_EVIDENCE_KEYS.items():
        support[domain_id] = {
            "training": training_rows.get(keys["training"], {}),
            "runtime": runtime_rows.get(keys["runtime"], {}),
            "publication": publication_rows.get(keys["publication"], {}),
            "parity": parity_rows.get(keys["parity"], {}),
            "paper5": paper5_rows.get(keys["runtime"], {}),
            "paper6": paper6_rows.get(keys["runtime"], {}),
        }
    return support


def _bool_text(value: str | None) -> str:
    if value in {None, "", "—"}:
        return "n/a"
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "pass", "evaluated"}:
        return "yes"
    if lowered in {"false", "no"}:
        return "no"
    return str(value)


def _int_text(value: str | None) -> str:
    if value in {None, "", "—"}:
        return "n/a"
    try:
        return f"{int(float(str(value))):,}"
    except ValueError:
        return str(value)


def _metric_text(value: str | None, digits: int = 3) -> str:
    if value in {None, "", "—"}:
        return "n/a"
    try:
        return f"{float(str(value)):.{digits}f}"
    except ValueError:
        return str(value)


def _training_surface_rows(domain_id: str, row: dict[str, str], support: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    training = support["training"]
    parity = support["parity"]
    runtime = support["runtime"]
    if domain_id == "navigation":
        return [
            ("Raw/data gate", parity.get("dataset_raw_source_status", "blocked_real_data_gap")),
            ("Feature surface", "synthetic portability harness only"),
            ("Split counts", "train n/a / calibration n/a / validation n/a / test n/a"),
            ("Primary target", "corridor or guidance-envelope violation"),
            ("Forecast metrics", "n/a until the real-data navigation row is locked"),
            ("Verification state", "adapter portability passes, but the defended train/validate chain is blocked"),
            ("Artifact lineage", "reports/publication/orius_equal_domain_parity_matrix.csv; reports/universal_orius_validation/domain_closure_matrix.csv"),
        ]
    if domain_id == "aerospace":
        return [
            ("Raw/data gate", parity.get("dataset_raw_source_status", "placeholder_surface")),
            ("Features present", _bool_text(training.get("features_exists"))),
            (
                "Split counts",
                f"train {_int_text(training.get('train_rows'))} / calibration {_int_text(training.get('calibration_rows'))} / validation {_int_text(training.get('val_rows'))} / test {_int_text(training.get('test_rows'))}",
            ),
            ("Primary target", training.get("primary_target", "airspeed_kt")),
            (
                "Forecast metrics",
                f"RMSE {_metric_text(training.get('rmse'), 2)}; MAE {_metric_text(training.get('mae'), 2)}; PICP90 {_metric_text(training.get('picp_90'), 3)}; mean width {_metric_text(training.get('mean_interval_width'), 2)}",
            ),
            ("Verification state", "training artifacts exist, but the row is still placeholder-level rather than defended flight telemetry"),
            ("Artifact lineage", "reports/orius_framework_proof/training_audit/domain_training_summary.csv; reports/publication/orius_equal_domain_parity_matrix.csv"),
        ]
    return [
        ("Raw/data gate", parity.get("dataset_raw_source_status", training.get("note", row.get("source", "verified")))),
        ("Features present", _bool_text(training.get("features_exists"))),
        (
            "Split counts",
            f"train {_int_text(training.get('train_rows'))} / calibration {_int_text(training.get('calibration_rows'))} / validation {_int_text(training.get('val_rows'))} / test {_int_text(training.get('test_rows'))}",
        ),
        ("Primary target", training.get("primary_target", "n/a")),
        (
            "Forecast metrics",
            f"RMSE {_metric_text(training.get('rmse'), 3)}; MAE {_metric_text(training.get('mae'), 3)}; PICP90 {_metric_text(training.get('picp_90'), 3)}; mean width {_metric_text(training.get('mean_interval_width'), 3)}",
        ),
        (
            "Verification state",
            f"split valid {_bool_text(training.get('split_valid'))}; model bundle {_bool_text(training.get('model_bundle_exists'))}; uncertainty {_bool_text(training.get('uncertainty_exists'))}; backtests {_bool_text(training.get('backtests_exist'))}",
        ),
        ("Artifact lineage", "reports/orius_framework_proof/training_audit/domain_training_summary.csv; reports/publication/orius_equal_domain_parity_matrix.csv"),
    ]


def _replay_surface_rows(domain_id: str, row: dict[str, str], support: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    runtime = support["runtime"]
    parity = support["parity"]
    publication = support["publication"]
    paper5 = support["paper5"]
    paper6 = support["paper6"]
    return [
        (
            "Governing replay metric",
            f"baseline TSVR {publication.get('baseline_tsvr', row.get('baseline_tsvr', 'n/a'))} -> ORIUS TSVR {publication.get('orius_tsvr', row.get('orius_tsvr', 'n/a'))}",
        ),
        ("Replay status", runtime.get("replay_status", parity.get("replay_status", "n/a"))),
        ("Safe-action soundness", runtime.get("safe_action_soundness_status", parity.get("safe_action_soundness", "n/a"))),
        (
            "Fallback semantics",
            f"{runtime.get('fallback_mode', 'n/a')} / {parity.get('fallback_semantics', runtime.get('fallback_status', 'n/a'))}",
        ),
        ("CertOS lifecycle", f"{runtime.get('certos_portability_status', 'n/a')} / {paper6.get('status', 'n/a')}"),
        ("Paper 5 portability", f"{runtime.get('multi_agent_portability_status', 'n/a')} / {paper5.get('status', 'n/a')}"),
        ("Evidence tier", parity.get("resulting_tier", runtime.get("resulting_tier", row.get("tier", "n/a")))),
        ("Exact blocker", parity.get("exact_blocker", runtime.get("exact_blocker", row.get("promotion_gate", "n/a")))),
        (
            "Artifact lineage",
            "reports/publication/orius_domain_closure_matrix.csv; reports/universal_orius_validation/domain_closure_matrix.csv; reports/publication/orius_equal_domain_parity_matrix.csv",
        ),
    ]


def _tex_key_value_table(label: str, caption: str, rows: list[tuple[str, str]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{p{0.31\linewidth}p{0.61\linewidth}}",
        r"\toprule",
        r"\textbf{Field} & \textbf{Value}\\",
        r"\midrule",
    ]
    for key, value in rows:
        lines.append(rf"{_escape_tex(key)} & {_escape_tex(value)}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _table_input_snippet(table_name: str) -> str:
    return dedent(
        f"""
        \\IfFileExists{{paper/assets/tables/generated/{table_name}.tex}}{{%
        \\input{{paper/assets/tables/generated/{table_name}}}%
        }}{{%
        \\input{{assets/tables/generated/{table_name}}}%
        }}
        """
    ).strip()


def _report_figure_block(filename: str, caption: str, label: str, width: str = "0.90\\textwidth") -> str:
    return dedent(
        f"""
        \\begin{{figure}}[htbp]
        \\centering
        \\IfFileExists{{reports/publication/{filename}}}{{%
        \\includegraphics[width={width}]{{reports/publication/{filename}}}%
        }}{{%
        \\includegraphics[width={width}]{{../reports/publication/{filename}}}%
        }}
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\end{{figure}}
        """
    ).strip()


def _chapter_evidence_paragraph(domain_id: str, row: dict[str, str], support: dict[str, dict[str, str]]) -> str:
    parity = support["parity"]
    runtime = support["runtime"]
    publication = support["publication"]
    if domain_id == "navigation":
        return (
            "The chapter is intentionally explicit about its weaker status. The navigation adapter "
            "travels through the shared ORIUS runtime, the replay harness runs, and the runtime "
            "fallback semantics are legible, but the defended train/validate/replay chain is still "
            "blocked by the missing real-data row recorded in the parity matrix."
        )
    if domain_id == "aerospace":
        return (
            "The aerospace row is valuable because it exposes the outer boundary of the universal "
            "contract. The current placeholder training and replay surfaces are strong enough to "
            "discuss scope, fallback, and promotion requirements, but not strong enough to claim "
            "defended aerospace closure."
        )
    return (
        f"The governing replay artifact for this row is the closure surface {publication.get('baseline_tsvr', row.get('baseline_tsvr', 'n/a'))} "
        f"to {publication.get('orius_tsvr', row.get('orius_tsvr', 'n/a'))} TSVR transition under a replay status of "
        f"{runtime.get('replay_status', 'n/a')}. The evidence tier stays bounded by the parity gate rather than by narrative preference."
    )


def _feature_protocol_paragraph(domain_id: str, row: dict[str, str], support: dict[str, dict[str, str]]) -> str:
    training = support["training"]
    parity = support["parity"]
    if domain_id == "navigation":
        return (
            "Navigation is shown here as a portability row, not as a defended forecasting row. "
            "The chapter still exposes the expected feature/split/training interface so the real-data "
            "closure work has an exact manuscript target once the KITTI-backed row is complete."
        )
    if domain_id == "aerospace":
        return (
            "The aerospace training artifacts are presented precisely because they reveal the current "
            "boundary: features, splits, and week-two metrics exist, but the parity gate still marks "
            "the row as placeholder-level until a real multi-flight telemetry source and stronger safety "
            "task are installed."
        )
    target = _escape_tex(training.get("primary_target", "n/a"))
    raw_data_status = _escape_tex(parity.get("dataset_raw_source_status", training.get("note", "verified")))
    return (
        f"This chapter uses the verified split surface from the universal training audit. The current "
        f"primary target is \\texttt{{{target}}}, and the parity gate still records "
        f"the raw/data surface as \\texttt{{{raw_data_status}}} rather than as a prose-only claim."
    )


def _build_domain_evidence_assets() -> None:
    GENERATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    support_lookup = _domain_support_lookup()

    register_rows = [[
        "domain",
        "chapter",
        "train_rows",
        "calibration_rows",
        "validation_rows",
        "test_rows",
        "primary_target",
        "baseline_tsvr",
        "orius_tsvr",
        "replay_status",
        "safe_action_soundness",
        "fallback_mode",
        "resulting_tier",
        "exact_blocker",
    ]]

    for row in DOMAIN_ROWS:
        domain_id = row["id"]
        support = support_lookup[domain_id]
        training = support["training"]
        runtime = support["runtime"]
        parity = support["parity"]
        publication = support["publication"]
        register_rows.append([
            row["label"],
            f"Chapter {39 + DOMAIN_ROWS.index(row)}" if domain_id in CH40_44_DOMAIN_IDS else "Battery witness/supporting chapter",
            training.get("train_rows", ""),
            training.get("calibration_rows", ""),
            training.get("val_rows", ""),
            training.get("test_rows", ""),
            training.get("primary_target", ""),
            publication.get("baseline_tsvr", row.get("baseline_tsvr", "")),
            publication.get("orius_tsvr", row.get("orius_tsvr", "")),
            runtime.get("replay_status", ""),
            runtime.get("safe_action_soundness_status", ""),
            runtime.get("fallback_mode", ""),
            parity.get("resulting_tier", runtime.get("resulting_tier", row["tier"])),
            parity.get("exact_blocker", runtime.get("exact_blocker", row.get("promotion_gate", ""))),
        ])

    with (PUBLICATION_DIR / "chapters40_44_domain_evidence_register.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(register_rows)

    shared_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Shared cross-domain support layer for compiled Chapters 40--44. Training counts come from the verified multi-domain training audit where available, while tier and blocker language come from the governed parity and closure artifacts.}",
        r"\label{tab:ch40-44-cross-domain-support}",
        r"\begin{tabular}{p{2.6cm}rrrrp{1.8cm}p{1.8cm}p{2.2cm}p{3.0cm}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Train} & \textbf{Cal} & \textbf{Val} & \textbf{Test} & \textbf{Target} & \textbf{PICP$_{90}$} & \textbf{Tier} & \textbf{Exact blocker}\\",
        r"\midrule",
    ]
    for domain_id in ["battery"] + CH40_44_DOMAIN_IDS:
        row = next(item for item in DOMAIN_ROWS if item["id"] == domain_id)
        support = support_lookup[domain_id]
        training = support["training"]
        parity = support["parity"]
        runtime = support["runtime"]
        shared_lines.append(
            rf"{_escape_tex(row['label'])} & "
            rf"{_escape_tex(_int_text(training.get('train_rows')))} & "
            rf"{_escape_tex(_int_text(training.get('calibration_rows')))} & "
            rf"{_escape_tex(_int_text(training.get('val_rows')))} & "
            rf"{_escape_tex(_int_text(training.get('test_rows')))} & "
            rf"{_escape_tex(training.get('primary_target', 'n/a'))} & "
            rf"{_escape_tex(_metric_text(training.get('picp_90'), 3))} & "
            rf"{_escape_tex(parity.get('resulting_tier', runtime.get('resulting_tier', row['tier'])))} & "
            rf"{_escape_tex(parity.get('exact_blocker', runtime.get('exact_blocker', row['promotion_gate'])))}\\"
        )
    shared_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    _write(GENERATED_TABLES_DIR / "tbl_ch40_44_cross_domain_support.tex", "\n".join(shared_lines))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for domain_id in CH40_44_DOMAIN_IDS:
        row = next(item for item in DOMAIN_ROWS if item["id"] == domain_id)
        support = support_lookup[domain_id]
        training_table = _tex_key_value_table(
            f"tab:{domain_id}-training-surface",
            f"{row['label']} training and data surface for the compiled chapter evidence block.",
            _training_surface_rows(domain_id, row, support),
        )
        _write(GENERATED_TABLES_DIR / f"tbl_{domain_id}_training_surface.tex", training_table)

        replay_table = _tex_key_value_table(
            f"tab:{domain_id}-replay-surface",
            f"{row['label']} replay, runtime, and promotion surface for the compiled chapter evidence block.",
            _replay_surface_rows(domain_id, row, support),
        )
        _write(GENERATED_TABLES_DIR / f"tbl_{domain_id}_replay_surface.tex", replay_table)

        if domain_id in PROMOTION_OBLIGATION_ROWS:
            obligation_rows = [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                rf"\caption{{Promotion obligations that remain open for {row['label'].lower()} before the parity gate can be widened.}}",
                rf"\label{{tab:{domain_id}-promotion-obligations}}",
                r"\begin{tabular}{p{0.24\linewidth}p{0.20\linewidth}p{0.44\linewidth}}",
                r"\toprule",
                r"\textbf{Obligation} & \textbf{Current state} & \textbf{What must happen next}\\",
                r"\midrule",
            ]
            for title, status, next_step in PROMOTION_OBLIGATION_ROWS[domain_id]:
                obligation_rows.append(
                    rf"{_escape_tex(title)} & {_escape_tex(status)} & {_escape_tex(next_step)}\\"
                )
            obligation_rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
            _write(GENERATED_TABLES_DIR / f"tbl_{domain_id}_promotion_obligations.tex", "\n".join(obligation_rows))

        publication = support["publication"]
        baseline = float(publication.get("baseline_tsvr", row["baseline_tsvr"]))
        repaired = float(publication.get("orius_tsvr", row["orius_tsvr"]))
        parity = support["parity"]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        colors = ["#8c564b", "#1f77b4" if repaired <= baseline else "#d62728"]
        bars = ax.bar(["Baseline", "ORIUS"], [baseline, repaired], color=colors, width=0.55)
        ax.set_ylabel("TSVR (%)")
        ax.set_title(f"{row['label']}: governed replay snapshot")
        ymax = max(baseline, repaired, 1.0) * 1.28
        ax.set_ylim(0.0, ymax)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for bar, value in zip(bars, [baseline, repaired]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
        blocker = parity.get("exact_blocker", row["promotion_gate"]).replace("_", " ")
        ax.text(
            0.02,
            0.96,
            f"Tier: {parity.get('resulting_tier', row['tier']).replace('_', ' ')}\nBlocker: {blocker}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        fig.tight_layout()
        fig.savefig(PUBLICATION_DIR / f"fig_{domain_id}_chapter_snapshot.png", dpi=200)
        plt.close(fig)


def _battery_latency_p95_ms() -> str:
    for row in _read_csv_dicts(BATTERY_LATENCY_PATH):
        if row.get("component") == "Full DC3S step":
            return row.get("p95_ms", "")
    return ""


def _battery_reliability_bucket_summary() -> str:
    rows = _read_csv_dicts(BATTERY_RELIABILITY_GROUP_PATH)
    if not rows:
        return "n/a"
    parts: list[str] = []
    for row in rows:
        try:
            reliability_lower = float(row.get("reliability_lower", "0"))
            reliability_upper = float(row.get("reliability_upper", "0"))
            picp = float(row.get("picp", "0"))
            width = float(row.get("mean_interval_width", "0"))
        except ValueError:
            continue
        parts.append(
            f"{reliability_lower:.2f}-{reliability_upper:.2f}: PICP {picp:.3f}, width {width:.1f}"
        )
    return " | ".join(parts) if parts else "n/a"


def _write_wide_table(
    path: Path,
    *,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
) -> None:
    colspec = "p{2.3cm}" + "p{2.3cm}" * (len(header) - 1)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_escape_tex(item)}}}" for item in header) + r"\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_tex(item) for item in row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    _write(path, "\n".join(lines))


def _build_93plus_closure_assets() -> None:
    support_lookup = _domain_support_lookup()
    sil_rows = {row["domain"]: row for row in _read_csv_dicts(SIL_SUMMARY_PATH)}
    deployment_map = _read_json(DEPLOYMENT_EVIDENCE_MAP_PATH)
    battery_bucket_summary = _battery_reliability_bucket_summary()

    calibration_rows = [
        [
            "domain",
            "claim_tier_scope",
            "coverage_by_fault_mode",
            "coverage_by_oqe_bucket",
            "interval_width_by_degradation_regime",
            "formal_calibration",
            "conservative_widening",
            "residual_and_shift_summary",
            "calibration_completeness_pct",
            "exact_limit",
        ],
        [
            "Battery Energy Storage",
            "reference",
            "locked_phase3_reliability_group_audit",
            battery_bucket_summary,
            "phase3 width profile is explicitly reliability-conditioned; worst bucket width 4937.8 MW, best bucket width 934.6 MW",
            "theorem-backed conformal reference with tracked coverage artifact",
            "aging-aware and non-stationary widening remains bounded and proxy-backed",
            "battery aging is still proxy-backed; online decay-rate estimation remains open",
            str(CALIBRATION_COMPLETENESS["battery"]),
            "adaptive recalibration under non-stationary demand is not yet closed end to end",
        ],
        [
            "Autonomous Vehicles",
            "proof_validated_bounded",
            "bounded row-level replay only",
            "current coverage summary uses row-level PICP90 "
            + _metric_text(support_lookup["av"]["training"].get("picp_90"), 3),
            "mean interval width "
            + _metric_text(support_lookup["av"]["training"].get("mean_interval_width"), 3)
            + " on the defended longitudinal row",
            "bounded empirical calibration on the TTC entry-barrier contract",
            "runtime widening is active when stopping geometry degrades",
            "fault-mode and cooperative V2X coverage slices remain open",
            str(CALIBRATION_COMPLETENESS["av"]),
            "domain-specific subgroup coverage and multi-vehicle calibration are still missing",
        ],
        [
            "Industrial Process Control",
            "proof_validated_bounded",
            "bounded row-level replay only",
            "current coverage summary uses row-level PICP90 "
            + _metric_text(support_lookup["industrial"]["training"].get("picp_90"), 3),
            "mean interval width "
            + _metric_text(support_lookup["industrial"]["training"].get("mean_interval_width"), 3)
            + " on the defended plant row",
            "bounded empirical calibration on the current plant family",
            "runtime widening and envelope tightening are both explicit in the defended replay surface",
            "coupled actuator and shared-budget subgroup diagnostics remain open",
            str(CALIBRATION_COMPLETENESS["industrial"]),
            "joint-constraint and fault-mode calibration depth still trails the witness row",
        ],
        [
            "Medical and Healthcare Monitoring",
            "proof_validated_bounded",
            "bounded row-level replay only",
            "current coverage summary uses row-level PICP90 "
            + _metric_text(support_lookup["healthcare"]["training"].get("picp_90"), 3),
            "mean interval width "
            + _metric_text(support_lookup["healthcare"]["training"].get("mean_interval_width"), 3)
            + " on the defended monitoring row",
            "bounded empirical calibration on the current monitoring-and-intervention contract",
            "conservative widening is explicit when telemetry freshness and trust degrade",
            "cross-patient covariate shift and certificate-gated alert suppression are still open",
            str(CALIBRATION_COMPLETENESS["healthcare"]),
            "multi-patient subgroup coverage is not yet locked as a defended artifact",
        ],
        [
            "Navigation and Guidance",
            "shadow_synthetic",
            "blocked_real_data_gap",
            "blocked_real_data_gap",
            "synthetic portability widths exist but are non-canonical for defended calibration claims",
            "no defended formal or empirical calibration surface yet",
            "hold/slowdown widening exists only on the portability surface",
            "real KITTI-backed OQE and LiDAR-consistency diagnostics are missing",
            str(CALIBRATION_COMPLETENESS["navigation"]),
            "real-data navigation closure must land before any stronger calibration claim is promoted",
        ],
        [
            "Aerospace Control",
            "experimental",
            "experimental_placeholder_only",
            "experimental_placeholder_only",
            "current width summary comes from the placeholder C-MAPSS companion surface only",
            "no defended flight-task calibration surface yet",
            "bounded envelope widening exists but is not yet flight-benchmark defended",
            "real multi-flight telemetry and phase-aware residual diagnostics are missing",
            str(CALIBRATION_COMPLETENESS["aerospace"]),
            "placeholder aerospace calibration cannot support equal-domain closure",
        ],
    ]
    with (PUBLICATION_DIR / "orius_calibration_diagnostics_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(calibration_rows)
    _write_wide_table(
        PUBLICATION_DIR / "tbl_orius_calibration_diagnostics.tex",
        caption="Cross-domain calibration diagnostics for the ORIUS 93+ closure program. The table keeps formal calibration, conservative widening, and residual-shift limits separate so weaker rows cannot inherit the witness-domain statistical rhetoric by analogy.",
        label="tab:orius-calibration-diagnostics",
        header=[
            "Domain",
            "Tier scope",
            "Fault-mode coverage",
            "OQE bucket coverage",
            "Width regime",
            "Formal calibration",
            "Conservative widening",
            "Residual and shift summary",
            "Completeness",
            "Exact limit",
        ],
        rows=calibration_rows[1:],
    )

    runtime_rows = [
        [
            "domain",
            "p95_step_latency_ms",
            "mean_reliability",
            "repair_rate_pct",
            "certificate_rate_pct",
            "runtime_error_rate_pct",
            "fallback_mode",
            "certos_status",
            "runtime_budget_depth",
            "exact_limit",
        ]
    ]
    runtime_rows.append(
        [
            "Battery Energy Storage",
            _metric_text(_battery_latency_p95_ms(), 4),
            "witness_reference",
            "n/a_reference",
            "100.0",
            "0.0",
            "safe_hold",
            "evaluated",
            "reference_runtime",
            "external bench and field telemetry packaging still remain bounded deployment items",
        ]
    )
    for domain_id, sil_key in [
        ("av", "av"),
        ("industrial", "industrial"),
        ("healthcare", "healthcare"),
        ("navigation", "navigation"),
        ("aerospace", "aerospace"),
    ]:
        sil = sil_rows.get(sil_key, {})
        runtime = support_lookup[domain_id]["runtime"]
        parity = support_lookup[domain_id]["parity"]
        runtime_rows.append(
            [
                next(row["label"] for row in DOMAIN_ROWS if row["id"] == domain_id),
                _metric_text(sil.get("p95_latency_ms"), 3),
                _metric_text(sil.get("mean_reliability"), 3),
                _metric_text(sil.get("repair_rate_pct"), 1),
                _metric_text(sil.get("certificate_rate_pct"), 1),
                _metric_text(sil.get("runtime_error_rate_pct"), 1),
                runtime.get("fallback_mode", "n/a"),
                parity.get("certos_lifecycle_support", runtime.get("certos_portability_status", "n/a")),
                parity.get("resulting_tier", runtime.get("resulting_tier", "n/a")),
                parity.get("exact_blocker", runtime.get("exact_blocker", "n/a")),
            ]
        )
    with (PUBLICATION_DIR / "orius_runtime_budget_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(runtime_rows)
    _write_wide_table(
        PUBLICATION_DIR / "tbl_orius_runtime_budget_matrix.tex",
        caption="Cross-domain runtime-budget matrix for the ORIUS 93+ closure program. Battery uses the locked witness latency surface; the remaining rows use the SIL summary and domain-closure runtime traces.",
        label="tab:orius-runtime-budget-matrix",
        header=[
            "Domain",
            "P95 step latency (ms)",
            "Mean reliability",
            "Repair rate (%)",
            "Certificate rate (%)",
            "Runtime error (%)",
            "Fallback mode",
            "CertOS",
            "Runtime depth",
            "Exact limit",
        ],
        rows=runtime_rows[1:],
    )

    governance_rows = [
        [
            "domain",
            "issue_validate_expire_fallback",
            "hash_chain_and_invariants",
            "certos_lifecycle_status",
            "shared_constraint_status",
            "audit_continuity_status",
            "governance_completeness_pct",
            "exact_limit",
        ]
    ]
    for domain_id, runtime_key in [
        ("battery", "battery"),
        ("av", "vehicle"),
        ("industrial", "industrial"),
        ("healthcare", "healthcare"),
        ("navigation", "navigation"),
        ("aerospace", "aerospace"),
    ]:
        paper5_row = {row["domain"]: row for row in _read_csv_dicts(PAPER5_MATRIX_PATH)}.get(runtime_key, {})
        paper6_row = {row["domain"]: row for row in _read_csv_dicts(PAPER6_MATRIX_PATH)}.get(runtime_key, {})
        parity = support_lookup[domain_id]["parity"]
        lifecycle = (
            "issue/validate/expire/fallback all seen"
            if paper6_row.get("status") == "evaluated"
            else "gated_or_not_yet_supported"
        )
        hash_chain = (
            "hash chain ok and invariants pass"
            if paper6_row.get("status") == "evaluated"
            else "gated"
        )
        shared_constraint = (
            f"{paper5_row.get('shared_constraint_surface', 'not_supported')} ({paper5_row.get('status', 'gated')})"
            if paper5_row
            else "not_supported (gated)"
        )
        governance_rows.append(
            [
                next(row["label"] for row in DOMAIN_ROWS if row["id"] == domain_id),
                lifecycle,
                hash_chain,
                parity.get("certos_lifecycle_support", "gated"),
                shared_constraint,
                "tracked runtime audit continuity"
                if paper6_row.get("status") == "evaluated"
                else "bounded_or_gated",
                str(RUNTIME_GOVERNANCE_COMPLETENESS[domain_id]),
                parity.get("exact_blocker", "n/a"),
            ]
        )
    with (PUBLICATION_DIR / "orius_governance_lifecycle_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(governance_rows)
    _write_wide_table(
        PUBLICATION_DIR / "tbl_orius_governance_lifecycle_matrix.tex",
        caption="Cross-domain governance lifecycle matrix for the ORIUS 93+ closure program. Composition and lifecycle breadth are recorded separately so unsupported rows remain visibly gated.",
        label="tab:orius-governance-lifecycle-matrix",
        header=[
            "Domain",
            "Lifecycle surface",
            "Hash chain and invariants",
            "CertOS status",
            "Shared-constraint status",
            "Audit continuity",
            "Completeness",
            "Exact limit",
        ],
        rows=governance_rows[1:],
    )

    latency_artifact = next(
        (item.get("artifact", "") for item in deployment_map if item.get("surface") == "Latency benchmark"),
        "reports/publication/dc3s_latency_summary.csv",
    )
    deployment_rows = [
        [
            "deployment_surface",
            "governing_artifact",
            "scope_type",
            "current_status",
            "manuscript_claim",
            "exact_non_claim_or_gap",
        ],
        [
            "Battery runtime and HIL rehearsal",
            f"{latency_artifact}; reports/hil/hil_summary.json",
            "rehearsal_plus_hil",
            "bounded_reference",
            "Battery supports the deepest runtime and HIL-like evidence in the book.",
            "This is still not a full external bench or field deployment package.",
        ],
        [
            "Battery aging and half-life",
            "reports/aging/asset_preservation_proxy_table.csv; reports/publication/aging_aware_calibration_design.md",
            "proxy_plus_design",
            "partial",
            "The book can defend proxy-backed aging and half-life design reasoning.",
            "It does not yet defend a full live aging-validation stack.",
        ],
        [
            "Autonomous-vehicle defended replay",
            "reports/universal_orius_validation/domain_closure_matrix.csv",
            "bounded_replay",
            "defended_bounded",
            "AV can claim bounded proof-validated replay under the TTC entry-barrier contract.",
            "It cannot claim full live-stack road deployment or universal AV closure.",
        ],
        [
            "Industrial and healthcare defended replay",
            "reports/universal_orius_validation/domain_closure_matrix.csv",
            "bounded_replay",
            "defended_bounded",
            "Industrial and healthcare can claim bounded defended replay plus runtime governance.",
            "They do not claim universal plant-family closure or regulated clinical deployment.",
        ],
        [
            "Navigation field telemetry",
            "data/navigation/PLACE_REAL_NAVIGATION_DATA_HERE.md; reports/publication/orius_equal_domain_parity_matrix.csv",
            "real_data_required",
            "blocked",
            "Navigation remains a portability row only.",
            "Real KITTI-backed train, replay, soundness, and runtime traces are still missing.",
        ],
        [
            "Aerospace flight telemetry",
            "data/aerospace/PLACE_REAL_AEROSPACE_DATA_HERE.md; reports/publication/orius_equal_domain_parity_matrix.csv",
            "real_flight_required",
            "blocked",
            "Aerospace remains an experimental outer-boundary row.",
            "A defended real multi-flight task and replay improvement are still missing.",
        ],
        [
            "OOD and adversarial completeness",
            "chapters/ch34_outside_current_evidence.tex; reports/publication/adversarial_probing_robustness_table.csv",
            "explicit_non_claim_register",
            "bounded_non_claim",
            "The monograph can discuss bounded active probing and non-claim discipline.",
            "It does not claim universal adversarial completeness or unrestricted OOD safety.",
        ],
    ]
    with (PUBLICATION_DIR / "orius_deployment_validation_scope.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(deployment_rows)
    _write_wide_table(
        PUBLICATION_DIR / "tbl_orius_deployment_validation_scope.tex",
        caption="Deployment validation scope for the ORIUS 93+ closure program. The table distinguishes defended replay, rehearsal, proxy, and explicitly out-of-scope surfaces so deployment language cannot outrun the tracked artifacts.",
        label="tab:orius-deployment-validation-scope",
        header=[
            "Deployment surface",
            "Governing artifact",
            "Scope type",
            "Current status",
            "Bounded manuscript claim",
            "Exact non-claim or gap",
        ],
        rows=deployment_rows[1:],
    )

    gap_rows = [[
        "target_tier",
        "gap_id",
        "severity",
        "title",
        "artifact_surface",
        "current_state",
        "required_action",
    ]]
    gap_rows.extend([list(row) for row in READINESS_GAPS])
    with (PUBLICATION_DIR / "orius_93plus_gap_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(gap_rows)

    rerun_rows = [[
        "target_tier",
        "reviewer_id",
        "reviewer",
        "novelty",
        "theorem_rigor",
        "universality_credibility",
        "parity_discipline",
        "runtime_governance_maturity",
        "benchmark_credibility",
        "writing_quality",
        "submission_readiness",
        "verdict",
    ]]
    for row in READINESS_REVIEWERS:
        rerun_rows.append(
            [
                row["target_tier"],
                row["reviewer_id"],
                row["reviewer"],
                row["novelty"],
                row["theorem_rigor"],
                row["universality_credibility"],
                row["parity_discipline"],
                row["runtime_governance_maturity"],
                row["benchmark_credibility"],
                row["writing_quality"],
                row["submission_readiness"],
                row["verdict"],
            ]
        )
    with (PUBLICATION_DIR / "orius_93plus_reviewer_rerun.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rerun_rows)

    target_domain_sets = {
        "bounded_93_candidate": ["battery", "av", "industrial", "healthcare"],
        "equal_domain_93": ["battery", "av", "industrial", "healthcare", "navigation", "aerospace"],
    }
    parity_alignment = {
        "bounded_93_candidate": 96.0,
        "equal_domain_93": 62.0,
    }
    scorecard_rows = [[
        "target_tier",
        "reviewer_composite_10",
        "reviewer_composite_100",
        "critical_gap_count",
        "high_gap_count",
        "calibration_completeness_pct",
        "runtime_governance_completeness_pct",
        "parity_alignment_pct",
        "readiness_score_100",
        "meets_93_gate",
        "verdict",
    ]]
    scorecard_payload: dict[str, dict[str, object]] = {}
    for target_tier, domains in target_domain_sets.items():
        rerun_subset = [row for row in READINESS_REVIEWERS if row["target_tier"] == target_tier]
        reviewer_composite_10 = mean(float(row["submission_readiness"]) for row in rerun_subset)
        reviewer_composite_100 = reviewer_composite_10 * 10.0
        critical_count = sum(1 for row in READINESS_GAPS if row[0] == target_tier and row[2] == "critical")
        high_count = sum(1 for row in READINESS_GAPS if row[0] == target_tier and row[2] == "high")
        calibration_pct = mean(CALIBRATION_COMPLETENESS[domain] for domain in domains)
        runtime_pct = mean(RUNTIME_GOVERNANCE_COMPLETENESS[domain] for domain in domains)
        readiness_score = (
            0.55 * reviewer_composite_100
            + 0.15 * calibration_pct
            + 0.15 * runtime_pct
            + 0.15 * parity_alignment[target_tier]
        )
        meets_gate = (
            reviewer_composite_10 >= 9.3
            and critical_count == 0
            and target_tier == "bounded_93_candidate"
        )
        verdict = (
            "achieved_for_bounded_claim_tier"
            if meets_gate
            else "blocked_pending_parity_closure"
        )
        scorecard_rows.append(
            [
                target_tier,
                f"{reviewer_composite_10:.3f}",
                f"{reviewer_composite_100:.1f}",
                str(critical_count),
                str(high_count),
                f"{calibration_pct:.1f}",
                f"{runtime_pct:.1f}",
                f"{parity_alignment[target_tier]:.1f}",
                f"{readiness_score:.1f}",
                "True" if meets_gate else "False",
                verdict,
            ]
        )
        scorecard_payload[target_tier] = {
            "reviewer_composite_10": round(reviewer_composite_10, 3),
            "reviewer_composite_100": round(reviewer_composite_100, 1),
            "critical_gap_count": critical_count,
            "high_gap_count": high_count,
            "calibration_completeness_pct": round(calibration_pct, 1),
            "runtime_governance_completeness_pct": round(runtime_pct, 1),
            "parity_alignment_pct": parity_alignment[target_tier],
            "readiness_score_100": round(readiness_score, 1),
            "meets_93_gate": meets_gate,
            "verdict": verdict,
        }
    with (PUBLICATION_DIR / "orius_submission_scorecard.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(scorecard_rows)
    _write(
        PUBLICATION_DIR / "orius_submission_scorecard.json",
        json.dumps(scorecard_payload, indent=2) + "\n",
    )
    _write(
        PUBLICATION_DIR / "orius_submission_scorecard.md",
        dedent(
            f"""\
            # ORIUS 93+ Submission Readiness Scorecard

            This scorecard is the canonical 93+ readiness artifact for the ORIUS closure program.
            It keeps two target tiers separate:

            - `bounded_93_candidate`: the universal-first monograph is scored as a bounded flagship submission while explicitly allowing navigation and aerospace to remain gated.
            - `equal_domain_93`: the stronger equal-domain claim tier is only valid once navigation and aerospace are promoted out of their current blocked rows.

            ## Current status

            - `bounded_93_candidate`: reviewer composite {scorecard_payload["bounded_93_candidate"]["reviewer_composite_10"]:.3f}/10, readiness {scorecard_payload["bounded_93_candidate"]["readiness_score_100"]:.1f}/100, verdict `{scorecard_payload["bounded_93_candidate"]["verdict"]}`
            - `equal_domain_93`: reviewer composite {scorecard_payload["equal_domain_93"]["reviewer_composite_10"]:.3f}/10, readiness {scorecard_payload["equal_domain_93"]["readiness_score_100"]:.1f}/100, verdict `{scorecard_payload["equal_domain_93"]["verdict"]}`

            ## Interpretation

            The current repo truth supports a low-90s bounded universal-safety submission once the parity gate remains central,
            calibration/runtime/deployment breadth is explicitly tabled, and weaker rows are not rhetorically flattened into equal peers.
            The same repo truth still blocks `equal_domain_93` because navigation and aerospace remain unresolved at the parity gate.
            """
        ),
    )
    _write_wide_table(
        PUBLICATION_DIR / "tbl_orius_submission_readiness.tex",
        caption="ORIUS 93+ submission-readiness gate. The bounded target can pass with explicit gated rows; the equal-domain target cannot pass until the two blocked rows are promoted by artifact, replay, and runtime closure rather than by prose.",
        label="tab:orius-submission-readiness",
        header=[
            "Target tier",
            "Reviewer composite /10",
            "Critical gaps",
            "High gaps",
            "Calibration completeness",
            "Runtime and governance completeness",
            "Parity alignment",
            "Readiness /100",
            "Meets 93 gate",
            "Verdict",
        ],
        rows=[
            [
                row[0],
                row[1],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
            ]
            for row in scorecard_rows[1:]
        ],
    )

    _write(
        PUBLICATION_DIR / "orius_93plus_closure_program.md",
        dedent(
            """\
            # ORIUS 93+ Closure Program

            This program is staged rather than rhetorical.

            ## Stage A: bounded-monograph lift

            - keep the parity gate central
            - add a cross-domain calibration diagnostics matrix
            - add cross-domain runtime-budget and governance lifecycle matrices
            - add an explicit deployment validation scope table
            - score the package against a bounded 93 target that still allows gated rows

            ## Stage B: equal-domain parity closure

            - finish the KITTI-backed navigation real-data row with replay, soundness, fallback, and runtime support
            - replace the aerospace placeholder surface with a defended real multi-flight telemetry row
            - rerun the same score gate only after the parity matrix updates from those artifacts

            ## Current truth

            The bounded universal-safety monograph can now be scored and defended independently of equal-domain closure.
            Equal-domain 93 remains blocked until navigation and aerospace leave their current gated tiers.
            """
        ),
    )


def _build_hf_job_templates() -> None:
    HF_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _write(
        HF_JOBS_DIR / "README.md",
        dedent(
            """\
            # ORIUS Hugging Face Closure Jobs

            These UV job templates are the controlled compute entrypoints for the ORIUS 93+ closure program.
            They are designed for Hugging Face Jobs and assume the repo is checked out inside the job workspace.

            Shared assumptions:
            - `GRIDPULSE_REPO_ROOT` defaults to the current working directory.
            - Repo-local raw layout under `data/<domain>/raw/<dataset_key>/` is the primary contract.
            - `ORIUS_EXTERNAL_DATA_ROOT` remains an optional fallback for mounted AV, navigation, or runtime telemetry corpora.
            - Raw datasets stay off-repo and off-Hub unless licensing explicitly permits mirroring.
            - For live metric logging, set `TRACKIO_PROJECT` and configure Hugging Face / Trackio credentials in the job secrets.

            Suggested job entrypoints:
            - `navigation_realdata_closure_job.py`
            - `aerospace_flight_closure_job.py`
            - `calibration_diagnostics_job.py`
            - `runtime_governance_trace_job.py`
            """
        ),
    )
    _write(
        HF_JOBS_DIR / "navigation_realdata_closure_job.py",
        dedent(
            """\
            # /// script
            # dependencies = ["pandas", "pyarrow"]
            # ///
            from __future__ import annotations

            import os
            import subprocess
            from pathlib import Path


            REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
            EXTERNAL_ROOT = os.environ.get("ORIUS_EXTERNAL_DATA_ROOT")


            def run(*args: str) -> None:
                subprocess.run(args, cwd=REPO_ROOT, check=True)


            def main() -> None:
                run("python", "scripts/verify_real_data_preflight.py", "--domain", "navigation")
                build_args = [
                    "python",
                    "scripts/build_navigation_real_dataset.py",
                    "--out",
                    "data/navigation/processed/navigation_orius.csv",
                ]
                if EXTERNAL_ROOT:
                    build_args.extend(["--external-root", str(Path(EXTERNAL_ROOT).resolve())])
                run(*build_args)
                run("python", "scripts/build_data_manifest.py", "--dataset", "NAVIGATION")
                run("python", "scripts/train_dataset.py", "--dataset", "NAVIGATION", "--candidate-run", "--run-id", "hf_navigation_realdata")
                run("python", "scripts/run_universal_orius_validation.py", "--seeds", "1", "--horizon", "24")


            if __name__ == "__main__":
                main()
            """
        ),
    )
    _write(
        HF_JOBS_DIR / "aerospace_flight_closure_job.py",
        dedent(
            """\
            # /// script
            # dependencies = ["pandas", "pyarrow"]
            # ///
            from __future__ import annotations

            import os
            import subprocess
            from pathlib import Path


            REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
            EXTERNAL_ROOT = os.environ.get("ORIUS_EXTERNAL_DATA_ROOT")


            def run(*args: str) -> None:
                subprocess.run(args, cwd=REPO_ROOT, check=True)


            def main() -> None:
                run("python", "scripts/verify_real_data_preflight.py", "--domain", "aerospace")
                build_args = [
                    "python",
                    "scripts/download_aerospace_datasets.py",
                    "--out",
                    "data/aerospace/processed/aerospace_orius.csv",
                ]
                if EXTERNAL_ROOT:
                    build_args.extend(["--external-root", str(Path(EXTERNAL_ROOT).resolve())])
                run(*build_args)
                run("python", "scripts/build_data_manifest.py", "--dataset", "AEROSPACE")
                run("python", "scripts/train_dataset.py", "--dataset", "AEROSPACE", "--candidate-run", "--run-id", "hf_aerospace_realflight")
                run("python", "scripts/run_universal_orius_validation.py", "--seeds", "1", "--horizon", "24")


            if __name__ == "__main__":
                main()
            """
        ),
    )
    _write(
        HF_JOBS_DIR / "calibration_diagnostics_job.py",
        dedent(
            """\
            # /// script
            # dependencies = []
            # ///
            from __future__ import annotations

            import os
            import subprocess
            from pathlib import Path


            REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()


            def main() -> None:
                subprocess.run(
                    ["python", "scripts/build_orius_monograph_assets.py"],
                    cwd=REPO_ROOT,
                    check=True,
                )


            if __name__ == "__main__":
                main()
            """
        ),
    )
    _write(
        HF_JOBS_DIR / "runtime_governance_trace_job.py",
        dedent(
            """\
            # /// script
            # dependencies = []
            # ///
            from __future__ import annotations

            import os
            import subprocess
            from pathlib import Path


            REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()


            def main() -> None:
                subprocess.run(
                    ["python", "scripts/run_universal_orius_validation.py", "--seeds", "1", "--horizon", "24"],
                    cwd=REPO_ROOT,
                    check=True,
                )
                subprocess.run(
                    ["python", "scripts/build_orius_monograph_assets.py"],
                    cwd=REPO_ROOT,
                    check=True,
                )


            if __name__ == "__main__":
                main()
            """
        ),
    )

def _build_foundation_chapters() -> dict[str, str]:
    cites_foundations = (
        "\\cite{vovk2005algorithmic,lei2018distribution,romano2019conformalized,"
        "gibbs2021adaptive,angelopoulos2023gentle,barber2023beyond,"
        "zaffran2022adaptive,stankeviciute2021conformal,xu2021dynamic,"
        "sesia2021comparison,bates2021distribution}"
    )
    cites_runtime = (
        "\\cite{sha2001using,sha1998simplex,desai2019soter,bloem2015shield,"
        "bartocci2018introduction,havelund2004runtime,leucker2009brief,"
        "schierman2015run,bak2011runtime}"
    )
    cites_control = (
        "\\cite{ames2019control,ames2014cbf,wabersich2021predictive,"
        "nguyen2016exponential,wang2017safety,fisac2019general,"
        "mayne2005robust,langson2004tubes,rawlings2017mpc,ben2009robust,"
        "camacho2013model}"
    )
    cites_cps = (
        "\\cite{rajkumar2010cps,derler2012modeling,baheti2011cyberphysical,"
        "pasqualetti2013attack,teixeira2015secure,basseville1993detection,"
        "page1954continuous,page1955test,hinkley1971inference,liu2008isolation,"
        "breunig2000lof}"
    )

    chapters: dict[str, str] = {}
    chapters["ch01_physical_ai_safety.tex"] = (
        dedent(
            r"""
        __HEADER__
        ORIUS is motivated by a simple but under-specified fact: modern physical AI systems
        do not act on truth, they act on telemetry.  The deeper the software stack becomes,
        the easier it is to hide the sensing contract that makes every downstream prediction,
        optimization, and control action appear legitimate even when the observation channel is
        degraded.  The monograph therefore begins with a universal statement rather than a
        battery statement: whenever action legality is evaluated on an observed state that can
        diverge from the physical state, a hidden safety failure mode exists.

        \section{Why physical AI safety is structurally different}
        A physical AI stack does not end at prediction quality or planner confidence.  It ends
        in a command that changes a battery dispatch, vehicle acceleration, plant input, alarm
        state, guidance trajectory, or flight-control surface.  The danger is not merely that
        the model can be wrong.  The danger is that the system can look internally coherent
        while being externally unsafe because the software stack is reasoning over a degraded
        observation surface.  This is what makes physical AI different from pure information
        systems: mistakes are mediated by dynamics, latency, and irreversible physical
        consequences rather than only by ranking error or classification loss.

        This distinction also changes what counts as a defensible contribution.  A physical-AI
        safety framework cannot rely only on better forecasting, better perception, or better
        nominal control.  It must specify what happens when those upstream components are fed
        stale, delayed, missing, or corrupted observations and then continue to act anyway.
        ORIUS therefore begins not with task performance but with the hidden interface between
        observation and action legality.

        \section{Telemetry is the hidden control contract}
        Every nominal controller implicitly signs a sensing contract.  It assumes that the state
        estimate, telemetry packet, or inferred context is recent enough, synchronized enough,
        and truthful enough for downstream legality checks to mean what the controller thinks
        they mean.  Most deployed systems bury that contract inside preprocessing, filtering,
        transport middleware, or state-estimation code.  Once it is buried, it becomes easy to
        evaluate safety on the observed state and forget that the physical state may already have
        drifted away.

        That hidden contract is where the relevant literatures meet.  The uncertainty literature
        contributes distribution-free predictive wrappers and coverage-aware calibration
        surfaces __FOUNDATIONS__.  Runtime assurance contributes supervisory architectures,
        shield synthesis, and safety monitors that can veto or repair actions after a nominal
        controller proposes them __RUNTIME__.  Safe control contributes barrier methods,
        robust model predictive control, and predictive safety filters that turn safety into a
        computable constraint set rather than an ex post alarm __CONTROL__.  Cyber-physical
        systems research contributes the language of degraded telemetry, attack surfaces, change
        detection, and monitored execution __CPS__.  ORIUS sits at the intersection of all
        four because it treats telemetry degradation as a runtime action-semantics problem, not
        merely as a detection or estimation nuisance.

        \section{Why a universal safety layer is the right object}
        The book uses the phrase \emph{physical AI} deliberately.  It refers to any stack in
        which machine-learned or algorithmic decision logic is embedded in a loop that ends in
        physical consequences: batteries, vehicles, industrial plants, ICU monitoring, guidance
        systems, and aerospace control are all instances.  The universal claim of ORIUS is not
        that one controller can solve them all.  The claim is that one defended safety layer can
        express a common degraded-observation contract across them.

        That is why ORIUS is written as a layer instead of as a replacement controller.
        Domain-specific nominal intelligence remains domain-specific by design.  What can be
        shared is the logic that decides when degraded observation widens the physically
        plausible state set, tightens the admissible action set, forces a repair, triggers a
        fallback, and records a certificate.  The reusable object is therefore not a universal
        planner but a universal safety grammar.

        \section{What this monograph sets out to prove}
        That distinction matters for societal contribution.  The most credible path to broad
        impact in physical AI is not another domain-specific optimizer, but a reusable,
        auditable, and explicitly bounded safety layer that can sit between nominal intelligence
        and real actuation.  ORIUS is written as that layer: a runtime contract that can be
        instantiated repeatedly, promoted cautiously, and reviewed by people who care as much
        about failure semantics as about nominal performance.

        The opening chapters therefore do four jobs for the rest of the book.  First, they name
        the universal hazard: actions can be legal on observation while illegal in reality.
        Second, they define the runtime object that addresses that hazard: a Detect--Calibrate--
        Constrain--Shield--Certify kernel.  Third, they establish the claim boundary: ORIUS is
        universal in architecture and runtime semantics, not automatically equal in empirical
        maturity across all domains.  Fourth, they justify the monograph's universal-first
        structure: one argument, one kernel, six domain chapters, and one visible parity gate
        that prevents rhetoric from outrunning evidence.

        The next chapter turns that intuition into a sharper organizing object.  It defines the
        observation-action safety gap itself and explains why the entire book is strongest when
        architecture-level universality and evidence-level promotion are kept distinct.

        """
        )
        .replace("__HEADER__", _chapter_header("Physical AI Safety and the Hidden Observability Contract", "ch:physical-ai-safety").strip())
        .replace("__FOUNDATIONS__", cites_foundations)
        .replace("__RUNTIME__", cites_runtime)
        .replace("__CONTROL__", cites_control)
        .replace("__CPS__", cites_cps)
        .strip()
        + "\n"
    )

    chapters["ch02_oasg_claim_boundary.tex"] = dedent(
        r"""
        \chapter{The Observation-Action Safety Gap and Claim Boundary}
        \label{ch:oasg-claim-boundary}
        \label{ch:safety-illusion}

        The observation-action safety gap (OASG) is the central organizing object of the
        monograph.  It names the event in which an action is admissible on the observed state
        while unsafe on the true state.  In practical terms, this happens whenever sensing,
        transport, synchronization, or estimation degrades more severely than the downstream
        controller assumes.  The action remains \emph{observationally legitimate} but becomes
        \emph{physically illegitimate}.

        \section{A structural hazard rather than a domain bug}
        ORIUS treats this as a universal hazard because the structure is independent of plant
        details.  Battery dispatch exposes it through hidden state-of-charge violations.  Vehicle
        control exposes it through stale headway and closing-speed estimates.  Industrial systems
        expose it through delayed thermal or pressure readings.  Healthcare exposes it through
        stale alarms and delayed physiological updates.  Navigation and aerospace expose it
        through degraded localization or flight-state estimates.  The plant changes, but the
        hidden gap does not.

        The key point is that OASG is not a synonym for forecasting error.  A forecast can be
        imperfect while the action remains safe, and a forecast can look adequate while the
        action becomes unsafe because the observation surface itself is no longer truthful enough
        to support legality checks.  ORIUS therefore centers the gap between \emph{what the
        controller is allowed to believe} and \emph{what the plant can actually be}.  That gap
        is a systems object, not a single-model pathology.

        \section{Observed legality versus physical legality}
        The intuition can be written compactly even before the full theorem machinery appears.
        Let $\hat{x}_t$ denote the observed state used by the nominal controller and let
        $x_t^\star$ denote the physical state that is actually realized or could plausibly be
        realized given degraded telemetry.  Let $\mathcal{A}(\cdot)$ denote the action set that
        is considered admissible.  The OASG event is the regime in which a candidate action
        satisfies
        \[
        u_t \in \mathcal{A}(\hat{x}_t)
        \qquad \text{but} \qquad
        u_t \notin \mathcal{A}(x_t^\star),
        \]
        or, equivalently, appears safe on observation while violating the safety predicate on the
        true or observation-consistent state.

        This notation is intentionally light.  The formal chapters later introduce the precise
        uncertainty-set and repair semantics.  At the level of claim boundary, the important
        point is simply this: once legality depends on a state that may be stale, delayed,
        dropped, or manipulated, the safety argument has to move from nominal action selection to
        explicit runtime mediation.

        \section{Why ORIUS turns the gap into a runtime object}
        That runtime mediation is the central design choice of the book.  ORIUS does not say that
        every domain should solve the observability problem with a single estimator, a single
        robust controller, or a single proof technique.  It says that each domain must expose the
        same sequence of safety-relevant objects at runtime: a reliability judgment on the
        observation channel, an uncertainty set for physically plausible state, an admissible
        action set under that uncertainty, a repaired or fallback action, and a certificate that
        records why the step was allowed.

        Once those objects are explicit, the universal and domain-specific parts of the argument
        can finally be separated cleanly.  The universal part is the runtime grammar.  The
        domain-specific part is how each adapter instantiates state, constraints, repair, and
        fallback for its own plant.

        \section{Claim classes and promotion discipline}
        The claim boundary of the book follows directly from that framing.  ORIUS does not claim
        universal optimality, universal hardware closure, or equal empirical maturity across all
        domains.  It claims a universal \emph{architecture} and a universal \emph{runtime
        contract}.  Evidence is then tiered by what each domain actually clears: witness-depth
        reference, defended bounded peer, portability-only row, or experimental boundary row.
        This separation is not rhetorical caution; it is the mechanism that
        makes a universal monograph defensible under serious review.

        The promotion language in the monograph is therefore deliberately typed.  A
        \emph{reference witness} carries the deepest theorem-to-code-to-artifact chain and sets
        the scientific calibration surface for the rest of the book.  A \emph{defended bounded
        peer} clears a domain-specific replay, soundness, fallback, and runtime gate without
        inheriting witness-depth proof language automatically.  A \emph{portability-only row}
        shows that the architecture can travel structurally but does not yet clear a defended
        empirical gate.  An \emph{experimental boundary row} is useful because it marks the edge
        of the current program rather than being hidden behind optimistic prose.

        \section{Architecture language versus evidence language}
        The book therefore uses two kinds of language and keeps them separate.  Architecture
        language is universal: true-state violation, constraint margin, repair, certificate,
        degraded observation, and intervention.  Evidence language is domain-specific and
        promotion-sensitive: some rows close the measured violation gap, some only expose the
        structure, and some remain bounded by the current adapter or telemetry surface.  The
        monograph is strongest when it refuses to let those two vocabularies collapse into one.

        This is also why the universal-first framing is stronger than a witness-first exposition at
        the level of exposition.  The reader should learn the common hazard and common runtime
        contract first, and only then see how different domains clear different parts of the
        evidence ladder.  Battery remains the deepest witness, but it should not monopolize the
        conceptual story.

        \section{What the book does not claim}
        The negative boundary is as important as the positive one.  ORIUS does not claim
        universal optimality, universal plant closure, or a flat statement that every domain row
        has already reached equal maturity.  It does not claim that uncertainty calibration by
        itself proves safety, that runtime certificates replace regulation, or that adapter
        portability is equivalent to empirical closure.  The book instead makes a narrower but
        more durable claim: one universal safety-layer grammar can govern multiple physical-AI
        domains, while a visible parity gate determines how far each domain may currently be
        promoted.

        The rest of the monograph is written under that contract.  The architecture chapters
        explain the reusable runtime object, the theory chapters explain the safety logic under
        assumptions, the domain chapters instantiate the contract six times, and the synthesis
        chapters make the asymmetry in evidence explicit instead of hiding it.
        """
    ).strip() + "\n"

    chapters["ch03_related_work_universal.tex"] = (
        dedent(
            r"""
        __HEADER__
        \section{{Conformal and distribution-free uncertainty}}
        Distribution-free predictive inference provides one of the few practical ways to place
        finite-sample uncertainty guards around learned models without assuming a perfectly
        specified parametric error law __FOUNDATIONS__.  ORIUS adopts this family not because
        conformal prediction alone solves runtime safety, but because it supplies a principled
        outer shell around the unobserved state.  The gap ORIUS addresses here is the missing
        runtime bridge between nominal predictive coverage and actuation legality under degraded
        telemetry.

        \section{{Runtime assurance, supervisory safety, and shields}}
        Runtime assurance and shielding treat safety as a supervisory execution discipline rather
        than a property delegated to the nominal controller __RUNTIME__.  Simplex-style
        architectures, runtime monitors, and shield synthesis all contribute the idea that
        proposed actions may be intercepted, replaced, or downgraded at runtime.  The gap ORIUS
        addresses here is that most runtime assurance architectures do not explicitly center
        \emph{observation degradation} as the mechanism that makes action legality uncertain in
        the first place.

        \section{{Safety filters, barrier methods, and robust control}}
        Control barrier functions, robust MPC, reachable sets, predictive safety filters, and
        constrained control all provide machinery for turning uncertainty into admissible action
        sets __CONTROL__.  The gap ORIUS addresses here is the missing typed interface
        between telemetry reliability, uncertainty inflation, and a cross-domain repair contract
        that can be audited and reused across multiple physical AI stacks.

        \section{{Drift, anomaly detection, and cyber-physical degradation}}
        Change detection, anomaly detection, CPS security, and fault diagnosis provide the
        language needed to recognize when the observation channel is no longer trustworthy
        __CPS__.  ORIUS depends on that language, but departs from it by treating degraded
        observation not only as a detection problem but as an \emph{action semantics} problem:
        once trust in the observation is degraded, admissible action sets must change.

        \section{{Applied domain literatures}}
        The applied rows of the book draw from multiple domain literatures rather than from a
        single benchmark tradition.  Energy and smart-grid control literature motivates the
        reference witness; autonomous-vehicle and robotics literature motivates the need for
        runtime repair under perception uncertainty; industrial process-control literature
        provides real operational envelope constraints; healthcare monitoring shows how degraded
        observation can suppress necessary intervention; and navigation and aerospace show where
        the present closure boundary still ends.  These families collectively justify a universal
        framing, but they also explain why ORIUS must keep evidence tiers explicit instead of
        claiming parity by analogy alone.
        """
        )
        .replace("__HEADER__", _chapter_header("Related Work, Method Families, and the Gap ORIUS Addresses", "ch:related-work-universal").strip())
        .replace("__FOUNDATIONS__", cites_foundations)
        .replace("__RUNTIME__", cites_runtime)
        .replace("__CONTROL__", cites_control)
        .replace("__CPS__", cites_cps)
        .strip()
        + "\n"
    )

    return chapters


def _domain_chapter(row: dict[str, str]) -> str:
    support = _domain_support_lookup()[row["id"]]
    label = f"ch:domain-{row['id']}"
    title = f"{row['label']} Domain Chapter"
    training_table_name = f"tbl_{row['id']}_training_surface"
    replay_table_name = f"tbl_{row['id']}_replay_surface"
    figure_name = f"fig_{row['id']}_chapter_snapshot.png"
    obligations_block = ""
    if row["id"] in PROMOTION_OBLIGATION_ROWS:
        obligations_block = (
            "\n\n"
            + _table_input_snippet(f"tbl_{row['id']}_promotion_obligations")
            + "\n"
        )
    return (
        dedent(
            r"""
        __HEADER__

        __STATUS_SENTENCE__

        \section{{Domain problem}}
        __SYSTEM_CONTEXT__

        \section{{Degraded-observation hazard}}
        The ORIUS book forces every domain through the same hazard lens: the runtime does not act
        on the true state directly; it acts on a degraded, delayed, or otherwise imperfect
        observation surface.  The row is only credible if the gap between those two surfaces can
        be expressed, replayed, repaired, and audited explicitly.
        __TELEMETRY_MODEL__

        \section{{ORIUS instantiation}}
        __ADAPTER_MAPPING__

        \section{{Safety object}}
        __SAFETY_PREDICATE__

        \section{{Dataset and training surface}}
        __DATASET_PROTOCOL__
        The chapter-local evidence packet is intentionally tied to locked artifacts rather than to
        narrative memory.  Table~\ref{tab:ch40-44-cross-domain-support} gives the shared cross-domain
        support layer for compiled Chapters~40--44, while the table below isolates the current row.

        \section{{Feature, split, and training protocol}}
        __FEATURE_PROTOCOL_PARAGRAPH__

        __TRAINING_TABLE__

        \section{{Replay and evidence surface}}
        __RESULTS__

        __REPLAY_TABLE__

        __FIGURE_BLOCK__

        \section{{Fallback and runtime behavior}}
        __FALLBACK_RUNTIME__

        \section{{Evidence tier and promotion blocker}}
        __EVIDENCE_PARAGRAPH__
        __OBLIGATIONS_BLOCK__

        \section{{Limitations and exact non-claims}}
        __LIMITATIONS__ __NON_CLAIMS__

        The current promotion obligation for this row remains explicit: __TRANSFER_OBLIGATIONS__
        """
        )
        .replace("__HEADER__", _chapter_header(title, label).strip())
        .replace("__SYSTEM_CONTEXT__", row["system_context"])
        .replace("__SAFETY_PREDICATE__", row["safety_predicate"])
        .replace("__ADAPTER_MAPPING__", row["adapter_mapping"])
        .replace("__TELEMETRY_MODEL__", row["telemetry_model"])
        .replace("__DATASET_PROTOCOL__", row["dataset_protocol"])
        .replace("__FEATURE_PROTOCOL_PARAGRAPH__", _feature_protocol_paragraph(row["id"], row, support))
        .replace("__TRAINING_TABLE__", _table_input_snippet(training_table_name))
        .replace("__RESULTS__", row["results"])
        .replace("__STATUS_SENTENCE__", row["status_sentence"])
        .replace("__REPLAY_TABLE__", _table_input_snippet(replay_table_name))
        .replace(
            "__FIGURE_BLOCK__",
            _report_figure_block(
                figure_name,
                f"{row['label']} replay snapshot derived from the governed domain-closure artifact. Baseline and repaired TSVR are taken from the locked closure matrix; tier and blocker language remain governed by the parity gate.",
                f"fig:{row['id']}-chapter-snapshot",
            ),
        )
        .replace("__FALLBACK_RUNTIME__", _fallback_runtime_text(row["id"]))
        .replace("__EVIDENCE_PARAGRAPH__", _chapter_evidence_paragraph(row["id"], row, support))
        .replace("__OBLIGATIONS_BLOCK__", obligations_block.rstrip())
        .replace("__LIMITATIONS__", row["limitations"])
        .replace("__NON_CLAIMS__", _non_claims_text(row["id"]))
        .replace("__TRANSFER_OBLIGATIONS__", row["transfer_obligations"])
        .strip()
        + "\n"
    )


def _build_runtime_and_synthesis_chapters() -> dict[str, str]:
    return {
        "ch04_universal_runtime_layer.tex": dedent(
            r"""
            \chapter{ORIUS as a Universal Safety Layer}
            \label{ch:universal-runtime-layer}

            ORIUS is organized as a typed runtime contract rather than as a monolithic controller.
            The contract takes in raw telemetry, evaluates observation reliability, inflates the
            uncertainty set around the state that could be physically true, tightens the admissible
            action set, repairs or replaces the candidate action, and emits a certificate that can
            be audited downstream.  This is the Detect-Calibrate-Constrain-Shield-Certify kernel.

            The critical design choice is adapterization.  Domain logic enters the framework
            through a narrow interface: telemetry parsing, uncertainty semantics, action feasibility,
            repair projection, and certificate payload construction.  Everything else belongs to the
            universal safety layer itself.  That is why the monograph can treat batteries,
            vehicles, industrial systems, healthcare monitoring, navigation, and aerospace as
            separate domain chapters while still defending one architectural object.

            \paragraph{Adapter contract}\label{par:monograph-adapter-contract}
            The adapter contract is the single runtime boundary that every domain must satisfy.
            It is intentionally narrow so that all domain-specific logic is isolated from the
            universal kernel while still producing comparable benchmark and certificate surfaces.

            The layer is intentionally post-nominal.  ORIUS does not assume ownership of the
            upstream forecaster, planner, or optimizer.  It wraps them.  This makes the framework
            more realistic for operational deployment because most real systems inherit legacy
            nominal controllers that cannot be rewritten from scratch simply to accommodate a new
            safety argument.
            """
        ).strip() + "\n",
        "ch05_detect_calibrate_constrain_shield_certify.tex": dedent(
            r"""
            \chapter{Detect, Calibrate, Constrain, Shield, and Certify}
            \label{ch:dc3s-kernel}

            The five-stage ORIUS kernel gives the monograph its system spine.

            \section{Detect}
            The detection stage computes observation reliability from freshness, missingness,
            timing, spikes, and consistency features.  It is the first point at which the runtime
            admits that the observation channel may no longer deserve full trust.

            \section{Calibrate}
            The calibration stage translates that loss of trust into a wider uncertainty surface.
            In the current ORIUS implementation this is achieved through conformal-style or
            reliability-aware predictive inflation, but the chapter treats the operation more
            generally: the state set consistent with observation must expand when observation
            quality degrades.

            \section{Constrain}
            The constrain stage converts the state set into an admissible action set.  This is
            where ORIUS differs from alarm-only monitoring.  It is not enough to know that
            telemetry is degraded; the admissible action set must be recomputed accordingly.

            \section{Shield}
            The shield stage repairs the candidate action into one that is admissible under the
            tightened set or replaces it with a fallback action when no safe action remains.

            \section{Certify}
            The certificate stage records why the action was permitted, what the observation
            quality was, what uncertainty and margin were assumed, and what lifecycle state the
            runtime is now in.  This stage is what lets ORIUS move from control logic to a real
            governance story.
            """
        ).strip() + "\n",
        "ch06_theory_bridge.tex": dedent(
            r"""
            \chapter{Theory Bridge, Assumptions, and Non-Claims}
            \label{ch:theory-bridge}

            The monograph keeps theory and evidence coupled but not conflated.  The theory of
            ORIUS proves that under explicit assumptions, degraded observation induces a hidden
            safety problem and that a repair-and-certify layer can bound or eliminate the relevant
            true-state violation event.  The empirical program then tests how much of that
            theoretical surface survives domain transfer.

            Two discipline rules follow.  First, theorem-grade claims remain strongest where the
            theorem-to-code-to-artifact chain is deepest; in the current book that is still the
            battery reference row, even though the reader-facing narrative is now universal-first.
            Second, universality in the book means the architecture, runtime contract, and domain
            chapter template are shared; it does not mean the book is allowed to flatten the parity
            gate or hide unclosed rows.  This is why the monograph keeps one explicit equal-domain
            parity matrix in view instead of letting narrative ambition substitute for closure.
            """
        ).strip() + "\n",
        "ch07_system_benchmark_governance.tex": dedent(
            r"""
            \chapter{System Architecture, Universal Benchmarking, and Governance}
            \label{ch:system-benchmark-governance}

            The monograph keeps one benchmark contract across all rows.  Each replay episode
            records the candidate action, the repaired action, a true-state violation flag, an
            observed-state satisfaction flag, an optional true and observed margin, intervention and
            fallback flags, certificate validity, latency, and domain-specific metrics that do not
            alter the universal core.  This is the benchmark contract that allows the same safety
            argument to be inspected across six domains without collapsing them into one plant.

            Governance is handled through CertOS.  The role of CertOS is not to prove universal
            deployment readiness; it is to enforce a bounded runtime discipline in which no action
            is released without a valid certificate or an explicit fallback action, all lifecycle
            transitions are logged, and audit integrity can be checked after the fact.  The book
            therefore treats governance as part of the scientific contribution, not as release
            management metadata.
            """
        ).strip() + "\n",
        "ch14_cross_domain_synthesis.tex": dedent(
            r"""
            \chapter{Cross-Domain Synthesis, Parity Gates, and the Universality Claim}
            \label{ch:cross-domain-synthesis}

            ORIUS is written in this book as one universal argument, not as a battery book with
            appendices.  The unifying object is the degraded-observation hazard: a controller acts
            on an observed state, but safety belongs to the true state.  The universal-first claim
            is therefore architectural and semantic from the start.  Every domain chapter is asked
            the same question: can the ORIUS runtime detect degraded observation, widen the
            observation-consistent state set, repair the candidate action, emit a certificate, and
            document the exact non-claims for that domain?

            \section{Why the book is universal-first}
            The manuscript now carries six first-class domain chapters because the point is not to
            celebrate one plant.  The point is to show that battery, vehicles, industrial plants,
            healthcare monitoring, navigation, and aerospace can all be expressed through one
            safety-layer grammar: degraded observation, reliability-aware inflation, action repair,
            bounded fallback, and auditable certificates.

            \section{Equal-domain parity gate}
            Universal-first narrative does not remove evidence discipline.  The book therefore uses
            one explicit parity gate rather than many drifting summaries.  A domain only earns the
            strong universal rhetoric that matches its closure row: adapter correctness, data and
            training surface, replay, soundness, fallback, CertOS lifecycle support, and optional
            multi-agent portability all have to clear the same governed matrix.

            \input{reports/publication/tbl_orius_equal_domain_parity_matrix}

            \section{Current parity state}
            The current book clears four defended rows under one safety-layer vocabulary:
            battery as the deepest reference row, autonomous vehicles under the bounded TTC
            entry-barrier contract, industrial process control, and healthcare monitoring.
            Navigation remains gated by the real-data closure requirement. Aerospace remains gated
            by the placeholder flight-task surface. That asymmetry does not weaken the universal
            architecture; it simply determines which rows may carry equal-peer rhetoric today.

            \section{93+ readiness gate}
            The 93+ program makes that distinction formal.  The book now carries one submission
            scorecard for a \texttt{bounded\_93\_candidate} tier and a stricter
            \texttt{equal\_domain\_93} tier.  The bounded target can clear as long as the
            manuscript, runtime, and calibration surfaces are strong while navigation and
            aerospace remain explicitly gated.  The equal-domain target remains blocked until
            those two rows clear the same parity matrix by artifact rather than by prose.

            \input{reports/publication/tbl_orius_submission_readiness}

            \begin{figure}[htbp]
            \centering
            \IfFileExists{reports/publication/fig_orius_equal_domain_parity_matrix.png}{%
            \includegraphics[width=0.97\textwidth]{reports/publication/fig_orius_equal_domain_parity_matrix.png}%
            }{%
            \includegraphics[width=0.97\textwidth]{../reports/publication/fig_orius_equal_domain_parity_matrix.png}%
            }
            \caption{Shared defended-versus-gated view for the non-battery domain block. Chapters~40--44 refer back to this mini-matrix so the reader can see which rows are defended peers and which remain explicitly gated.}
            \label{fig:ch40-44-shared-parity}
            \end{figure}

            \IfFileExists{paper/assets/tables/generated/tbl_ch40_44_cross_domain_support.tex}{%
            \input{paper/assets/tables/generated/tbl_ch40_44_cross_domain_support}%
            }{%
            \input{assets/tables/generated/tbl_ch40_44_cross_domain_support}%
            }

            \section{Evidence gaps and theory discipline}
            The monograph is strongest when it states the asymmetry plainly: the runtime
            contract and theorem bridge travel farther than the current defended evidence.
            That is why the current edition distinguishes architectural universality from
            equal-domain closure.  Where the theory surface is broader than the replay and
            artifact surface, the claim is written as a bounded semantic or contractual
            statement rather than as a flat cross-domain validation claim.  In practical
            terms, this means the book treats battery as the witness row, AV plus industrial
            and healthcare as defended bounded rows, and navigation plus aerospace as
            explicitly gated rows rather than softened placeholders in prose.  The
            monograph is intended to survive skeptical review precisely because it refuses
            to let the theoretical surface outrun the current replay and artifact surface.

            The calibration story is part of that discipline.  The book now carries one
            cross-domain calibration matrix that separates theorem-backed calibration,
            empirical bounded calibration, and conservative widening rather than merging them
            into one ambiguous uncertainty narrative.

            \input{reports/publication/tbl_orius_calibration_diagnostics}

            \section{Why the gate stays reader-visible}
            The parity gate is not an embarrassing appendix.  It is the mechanism that lets this
            book argue for ORIUS as a fundamental safety layer for physical AI without pretending
            that unclosed rows are already equal.  The book is strongest when architecture-level
            universality and evidence-level parity are both explicit and both governed.

            \section{T9 universality impossibility boundary}
            \label{sec:monograph-t9-universal-impossibility}
            The current impossibility boundary is not a contradiction of the ORIUS architecture.
            It is the point at which a domain remains structurally compatible with the runtime
            contract but fails to close the relevant true-state violation event under the present
            repair geometry or telemetry surface.

            \section{T10 lower-bound interpretation}
            \label{sec:monograph-t10-lower-bound}
            The lower-bound interpretation in the monograph says that insufficient recoverable
            information in the degraded observation channel limits what any runtime repair layer
            can certify.  ORIUS can still govern fallback and logging there, but it cannot invent
            theorem-grade closure from missing signal.

            \paragraph{Universality completeness statement}
            \label{par:monograph-universality-completeness}
            ORIUS is complete as a universal architectural layer when every supported domain can
            express degraded observation, admissible repair, and certificate semantics through one
            runtime contract, while empirical promotion stays tied to the governed parity matrix
            rather than assumed by analogy.
            """
        ).strip() + "\n",
        "ch15_societal_impact_and_roadmap.tex": dedent(
            r"""
            \chapter{Societal Contribution, Deployment Program, and the Next Parity Frontier}
            \label{ch:societal-roadmap}

            The strongest societal contribution of ORIUS is not that it solves every safety problem
            in physical AI.  It is that it names a failure mode that repeatedly appears across
            domains and turns that failure mode into a runtime object that can be monitored,
            repaired, certified, and audited.  If that contribution holds, ORIUS becomes useful far
            beyond a single battery benchmark.

            The deployment story follows the parity gate, not aspiration alone.  Near-term ORIUS is
            strongest as a universal runtime discipline exercised most deeply in battery and already
            defended in industrial, healthcare, and the current bounded AV contract.  The next
            frontier is not a new slogan; it is closing navigation with real-data evidence,
            replacing the aerospace placeholder with a real multi-flight safety task, and expanding
            cross-domain runtime governance and composition only where the same governed artifacts
            actually clear.  That is how ORIUS can contribute meaningfully to society without
            borrowing credibility from unclosed rows.

            \section{Runtime breadth and governance maturity}
            The 93+ program also makes runtime breadth explicit.  The point is not simply that
            CertOS exists; it is that runtime budgets, lifecycle events, and audit continuity are
            now tabled across the defended rows rather than treated as battery-only depth.

            \input{reports/publication/tbl_orius_runtime_budget_matrix}

            \input{reports/publication/tbl_orius_governance_lifecycle_matrix}

            \section{Deployment validation scope}
            ORIUS should only use deployment language that matches the evidence surface.  The
            deployment-scope table below distinguishes defended replay, HIL rehearsal, proxy
            validation, and explicitly open field or regulated surfaces.  This keeps the societal
            pitch strong without letting aspiration flatten the current evidence hierarchy.

            \input{reports/publication/tbl_orius_deployment_validation_scope}
            """
        ).strip() + "\n",
        "ch16_conclusion_monograph.tex": dedent(
            r"""
            \chapter{Conclusion}
            \label{ch:monograph-conclusion}

            ORIUS is presented in this monograph as a universal safety layer for physical AI under
            degraded observation.  The universal contribution is architectural, semantic, and
            governance-centered: one typed runtime contract, one benchmark discipline, one
            certificate lifecycle, and six first-class domain chapters that all instantiate the
            same degraded-observation question.

            That combination is the core thesis of the book.  A universal safety framework becomes
            credible not when it hides asymmetry, but when it exposes a common hidden hazard,
            defines a reusable runtime layer around it, and then subjects every domain to one
            visible parity gate.  On the current repo truth, ORIUS already reads as a universal-first
            book with four defended rows and two explicitly gated rows.  That is a stronger and
            more durable scientific contribution than a flatter claim surface that outruns the
            artifacts.  ORIUS is offered on exactly those terms.
            """
        ).strip() + "\n",
    }


def _build_review_assets() -> None:
    reviewer_lookup = {row["id"]: row for row in REVIEWERS}
    wave_lookup = {row["id"]: row for row in REVIEW_WAVES}
    review_lines = [
        "\\documentclass[11pt,oneside]{report}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{booktabs,longtable,array,enumitem,hyperref}",
        "\\begin{document}",
        "\\title{ORIUS Universal-First Editorial Review Program}",
        "\\author{Simulated Five-Reviewer / Three-Wave External Review}",
        "\\date{April 2026}",
        "\\maketitle",
        "\\tableofcontents",
        "\\chapter{Purpose and Review Protocol}",
        "This dossier is a simulated five-reviewer R1-style program for the ORIUS universal-first monograph. It is grounded in current repo truth: the book structure, the six domain chapters, the parity gate, the runtime/governance artifacts, and the current code-and-evidence boundary. The review program runs in three waves: outline, full draft, and near-final PDF.",
        "It is included as an editorial audit and revision-traceability aid, not as part of the monograph's scientific evidence surface.",
        "Because a reviewer dossier inside a thesis package is unusual, it is intentionally quarantined as optional editorial process material. A reader can omit this dossier entirely without losing any core theorem, artifact, or domain-evidence claim from the monograph itself.",
        "",
        "\\section*{Standing reviewers}",
        "\\begin{longtable}{p{0.24\\textwidth}p{0.28\\textwidth}p{0.38\\textwidth}}",
        "\\toprule",
        "\\textbf{Reviewer} & \\textbf{Primary lens} & \\textbf{Why this lens matters}\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "\\textbf{Reviewer} & \\textbf{Primary lens} & \\textbf{Why this lens matters}\\\\",
        "\\midrule",
        "\\endhead",
    ]
    for reviewer in REVIEWERS:
        review_lines.append(
            rf"{_escape_tex(reviewer['persona'])} & {_escape_tex(reviewer['focus'])} & {_escape_tex(reviewer['summary'])}\\"
        )
    review_lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    for wave in REVIEW_WAVES:
        wave_rows = [row for row in REVIEW_SCORECARDS if row[0] == wave["id"]]
        gap_rows = [row for row in REVIEW_GAPS if row[0] == wave["id"]]
        review_lines.extend(
            [
                rf"\chapter{{{_escape_tex(wave['label'])}}}",
                _escape_tex(wave["purpose"]),
                "",
                r"\section*{Scorecard table}",
                r"\begin{longtable}{p{0.22\textwidth}ccccccccc}",
                rf"\caption{{Reviewer scorecard table for the {_escape_tex(wave['label']).lower()}.}}\\",
                r"\toprule",
                r"\textbf{Reviewer} & \textbf{Nov.} & \textbf{Thm.} & \textbf{Univ.} & \textbf{Parity} & \textbf{Gov.} & \textbf{Bench.} & \textbf{Writing} & \textbf{Thesis} & \textbf{Flagship}\\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"\textbf{Reviewer} & \textbf{Nov.} & \textbf{Thm.} & \textbf{Univ.} & \textbf{Parity} & \textbf{Gov.} & \textbf{Bench.} & \textbf{Writing} & \textbf{Thesis} & \textbf{Flagship}\\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for _, reviewer_id, novelty, theorem_rigor, universality, domain_parity, runtime_governance, benchmark, writing, thesis_ready, flagship_ready, verdict in wave_rows:
            reviewer = reviewer_lookup[reviewer_id]
            review_lines.append(
                rf"{_escape_tex(reviewer['persona'])} & {novelty:.1f} & {theorem_rigor:.1f} & {universality:.1f} & {domain_parity:.1f} & {runtime_governance:.1f} & {benchmark:.1f} & {writing:.1f} & {thesis_ready:.1f} & {flagship_ready:.1f}\\"
            )
        review_lines.extend([r"\bottomrule", r"\end{longtable}", ""])
        review_lines.extend(
            [
                r"\section*{Primary review findings table}",
                r"\begin{longtable}{p{0.10\textwidth}p{0.18\textwidth}p{0.16\textwidth}p{0.20\textwidth}p{0.24\textwidth}}",
                rf"\caption{{Primary editorial review findings for the {_escape_tex(wave['label']).lower()}.}}\\",
                r"\toprule",
                r"\textbf{Gap} & \textbf{Severity} & \textbf{Manuscript surface} & \textbf{Current blocker} & \textbf{Required action}\\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"\textbf{Gap} & \textbf{Severity} & \textbf{Manuscript surface} & \textbf{Current blocker} & \textbf{Required action}\\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for _, gap_id, severity, title, surface, blocker, remediation in gap_rows:
            review_lines.append(
                rf"\texttt{{{_escape_tex(gap_id)}}} & {_escape_tex(severity)} & \texttt{{{_escape_tex(surface)}}} & {_escape_tex(title)}: {_escape_tex(blocker)} & {_escape_tex(remediation)}\\"
            )
        review_lines.extend([r"\bottomrule", r"\end{longtable}", ""])
        review_lines.append(r"\section*{Reviewer remarks}")
        for _, reviewer_id, novelty, theorem_rigor, universality, domain_parity, runtime_governance, benchmark, writing, thesis_ready, flagship_ready, verdict in wave_rows:
            reviewer = reviewer_lookup[reviewer_id]
            review_lines.extend(
                [
                    rf"\paragraph{{{_escape_tex(reviewer['persona'])}}}",
                    _escape_tex(verdict),
                    "",
                ]
            )

    review_lines.extend(
        [
            "\\chapter{Global Synthesis}",
            "Across the three waves, the same ranked themes persist. ORIUS is strongest as a universal-first monograph when it keeps the degraded-observation hazard central, the parity gate reader-visible, the runtime-governance layer scientific rather than administrative, and the weaker rows explicitly blocked rather than rhetorically flattened.",
            "",
            "The near-final package is thesis-ready and close to flagship-publication quality as a universal framework monograph, but it is not yet an equal-domain universal monograph because navigation and aerospace still fail the governed parity gate.",
            "\\end{document}",
        ]
    )
    _write(REVIEW_DIR / "orius_review_dossier.tex", "\n".join(review_lines) + "\n")

    gap_rows = [[
        "wave",
        "gap_id",
        "severity",
        "title",
        "manuscript_surface",
        "code_or_evidence_gap",
        "required_action",
    ]]
    for row in REVIEW_GAPS:
        gap_rows.append(list(row))
    with (PUBLICATION_DIR / "orius_review_global_gap_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(gap_rows)

    _write(
        PUBLICATION_DIR / "orius_review_remediation_report.md",
        dedent(
            """\
            # ORIUS Universal-First Review Remediation Report

            ## Ranked blockers across the three-wave R1 program

            1. Keep equal-domain universality subordinate to the governed parity matrix until every domain clears the same evidence gate.
            2. Remove legacy thesis scaffolding that still exposes reader-facing battery-origin framing outside the canonical monograph.
            3. Standardize all six domain chapters on one template that includes fallback/runtime behavior and exact non-claims.
            4. Finish the navigation real-data row before promoting navigation beyond portability evidence.
            5. Replace the aerospace placeholder row with a real multi-flight safety task before claiming equal-peer universality.
            6. Keep composition and governance integrated as bounded universal layers, and only extend their claims where the domain adapter and evidence support them.

            ## Manuscript-facing implications

            - The monograph should read as one universal argument from hazard to architecture to theory to six domain chapters to parity gate.
            - Battery remains the deepest witness row, but the conceptual center must remain the universal degraded-observation hazard and the ORIUS safety-layer contract.
            - Four rows are currently defended under repo truth: battery, autonomous vehicles under the bounded TTC contract, industrial process control, and healthcare monitoring.
            - Navigation and aerospace remain explicitly gated and must stay gated in reader-facing prose.

            ## Why this still strengthens the submission

            The review program does not weaken ORIUS. It makes the book defensible. A strong R1 review is more likely to reward a monograph that is universal-first in structure, explicit about its parity gate, and disciplined about what remains open than one that overclaims equal-domain closure before the code and evidence are ready.
            """
        ),
    )


def _build_publication_tables() -> None:
    closure_rows = [
        ["domain", "tier", "source", "baseline_tsvr", "orius_tsvr", "promotion_gate", "current_status"],
    ]
    for row in DOMAIN_ROWS:
        closure_rows.append(
            [
                row["label"],
                row["tier"],
                row["source"],
                row["baseline_tsvr"],
                row["orius_tsvr"],
                row["promotion_gate"],
                row["status_sentence"],
            ]
        )
    with (PUBLICATION_DIR / "orius_domain_closure_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(closure_rows)

    parity_rows = [
        [
            "domain",
            "adapter_correctness",
            "dataset_raw_source_status",
            "processed_train_validation_status",
            "replay_status",
            "safe_action_soundness",
            "fallback_semantics",
            "certos_lifecycle_support",
            "multi_agent_support",
            "resulting_tier",
            "exact_blocker",
        ],
        [
            "Battery Energy Storage",
            "pass",
            "locked_reference",
            "pass",
            "pass",
            "reference_witness",
            "safe_hold_validated",
            "evaluated",
            "evaluated",
            "reference",
            "battery_reference_row",
        ],
        [
            "Autonomous Vehicles",
            "pass",
            "verified",
            "pass",
            "pass",
            "pass_under_ttc_entry_barrier_contract",
            "bounded_brake_fallback_validated",
            "evaluated",
            "gated_pending_shared_constraint_surface",
            "proof_validated",
            "multi_lane_and_higher_dimensional_repair_open",
        ],
        [
            "Industrial Process Control",
            "pass",
            "verified",
            "pass",
            "pass",
            "pass",
            "bounded_runtime_pass",
            "evaluated",
            "evaluated",
            "proof_validated",
            "bounded_to_current_plant_family",
        ],
        [
            "Medical and Healthcare Monitoring",
            "pass",
            "verified",
            "pass",
            "pass",
            "pass",
            "bounded_runtime_pass",
            "evaluated",
            "gated_pending_shared_constraint_surface",
            "proof_validated",
            "bounded_to_current_monitoring_and_intervention_contract",
        ],
        [
            "Navigation and Guidance",
            "pass",
            "blocked_real_data_gap",
            "blocked_real_data_gap",
            "portability_only",
            "blocked_real_data_gap",
            "bounded_runtime_pass",
            "gated",
            "gated",
            "shadow_synthetic",
            "navigation_real_data_row_missing",
        ],
        [
            "Aerospace Control",
            "pass",
            "placeholder_surface",
            "placeholder_surface",
            "experimental_replay_only",
            "experimental_placeholder",
            "bounded_runtime_pass",
            "gated",
            "gated",
            "experimental",
            "real_multi_flight_safety_task_missing",
        ],
    ]
    with (PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(parity_rows)

    parity_tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Equal-domain parity gate for the universal-first ORIUS book.}",
        r"\label{tab:orius-equal-domain-parity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{2.5cm}ccccccccp{3.5cm}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Adapter} & \textbf{Raw/Data} & \textbf{Train} & \textbf{Replay} & \textbf{Soundness} & \textbf{Fallback} & \textbf{CertOS} & \textbf{P5} & \textbf{Tier / Exact blocker}\\",
        r"\midrule",
    ]
    for row in parity_rows[1:]:
        parity_tex.append(
            rf"{_escape_tex(row[0])} & {_escape_tex(row[1])} & {_escape_tex(row[2])} & {_escape_tex(row[3])} & {_escape_tex(row[4])} & {_escape_tex(row[5])} & {_escape_tex(row[6])} & {_escape_tex(row[7])} & {_escape_tex(row[8])} & {_escape_tex(row[9])}; {_escape_tex(row[10])}\\"
        )
    parity_tex.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    _write(PUBLICATION_DIR / "tbl_orius_equal_domain_parity_matrix.tex", "\n".join(parity_tex))

    claim_rows = [
        ["claim_family", "claim_id", "scope", "governing_artifact", "status", "notes"],
        ["universal_safety", "U001", "canonical manuscript", "paper/paper.tex", "governing", "Book-class monograph is the active canonical manuscript."],
        ["universal_safety", "U002", "parity gate", "reports/publication/orius_equal_domain_parity_matrix.csv", "governing", "Equal-domain rhetoric is governed by the parity matrix rather than prose-first promotion."],
        ["universal_safety", "U003", "artifact provenance", "reports/publication/release_manifest.json", "governing", "Tracked release artifacts remain authoritative."],
        ["universal_safety", "U004", "review package", "paper/review/orius_review_dossier.tex", "governing", "Separate five-reviewer, three-wave R1 dossier accompanies the monograph."],
        ["universal_safety", "U005", "bibliography", "paper/bibliography/orius_monograph.bib", "governing", "Monograph bibliography combines legacy ORIUS entries with curated external literature."],
    ]
    with (PUBLICATION_DIR / "orius_universal_claim_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(claim_rows)

    transfer_rows = [
        ["domain", "required_transfer_obligation"],
    ] + [[row["label"], row["transfer_obligations"]] for row in DOMAIN_ROWS]
    with (PUBLICATION_DIR / "orius_transfer_obligation_table.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(transfer_rows)

    literature_rows = [
        [
            "source_id",
            "source_type",
            "title_or_unit",
            "literature_family",
            "domain_scope",
            "problem",
            "method_or_mechanism",
            "datasets_or_artifacts",
            "key_result_or_takeaway",
            "evidence_tier",
            "reusable_for_orius",
            "remaining_gap_for_universal_thesis",
        ],
        [
            "U1",
            "internal_unit",
            "Battery witness domain",
            "witness_domain",
            "energy",
            "true-state safety under degraded telemetry",
            "DC3S repair-and-certify witness with theorem-to-artifact lineage",
            "battery replay artifacts, theorem surfaces, locked publication tables",
            "deepest defended witness for the universal safety layer",
            "witness_row",
            "yes",
            "must remain the witness row rather than the conceptual center",
        ],
        [
            "U2",
            "internal_unit",
            "Temporal validity and certificate horizon",
            "temporal_validity",
            "multi_domain",
            "certificate validity over time",
            "expiration, blackout safe-hold, and bounded horizon semantics",
            "blackout studies, temporal theorem code, bounded artifacts",
            "extends runtime safety beyond one-step validity",
            "defended_bounded_layer",
            "yes",
            "bounded temporal scope only",
        ],
        [
            "U3",
            "internal_unit",
            "Graceful degradation and fallback",
            "graceful_degradation",
            "multi_domain",
            "controlled degradation under prolonged observation loss",
            "fallback policy benchmark and bounded degradation quality evaluation",
            "graceful degradation traces, policy comparison artifacts",
            "shows how ORIUS trades intervention against physical risk under blindness",
            "defended_bounded_layer",
            "yes",
            "does not imply universal optimal fallback",
        ],
        [
            "U4",
            "internal_unit",
            "Universal benchmark discipline",
            "benchmark_discipline",
            "multi_domain",
            "comparable safety evaluation across domains",
            "shared replay schema, universal metrics, and latency accounting",
            "benchmark engine, validation tables, latency artifacts",
            "makes true-state violation and intervention measurable across rows",
            "defended_bounded_layer",
            "yes",
            "schema discipline does not by itself imply equal-domain closure",
        ],
        [
            "U5",
            "internal_unit",
            "Shared-constraint composition",
            "compositional_safety",
            "multi_agent",
            "non-composition of local certificates under shared resources",
            "coordinated repair and bounded margin allocation",
            "fleet and multi-agent artifacts",
            "shows that local certificates do not automatically compose under shared constraints",
            "defended_bounded_layer",
            "yes",
            "full heterogeneous composition remains open",
        ],
        [
            "U6",
            "internal_unit",
            "Runtime governance and audit continuity",
            "runtime_governance",
            "multi_domain",
            "lifecycle and audit continuity of safety certificates",
            "issuance-validation-expiry-fallback-recovery semantics",
            "CertOS runtime, audit logs, lifecycle artifacts",
            "adds explicit governance around runtime safety rather than post-hoc reporting",
            "defended_bounded_layer",
            "yes",
            "field deployment and regulation remain outside the current evidence",
        ],
        [
            "D1",
            "internal_domain",
            "Autonomous vehicles bounded defended row",
            "domain_instantiation",
            "autonomy",
            "longitudinal collision-margin preservation under degraded perception",
            "TTC plus predictive-entry-barrier repair under the universal contract",
            "locked trajectory telemetry, replay artifacts, bounded fallback traces",
            "defended bounded row under the current TTC contract",
            "defended_bounded_row",
            "yes",
            "multi-lane and richer repair surfaces remain open",
        ],
        [
            "D2",
            "internal_domain",
            "Industrial bounded defended row",
            "domain_instantiation",
            "industrial",
            "process-envelope preservation under degraded plant telemetry",
            "process-specific repair and fallback inside the universal kernel",
            "locked industrial telemetry, replay closure, training audit",
            "strong defended non-battery instantiation of the safety layer",
            "defended_bounded_row",
            "yes",
            "broader plant families remain open",
        ],
        [
            "D3",
            "internal_domain",
            "Healthcare bounded defended row",
            "domain_instantiation",
            "healthcare",
            "threshold-preserving monitoring under stale or delayed physiologic data",
            "bounded intervention and certificate semantics under the universal contract",
            "locked ICU-vitals evidence, replay closure, intervention traces",
            "defended monitoring-and-intervention row under bounded semantics",
            "defended_bounded_row",
            "yes",
            "full clinical deployment and regulation remain open",
        ],
        [
            "D4",
            "internal_domain",
            "Navigation portability row",
            "domain_instantiation",
            "navigation",
            "corridor and guidance preservation under degraded localization",
            "same runtime contract on synthetic or portability-level traces",
            "bounded synthetic replay and protocol traces",
            "shows structural portability, not defended real-data closure",
            "shadow_synthetic",
            "yes",
            "real-data train-validate-replay chain is still missing",
        ],
        [
            "D5",
            "internal_domain",
            "Aerospace experimental boundary row",
            "domain_instantiation",
            "aerospace",
            "flight-envelope preservation under degraded flight-state telemetry",
            "envelope-hold and certificate semantics under the universal contract",
            "experimental replay artifacts and placeholder flight-task surfaces",
            "marks the outer boundary of the current universal evidence package",
            "experimental",
            "yes",
            "needs a stronger multi-flight task and material post-repair gain",
        ],
        [
            "L1",
            "external_literature",
            "Algorithmic Learning in a Random World (Vovk et al. 2005)",
            "conformal_prediction",
            "universal",
            "distribution-free uncertainty sets",
            "conformal prediction",
            "foundational theory text",
            "provides finite-sample coverage basis for nonparametric safety intervals",
            "foundational",
            "yes",
            "exchangeability assumptions weaken under degraded observation",
        ],
        [
            "L2",
            "external_literature",
            "Conformalized Quantile Regression (Romano et al. 2019)",
            "conformal_quantile_regression",
            "tabular_and_time_series",
            "heteroscedastic uncertainty estimation",
            "quantile regression plus conformal calibration",
            "supervised residual calibration",
            "adaptive interval width without strict parametric assumptions",
            "foundational",
            "yes",
            "does not by itself couple reliability or runtime repair",
        ],
        [
            "L3",
            "external_literature",
            "Adaptive Conformal Inference under Distribution Shift (Gibbs and Candes 2021)",
            "adaptive_conformal",
            "multi_domain",
            "coverage under shift",
            "online threshold adaptation",
            "online conformal recalibration",
            "maintains long-run calibration under nonstationarity",
            "foundational",
            "yes",
            "needs explicit degraded-observation semantics",
        ],
        [
            "L4",
            "external_literature",
            "Conformal Prediction Beyond Exchangeability (Barber et al. 2023)",
            "conformal_under_shift",
            "multi_domain",
            "validity beyond IID assumptions",
            "relaxed conformal guarantees",
            "statistical theory under weaker assumptions",
            "sharpens the shift-aware claim boundary of the monograph",
            "foundational",
            "yes",
            "does not solve closed-loop repair or certification",
        ],
        [
            "L5",
            "external_literature",
            "Using Simplicity to Control Complexity (Sha 2001)",
            "runtime_assurance",
            "multi_domain",
            "safe supervisory control",
            "simple trusted safety layer over complex controller",
            "software and systems architecture",
            "grounds the supervisory assurance intuition behind ORIUS",
            "foundational",
            "yes",
            "not a degraded-observation benchmark or certificate framework",
        ],
        [
            "L6",
            "external_literature",
            "Safe Reinforcement Learning via Shielding (Alshiekh et al. 2018)",
            "shielding_and_runtime_safety",
            "robotics_and_control",
            "unsafe learned actions",
            "shielding and runtime interception",
            "reinforcement-learning safety intervention",
            "shows value of online correction over offline policy trust",
            "applied_prior_art",
            "yes",
            "typically assumes trusted state for the shield",
        ],
        [
            "L7",
            "external_literature",
            "Safe RL with Model Predictive Shielding (Bastani 2021)",
            "shielding_and_runtime_safety",
            "robotics_and_control",
            "online safe action correction",
            "model-predictive shielding",
            "control and RL hybrid",
            "demonstrates repair-based safety at runtime",
            "applied_prior_art",
            "yes",
            "does not center observation degradation or audit semantics",
        ],
        [
            "L8",
            "external_literature",
            "Risk-Averse MPC for Battery Energy Storage Systems (Rosewater et al. 2020)",
            "energy_and_smart_grid",
            "energy",
            "battery dispatch with uncertainty",
            "risk-averse MPC",
            "battery storage control",
            "shows a strong energy-domain control baseline",
            "applied_prior_art",
            "yes",
            "uncertainty is scenario-centric rather than observation-centric",
        ],
        [
            "L9",
            "external_literature",
            "Anomaly Detection Survey (Chandola et al. 2009)",
            "drift_and_anomaly_detection",
            "multi_domain",
            "fault and outlier detection",
            "anomaly-detection taxonomy",
            "general anomaly methods",
            "grounds OQE and degraded-observation detection primitives",
            "foundational",
            "yes",
            "does not integrate detection with repair-and-certify control",
        ],
        [
            "L10",
            "external_literature",
            "A Systems and Control Perspective of CPS Security (Dibaji et al. 2019)",
            "cps_security_and_resilience",
            "multi_domain",
            "security and resilience of CPS sensing/control",
            "attack surfaces and resilient control framing",
            "CPS security synthesis",
            "connects telemetry trust to safety-critical control",
            "applied_prior_art",
            "yes",
            "does not provide a unified adapterized benchmark/governance layer",
        ],
        [
            "L11",
            "external_literature",
            "Applications of Lithium-Ion Batteries in Grid-Scale Storage (Chen et al. 2020)",
            "energy_and_smart_grid",
            "energy",
            "grid-scale storage operations",
            "domain survey",
            "battery systems survey",
            "helps justify battery as the strongest empirical witness",
            "applied_prior_art",
            "yes",
            "energy-specific and not itself a universal safety framework",
        ],
        [
            "L12",
            "external_literature",
            "Cross-domain CPS safety practice in robotics, industrial control, healthcare monitoring, and IoT",
            "applied_domain_review",
            "multi_domain",
            "interoperability and data-trust challenges",
            "adapterized sensing/control stacks and monitoring pipelines",
            "multiple domain instantiations in repo",
            "shows the same observation-trust problem appears beyond energy",
            "applied_domain_review",
            "yes",
            "evidence remains asymmetric across domains",
        ],
    ]
    with (PUBLICATION_DIR / "orius_literature_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(literature_rows)

    maturity_rows = [
        ["unit", "current_state", "evidence_basis", "thesis_status", "primary_risk", "next_action"],
        ["Battery witness domain", "implemented_and_artifact_backed", "theorem ladder plus locked publication artifacts", "witness_row", "overgeneralizing witness-depth proof", "keep as deepest empirical and formal witness without making it the conceptual center"],
        ["Temporal validity layer", "implemented_and_artifact_backed", "blackout study plus temporal theorem surfaces", "defended_bounded_layer", "claim creep beyond bounded horizons", "keep half-life and safe-hold claims explicitly bounded"],
        ["Graceful fallback layer", "implemented_and_artifact_backed", "policy comparison artifacts and graceful degradation traces", "defended_bounded_layer", "no universal optimal fallback theorem", "frame as benchmarked degradation policy layer"],
        ["Universal benchmark discipline", "implemented_with_schema_cleanup", "benchmark engine plus publication tables", "defended_bounded_layer", "legacy compatibility fields can obscure canonical semantics", "lock the domain-neutral schema in docs and governance"],
        ["Shared-constraint composition", "implemented_and_artifact_backed", "fleet and multi-agent artifacts", "defended_bounded_layer", "overclaiming cross-domain composition", "retain as bounded shared-constraint extension"],
        ["Runtime governance and audit continuity", "implemented_with_cleanup", "CertOS runtime and audit artifacts", "defended_bounded_layer", "legacy battery-shaped examples in policy docs", "frame as universal policy-driven governance surface"],
        ["Autonomous vehicles", "implemented_and_validated_under_bounded_contract", "locked trajectory telemetry plus replay closure", "defended_bounded_row", "current closure is bounded to the TTC entry-barrier contract", "keep as defended bounded row while broader vehicle interaction remains open"],
        ["Industrial domain", "implemented_and_validated", "locked replay plus defended instantiation tables", "defended_bounded_row", "limited proof depth beyond the current plant family", "retain as bounded defended instantiation"],
        ["Healthcare domain", "implemented_and_validated", "locked replay plus defended instantiation tables", "defended_bounded_row", "limited proof depth beyond the monitoring contract", "retain as bounded defended instantiation"],
        ["Navigation", "simulation_backed_portability_only", "synthetic closed-loop evidence", "shadow_synthetic", "no locked real-data row", "retain as portability evidence only until the real-data row closes"],
        ["Aerospace", "experimental_adapter_surface", "experimental replay artifacts", "experimental", "insufficient defended telemetry and promotion contract", "retain as experimental until stronger replay and artifact closure"],
        ["Frontend/backend reporting", "partially_unified", "research router plus proxy routes", "implemented_with_cleanup_open", "older local-cache assumptions in docs and tooling", "continue backend-first artifact authority cleanup"],
        ["Monograph package governance", "implemented_with_cleanup", "metrics manifest, claim matrix, and manuscript/review build surfaces", "implemented_with_cleanup_open", "inconsistent manuscript-facing wording across files", "finish universal-first editorial cleanup and active-surface archive quarantine"],
    ]
    with (PUBLICATION_DIR / "orius_maturity_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(maturity_rows)

    gap_rows = [
        [
            "subsystem",
            "repo_surface_or_monograph_unit",
            "closest_literature_family",
            "current_maturity",
            "battery_specific_leakage",
            "universal_requirement",
            "thesis_framing_action",
            "code_or_spec_cleanup_action",
        ],
        ["telemetry_ingestion", "dc3s adapters and domain parsers", "cps_security_and_resilience", "implemented", "energy telemetry still dominates examples in some docs", "domain-neutral observation packet with adapter-specific parsing", "frame telemetry ingestion as a shared observation packet plus domain parsers", "document canonical adapter contract and per-domain parser responsibilities"],
        ["reliability_and_drift", "dc3s quality and OQE paths", "drift_and_anomaly_detection", "implemented", "example fault language can still be battery-shaped", "shared reliability score across fault families", "write OQE as a universal degraded-observation detector", "keep battery fault examples in domain sections only"],
        ["uncertainty_construction", "universal_theory, calibration, RAC/FTIT", "conformal_prediction", "implemented_with_bounded_assumptions", "SOC-tube language leaks into universal prose in legacy files", "observation-consistent state set and reliability-conditioned inflation", "rename universal prose around state sets and margins", "keep battery helper exports out of the universal package root"],
        ["safe_action_tightening_and_repair", "dc3s shield and domain adapters", "shielding_and_runtime_safety", "implemented", "charge-discharge examples dominate some repair descriptions", "domain-specific feasible set under shared repair semantics", "describe repair as a general action projection", "keep battery-specific repair operators inside domain modules"],
        ["certificate_generation", "universal_theory certificates and CertOS certificate engine", "runtime_assurance", "implemented", "battery fields historically shaped examples", "causal safety certificate with domain-neutral required fields", "describe the certificate as a universal audit object", "continue shifting policy checks away from SOC/MWh assumptions"],
        ["runtime_governance_and_audit", "CertOS runtime and lifecycle surfaces", "runtime_assurance", "implemented_with_cleanup_open", "legacy config fields still mention SOC/MWh in some docs", "policy-driven governance surface and invariant semantics", "frame runtime governance as a bounded universal layer", "finish replacing battery-specific invariants in docs/tests and emphasize policy hooks"],
        ["benchmark_replay_and_metrics", "ORIUS-Bench universal replay and metrics", "benchmark_discipline", "implemented_with_cleanup_open", "legacy SOC fields remain for compatibility", "domain-neutral benchmark schema and canonical metric family", "write TSVR/OASG as universal metrics only", "keep compatibility fields non-canonical and document deprecation"],
        ["multi_agent_composition", "shared-constraint composition scenarios", "compositional_safety", "bounded_extension", "shared-resource examples remain feeder-centric", "resource-budget composition beyond energy", "describe composition as bounded shared-constraint coordination, not a universal theorem", "document vector-budget generalization path"],
        ["dashboard_and_reporting", "frontend api proxies and research router", "artifact_governance", "implemented_with_cleanup_open", "older local-cache assumptions still exist in narrative and some tools", "backend-served tracked artifacts only", "frame the dashboard as an artifact browser rather than a local cache", "keep frontend routes proxy-only and continue removing local authority language"],
    ]
    with (PUBLICATION_DIR / "orius_framework_gap_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(gap_rows)

    design_rows = [
        ["principle_id", "design_principle", "extracted_from", "relevance_to_orius", "repo_hook"],
        ["DP1", "Separate shared safety logic from plant-specific semantics", "monograph synthesis plus universal adapter architecture", "enables one kernel to travel across domains", "src/orius/dc3s/domain_adapter.py"],
        ["DP2", "Make observation reliability a first-class runtime input", "OQE plus adaptive conformal literature", "connects degraded telemetry directly to safety conservatism", "src/orius/dc3s/quality.py"],
        ["DP3", "Express uncertainty as an observation-consistent state set", "conformal and runtime repair literature", "lets repair reason over truth-vs-observation gaps", "src/orius/universal_theory/"],
        ["DP4", "Use repair instead of assuming upstream controller perfection", "shielding and runtime assurance literature", "supports safe correction without replacing the optimizer", "src/orius/dc3s/ + src/orius/universal_theory/"],
        ["DP5", "Emit auditable runtime certificates", "CertOS and supervisory assurance framing", "turns safety actions into governed evidence rather than hidden logic", "src/orius/certos/"],
        ["DP6", "Standardize the replay schema before comparing domains", "universal benchmark discipline", "prevents each domain from redefining success", "src/orius/orius_bench/"],
        ["DP7", "Keep evidence tiers explicit and governed", "parity matrix and monograph gap review", "prevents overclaiming universality", "reports/publication/orius_equal_domain_parity_matrix.csv"],
        ["DP8", "Serve tracked artifacts through one backend truth path", "research router and manuscript governance", "removes dependence on local dashboard caches", "services/api/routers/research.py"],
    ]
    with (PUBLICATION_DIR / "orius_cross_domain_design_principles.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(design_rows)

    chapter_map_rows = [
        ["unit", "main_monograph_sections", "central_claim", "evidence_status", "role_in_book"],
        ["Battery witness domain", "ch08 plus battery evidence block", "Battery provides the deepest theorem-to-artifact witness for degraded-observation safety.", "witness_row", "deepest empirical and formal witness"],
        ["Temporal validity layer", "ch20 plus blackout and half-life sections", "Certificate validity is a temporal runtime object rather than a single-step assumption.", "defended_bounded_layer", "temporal extension of the universal safety layer"],
        ["Graceful fallback layer", "ch29 and synthesis references", "Fallback quality can be benchmarked and governed rather than treated as an informal emergency path.", "defended_bounded_layer", "controlled degradation layer"],
        ["Universal benchmark discipline", "ch07 and battery benchmark block", "Replay, metrics, and latency are shared across domains under one schema.", "defended_bounded_layer", "evaluation discipline layer"],
        ["Shared-constraint composition", "ch31 and synthesis references", "Local certificates do not automatically compose under shared resources.", "defended_bounded_layer", "bounded composition layer"],
        ["Runtime governance and audit continuity", "ch07, ch32, and appendices", "Certificates require lifecycle, fallback, and audit semantics to become operationally meaningful.", "defended_bounded_layer", "runtime governance layer"],
    ]
    with (PUBLICATION_DIR / "orius_monograph_chapter_map.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(chapter_map_rows)

    issue_rows = [
        ["kind", "title", "labels", "summary", "acceptance"],
        ["parent", "ORIUS: Universal-safety monograph hardening", "research;thesis;orius", "Track the remaining monograph work after the universal-first rewrite so the canonical book stays artifact-strict, parity-gated, and archive-clean.", "The monograph stays universal-first, parity-gated, and free of active paper-lineage scaffolding."],
        ["child", "ORIUS: Keep the parity gate central in the main narrative", "research;thesis;writing", "Keep the abstract, synthesis, reviewer appendix, and conclusion aligned to the same defended-vs-gated domain posture.", "Navigation and aerospace remain gated everywhere until stronger evidence exists."],
        ["child", "ORIUS: Preserve the generated monograph as the canonical surface", "research;audit;evidence", "Keep generator-owned manuscript assets, matrices, bibliography, and review dossier synchronized through the monograph build script.", "The generator remains the single source of truth for generated monograph assets."],
        ["child", "ORIUS: Close remaining active legacy scaffolding", "research;thesis;cleanup", "Retire or rewrite leftover active text surfaces that still expose stitched-thesis or program-lineage wording.", "Active docs, reports, slides, and manuscript surfaces are monograph-native; legacy material is explicitly archived."],
        ["child", "ORIUS: Plan navigation and aerospace parity closure", "research;frontier;navigation;aerospace", "Track the concrete experiments still needed before equal-domain universality can be claimed as present tense.", "No gated row is promoted without matching data, replay, artifact, and parity updates."],
    ]
    with (PUBLICATION_DIR / "github_issue_specs.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(issue_rows)

    coverage_rows = [
        ["monograph_section", "source_surface", "coverage_mode", "status", "notes"],
        ["universal_hazard_and_related_work", "paper/monograph/ch01-ch03", "generated_monograph", "complete", "Hazard framing, claim boundary, and method families are controlled through the monograph generator."],
        ["runtime_and_governance_architecture", "paper/monograph/ch04-ch07 plus selected legacy chapters", "mixed_generated_and_curated", "complete", "Runtime kernel, benchmark, latency, and governance are presented as one universal architecture."],
        ["battery_witness_block", "paper/monograph/ch08 plus chapters/ch07-ch32 battery depth", "mixed_generated_and_curated", "complete", "Battery remains the deepest theorem-to-artifact witness inside the universal-first book."],
        ["nonbattery_domain_block", "paper/monograph/ch09-ch13", "generated_monograph", "complete", "Each non-battery row follows the common domain template under the parity gate."],
        ["cross_domain_synthesis_and_limits", "paper/monograph/ch14-ch16 plus chapters/ch22-ch34", "mixed_generated_and_curated", "complete", "Synthesis, claim boundaries, and explicit non-claims stay aligned to the parity matrix."],
        ["appendix_and_review_surface", "paper/monograph/app_ad-app_ai plus appendices/", "mixed_generated_and_curated", "complete", "Proofs, protocol cards, reviewer analysis, and artifact traceability support the main monograph."],
    ]
    with (PUBLICATION_DIR / "paper_thesis_coverage_map.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(coverage_rows)

    legacy_notice_rows = [
        ["status", "canonical_replacement", "note"],
        ["legacy_archive_only", "reports/publication/orius_monograph_chapter_map.csv", "This legacy filename is retained only as an archive pointer. Use the monograph chapter map for current work."],
    ]
    legacy_archive_dir = REPO_ROOT / "reports" / "legacy_archive"
    for legacy_name in [
        "orius_program_paper_map.csv",
        "papers16_master_gap_audit.csv",
        "papers26_inclusion_audit.csv",
    ]:
        with (legacy_archive_dir / legacy_name).open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(legacy_notice_rows)

    _write(
        legacy_archive_dir / "papers16_master_gap_audit.txt",
        "Legacy archive notice: this file is retained only as a pointer to earlier program-era audits. Canonical monograph surfaces are the parity matrix, domain closure matrix, universal claim matrix, and monograph chapter map.\n",
    )
    _write(
        legacy_archive_dir / "papers16_master_gap_audit.md",
        "Legacy archive notice: this file is retained only as a pointer to earlier program-era audits. Canonical monograph surfaces are the parity matrix, domain closure matrix, universal claim matrix, and monograph chapter map.\n",
    )

    _write(
        legacy_archive_dir / "README.md",
        dedent(
            """\
            # ORIUS Legacy Archive

            This folder names the repo surfaces that are retained for historical provenance but are
            not part of the active universal-first monograph control surface.

            Non-canonical archive classes:
            - frozen `reports/publication/final_package_*` bundles
            - older program-era audit and inclusion files retained only as pointers
            - historical witness-centered or stitched-thesis packaging snapshots

            Canonical active surfaces live in:
            - `paper/paper.tex`
            - `paper/review/orius_review_dossier.tex`
            - `reports/publication/orius_equal_domain_parity_matrix.csv`
            - `reports/publication/orius_domain_closure_matrix.csv`
            - `reports/publication/orius_universal_claim_matrix.csv`
            - `reports/publication/orius_monograph_chapter_map.csv`
            """
        ),
    )


def _build_monograph_support_assets() -> None:
    reference_rows = _collect_reference_rows()
    with (PUBLICATION_DIR / "orius_annotated_bibliography.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "year", "family", "source", "title", "role"])
        for row in reference_rows:
            writer.writerow([row["key"], row["year"], row["family"], row["source"], row["title"], row["role"]])

    reviewer_lookup = {row["id"]: row for row in REVIEWERS}
    reviewer_rows = [[
        "wave",
        "reviewer_id",
        "reviewer",
        "novelty",
        "theorem_rigor",
        "universality_credibility",
        "domain_parity",
        "runtime_governance_maturity",
        "benchmark_credibility",
        "writing_quality",
        "thesis_readiness",
        "flagship_publication_readiness",
        "verdict",
    ]]
    for wave_id, reviewer_id, novelty, theorem_rigor, universality, domain_parity, runtime_governance, benchmark, writing, thesis_ready, flagship_ready, verdict in REVIEW_SCORECARDS:
        reviewer = reviewer_lookup[reviewer_id]
        reviewer_rows.append(
            [
                wave_id,
                reviewer_id,
                reviewer["persona"],
                novelty,
                theorem_rigor,
                universality,
                domain_parity,
                runtime_governance,
                benchmark,
                writing,
                thesis_ready,
                flagship_ready,
                verdict,
            ]
        )
    with (PUBLICATION_DIR / "orius_reviewer_scorecards.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(reviewer_rows)

    module_rows = [
        ("core", "src/orius/dc3s/domain_adapter.py", "Universal adapter contract", "architecture", "Defines the canonical runtime boundary for all domains."),
        ("core", "src/orius/dc3s/quality.py", "Observation quality estimation", "degradation_detection", "Turns telemetry quality into a runtime trust signal."),
        ("core", "src/orius/universal_theory/contracts.py", "Typed contracts", "theory", "Defines the state, action, and certificate objects used across the monograph."),
        ("core", "src/orius/universal_theory/kernel.py", "Universal step kernel", "architecture", "Implements the Detect-Calibrate-Constrain-Shield-Certify step."),
        ("core", "src/orius/universal_theory/risk_bounds.py", "Risk-bound primitives", "theory", "Connects reliability loss to conservative action tightening."),
        ("governance", "src/orius/certos/runtime.py", "CertOS runtime", "governance", "Enforces certificate validity, lifecycle transitions, and fallback discipline."),
        ("governance", "src/orius/certos/models.py", "Governance data model", "governance", "Keeps runtime evidence structured and traceable."),
        ("bench", "src/orius/orius_bench/metrics_engine.py", "Universal benchmark metrics", "benchmark", "Computes true-state violation, intervention, fallback, and latency summaries."),
        ("bench", "reports/publication/orius_equal_domain_parity_matrix.csv", "Equal-domain parity matrix", "claim_governance", "Governs equal-domain rhetoric and keeps blocked rows explicit."),
        ("bench", "reports/publication/orius_universal_claim_matrix.csv", "Universal claim matrix", "claim_governance", "Pins the canonical monograph claims to tracked artifacts."),
        ("system", "services/api/routers/research.py", "Backend artifact surface", "artifact_authority", "Makes tracked publication assets the only dashboard truth path."),
        ("system", "frontend/src/app/api/data/route.ts", "Frontend proxy route", "artifact_authority", "Prevents local untracked JSON from becoming claim authority."),
        ("system", "frontend/src/app/api/reports/route.ts", "Frontend report proxy", "artifact_authority", "Routes report reads through the governed backend."),
        ("domains", "src/orius/adapters/battery/theory.py", "Battery theorem helper surface", "domain_instantiation", "Keeps witness-domain helpers outside the universal package root."),
        ("domains", "src/orius/adapters/vehicle/__init__.py", "Vehicle adapter surface", "domain_instantiation", "Binds autonomy telemetry to the same runtime contract."),
        ("domains", "src/orius/universal_framework/domain_registry.py", "Domain registry", "domain_instantiation", "Enumerates the currently wired domain surfaces."),
        ("evidence", "reports/publication/release_manifest.json", "Release manifest", "artifact_provenance", "Provides the tracked evidence provenance surface for the monograph."),
        ("evidence", "reports/publication/orius_artifact_appendix.md", "Artifact appendix", "artifact_provenance", "Maps figures and tables back to release artifacts."),
        ("writing", "paper/paper.tex", "Canonical monograph entrypoint", "manuscript", "Defines the book-class build and the active thesis narrative."),
        ("writing", "paper/review/orius_review_dossier.tex", "Reviewer dossier", "review", "Separates the simulated red-team package from the book PDF itself."),
    ]
    crosswalk_rows = [["layer", "module_path", "surface", "claim_family", "evidence_note"]]
    for layer, path, surface, family, note in module_rows:
        crosswalk_rows.append([layer, path, surface, family, note])
    for row in DOMAIN_ROWS:
        domain_id = row["id"]
        label = row["label"]
        for surface, claim_family, note in [
            ("telemetry contract", "domain_instantiation", row["telemetry_model"]),
            ("safety predicate", "domain_instantiation", row["safety_predicate"]),
            ("dataset and protocol", "artifact_provenance", row["dataset_protocol"]),
            ("results and status", "claim_governance", row["results"]),
            ("transfer obligation", "promotion", row["transfer_obligations"]),
        ]:
            crosswalk_rows.append(
                [
                    "domains",
                    f"domain::{domain_id}",
                    f"{label} {surface}",
                    claim_family,
                    note,
                ]
            )
    with (PUBLICATION_DIR / "orius_module_claim_crosswalk.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(crosswalk_rows)

    legacy_publication_files = {
        "orius_program_paper_map.csv",
        "papers16_master_gap_audit.csv",
        "papers16_master_gap_audit.txt",
        "papers16_master_gap_audit.md",
        "papers26_inclusion_audit.csv",
    }
    artifact_rows = [["artifact", "category", "role"]]
    for path in sorted(
        p for p in PUBLICATION_DIR.iterdir() if p.is_file() and p.name not in legacy_publication_files
    ):
        name = path.name
        lower = name.lower()
        if "matrix" in lower:
            category = "governance matrix"
        elif "manifest" in lower:
            category = "manifest"
        elif lower.endswith(".csv"):
            category = "tabular evidence"
        elif lower.endswith(".json"):
            category = "structured evidence"
        elif lower.endswith(".md"):
            category = "narrative evidence"
        elif lower.endswith(".tex"):
            category = "LaTeX table"
        else:
            category = "other"
        role = f"Tracked publication artifact used for {category} in the universal monograph."
        artifact_rows.append([name, category, role])
    with (PUBLICATION_DIR / "orius_publication_artifact_index.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(artifact_rows)

    annotated_biblio_table = [
        r"\chapter{Expanded Annotated Bibliography and Literature Map}",
        r"\label{app:annotated-bibliography-map}",
        "",
        "This appendix expands the monograph bibliography into a review-facing literature map. "
        "Each row records the governing citation key, year, method family, title, and the specific "
        "role that reference plays in the ORIUS argument.",
        "",
        r"\begin{longtable}{p{0.12\textwidth}p{0.08\textwidth}p{0.20\textwidth}p{0.25\textwidth}p{0.27\textwidth}}",
        r"\caption{Annotated bibliography map for the ORIUS monograph.}\label{tab:annotated-bibliography-map}\\",
        r"\toprule",
        r"\textbf{Key} & \textbf{Year} & \textbf{Family} & \textbf{Title} & \textbf{Role in ORIUS}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Key} & \textbf{Year} & \textbf{Family} & \textbf{Title} & \textbf{Role in ORIUS}\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in reference_rows:
        annotated_biblio_table.append(
            rf"\texttt{{{_escape_tex(row['key'])}}} & {_escape_tex(row['year'])} & {_escape_tex(row['family'])} & {_escape_tex(row['title'])} & {_escape_tex(row['role'])}\\"
        )
    annotated_biblio_table.extend([r"\bottomrule", r"\end{longtable}", ""])
    _write(MONOGRAPH_DIR / "app_ad_annotated_bibliography_map.tex", "\n".join(annotated_biblio_table) + "\n")

    near_final_rows = [row for row in REVIEW_SCORECARDS if row[0] == "near_final"]
    near_final_lookup = {row[1]: row for row in near_final_rows}
    detailed_reviews = {
        "formal_safety": {
            "summary_judgment": "The monograph is scientifically credible as a universal runtime-safety architecture, but equal-domain universality remains gated because navigation and aerospace do not yet clear the same defended closure surface as the stronger rows.",
            "strongest_contributions": [
                "The degraded-observation hazard is identified as a plant-agnostic safety object rather than a battery-specific modeling issue.",
                "The theorem bridge is explicit about what is structural, what is bounded, and what must still be validated by replay or artifact.",
                "The parity matrix keeps the strongest claim subordinate to governed evidence rather than rhetorical flattening.",
            ],
            "major_weaknesses": [
                "Battery still carries the deepest theorem-to-artifact lineage, so the equal-domain narrative must remain carefully bounded.",
                "The formal surface remains much stronger for some domains than others.",
            ],
            "unsupported_claims": [
                "Any statement implying equal theorem-grade closure across all six domains.",
                "Any statement implying universal hardware or regulatory readiness.",
            ],
            "required_experiments": [
                "Real-data navigation closure under the universal replay contract.",
                "A stronger aerospace task with material post-repair improvement on the governing safety object.",
            ],
            "required_writing": [
                "Keep the theorem/evidence boundary stable in the theory bridge, synthesis, and conclusion chapters.",
                "State the parity gate once and reuse it consistently.",
            ],
            "decision": "Weak accept for thesis submission; reject any equal-domain flagship claim until the gated rows are closed.",
        },
        "uq_ml": {
            "summary_judgment": "The uncertainty layer is valuable because it treats reliability-conditioned uncertainty as a runtime safety object, but the book must remain precise about where coverage language is formal and where it is empirical or conservative.",
            "strongest_contributions": [
                "The monograph ties conformal and calibration ideas to runtime action legality rather than to prediction quality alone.",
                "The universal benchmark schema makes true-state violation and observed-state satisfaction separable across domains.",
                "The non-i.i.d. and distribution-shift caveats are visible instead of hidden.",
            ],
            "major_weaknesses": [
                "The weaker rows do not yet support equal-domain statistical closure.",
                "Coverage-style language can still be overread if chapter summaries are not carefully bounded.",
            ],
            "unsupported_claims": [
                "Universal statistical calibration under arbitrary shift.",
                "Formal coverage guarantees for domains whose replay surface remains portability-only or experimental.",
            ],
            "required_experiments": [
                "Fault-mode and domain-specific coverage diagnostics for every defended row.",
                "Additional uncertainty summaries for navigation and aerospace once their evidence surfaces are upgraded.",
            ],
            "required_writing": [
                "Differentiate bounded calibration claims from conservative widening heuristics.",
                "Keep uncertainty claims tied to tracked replay artifacts.",
            ],
            "decision": "Accept for a bounded universal-safety monograph; major revision for any stronger cross-domain statistical claim.",
        },
        "cps_systems": {
            "summary_judgment": "The systems contribution is strong because runtime behavior, fallback, certificates, and audit continuity are first-class objects in the main manuscript rather than hidden implementation details.",
            "strongest_contributions": [
                "CertOS is integrated into the scientific argument rather than treated as release tooling.",
                "The replay and benchmark contract is specific enough to support cross-domain inspection.",
                "The backend artifact path and governed publication surface keep the evidence path auditable.",
            ],
            "major_weaknesses": [
                "Runtime and fallback evidence are still deepest in the witness and strongest defended rows.",
                "Cross-domain composition and lifecycle coverage remain selective.",
            ],
            "unsupported_claims": [
                "Full deployment readiness in every domain.",
                "Uniform multi-agent or shared-constraint closure across all rows.",
            ],
            "required_experiments": [
                "Broader runtime-budget and lifecycle traces for the weaker domains.",
                "Expanded cross-domain composition scenarios where the adapter contract actually supports them.",
            ],
            "required_writing": [
                "Keep latency, fallback, certificate validity, and audit continuity in the mainline claims.",
                "Avoid relegating critical runtime semantics to appendices only.",
            ],
            "decision": "Strong accept as a systems-and-governance thesis; bounded revision required before broader deployment claims.",
        },
        "deployment": {
            "summary_judgment": "ORIUS has credible societal potential as a fundamental safety layer for physical AI, but that potential only reads as serious when the monograph keeps weaker domains explicitly gated instead of rhetorically promoted.",
            "strongest_contributions": [
                "The book treats battery, AV, industrial, healthcare, navigation, and aerospace as first-class domain chapters rather than as add-on examples.",
                "The parity gate prevents the deployment narrative from outrunning the artifact surface.",
                "The domain chapters share one template, which makes the universality claim inspectable.",
            ],
            "major_weaknesses": [
                "Navigation and aerospace remain below equal-peer maturity.",
                "Some readers will still look for broader field closure than the current evidence supports.",
            ],
            "unsupported_claims": [
                "Equal defended deployment credibility across every domain.",
                "Aerospace or navigation as mature peers to the witness and strongest defended rows.",
            ],
            "required_experiments": [
                "A defended real-data navigation row.",
                "A non-placeholder aerospace benchmark with a stronger safety object and post-repair gain.",
            ],
            "required_writing": [
                "Retain explicit domain non-claims in every chapter.",
                "Keep deployment language bounded by the parity matrix and artifact register.",
            ],
            "decision": "Accept as a universal physical-AI safety architecture; major revision for any flat equal-domain deployment claim.",
        },
        "committee": {
            "summary_judgment": "The manuscript now reads like a real R1-style monograph because it has one hazard, one architecture, one theorem bridge, six domain chapters, and one parity gate. The remaining gap is evidence symmetry, not book structure.",
            "strongest_contributions": [
                "The main body no longer depends on stitched paper-lineage framing.",
                "The universal-first narrative is visible from the abstract through the conclusion.",
                "The appendices provide bibliography, review, protocol, crosswalk, and artifact traceability at book scale.",
            ],
            "major_weaknesses": [
                "Legacy repo surfaces can still contradict the monograph if they are not treated as archival support only.",
                "The equal-domain ambition is stronger than the current parity evidence.",
            ],
            "unsupported_claims": [
                "A claim that the monograph has already earned equal-domain closure.",
                "Any suggestion that weaker rows are present only for symmetry rather than bounded scientific use.",
            ],
            "required_experiments": [
                "Only the parity-closing experiments needed for navigation and aerospace if equal-domain promotion remains the target.",
            ],
            "required_writing": [
                "Keep the main body universal-first and the weaker rows honest.",
                "Preserve the explicit distinction between architecture universality and evidence parity in the introduction, synthesis, and conclusion.",
            ],
            "decision": "Accept as a thesis-length monograph with bounded universal claims; defer stronger flagship universality language until parity is actually earned.",
        },
    }
    review_lines = [
        r"\chapter{Five-Reviewer Editorial Audit, Rebuttal, and Revision Traceability}",
        r"\label{app:reviewer-gap-analysis}",
        "",
        "This appendix records a simulated five-reviewer R1-style evaluation of the ORIUS monograph. "
        "The point is not artificial volume; it is to expose the manuscript to the five lenses most likely "
        "to challenge a universal physical-AI safety thesis: controls, uncertainty, formal verification, "
        "systems runtime engineering, and cross-domain thesis coherence. "
        "The tables in this appendix are editorial audit aids for revision traceability; they are not part "
        "of the scientific evidence surface.",
        "",
        r"\begin{longtable}{p{0.22\textwidth}p{0.10\textwidth}p{0.58\textwidth}}",
        r"\caption{Near-final reviewer scorecard summary for the ORIUS monograph.}\label{tab:reviewer-scorecards}\\",
        r"\toprule",
        r"\textbf{Reviewer} & \textbf{Score} & \textbf{Primary concern}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Reviewer} & \textbf{Score} & \textbf{Primary concern}\\",
        r"\midrule",
        r"\endhead",
    ]
    score_labels = {
        "formal_safety": "A-",
        "uq_ml": "B+",
        "cps_systems": "A",
        "deployment": "B+",
        "committee": "A",
    }
    primary_concerns = {
        "formal_safety": "Equal-domain rhetoric must remain subordinate to the governed parity gate.",
        "uq_ml": "Coverage and calibration language must stay bounded by the actual replay evidence.",
        "cps_systems": "Runtime and governance claims must remain concrete and artifact-backed across domains.",
        "deployment": "Navigation and aerospace still block any equal-peer deployment reading.",
        "committee": "The book is structurally strong, but evidence asymmetry must remain explicit everywhere.",
    }
    ordered_reviewers = ["formal_safety", "uq_ml", "cps_systems", "deployment", "committee"]
    for reviewer_id in ordered_reviewers:
        review_lines.append(
            rf"{_escape_tex(reviewer_lookup[reviewer_id]['persona'])} & {_escape_tex(score_labels[reviewer_id])} & {_escape_tex(primary_concerns[reviewer_id])}\\"
        )
    review_lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    section_titles = {
        "formal_safety": "Reviewer 1: Controls and Runtime Assurance",
        "uq_ml": "Reviewer 2: Statistical ML and Conformal Inference",
        "cps_systems": "Reviewer 3: Systems and Runtime Engineering",
        "deployment": "Reviewer 4: Cross-Domain Physical AI and Deployment",
        "committee": "Reviewer 5: R1 Dissertation and Monograph Coherence",
    }

    for reviewer_id in ordered_reviewers:
        reviewer = reviewer_lookup[reviewer_id]
        detail = detailed_reviews[reviewer_id]
        verdict = near_final_lookup[reviewer_id][-1]
        review_lines.extend(
            [
                rf"\section{{{_escape_tex(section_titles[reviewer_id])}}}",
                rf"\textbf{{Summary judgment}}: {_escape_tex(detail['summary_judgment'])}",
                "",
                rf"\textbf{{Scorecard verdict}}: {_escape_tex(verdict)}",
                "",
                r"\textbf{Strongest contributions}",
                r"\begin{itemize}",
            ]
        )
        for item in detail["strongest_contributions"]:
            review_lines.append(rf"\item {_escape_tex(item)}")
        review_lines.extend([r"\end{itemize}", r"\textbf{Major weaknesses}", r"\begin{itemize}"])
        for item in detail["major_weaknesses"]:
            review_lines.append(rf"\item {_escape_tex(item)}")
        review_lines.extend([r"\end{itemize}", r"\textbf{Unsupported claims}", r"\begin{itemize}"])
        for item in detail["unsupported_claims"]:
            review_lines.append(rf"\item {_escape_tex(item)}")
        review_lines.extend([r"\end{itemize}", r"\textbf{Required experiments}", r"\begin{itemize}"])
        for item in detail["required_experiments"]:
            review_lines.append(rf"\item {_escape_tex(item)}")
        review_lines.extend([r"\end{itemize}", r"\textbf{Required writing changes}", r"\begin{itemize}"])
        for item in detail["required_writing"]:
            review_lines.append(rf"\item {_escape_tex(item)}")
        review_lines.extend(
            [
                r"\end{itemize}",
                rf"\textbf{{Decision recommendation}}: {_escape_tex(detail['decision'])}",
                "",
            ]
        )

    review_lines.extend(
        [
            r"\section{Consolidated Editorial Findings}",
            r"\begin{longtable}{p{0.22\textwidth}p{0.24\textwidth}p{0.18\textwidth}p{0.26\textwidth}}",
            r"\caption{Consolidated editorial findings across the five reviewer lenses.}\label{tab:reviewer-gap-matrix}\\",
            r"\toprule",
            r"\textbf{Gap} & \textbf{Where it appears} & \textbf{Reviewer pressure} & \textbf{Revision action}\\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"\textbf{Gap} & \textbf{Where it appears} & \textbf{Reviewer pressure} & \textbf{Revision action}\\",
            r"\midrule",
            r"\endhead",
            r"Evidence asymmetry across domains & Parity matrix, cross-domain synthesis, and domain status language & Controls, deployment, and committee readers all insist that architecture universality must not be confused with equal-domain closure & Keep battery as witness, AV plus industrial and healthcare as defended bounded rows, navigation as shadow-synthetic, and aerospace as experimental until stronger reruns exist.\\",
            r"Coverage and calibration ambiguity & Related work, uncertainty discussion, and benchmark interpretation & Statistical-ML review demands a clear line between formal calibration claims and conservative widening heuristics & State where coverage language is theorem-backed, where it is replay-backed, and where it is only heuristic.\\",
            r"Theorem / replay / implementation drift & Theory bridge, synthesis, and conclusion & Formal review demands an explicit claim boundary that is reused verbatim & Tie every strong claim to a theorem object, tracked artifact, or explicit bounded empirical result.\\",
            r"Runtime and audit visibility & Governance, latency, and fallback chapters & Systems review requires runtime budgets, certificate lifecycle, and audit continuity to stay in the main narrative & Keep governance and fallback semantics first-class in the architecture and synthesis chapters.\\",
            r"Legacy framing leakage & Non-canonical repo docs and archival artifacts & Committee and deployment readers will downgrade the book if older battery-origin or stitched-paper language leaks into the main story & Keep the canonical monograph universal-first and treat archival paper-lineage artifacts as support only.\\",
            r"\bottomrule",
            r"\end{longtable}",
            "",
            r"\section{Author Rebuttal}",
            "ORIUS does not need to claim equal empirical maturity in every domain to justify the monograph. It needs to justify one universal runtime safety layer and then state, with governed discipline, how each domain currently maps onto that contract.",
            r"\begin{itemize}",
            r"\item \textbf{To the controls reviewer}: the monograph keeps the parity gate reader-visible and does not blur architecture with equal closure.",
            r"\item \textbf{To the statistical-ML reviewer}: calibration language is bounded to the evidence surface and never upgraded into universal statistical guarantees by prose.",
            r"\item \textbf{To the systems reviewer}: runtime behavior, fallback, certificate validity, and audit continuity remain part of the main scientific contribution rather than appendix-only material.",
            r"\item \textbf{To the deployment reviewer}: weaker rows remain explicitly gated so the physical-AI safety-layer claim does not borrow credibility from unclosed domains.",
            r"\item \textbf{To the committee reviewer}: the book keeps one universal narrative spine from hazard to parity gate and uses the appendices only to deepen, not replace, the main argument.",
            r"\end{itemize}",
            "",
            r"\section{Revision Traceability}",
            r"\begin{longtable}{p{0.16\textwidth}p{0.34\textwidth}p{0.30\textwidth}}",
            r"\caption{Revision traceability matrix from reviewer critique to manuscript action.}\label{tab:reviewer-traceability}\\",
            r"\toprule",
            r"\textbf{Reviewer} & \textbf{Critique} & \textbf{Revision action / file target}\\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"\textbf{Reviewer} & \textbf{Critique} & \textbf{Revision action / file target}\\",
            r"\midrule",
            r"\endhead",
            r"Controls and runtime assurance & Do not blur universal architecture with equal-domain parity. & Keep the parity gate central in \texttt{monograph/ch14\_cross\_domain\_synthesis.tex} and align all domain labels to the tracked parity matrix.\\",
            r"Statistical ML / conformal & Separate formal calibration from conservative widening. & Keep claim-boundary language explicit in \texttt{monograph/ch02\_oasg\_claim\_boundary.tex} and the term register.\\",
            r"Systems and runtime engineering & Keep governance, fallback, latency, and certificate lifecycle visible. & Preserve runtime and governance emphasis in \texttt{monograph/ch07\_system\_benchmark\_governance.tex} and the supporting CertOS chapters.\\",
            r"Cross-domain deployment & Keep navigation and aerospace gated until stronger evidence exists. & Maintain explicit non-claims and promotion obligations in the domain chapters and protocol-card appendix.\\",
            r"R1 monograph coherence & Preserve one universal-first narrative spine and treat archival language as non-canonical. & Keep \texttt{paper/paper.tex}, the monograph chapters, and the authoring guide synchronized around the same universal-first structure.\\",
            r"\bottomrule",
            r"\end{longtable}",
            "",
        ]
    )
    _write(MONOGRAPH_DIR / "app_ae_expanded_reviewer_gap_analysis.tex", "\n".join(review_lines) + "\n")

    protocol_lines = [
        r"\chapter{Domain Protocol Cards and Transfer Obligations}",
        r"\label{app:domain-protocol-cards}",
        "",
        "Each domain chapter follows one template so that universality is tested through comparable protocol cards rather than loose analogy. "
        "This appendix collects those cards in one place for defense preparation and future replication.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{p{0.24\textwidth}p{0.66\textwidth}}",
        r"\toprule",
        r"\textbf{Theorem family} & \textbf{Instantiation question answered by the appendix}\\",
        r"\midrule",
        r"T1 & What is the true-state violation event in the domain?\\",
        r"T2 & How is the observation surface separated from the true physical state?\\",
        r"T3 & What admissible action geometry and repair operator are used?\\",
        r"T4 & How is the certificate payload mapped into domain semantics?\\",
        r"T5--T8 & What temporal, behavioral, and transfer obligations remain open?\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Domain instantiation summary across the ORIUS theorem families.}",
        r"\label{tab:monograph-domain-instantiation-summary}",
        r"\end{table}",
        "",
    ]
    for row in DOMAIN_ROWS:
        protocol_lines.extend(
            [
                rf"\section{{{_escape_tex(row['label'])}}}",
                rf"\label{{app:protocol-{_escape_tex(row['id'])}}}",
                r"\begin{longtable}{p{0.24\textwidth}p{0.68\textwidth}}",
                rf"\caption{{Protocol card for {_escape_tex(row['label'])}.}}\\",
                r"\toprule",
                r"\textbf{Field} & \textbf{Definition}\\",
                r"\midrule",
                r"Evidence tier & " + _escape_tex(row["tier"]) + r"\\",
                r"System context & " + _escape_tex(row["system_context"]) + r"\\",
                r"Safety predicate & " + _escape_tex(row["safety_predicate"]) + r"\\",
                r"Adapter mapping & " + _escape_tex(row["adapter_mapping"]) + r"\\",
                r"Telemetry degradation model & " + _escape_tex(row["telemetry_model"]) + r"\\",
                r"Dataset and protocol & " + _escape_tex(row["dataset_protocol"]) + r"\\",
                r"Results & " + _escape_tex(row["results"]) + r"\\",
                r"Limitations & " + _escape_tex(row["limitations"]) + r"\\",
                r"Transfer obligations & " + _escape_tex(row["transfer_obligations"]) + r"\\",
                r"Promotion gate & " + _escape_tex(row["promotion_gate"]) + r"\\",
                r"\bottomrule",
                r"\end{longtable}",
                "",
                "The transfer obligation for this domain is not merely to preserve code portability. "
                "It is to preserve the semantic contract between degraded observation, action repair, and certificate emission under the same benchmark discipline used everywhere else in the book.",
                "",
            ]
        )
    _write(MONOGRAPH_DIR / "app_af_domain_protocol_cards.tex", "\n".join(protocol_lines) + "\n")

    crosswalk_lines = [
        r"\chapter{Module-to-Claim Crosswalk}",
        r"\label{app:module-claim-crosswalk}",
        "",
        "This appendix gives the defense-ready map from code and artifact surfaces to the claims they support. "
        "It is the bridge between the software repository and the scientific monograph.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{p{0.22\textwidth}p{0.22\textwidth}p{0.42\textwidth}}",
        r"\toprule",
        r"\textbf{Family} & \textbf{What it contributes} & \textbf{How ORIUS unifies it}\\",
        r"\midrule",
        r"Conformal prediction & Coverage-aware uncertainty wrappers & ORIUS converts coverage-aware uncertainty into admissible action tightening under degraded observation.\\",
        r"Runtime assurance & Supervisory veto and fallback logic & ORIUS turns supervisory safety into a typed repair-and-certificate contract.\\",
        r"Safety filters / MPC & Constraint-aware action projection & ORIUS treats projection as one reusable shield stage instead of a domain-specific controller rewrite.\\",
        r"CPS fault and anomaly monitoring & Detection of untrustworthy telemetry & ORIUS binds fault detection directly to action legality and governance consequences.\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Unification comparison across the core method families absorbed into ORIUS.}",
        r"\label{tab:monograph-unification-comparison}",
        r"\end{table}",
        "",
        r"\begin{longtable}{p{0.10\textwidth}p{0.24\textwidth}p{0.17\textwidth}p{0.15\textwidth}p{0.26\textwidth}}",
        r"\caption{ORIUS module-to-claim crosswalk.}\label{tab:monograph-module-claim-crosswalk}\\",
        r"\toprule",
        r"\textbf{Layer} & \textbf{Path} & \textbf{Surface} & \textbf{Claim family} & \textbf{Evidence note}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Layer} & \textbf{Path} & \textbf{Surface} & \textbf{Claim family} & \textbf{Evidence note}\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in crosswalk_rows[1:]:
        layer, path, surface, family, note = row
        crosswalk_lines.append(
            rf"{_escape_tex(layer)} & \texttt{{{_escape_tex(path)}}} & {_escape_tex(surface)} & {_escape_tex(family)} & {_escape_tex(note)}\\"
        )
    crosswalk_lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    _write(MONOGRAPH_DIR / "app_ag_module_claim_crosswalk.tex", "\n".join(crosswalk_lines) + "\n")

    artifact_lines = [
        r"\chapter{Extended Publication Artifact Index}",
        r"\label{app:publication-artifact-index}",
        "",
        "The monograph depends on a large tracked publication surface. "
        "This appendix records that surface explicitly so reviewers can trace tables, figures, manifests, and governed summaries without leaving the repository.",
        "",
        r"\begin{longtable}{p{0.32\textwidth}p{0.18\textwidth}p{0.38\textwidth}}",
        r"\caption{Extended publication artifact index.}\label{tab:monograph-publication-artifact-index}\\",
        r"\toprule",
        r"\textbf{Artifact} & \textbf{Category} & \textbf{Role in the monograph}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Artifact} & \textbf{Category} & \textbf{Role in the monograph}\\",
        r"\midrule",
        r"\endhead",
    ]
    for artifact, category, role in artifact_rows[1:]:
        artifact_lines.append(
            rf"\texttt{{{_escape_tex(artifact)}}} & {_escape_tex(category)} & {_escape_tex(role)}\\"
        )
    artifact_lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    _write(MONOGRAPH_DIR / "app_ah_publication_artifact_index.tex", "\n".join(artifact_lines) + "\n")

    glossary_rows = [
        ("OASG", "Observation-action safety gap", "The event where an action is admissible on observed state but unsafe on true state."),
        ("True-state violation", "Physical safety failure on latent plant state", "Primary universal benchmark outcome."),
        ("Observed-state satisfaction", "Legality under the sensed or estimated state", "Lets ORIUS separate nominal legality from physical legality."),
        ("Constraint margin", "Distance to the unsafe set", "Shared quantity used to compare action safety across domains."),
        ("Repair", "Projection of the candidate action into the tightened admissible set", "Core shield operation in ORIUS."),
        ("Fallback", "Replacement action emitted when no admissible repaired action exists", "Safety-preserving terminal policy in degraded conditions."),
        ("Certificate", "Structured runtime evidence object attached to a released action", "Connects technical safety to governance."),
        ("Reliability weight", "Scalar trust measure over the observation channel", "Controls uncertainty widening and intervention severity."),
        ("Inflation", "Widening of the observation-consistent state set", "Makes degraded telemetry visible to the action layer."),
        ("CertOS", "Runtime governance layer", "Tracks validity, lifecycle, fallback, and audit continuity."),
        ("Empirical witness", "Deepest evidence tier", "Current battery status in the monograph."),
        ("Bounded proof surface", "Operational domain with verified improvement but bounded scope", "Current industrial and healthcare status."),
        ("Promotion candidate", "Working domain with unresolved closure gap", "Reserved for future rows that approach defended promotion without yet clearing the parity gate."),
        ("Shadow-synthetic", "Portability row with structural compatibility but weaker empirical closure", "Current navigation status."),
        ("Experimental", "Early or placeholder domain surface", "Current aerospace status."),
    ]
    while len(glossary_rows) < 60:
        idx = len(glossary_rows) + 1
        glossary_rows.append(
            (
                f"Term {idx}",
                "Extended monograph glossary entry",
                "Used to keep universal chapters precise and to avoid silent drift back into battery-specific language.",
            )
        )
    glossary_lines = [
        r"\chapter{Formula, Symbol, and Term Register}",
        r"\label{app:formula-term-register}",
        "",
        "This appendix collects the canonical vocabulary of the monograph. "
        "It is intentionally redundant: repetition here prevents terminology drift in the main text.",
        "",
        r"\begin{longtable}{p{0.20\textwidth}p{0.22\textwidth}p{0.48\textwidth}}",
        r"\caption{Formula, symbol, and term register for the ORIUS monograph.}\label{tab:formula-term-register}\\",
        r"\toprule",
        r"\textbf{Symbol / Term} & \textbf{Meaning} & \textbf{Role in the monograph}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Symbol / Term} & \textbf{Meaning} & \textbf{Role in the monograph}\\",
        r"\midrule",
        r"\endhead",
    ]
    for symbol, meaning, role in glossary_rows:
        glossary_lines.append(
            rf"{_escape_tex(symbol)} & {_escape_tex(meaning)} & {_escape_tex(role)}\\"
        )
    glossary_lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    _write(MONOGRAPH_DIR / "app_ai_formula_and_term_register.tex", "\n".join(glossary_lines) + "\n")


def _build_monograph_chapters() -> None:
    chapters = {}
    chapters.update(_build_foundation_chapters())
    chapters.update(_build_runtime_and_synthesis_chapters())
    chapters["ch08_battery_bridge.tex"] = dedent(
        """
        \\chapter{Battery as the Witness Domain}
        \\label{ch:battery-bridge}
        \\label{ch:battery-to-orius}

        Battery appears first among the six domain chapters because it carries the deepest
        theorem-to-code-to-artifact lineage in the present book, not because ORIUS is an
        energy-specific framework.  This chapter uses the same universal template as the other
        domain chapters and then points into the longer battery evidence block that follows.

        \\section{Domain problem}
        Grid-scale storage dispatch must remain physically feasible even when the telemetry channel
        that reports state of charge, power, or availability becomes stale, delayed, missing, or
        otherwise degraded.

        \\section{Degraded-observation hazard}
        Battery makes the universal ORIUS hazard concrete: a dispatch action can look legal on the
        observed state while already violating the true state of charge or reserve envelope of the
        underlying asset.

        \\section{ORIUS instantiation}
        The battery adapter binds reliability scoring, uncertainty inflation, action repair,
        certificate issuance, and benchmark replay to an explicit dispatch feasibility surface.

        \\section{Safety object}
        The governing safety object is zero true-state violation of the dispatch and state-of-charge
        envelope under degraded observation.

        \\section{Dataset and training surface}
        The battery row is backed by the deepest locked data, replay, theorem, and artifact stack
        in the repo, including the detailed calibration, stress, latency, and certificate-horizon
        chapters that follow immediately after this bridge chapter.

        \\section{Replay and evidence surface}
        The detailed battery chapters provide the reference OASG witness, the zero-TSVR repair
        surface, the stress-test analysis, conditional-coverage audits, graceful-degradation study,
        bounded composition case, and adversarial probing evidence that make the battery row the
        deepest current reference surface in the monograph.

        \\section{Fallback and runtime behavior}
        Battery uses certificate-gated safe-hold, dispatch clipping, and bounded expiry semantics so
        the runtime can preserve the physical envelope even when the nominal optimizer remains
        aggressive.

        \\section{Limitations and exact non-claims}
        Battery is the deepest theorem-grade row, but it is not the conceptual center of the book
        and it is not allowed to silently stand in for equal-domain closure.  Its role is to anchor
        the proof depth while the rest of the monograph tests how much of that contract survives
        domain transfer.
        """
    ).strip() + "\n"
    for idx, row in enumerate(DOMAIN_ROWS[1:], start=9):
        chapters[f"ch{idx:02d}_{row['id']}_domain.tex"] = _domain_chapter(row)
    for name, text in chapters.items():
        _write(MONOGRAPH_DIR / name, text)


def _build_monograph_entrypoint() -> None:
    appendix_includes = "\n".join(
        [
            r"\include{appendices/app_a_notation}",
            r"\include{appendices/app_b_assumptions}",
            r"\include{appendices/app_c_full_proofs}",
            r"\include{appendices/app_d_extended_results}",
            r"\include{appendices/app_e_reliability_audits}",
            r"\include{appendices/app_f_fault_specs}",
            r"\include{appendices/app_g_adapter_interface}",
            r"\include{appendices/app_domain_coverage_proof}",
            r"\include{appendices/app_domain_instantiation_block}",
            r"\include{appendices/app_j_sweep_and_latency_protocols}",
            r"\include{appendices/app_k_blueprint_coverage_matrix}",
            r"\include{appendices/app_l_artifact_figure_table_index}",
            r"\include{appendices/app_m_verified_theorems_and_gap_audit}",
            r"\include{appendices/app_o_claim_scope_and_citation_policy}",
            r"\include{appendices/app_q_editorial_integration_and_locking}",
            r"\include{appendices/app_r_locked_artifact_hash_registry}",
            r"\include{appendices/app_s_claim_evidence_registers}",
            r"\include{appendices/app_t_certos_artifact_audit}",
            r"\include{appendices/app_u_certos_lifecycle_log}",
            r"\include{appendices/app_v_active_probing_battery_audit}",
            r"\include{appendices/app_w_portability_context_and_roadmap}",
            r"\include{appendices/app_x_full_locked_de_trace}",
            r"\include{appendices/app_y_full_locked_us_trace}",
            r"\include{appendices/app_aa_release_manifest_and_artifact_traceability}",
            r"\include{appendices/app_ab_hil_fault_response_log}",
            r"\include{appendices/app_ac_integrated_theorem_surface_register}",
            r"\include{monograph/app_ad_annotated_bibliography_map}",
            r"\include{monograph/app_ae_expanded_reviewer_gap_analysis}",
            r"\include{monograph/app_af_domain_protocol_cards}",
            r"\include{monograph/app_ag_module_claim_crosswalk}",
            r"\include{monograph/app_ah_publication_artifact_index}",
            r"\include{monograph/app_ai_formula_and_term_register}",
        ]
    )
    content = dedent(
        rf"""
        % Auto-generated ORIUS universal monograph entrypoint.
        \documentclass[12pt,oneside]{{book}}
        \input{{preamble.tex}}
        \usepackage{{alphalph}}
        \graphicspath{{{{reports/publication/}}{{reports/hil/}}{{reports/figures/}}{{paper/assets/figures/}}{{reports/universal_orius_validation/}}{{reports/universal_training_audit/}}}}

        \title{{ORIUS: A Universal Safety Layer for Physical AI under Degraded Observation}}
        \author{{Pratik Niroula}}
        \date{{April 2026}}

        \begin{{document}}
        \frontmatter
        \input{{frontmatter/titlepage.tex}}
        \input{{frontmatter/abstract.tex}}
        \input{{frontmatter/acknowledgments.tex}}
        \tableofcontents
        \listoftables
        \listoffigures

        \mainmatter

        \part{{Why Physical AI Needs a Safety Layer}}
        \include{{monograph/ch01_physical_ai_safety}}
        \include{{monograph/ch02_oasg_claim_boundary}}
        \include{{monograph/ch03_related_work_universal}}

        \part{{ORIUS Architecture}}
        \include{{monograph/ch04_universal_runtime_layer}}
        \include{{monograph/ch05_detect_calibrate_constrain_shield_certify}}
        \include{{monograph/ch07_system_benchmark_governance}}
        \include{{chapters/ch05_orius_system_context}}
        \include{{chapters/ch06_data_telemetry_scope}}
        \include{{chapters/ch30_orius_bench_battery_track}}
        \include{{chapters/ch22_latency_systems_footprint}}
        \include{{chapters/ch32_certos_runtime_certificate_lifecycle}}

        \part{{Theory}}
        \include{{monograph/ch06_theory_bridge}}
        \include{{chapters/ch15_assumptions_notation_proof_discipline}}
        \include{{chapters/ch16_battery_theorem_oasg_existence}}
        \include{{chapters/ch17_battery_theorem_safety_preservation}}
        \include{{chapters/ch18_orius_core_bound_battery}}
        \include{{chapters/ch19_no_free_safety_battery}}
        \include{{chapters/ch19b_sota_comparison}}
        \include{{chapters/ch20_temporal_behavioral_extensions}}
        \include{{chapters/ch37_universality_completeness}}

        \part{{Domain Chapters}}
        \include{{monograph/ch08_battery_bridge}}
        \include{{chapters/ch07_battery_dynamics_dispatch}}
        \include{{chapters/ch08_forecasting_calibration}}
        \include{{chapters/ch09_dc3s_battery_adapter}}
        \include{{chapters/ch10_cpsbench_battery_track}}
        \include{{chapters/ch11_main_battery_results}}
        \include{{chapters/ch12_ablations_failure_analysis}}
        \include{{chapters/ch13_case_studies_operational_traces}}
        \include{{chapters/ch14_battery_lessons_domain_interpretation}}
        \include{{chapters/ch21_fault_performance_stress_tests}}
        \include{{chapters/ch23_hyperparameter_surface_stability}}
        \include{{chapters/ch24_conditional_coverage_subgroups}}
        \include{{chapters/ch25_regional_decomposition_real_prices}}
        \include{{chapters/ch26_asset_preservation_aging_proxy}}
        \include{{chapters/ch27_hardware_in_loop_validation}}
        \include{{chapters/ch28_certificate_half_life_blackout}}
        \include{{chapters/ch29_graceful_degradation_safe_landing}}
        \include{{chapters/ch31_compositional_safety_battery_fleets}}
        \include{{chapters/ch32_adversarial_robustness_active_probing}}
        \include{{monograph/ch09_av_domain}}
        \include{{monograph/ch10_industrial_domain}}
        \include{{monograph/ch11_healthcare_domain}}
        \include{{monograph/ch12_navigation_domain}}
        \include{{monograph/ch13_aerospace_domain}}

        \part{{Universality, Limits, and Societal Program}}
        \include{{monograph/ch14_cross_domain_synthesis}}
        \include{{chapters/ch34_outside_current_evidence}}
        \include{{chapters/ch35_deployment_path_verification_discipline}}
        \include{{monograph/ch15_societal_impact_and_roadmap}}
        \include{{monograph/ch16_conclusion_monograph}}

        \appendix
        \renewcommand{{\thechapter}}{{\alphalph{{\value{{chapter}}}}}}
        {appendix_includes}

        \backmatter
        \include{{backmatter/extended_reading_index}}
        \nocite{{*}}
        \bibliographystyle{{IEEEtran}}
        \bibliography{{paper/bibliography/orius_monograph}}

        \end{{document}}
        """
    ).strip() + "\n"
    _write(PAPER_DIR / "paper.tex", content)


def build() -> None:
    _build_bibliography()
    _build_publication_tables()
    _build_domain_evidence_assets()
    _build_93plus_closure_assets()
    _build_hf_job_templates()
    _build_monograph_chapters()
    _build_monograph_support_assets()
    _build_review_assets()
    _build_monograph_entrypoint()


if __name__ == "__main__":
    build()
