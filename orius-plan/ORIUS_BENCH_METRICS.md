# ORIUS-Bench Seven Standard Metrics (Paper 4)

**Source:** `src/orius/orius_bench/metrics_engine.py`  
**Claim:** paper §ORIUS-Bench, thesis Paper 4, `reports/orius_evidence_checklist_audit.md`

| ID | Name | Definition | Code |
|----|------|------------|------|
| M1 | TSVR | True-SOC Violation Rate | `compute_tsvr()` |
| M2 | OASG | Online Adaptation Speed Gap | `compute_oasg()` |
| M3 | CVA | Certificate Validity Accuracy | `compute_cva()` |
| M4 | GDQ | Graceful Degradation Quality | `compute_gdq()` |
| M5 | IR | Intervention Rate | `compute_intervention_rate()` |
| M6 | AC | Audit Completeness | `compute_audit_completeness()` |
| M7 | RL | Recovery Latency | `compute_recovery_latency()` |

**M7 (Recovery Latency):** Average number of steps to regain a valid certificate after expiry. Computed from `StepRecord` sequence by detecting expiry→recovery transitions.

**Export:** All seven metrics are written to `leaderboard.csv` and `artefact_bundle.json` by `orius_bench/export.py`.
