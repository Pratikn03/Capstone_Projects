# ORIUS PhD Thesis 300p — Code Alignment

Maps thesis structure, theorems, and components to the implementation.

---

## 1. Thesis Structure → Code

| Thesis Part/Chapter | Code Path | Status |
|---------------------|-----------|--------|
| Ch 4: GridPulse Decision Loop | `src/orius/` pipeline | ✓ ORIUS (rebranded) |
| Ch 5: Data Pipeline, Forecasting | `src/orius/forecasting/`, `data_pipeline/` | ✓ |
| Ch 6: DC3S Shield | `src/orius/dc3s/` | ✓ |
| Ch 7: Theory (T1–T6, No Free Safety) | `src/orius/dc3s/` (coverage_theorem, temporal_theorems, etc.) | ✓ |
| Ch 8: CPSBench | `src/orius/cpsbench_iot/` | ✓ |
| Ch 9: Certificate Half-Life | `src/orius/dc3s/half_life.py`, `reachability.py` | ✓ |
| Ch 10: Graceful Degradation | `src/orius/dc3s/shield.py` (safe_landing), `certos/graceful_planner.py` | ✓ |
| Ch 11: ORIUS-Bench, CertOS | `src/orius/orius_bench/`, `src/orius/certos/` | ✓ |
| Ch 16: Building an ORIUS Adapter | `src/orius/dc3s/domain_adapter.py`, `universal_framework/` | ✓ |
| Ch 18: Cross-Domain Analysis | `src/orius/universal_framework/` | ✓ |

---

## 2. Five Domains (Thesis Ch 18) → Code

| Thesis Domain | Code Adapter | Path | Status |
|---------------|--------------|------|--------|
| 1. Battery Energy Storage | BatteryDomainAdapter | `dc3s/battery_adapter.py` | ✓ Locked |
| 2. Autonomous Vehicles | VehicleDomainAdapter | `vehicles/vehicle_adapter.py` | ✓ |
| 3. Surgical Robotics | HealthcareDomainAdapter | `universal_framework/healthcare_adapter.py` | ✓ (vital signs in OR) |
| 4. Aerospace | AerospaceDomainAdapter | `universal_framework/aerospace_adapter.py` | Placeholder |
| 5. Industrial Process Control | IndustrialDomainAdapter | `universal_framework/industrial_adapter.py` | ✓ |

**Note:** Thesis lists Surgical Robotics and Aerospace. We implement Healthcare (vital signs) which covers surgical monitoring; Aerospace is a placeholder for future work.

---

## 3. ORIUS-Bench Seven Metrics (Thesis Table 11.1) → Code

| Thesis Metric | Target | Code | Path |
|---------------|--------|------|------|
| M1 TSVR | ≤ 0.001 | `compute_tsvr` | `orius_bench/metrics_engine.py` |
| M2 OASG_frac | — | `compute_oasg` | `orius_bench/metrics_engine.py` |
| M3 PICP | ≥ 0.90 | coverage in CPSBench | `cpsbench_iot/metrics.py` |
| M4 GDQ | ≥ 0.85 | `compute_gdq` | `orius_bench/metrics_engine.py` |
| M5 IR | ≤ 0.05 | `compute_intervention_rate` | `orius_bench/metrics_engine.py` |
| M6 trace_complete | — | `compute_audit_completeness` | `orius_bench/metrics_engine.py` |
| M7 τ_recovery | ≤ 30 | `compute_recovery_latency` | `orius_bench/metrics_engine.py` |

---

## 4. CertOS Six Lifecycle Operations (Thesis §11.3) → Code

| Operation | Trigger | Code | Path |
|-----------|---------|------|------|
| ISSUE | New certificate | `CertificateEngine.issue` | `certos/certificate_engine.py` |
| VALIDATE | Check validity | `CertificateEngine.validate` | `certos/certificate_engine.py` |
| EXPIRE | Horizon exhausted | `CertificateEngine.expire` | `certos/certificate_engine.py` |
| RENEW | Re-issue | `CertificateEngine.renew` | `certos/certificate_engine.py` |
| REVOKE | Invalidate | `CertificateEngine.revoke` | `certos/certificate_engine.py` |
| FALLBACK | No valid cert | `CertificateEngine.fallback` | `certos/certificate_engine.py` |

---

## 5. Theorems (Thesis Ch 7) → Code

| Theorem | Code | Path |
|---------|------|------|
| T1 Universal OASG Existence | CPSBench fault injection | `cpsbench_iot/scenarios.py` |
| T2 ORIUS Safety Preservation | `check_tightened_soc_invariance`, `evaluate_guarantee_checks` | `dc3s/guarantee_checks.py`, `safety_filter_theory.py` |
| T3 Core Safety Bound | `assert_coverage_guarantee`, `reliability_error_bound` | `dc3s/coverage_theorem.py`, `safety_filter_theory.py` |
| T4 Certificate Validity Horizon | `certificate_validity_horizon`, `forward_tube` | `dc3s/temporal_theorems.py` |
| T5 Infinite-Horizon Renewal | `certificate_expiration_bound` | `dc3s/temporal_theorems.py` |
| T6 Graceful Degradation Dominance | `evaluate_graceful_degradation_dominance` | `dc3s/temporal_theorems.py` |
| No Free Safety | Baseline comparison in CPSBench | `cpsbench_iot/baselines.py`, `runner.py` |

---

## 6. Universal Framework Pipeline

| Thesis | Code |
|--------|------|
| Domain Adapter interface | `dc3s/domain_adapter.py` (ingest_telemetry, compute_oqe, build_uncertainty_set, tighten_action_set, repair_action, emit_certificate) |
| run_universal_step | `universal_framework/pipeline.py` |
| Domain registry | `universal_framework/domain_registry.py` |
| Five-stage pipeline | Detect → Calibrate → Constrain → Shield → Certify |

---

## 7. GridPulse → ORIUS

Thesis uses "GridPulse" in Ch 1, 4. The codebase uses **ORIUS** throughout. All references in `orius-plan/`, `paper/`, and `chapters/` use ORIUS.

---

## 8. Run Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_orius_full_check.py` | Full pipeline check including universal framework (R1) |
| `scripts/run_multi_domain_framework.py` | Run all five thesis domains (energy, av, surgical_robotics, aerospace, industrial) with `run_universal_step` |
| `scripts/run_orius_bench_release.py` | ORIUS-Bench across battery + navigation tracks, seven metrics |
| `scripts/run_certos_lifecycle.py` | CertOS six lifecycle operations demo |

---

## 9. Datasets by Domain

| Domain | Dataset Path | Download | Status |
|--------|--------------|----------|--------|
| Energy | `data/raw/opsd/`, `data/raw/us_eia930/`, `data/processed/` | OPSD, EIA-930 | ✓ Locked |
| AV | `data/av/processed/av_trajectories_orius.csv` | `make av-datasets` | ✓ Synthetic/real |
| Industrial | `data/industrial/processed/industrial_orius.csv` | `make industrial-datasets` | ✓ Synthetic/CCPP |
| Healthcare | `data/healthcare/processed/healthcare_orius.csv` | `make healthcare-datasets` | ✓ Synthetic/BIDMC |
| Surgical Robotics | Same as Healthcare | `make healthcare-datasets` | ✓ |
| Aerospace | — | — | Placeholder (no dataset) |
