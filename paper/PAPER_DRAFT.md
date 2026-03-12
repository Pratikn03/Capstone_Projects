# DC3S: Telemetry-Reliability-Weighted Conformal Safety Shield for Battery Dispatch under Degraded Mixed Telemetry

**Author:** Pratik Niroula  
**Affiliation:** Minnesota State University, Mankato; CIT (Major), Mathematics (Minor)
**Date:** March 12, 2026

## Abstract
Battery storage dispatch under degraded telemetry exposes a critical safety gap: when sensor readings are delayed, dropped, or corrupted, uncertainty estimates become miscalibrated and optimization solvers issue unsafe commands. We introduce DC3S (Detect-Calibrate-Constrain-Certify-Shield), a conformal safety shield that scores telemetry reliability at each control step, inflates prediction intervals via a Reliability-Adaptive Conformal Certification (RAC-Cert) law, projects candidate dispatch actions into the feasible safe set, and emits an auditable certificate. Evaluated on a CPSBench harness covering four fault dimensions (dropout, delay-jitter, spikes, out-of-order) across Germany (OPSD, 17,377 rows) and three US balancing authorities (EIA-930 MISO/PJM/ERCOT, about 13,500 rows each), DC3S achieves **zero true-SOC violations** with a 2.8% intervention rate while the deterministic baseline violates at 3.9% with P95 severity of 333 MWh. Cost-side, DC3S-wrapped dispatch yields 7.11% cost savings, 0.30% carbon reduction, and 6.13% peak shaving on DE; stochastic value of the stochastic solution is positive in both regions (DE VSS = 2,708.61, US VSS = 297,092.71). A nine-row ablation with Wilcoxon signed-rank gates isolates RAC-Cert inflation as the critical safety component. All quantitative claims are locked to versioned run artifacts and validated by an automated claim-consistency checker.

## Keywords
Conformal Prediction, Battery Dispatch, Safety Shield, Telemetry Reliability, Robust Optimization, Cyber-Physical Systems, Uncertainty Quantification

## 1. Introduction

### 1.1 Operational Motivation
Grid-scale battery storage is now expected to support renewable integration, peak management, and flexible dispatch under noisy operational conditions. In practice, however, battery dispatch is driven by mixed telemetry rather than pristine state information: delayed measurements, stale SCADA-style feeds, packet reordering, and occasional spikes or dropouts are ordinary operational conditions, not edge cases. Under those conditions, prediction intervals can become miscalibrated and optimization solvers can issue commands that look feasible on observed state while violating the true battery envelope.

GridPulse is built as a closed decision loop:

`Forecast -> Optimize -> Dispatch -> Measure -> Monitor`

The thesis studies the safety gap at the boundary between uncertainty estimation and control. The problem is not only to forecast well, but to decide safely when telemetry quality is degraded.

### 1.2 Research Questions
1. Can one architecture maintain strong forecasting quality across DE and multi-region US regimes while exposing uncertainty in a dispatch-usable form?
2. Does uncertainty-aware dispatch provide measurable value relative to deterministic dispatch under locked artifacts?
3. Which shield components are actually responsible for safety: telemetry scoring, interval inflation, or action repair?
4. Why do direct cost and peak gains differ so sharply across DE and US even when stochastic value remains positive in both regions?

### 1.3 Contributions
1. A telemetry-aware conformal control wrapper that converts reliability, drift, and sensitivity signals into widened dispatch intervals.
2. A safe-dispatch runtime that combines robust optimization, action repair, and step-level certification.
3. Locked empirical evidence showing zero-violation safety under telemetry faults together with pinned DE and US run artifacts.
4. A bounded governance contribution in which manifests, claim matrices, and validator gates are treated as part of the scientific object for this paper.

### 1.4 Contribution Boundary
The main scientific contribution is not a deep new conformal theorem. It is the coupling of telemetry reliability, uncertainty inflation, safe dispatch repair, and truth-state safety evaluation inside one auditable control loop. The formal RAC-Cert result in this manuscript is therefore presented as a supporting correctness proposition rather than as the primary novelty claim.

### 1.5 Paper Roadmap
Section 2 positions the work in the literature. Sections 3 and 4 define the system context and dispatch problem. Sections 5 and 6 describe the forecasting, calibration, and shield mechanics. Sections 7 through 10 present the evidence, baseline diagnosis, and cross-region analysis. Sections 11 through 14 close with governance, validity, limitations, and conclusion. Section 15 lists references, followed by appendices.

## 2. Related Work

### 2.1 Conformal Prediction and Time-Series Calibration
Classical conformal prediction (Vovk, Gammerman, and Shafer, 2005; Shafer and Vovk, 2008) provides the base coverage logic used by this thesis. Conformalized quantile regression (Romano, Patterson, and Candès, 2019) is the direct calibration template used in the forecasting layer. Adaptive conformal inference under distribution shift (Gibbs and Candès, 2021) and adaptive conformal prediction for time series (Zaffran et al., 2022) motivate the online and temporal components of the uncertainty layer. Weighted and generalized conformal approaches beyond exchangeability (Barber et al., 2023) are relevant because the thesis explicitly acknowledges that distributional assumptions weaken under degraded telemetry.

### 2.2 Robust, Stochastic, and Risk-Aware Dispatch
The dispatch layer draws directly on robust optimization (Bertsimas, Brown, and Caramanis, 2011; Ben-Tal, El Ghaoui, and Nemirovski, 2009), stochastic programming (Birge and Louveaux, 2011), and CVaR linearization (Rockafellar and Uryasev, 2000). In the BESS domain, Rosewater, Baldick, and Santoso (2020) are the closest direct comparator because they study risk-averse model predictive control for battery energy storage under uncertainty.

### 2.3 Safety Shields and Runtime Monitoring
Shielding work in safe reinforcement learning (Alshiekh et al., 2018; Kőnighofer et al., 2020) and model-predictive shielding (Bastani, 2021) motivates the projection-and-repair step. Runtime verification (Leucker and Schallhart, 2009) and simple runtime safety architectures (Sha, 2001) motivate the certificate and audit interpretation of the control path.

### 2.4 Telemetry Reliability and CPS Fault Models
The telemetry fault vocabulary used here—dropout, delay, spikes, and reordered observations—follows the systems-and-control view of CPS security and reliability (Dibaji et al., 2019) together with classical anomaly-detection framing (Chandola, Banerjee, and Kumar, 2009). The physical importance of safe BESS dispatch is grounded in grid-scale storage reviews such as Chen et al. (2020).

### 2.5 Reproducibility and Governance
The governance layer is deliberately narrow. It is informed by reproducibility work such as Pineau et al. (2021), but it is not claimed as a universal framework. The thesis only claims that for this project, claim locking and validator gates are necessary to prevent manuscript-artifact drift.

## 3. Background, Preliminaries, and System Context

### 3.1 GridPulse System Context
GridPulse is implemented as a forecast-to-dispatch stack whose main layers are data preparation, forecasting, uncertainty calibration, optimization, monitoring, and API/dashboard serving. DC3S sits between uncertainty estimation and command execution. It receives observed telemetry and calibrated intervals, widens those intervals when quality deteriorates, solves or wraps a dispatch action, repairs the action if necessary, and emits a certificate for audit.

### 3.2 Component-to-Code Map
| Layer | Purpose | Primary paths |
|---|---|---|
| Data pipeline | Region features, splits, and profile locks | `src/gridpulse/data_pipeline/`, `src/gridpulse/pipeline/run.py` |
| Forecasting | GBM and deep-baseline training/inference | `src/gridpulse/forecasting/` |
| Uncertainty | Conformal and adaptive interval logic | `src/gridpulse/forecasting/uncertainty/conformal.py` |
| Optimization | Deterministic, robust, and baseline dispatch | `src/gridpulse/optimizer/` |
| DC3S | Reliability, drift, shield, certificate, and state | `src/gridpulse/dc3s/` |
| Monitoring | Drift checks and retraining logic | `src/gridpulse/monitoring/` |
| API and dashboard | Runtime routes and operator-facing inspection | `services/api/`, `frontend/` |

### 3.3 Artifact Contracts and Profile Lock
The thesis uses a locked artifact family rather than free-floating report values. The main publication interfaces are:
1. `data/dashboard/de_metrics.json` and `data/dashboard/us_metrics.json`
2. `data/dashboard/de_stats.json`, `data/dashboard/us_stats.json`, and `data/dashboard/manifest.json`
3. `reports/impact_summary.csv` and `reports/eia930/impact_summary.csv`
4. `reports/research_metrics_de.csv` and `reports/research_metrics_us.csv`
5. publication tables and figures under `reports/publication/`

### 3.4 Dataset Scope
The canonical profile lock is:
1. DE (OPSD): 17,377 hourly rows
2. US dashboard profile: 13,638 hourly rows for the canonical manuscript lock
3. release-family supporting assets: US_MISO, US_PJM, and US_ERCOT dataset cards

This means the thesis preserves a single locked dashboard story for canonical quantitative claims while acknowledging the broader multi-BA release family in the surrounding analysis.

### 3.5 Truth vs. Observed State
The most important preliminaries distinction is between observed and true state:
1. `SOC_obs` is what the controller sees after telemetry degradation
2. `SOC_true` is the physical state used for safety evaluation

This is what allows the benchmark to expose hidden safety failures that would disappear if evaluation were done only on observed state.

## 4. Problem Formulation

### 4.1 Dispatch Setting
At each step `t`, the controller observes telemetry `z_t` containing load, wind, solar, price, and battery-state signals, possibly degraded by dropout, delay, spikes, or reordering. It then chooses a dispatch action `a_t` inside a feasible set constrained by SOC, power, and ramp limits.

### 4.2 Battery Dynamics and Feasible Set
The true battery state evolves under charge/discharge efficiency and capacity constraints. The feasible set enforces:
1. SOC lower and upper bounds
2. battery power limits
3. ramp constraints relative to the previous action

The thesis evaluates safety on `SOC_true`, not on the degraded observed state.

### 4.3 Objective
The dispatch objective combines:
1. energy cost
2. carbon-weighted cost
3. peak-demand penalty
4. battery-throughput degradation penalty

This is a comparative engineering objective used to evaluate decision quality under locked artifacts. It is not presented as a complete representation of all market settlement rules.

### 4.4 Calibration Gap
Base conformal prediction constructs intervals targeting a nominal coverage level. Under degraded telemetry, the conformity-score distribution shifts and the base intervals can become too narrow for safe control. This failure mode is the calibration gap addressed by DC3S.

### 4.5 Baseline Policies
Three baseline notions appear throughout the thesis:
1. `B1`: deterministic LP dispatch without uncertainty intervals
2. `B2`: grid-only dispatch with no battery action
3. `B3`: naive fixed-window battery heuristic

Canonical DE and US impact percentages are always interpreted as B1-vs-B2 comparisons. Stochastic metrics come from pinned robust-versus-deterministic run summaries rather than B2.

## 5. Forecasting and Uncertainty Method

### 5.1 Locked Forecasting Layer
The locked comparison set contains GBM (LightGBM), LSTM, and TCN. The broader training stack already supports N-BEATS, TFT, and PatchTST, but those models are candidate baselines for future release-family reruns rather than part of the canonical dashboard tables.

### 5.2 Training Setup
The locked training configuration uses:
1. 24-hour forecast horizon
2. 168-hour lookback
3. 10-fold time-aware cross-validation
4. 24-hour temporal gap

These settings matter twice: they make the forecasting comparison leakage-safe, and they define the representation contract later consumed by optimization and calibration.

### 5.3 CQR and Adaptive Conformal Layer
The uncertainty layer follows conformalized quantile regression: lower and upper quantile models are calibrated on a held-out slice, then expanded by an empirical residual quantile. Adaptive conformal logic updates effective coverage behavior online. Regime-aware calibration bins the time series by volatility so the paper can inspect where the base calibration is near nominal and where it is not.

### 5.4 RAC-Cert Inflation Law
RAC-Cert widens the base conformal interval according to:
1. telemetry reliability score `w_t`
2. drift signal `d_t`
3. sensitivity probe `s_t`

The inflation factor is clipped below by 1, so the shield never narrows the base interval in response to degraded telemetry.

### 5.5 Supporting Proposition and Contribution Boundary
**Supporting proposition.** If the base conformal predictor satisfies marginal coverage under the clean distribution, and RAC-Cert only inflates intervals so that the inflated interval contains the base interval, then marginal coverage cannot decrease under that inflation.

This is intentionally presented as a supporting correctness result rather than a deep theoretical contribution. The thesis novelty is the control coupling and empirical truth-state safety evidence, not the set-inclusion argument itself.

## 6. Safe Dispatch and DC3S Shield

### 6.1 Robust Dispatch
Given the RAC-inflated lower and upper trajectories for load, wind, and solar, DC3S solves a two-scenario robust dispatch problem over those widened bounds.

### 6.2 CVaR Comparator
For comparison, the experiments also include a CVaR baseline using the standard Rockafellar-Uryasev linearization. This helps separate telemetry-aware safe control from generic tail-risk optimization.

### 6.3 Action Repair
If the optimizer proposes an action that violates SOC, power, or ramp constraints, DC3S projects it back into the feasible set. This step is what turns a risk-aware optimizer into a safe controller.

### 6.4 Certificate Emission and Audit
Each step emits a certificate containing:
1. time index
2. reliability score
3. inflated interval
4. candidate action
5. repaired safe action
6. solver status
7. observed SOC
8. distance to the nearest active constraint

This certificate chain is what makes the shield auditable rather than merely conservative.

### 6.5 One-Step Runtime Flow
The runtime order is:
1. observe telemetry
2. score reliability
3. detect drift and sensitivity
4. inflate intervals
5. solve dispatch
6. repair unsafe actions
7. emit certificate

## 7. Experimental Protocol

### 7.1 Forecasting Evaluation
The forecasting layer compares GBM, LSTM, and TCN on the three primary targets: load, wind, and solar. All models use the same 24-hour horizon, 168-hour lookback, and temporal cross-validation framing.

### 7.2 Fault Injection Benchmark
CPSBench injects four telemetry fault families:
1. dropout
2. delay-jitter
3. spikes
4. out-of-order telemetry

All safety is evaluated on true SOC rather than observed SOC.

### 7.3 Controllers
Five controllers are compared:
1. deterministic LP
2. robust fixed-interval
3. CVaR interval
4. DC3S wrapped
5. DC3S FTIT

### 7.4 Metrics
The main metrics are:
1. true-SOC violation rate
2. P95 violation severity
3. intervention rate
4. cost delta
5. VSS and EVPI
6. PICP, pinball loss, and Winkler score

### 7.5 Statistical Protocol
Randomized experiments use 10 seeds. Safety comparisons use Wilcoxon signed-rank tests at `p < 0.01`. Ablations use both relative-reduction and significance gates.

### 7.6 Evidence Lock and Run Selection
The thesis binds its main stochastic claims to:
1. DE run `20260217_165756`
2. US run `20260217_182305`
3. dashboard profile lock timestamp `2026-02-17T11:15:38.623283`

## 8. Results

### 8.1 Safety Under Telemetry Faults
The central result is that both DC3S variants maintain zero true-SOC violations across the fault sweep, while the deterministic baseline violates at 3.9% and the CVaR baseline violates at 25.6%. DC3S FTIT intervenes on only 2.8% of steps, showing that safety does not require constant override.

### 8.2 Locked Region-Level Impact and Stochastic Value
| Region | Cost Savings | Carbon Reduction | Peak Shaving | VSS | Run ID |
|---|---:|---:|---:|---:|---|
| DE | 7.11% | 0.30% | 6.13% | 2,708.61 | `20260217_165756` |
| US | 0.11% | 0.13% | 0.00% | 297,092.71 | `20260217_182305` |

The DE result is materially stronger than the locked US result in direct decision impact, even though stochastic value remains positive in both regions.

### 8.3 Ablation
The ablation shows that:
1. RAC-Cert inflation is necessary
2. action repair is also necessary
3. drift detection matters most under higher-severity faults
4. fixed static intervals are not enough

### 8.4 Calibration Quality
Calibration is mixed rather than uniformly strong. Wind high-volatility coverage is near nominal, while solar high-volatility coverage remains below target. This is an explicit negative finding and one reason the shield needs to adapt interval width at runtime.

### 8.5 Transfer and Runtime Behavior
Transfer experiments show that safety is more robust than cost optimality. Runtime overhead remains modest, with reliability scoring, interval inflation, and action repair adding less than 5% to solver wall-clock time.

## 9. Deep Baseline Diagnosis

### 9.1 What the Locked Forecast Results Say
Across both DE and US dashboard artifacts, GBM dominates LSTM and TCN on load, wind, and solar. The thesis does not interpret this as a universal statement about sequence models. It interprets it as a regime-specific result of the locked data, feature design, and training budget.

### 9.2 Sample Size and Tabular Regime
The locked datasets are moderate in size and heavily feature-engineered. In that regime, boosted trees can exploit strong precomputed predictors with less sample complexity than the deep baselines, especially after the temporal split policy reduces the number of effective training windows.

### 9.3 Feature Representation and Inductive Bias
The deep models are not operating on raw sensor streams alone. They are operating after the pipeline has already encoded lagged structure, calendar variables, and exogenous weather signals into the tabular dataset. That reduces the relative advantage of architectures whose main strength is learning sequential structure from less engineered inputs.

### 9.4 Lookback, Horizon, and Tuning Caveats
The same 168-hour lookback and 24-hour horizon are used for all locked baselines. That keeps the comparison clean, but it may not be optimal for LSTM and TCN. The current deep-model results are also legacy operational baselines rather than exhaustive deep-learning sweeps. The honest thesis interpretation is therefore: GBM is the operational winner in the locked evidence, and the deep baselines were not tuned to the same maturity as the tabular stack.

## 10. Cross-Region and US Regime Analysis

### 10.1 Why US Direct Gains Are Weak
The weak US direct gains are real, not a formatting artifact. The locked US result is 0.11% cost savings and 0.00% peak shaving even though the pinned stochastic run reports VSS = 297,092.71.

### 10.2 Arbitrage Headroom and Tariff Structure
The first explanation is economic headroom. The locked DE setting presents larger direct arbitrage and peak-management opportunities than the locked US setting. A region can still have large stochastic value because uncertainty matters inside the optimization, while showing weak realized direct gains against a grid-only baseline over the specific evaluation window.

### 10.3 Battery Utilization and Constraint Binding
The second explanation is control leverage. If reserve, terminal-SOC, or power-cap constraints bind more often in the US runs, the optimizer has less freedom to convert better forecasts into visibly larger direct gains.

### 10.4 Calibration and Optimization Leverage
The third explanation is calibration asymmetry. Under-coverage in some US targets reduces the economic leverage of uncertainty-aware optimization because the system spends more of its effort compensating for uncertainty quality than exploiting favorable operating windows.

### 10.5 Multi-BA Interpretation
The thesis lock still uses the canonical US dashboard profile, while release-family assets already describe MISO, PJM, and ERCOT separately. That means “US” is partly a compressed narrative surface. A stronger next revision would expose those balancing-authority differences in the main results rather than only in supporting assets.

## 11. Governance and Reproducibility

### 11.1 Source of Truth
The canonical manuscript source is `paper/PAPER_DRAFT.md`. The corresponding LaTeX thesis manuscript is `paper/paper.tex`, and the shorter conference derivative is `paper/paper_r1.tex`. Quantitative claims are valid only if they remain traceable to `paper/metrics_manifest.json`, `paper/claim_matrix.csv`, and the locked report artifacts.

### 11.2 Repo-to-Paper Traceability
| Repo artifact family | Primary role in the paper | Main paper touchpoint |
|---|---|---|
| README + setup docs | Environment bootstrap, run commands, and operator setup | Governance and appendices |
| Train/eval code | Forecasting, uncertainty, dispatch, and CPSBench evaluation | Methods and experimental protocol |
| Configs + seeds | Dataset scope, model settings, and randomization policy | Experimental protocol and appendices |
| Data + preprocessing | Dataset construction, split logic, and feature generation | Background and experimental protocol |
| Logs + checkpoints | Run IDs, solver outputs, metrics, and locked evidence values | Results and governance |
| Plotting + publication scripts | Figures, tables, and paper-facing release assets | Results and appendices |

### 11.3 Validation Gates
The release gates for this thesis are concrete:
1. `python3 scripts/validate_paper_claims.py`
2. `python3 scripts/sync_paper_assets.py --check`
3. title, core numbers, and run IDs must remain aligned across markdown and LaTeX

### 11.4 What Governance Does Not Claim
The governance layer is scoped to this paper. It does not claim a universal reproducibility framework for ML research. It claims that, for a multi-artifact cyber-physical systems project, automated claim locking is necessary to keep the method story and the evidence story synchronized.

## 12. Discussion and Threats to Validity

### 12.1 Implications
The results support a systems-level claim: telemetry reliability should be treated as part of the control state, not as a peripheral monitoring variable. The safety win comes not from maximum conservatism but from conditional conservatism.

### 12.2 Negative and Neutral Findings
1. US peak shaving remains 0.00%.
2. Some target-regime combinations remain under-covered.
3. GBM dominates the locked deep baselines.
4. Cross-region impact magnitudes vary sharply despite the shared architecture.

### 12.3 Internal Validity
The main internal-validity risk is manuscript-artifact drift. The validator and claim matrix reduce that risk, but they do not eliminate it forever. Another internal-validity risk is that the diagnosis sections are based on locked artifacts and configuration evidence rather than new targeted reruns.

### 12.4 External Validity
The thesis is intentionally scoped to the locked DE and US windows. Generalization to different markets, battery sizes, or telemetry infrastructures remains future work.

### 12.5 Construct and Conclusion Validity
Construct validity depends on the proxy cost and carbon signals being appropriate enough for comparative analysis. Conclusion validity depends on scenario design and calibration definitions remaining stable. The RAC-Cert proposition also should not be overstated: it gives monotone coverage preservation under inflation, not a complete theory of conditional safety.

## 13. Limitations and Future Work

### 13.1 Current Limitations
1. Evidence is limited to locked DE and US windows.
2. Evaluation is software-in-the-loop rather than hardware field validation.
3. Cost and carbon signals are engineering proxies.
4. The formal result is intentionally modest.
5. The deep-baseline diagnosis is evidence-based but not yet backed by a new dedicated rerun campaign.

### 13.2 Near-Term Future Work
Near-term work should focus on:
1. rerunning deeper forecasting baselines under comparable tuning budgets
2. decomposing the US narrative more explicitly into MISO/PJM/ERCOT in the main text
3. improving under-covered target-regime calibration before dispatch

### 13.3 Longer-Term Future Work
Longer-term work includes hardware-in-the-loop validation, richer market constraints, conditional-coverage analysis under degradation, and stronger release automation with signed evidence bundles.

## 14. Conclusion
We presented DC3S, a conformal safety shield that integrates telemetry-reliability scoring, adaptive interval inflation (RAC-Cert), robust dispatch, action repair, and per-step certification into a single online loop for battery dispatch under degraded telemetry. Across four fault types and parametric severity sweeps on DE and US data, DC3S achieves zero true-SOC violations while maintaining positive stochastic value in both regions and 7.11% cost savings in DE. The corresponding locked US result is more modest at 0.11% cost savings and 0.00% peak shaving, which the thesis interprets as a regime-analysis problem rather than something to hide.

The practical message is that trustworthy dispatch requires telemetry-aware uncertainty calibration rather than robust optimization over assumed-correct uncertainty sets. The thesis contribution is therefore both operational and methodological: it shows how degraded telemetry can be converted into bounded uncertainty, safe repaired actions, and auditable evidence without overstating the formal theory behind the inflation rule.

## 15. References
1. Vovk, V., Gammerman, A., and Shafer, G. *Algorithmic Learning in a Random World*. Springer, 2005.
2. Shafer, G., and Vovk, V. “A Tutorial on Conformal Prediction.” *Journal of Machine Learning Research*, 2008.
3. Romano, Y., Patterson, E., and Candès, E. “Conformalized Quantile Regression.” *NeurIPS*, 2019.
4. Gibbs, I., and Candès, E. “Adaptive Conformal Inference Under Distribution Shift.” *NeurIPS*, 2021.
5. Zaffran, M., Féron, O., Goude, Y., Josse, J., and Dieuleveut, A. “Adaptive Conformal Predictions for Time Series.” *ICML*, 2022.
6. Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. “Conformal Prediction Beyond Exchangeability.” *Annals of Statistics*, 2023.
7. Bertsimas, D., Brown, D. B., and Caramanis, C. “Theory and Applications of Robust Optimization.” *SIAM Review*, 2011.
8. Ben-Tal, A., El Ghaoui, L., and Nemirovski, A. *Robust Optimization*. Princeton University Press, 2009.
9. Birge, J. R., and Louveaux, F. *Introduction to Stochastic Programming*. Springer, 2011.
10. Rockafellar, R. T., and Uryasev, S. “Optimization of Conditional Value-at-Risk.” *Journal of Risk*, 2000.
11. Rosewater, D., Baldick, R., and Santoso, S. “Risk-Averse Model Predictive Control Design for Battery Energy Storage Systems.” *IEEE Transactions on Smart Grid*, 2020.
12. Alshiekh, M., Bloem, R., Ehlers, R., Kőnighofer, B., Niekum, S., and Topcu, U. “Safe Reinforcement Learning via Shielding.” *AAAI*, 2018.
13. Bastani, O. “Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding.” *ACC*, 2021.
14. Kőnighofer, B., Lorber, F., Jansen, N., and Bloem, R. “Shield Synthesis for Reinforcement Learning.” *ISoLA*, 2020.
15. Sha, L. “Using Simplicity to Control Complexity.” *IEEE Software*, 2001.
16. Leucker, M., and Schallhart, C. “A Brief Account of Runtime Verification.” *Journal of Logic and Algebraic Programming*, 2009.
17. Dibaji, S. M., Pirani, M., Flamholz, D. B., Annaswamy, A. M., Johansson, K. H., and Chakrabortty, A. “A Systems and Control Perspective of CPS Security.” *Annual Reviews in Control*, 2019.
18. Chandola, V., Banerjee, A., and Kumar, V. “Anomaly Detection: A Survey.” *ACM Computing Surveys*, 2009.
19. Chen, T., Jin, Y., Lv, H., Yang, A., Liu, M., Chen, B., Xie, Y., and Chen, Q. “Applications of Lithium-Ion Batteries in Grid-Scale Energy Storage Systems.” *Transactions of Tianjin University*, 2020.
20. Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Larochelle, H. “Improving Reproducibility in Machine Learning Research.” *Journal of Machine Learning Research*, 2021.

## Appendix A. Replication Checklist
1. Run `python3 scripts/validate_paper_claims.py`
2. Run `python3 scripts/sync_paper_assets.py --check`
3. Build the LaTeX manuscripts from `paper.tex` and `paper_r1.tex`
4. Verify that the canonical numbers and run IDs appear consistently across markdown and LaTeX

## Appendix B. Artifact Inventory
| Artifact | Purpose |
|---|---|
| `paper/PAPER_DRAFT.md` | Canonical thesis manuscript source |
| `paper/paper.tex` | Thesis LaTeX twin of the markdown source |
| `paper/paper_r1.tex` | Shorter conference derivative |
| `paper/metrics_manifest.json` | Canonical metric lock and display contract |
| `paper/claim_matrix.csv` | Claim-level provenance and status tracking |
| `scripts/validate_paper_claims.py` | Cross-file claim validation |
| `scripts/sync_paper_assets.py` | Paper-asset synchronization audit |
| `reports/impact_summary.csv` | Locked DE impact values |
| `reports/eia930/impact_summary.csv` | Locked US impact values |
| `reports/research_metrics_de.csv` | DE stochastic summary rows |
| `reports/research_metrics_us.csv` | US stochastic summary rows |

## Appendix C. Operational Failure Modes
| Failure Mode | Trigger | Detection | Mitigation | Fallback |
|---|---|---|---|---|
| Missing or stale data | Upstream delay | Readiness checks | Block unsafe dispatch | Hold current state |
| Forecast drift | Distribution shift | KS/model monitors | Retraining workflow | Conservative mode |
| Solver infeasible | Extreme intervals | Solver status flags | Safe infeasible response | Grid-only baseline |
| Unsafe command | SOC/power violation | BMS validation | Reject with detail | Maintain safe state |
| Control degradation | Heartbeat timeout | Watchdog detection | Lock remote control | Manual/local control |
 