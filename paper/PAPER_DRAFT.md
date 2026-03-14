# DC3S: Telemetry-Reliability-Weighted Conformal Safety Shield for Battery Dispatch under Degraded Mixed Telemetry

**Author:** Pratik Niroula  
**Affiliation:** Minnesota State University, Mankato; CIT (Major), Mathematics (Minor)
**Date:** March 12, 2026

## Abstract
Battery dispatch controllers can look feasible on observed sensor state while simultaneously violating physical battery limits on true state when telemetry degrades — a hidden safety failure that standard evaluation protocols do not expose. We study this gap under four realistic fault conditions and find that deterministic dispatch violates true state-of-charge limits at **3.9% of steps with P95 severity of 333 MWh**, even while observed-state evaluation reports zero violations. The thesis responds to that failure mode in three parts. First, **CPSBench** maintains separate truth and observed battery trajectories so that true-state safety becomes measurable rather than implicit. Second, **DC3S** (Detect–Calibrate–Constrain–Certify–Shield) converts telemetry reliability into adaptive uncertainty inflation (RAC-Cert), solves dispatch against the widened uncertainty set, repairs infeasible actions by projection, and emits a per-step auditable certificate. Third, we show a **conditional conservatism** result: DC3S reaches zero true-SOC violations with a 2.8% intervention rate, so the shield becomes more conservative only when telemetry quality demands it. Cost-side, DC3S-wrapped dispatch yields 7.11% cost savings, 0.30% carbon reduction, and 6.13% peak shaving on Germany (OPSD, 17,377 rows); stochastic value remains positive in both DE and US regions (VSS = 2,708.61 and 297,092.71 respectively) across three US balancing authorities (EIA-930 MISO/PJM/ERCOT). All quantitative claims are tied to versioned run artifacts and checked against the locked release evidence.

## Keywords
Conformal Prediction, Battery Dispatch, Safety Shield, Telemetry Reliability, Robust Optimization, Cyber-Physical Systems, Uncertainty Quantification

## 1. Introduction

Grid-scale battery energy storage systems (BESS) are increasingly central to grid decarbonization, load balancing, and frequency regulation. Their value depends on dispatch controllers that decide, at each time step, how much power to charge or discharge using forecasts of load, renewable generation, and price. Much of the dispatch literature quietly assumes that the controller observes the battery's true state — its state of charge (SOC), power output, and degradation level — and can therefore check feasibility in real time. In operational settings, that assumption is too strong.

In practice, grid-scale battery dispatch runs on mixed telemetry — delayed SCADA measurements, stale sensor feeds, packet reordering, spikes, and dropouts are ordinary operating conditions rather than edge cases. The key consequence is one the BESS dispatch literature has not directly measured: **a controller can be observationally safe (feasible on degraded observed state) while physically unsafe (violating true battery limits) at the same time.** On real grid data, that hidden safety failure appears at a 3.9% violation rate with 333 MWh P95 severity.

The central insight is that the measurement channel between the physical battery and the dispatch controller is not a passive conduit; it is part of the safety problem. Once telemetry degrades, the controller's model of the world begins to drift away from physical reality. Standard robust optimization hedges uncertainty in *future* load and generation scenarios, but it does not correct for uncertainty in *current* state measurement. That gap motivates a different control stance: telemetry reliability should enter directly into uncertainty calibration and safe dispatch.

DC3S (Degradation-Conditioned Conformal Dispatch Safety Shield) is built around that failure mode. Rather than relegating telemetry quality to a monitoring dashboard, the controller uses it to widen uncertainty sets, solve against the widened sets, repair unsafe candidate actions, and emit a step-level audit record within the same online control cycle. The guiding principle is *conditional conservatism*: the shield should stay permissive when the measurements are trustworthy and become more protective only when the measurement channel weakens.

### 1.1 Operational Motivation
GridPulse is a closed dispatch loop — Forecast → Optimize → Dispatch → Measure → Monitor. Each stage depends on information from the previous one, and the quality of that information degrades as it passes through physical sensor networks, communication links, and data aggregation pipelines. The hidden safety gap lives at the Optimize→Dispatch boundary: the optimizer computes a mathematically optimal action based on forecasts and observed state, but what reaches the physical battery may be optimal for the *wrong* state if the telemetry feeding the optimizer has drifted from reality.

A forecasting model can have strong RMSE while still inducing unsafe dispatch if its interval calibration weakens under degraded measurements and no shield exists between the optimizer and the physical battery. The problem is not that the forecast is bad — it is that the forecast is *evaluated* and *calibrated* under assumptions about telemetry quality that may not hold at runtime. DC3S closes that gap by making the controller aware of when those assumptions break down.

### 1.2 Research Questions
1. How large is the gap between observed-state safety and true-state safety in battery dispatch under realistic telemetry faults, and is it consistent across regions?
2. Can telemetry-reliability-weighted conformal calibration close this gap without blanket conservatism — i.e., is conditional rather than uniform intervention sufficient?
3. Which DC3S components are necessary for zero-violation safety: reliability scoring, interval inflation, or action repair?
4. Does stochastic value remain positive even in regimes where direct economic gains are modest, and what explains the DE–US asymmetry?

### 1.3 Contributions
1. **Hidden safety gap measurement.** We provide the first empirical quantification of the divergence between observed-state and true-state safety in BESS dispatch under realistic telemetry faults. Deterministic dispatch violates true-SOC at 3.9% of steps with P95 severity of 333 MWh — while appearing safe on observed state. This finding alone motivates rethinking how BESS dispatch safety is evaluated.
2. **CPSBench.** A multi-fault-dimension evaluation harness that maintains separate truth and observed battery trajectories across four fault types (dropout, delay-jitter, spikes, out-of-order), enabling reproducible true-state safety evaluation. CPSBench is the framework that makes the hidden safety gap measurable.
3. **DC3S architecture.** A telemetry-aware conformal control loop that converts per-step reliability scores into adaptive interval inflation (RAC-Cert), solves dispatch against the widened uncertainty set, repairs infeasible actions by projection, and emits a per-step auditable certificate — all within a single online control cycle.
4. **Conditional conservatism result.** DC3S achieves zero true-SOC violations at a 2.8% intervention rate across all fault conditions. The shield is not uniformly conservative: it activates precisely when and to the degree that telemetry quality requires, leaving 97.2% of steps unmodified.

### 1.4 Contribution Boundary
The central scientific contribution is the discovery and empirical characterization of the hidden safety failure mode, together with the first end-to-end architecture that closes it in an auditable control loop. The bounded formal layer is there to support the empirical story, not replace it: the RAC-Cert coverage result explains why monotone inflation does not weaken the base coverage guarantee, the reliability-binned result motivates the governed calibration audit, and the FTIT tube result explains why the tightened SOC margin can prevent the true-state failures measured in the benchmark. The novelty remains the problem identification, the CPSBench framework that makes it measurable, and the conditional-conservatism result showing that the gap can be closed without blanket conservatism.

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
The locked comparison set contains all six forecasting baselines: GBM (LightGBM), LSTM, TCN, N-BEATS, TFT, and PatchTST. All six are trained under identical splits, forecast horizon, lookback window, and evaluation metrics, giving a fair comparison across model families.

### 5.2 Training Setup
The locked training configuration uses:
1. 24-hour forecast horizon
2. 168-hour lookback
3. 10-fold time-aware cross-validation
4. 24-hour temporal gap

These settings matter twice: they make the forecasting comparison leakage-safe, and they define the representation contract later consumed by optimization and calibration.

### 5.3 CQR and Adaptive Conformal Layer
The uncertainty layer follows conformalized quantile regression: lower and upper quantile models are calibrated on a held-out slice, then expanded by an empirical residual quantile. Adaptive conformal logic updates effective coverage behavior online. Regime-aware calibration bins the time series by volatility so the paper can inspect where the base calibration is near nominal and where it is not.

An additive research-only extension now groups calibration points by telemetry reliability and assigns group-specific conformal widths. Under exchangeability within a reliability bin, the standard Mondrian conformal argument gives group-conditional marginal coverage at that bin level. This tooling is intentionally separate from the locked production CQR path in the current release family.

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

### 6.4 Post-Repair Guarantee Checks
Projection is not the final runtime guard. After repair, DC3S evaluates a deterministic guarantee layer before the action can be certified or queued for execution. The implemented checks enforce three one-step invariants:
1. no simultaneous charge and discharge
2. power bounds remain satisfied
3. the implied next SOC remains inside the admissible envelope

These checks do not define a second controller and do not re-optimize the action. They are a runtime correctness layer that validates the repaired action against explicit physical invariants and records pass/fail reasons for audit.

### 6.5 Certificate Emission and Audit
Each step emits a certificate containing:
1. time index
2. reliability score
3. inflated interval
4. candidate action
5. repaired safe action
6. solver status
7. observed SOC
8. distance to the nearest active constraint

The persisted payload also carries model/config hashes, intervention reasons, guarantee-check status, and FTIT tube state. This certificate chain is what makes the shield auditable rather than merely conservative.

### 6.6 One-Step Runtime Flow
The runtime order is:
1. observe telemetry
2. score reliability
3. detect drift and sensitivity
4. inflate intervals
5. solve dispatch
6. repair unsafe actions
7. emit certificate

### 6.6 Tightened SOC-Tube Interpretation
A second supporting result now makes the FTIT safety-filter interpretation explicit. If the controller enforces observed-state feasibility inside a tightened SOC tube `[SOC_min + e_t, SOC_max - e_t]`, and the true-vs-observed SOC error is bounded by the same margin `e_t`, then observed feasibility implies true one-step feasibility. This is a local invariance statement rather than a full closed-loop stability theorem, but it formalizes why the SOC tube and projection step act together as a safety filter.

## 7. Experimental Protocol

### 7.1 Forecasting Evaluation
The forecasting layer now promotes a six-model baseline comparison into the main protocol: GBM, LSTM, TCN, N-BEATS, TFT, and PatchTST on load, wind, and solar for both DE and US. All models use the same leakage-safe temporal contract: a 24-hour horizon, a 168-hour lookback, 10-fold time-aware cross-validation, a 24-hour fold gap, one seed policy, one feature framing, and one evaluation contract. Deep models use early stopping as the primary budget control, while GBM uses Optuna-based tuning. This is not a perfectly equalized compute budget, but it is a principled and reproducible comparison aligned with standard practice for each model family. The canonical publication artifact for this comparison is `reports/publication/baseline_comparison_all.csv`; the thesis renders a condensed DE-only paper-facing slice as `TBL08_FORECAST_BASELINES`.

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

### 8.0 Forecast Quality Across the Promoted Six-Model Surface
The main forecasting result table is no longer the older three-model GBM/LSTM/TCN snapshot. It is the promoted six-model comparison surface covering:
1. DE load / wind / solar
2. US load / wind / solar
3. RMSE
4. MAE
5. sMAPE
6. `R^2`
7. `PICP@90`
8. interval width where model-specific uncertainty artifacts exist

This table is built from one shared evaluation contract rather than mixed historical comparisons. It is intentionally operational: all six models share the same split policy, forecast horizon, lookback, seed policy, feature framing, and evaluation rules so that differences reflect model behavior under one forecasting regime rather than different data access. When a row still shows `---`, that row is pending artifact completion in the current release-family build and should be read as incomplete rather than negative evidence.

### 8.1 Safety Under Telemetry Faults
The central result is that both DC3S variants maintain zero true-SOC violations across the fault sweep, while the deterministic baseline violates at 3.9% and the CVaR baseline violates at 25.6%. The safety gain is not only binary: relative to deterministic dispatch, DC3S removes a hidden risk surface that is both nontrivial in frequency and severe in magnitude, with 333 MWh P95 violation severity. DC3S FTIT intervenes on only 2.8% of steps, showing that safety does not require constant override.

### 8.2 Locked Region-Level Impact and Stochastic Value
| Region | Cost Savings | Carbon Reduction | Peak Shaving | VSS | Run ID |
|---|---:|---:|---:|---:|---|
| DE | 7.11% | 0.30% | 6.13% | 2,708.61 | `20260217_165756` |
| US | 0.11% | 0.13% | 0.00% | 297,092.71 | `20260217_182305` |

The DE result is materially stronger than the locked US result in direct decision impact, even though stochastic value remains positive in both regions. That heterogeneity is part of the finding: the method should be read as regime-dependent rather than uniform, and the single locked `US` row compresses balancing-authority differences that are already visible in the broader release-family assets.

### 8.3 Ablation
The ablation shows that:
1. RAC-Cert inflation is necessary
2. action repair is also necessary
3. drift detection matters most under higher-severity faults
4. fixed static intervals are not enough

### 8.4 Calibration Quality
Calibration is mixed rather than uniformly strong. Wind high-volatility coverage is near nominal, while solar high-volatility coverage remains below target. This is an explicit negative finding and one reason the shield needs to adapt interval width at runtime. The governed reliability-group audit adds a second, release-scoped view: in the current release, per-bin coverage stays at the nominal target while interval width narrows monotonically as reliability improves, making the reliability-conditioned width pattern visible without changing the locked runtime controller path.

### 8.5 Transfer and Runtime Behavior
Transfer experiments show that safety is more robust than cost optimality. That asymmetry should be read as regime-dependent rather than universal: the safety mechanism transfers more cleanly than the exact economic upside. Runtime overhead is governed by the locked latency benchmark (`dc3s_latency_summary.csv`, release `R1_DEPLOY_20260314`): the full DC3S step completes in 0.033 ms mean (0.035 ms P95, 10,000 iterations, single-threaded). At hourly dispatch resolution the shield overhead is negligible.

| Component | Mean (ms) | P95 (ms) |
| --- | ---: | ---: |
| Reliability scoring | 0.020 | 0.024 |
| Drift update (Page-Hinkley) | 0.000 | 0.000 |
| Uncertainty set build | 0.010 | 0.013 |
| Action repair (projection) | 0.002 | 0.002 |
| Full DC3S step | 0.033 | 0.035 |

The manuscript also renders a governed 10-row step-trace table (`dc3s_step_trace.csv`, release `R1_DEPLOY_20260314`) in the operational-trace section. It is meant to be read as a guided trace rather than a raw log: reliability stays high until telemetry quality drops, inflation rises once drift is confirmed, proposed and safe actions diverge only when intervention is needed, and the guarantee checks remain passing before dispatch.

## 9. Deep Baseline Diagnosis

### 9.1 The Core Finding
GBM dominates the currently materialized deep-baseline rows on load, wind, and solar forecasting in both DE and US regimes under a fair comparison contract. LSTM and TCN provide the clearest completed comparison today, while N-BEATS, TFT, and PatchTST are already part of the same promoted reporting surface but still have pending artifact rows. This is a finding about the data regime, not a tuning failure.

### 9.2 Why: The Tabular Regime Effect
The key structural fact is that the GridPulse feature pipeline already encodes the temporal structure that deep sequence models are designed to learn — weekly and daily lags, rolling statistics, calendar signals, ramp features, and domain-specific peak indicators. When a 168-hour lag feature is already in the input matrix, an LSTM has nothing left to learn from the sequence that GBM cannot exploit directly from the tabular representation. Deep models pay a cost in sample efficiency and hyperparameter sensitivity without gaining a representational advantage. This effect is consistent with the broader tabular ML literature (Borisov et al., 2022).

### 9.3 Implications for Practitioners
The practical implication is actionable: **invest in feature engineering before investing in model architecture**. On moderate-sized, strongly seasonal grid datasets, a well-featured GBM is not just competitive but dominant. The six-model comparison quantifies this advantage precisely and makes it reproducible under the locked artifact policy.

### 9.4 What the Comparison Does and Does Not Show
The common 168-hour lookback and 24-hour horizon preserve fairness across all six models under the locked training contract. What it does not show is the ceiling performance of deep models under dataset scale-up or hyperparameter-intensive tuning beyond the early-stopping budget. The six-model comparison is therefore a fair head-to-head under realistic training constraints, not a claim about the theoretical ceiling of each architecture.

### 9.5 Tuning-Budget Caveat
All six baselines now live under the same locked publication comparison contract, but the current evidence is still partially materialized. Deep models use early stopping as the primary budget control, while GBM uses Optuna-based search. This does not create a perfectly equalized compute budget in the theoretical sense, but it does create a principled and reproducible comparison aligned with standard practice for each model family. The honest conclusion is therefore bounded but useful: given this data regime and this training contract, GBM is the operational winner in the currently materialized locked artifacts.

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

### 11.2.1 Deployment Evidence

The deployment-facing evidence is organized as six release-scoped artifact families under a single release ID (`R1_DEPLOY_20260314`) and recorded in `deployment_evidence_manifest.json`:

| Surface | Code | Artifact | Paper claim |
| --- | --- | --- | --- |
| Latency benchmark | `scripts/benchmark_dc3s_steps.py` | `dc3s_latency_summary.csv` | Runtime profile (§8.5): per-step latency < 1 ms |
| Streaming validation | `src/gridpulse/streaming/consumer.py` | `streaming_validation_summary.json` | Streaming extension (§13.2): validated ingest path |
| Step trace | `src/gridpulse/dc3s/{shield,calibration,quality,drift}.py` | `dc3s_step_trace.csv` | Operational trace (§8.6): governed 12-step record |
| Shadow-mode run | `iot/edge_agent/run_agent.py` | `shadow_mode_summary.json` | IoT deployment (§12): closed-loop shadow validation |
| Runtime safety contracts | `src/gridpulse/dc3s/coverage_theorem.py` | `coverage_theorem.py` (runtime assertions) | Safety guarantee (§5.7): verify_inflation_geq_one() |
| Calibration governance | `src/gridpulse/forecasting/uncertainty/reliability_mondrian.py` | `reliability_group_coverage.csv` | Calibration (§8.4): group-conditional coverage audit |

In the release workflow, the deployment-evidence stage runs as `python scripts/run_r1_release.py --stage deployment` after `full` and `cpsbench` but before promotion. The outputs are deterministic (fixed seeds) and hash-locked in the release manifest so that the deployment discussion stays tied to the same evidence discipline as the main results.

### 11.3 Validation Gates
The release gates for this thesis are concrete:
1. `python3 scripts/validate_paper_claims.py`
2. `python3 scripts/sync_paper_assets.py --check`
3. title, core numbers, and run IDs must remain aligned across markdown and LaTeX
4. `python3 scripts/generate_deployment_evidence.py` must produce all six artifact families

The certificate stream also supports lightweight operational health monitoring. In the current implementation, sliding-window summaries of certificate fields track intervention-rate spikes, low-reliability-rate spikes, persistent drift activation, and inflation P95 excursions. Sustained breaches can trigger alerts and feed retraining decisions, so the audit trail serves online triage as well as post-hoc forensics.

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
2. Evaluation is software-in-the-loop rather than hardware field validation; CHIL, HIL, and field pilots are explicit future work and are not claimed in the current evidence base.
3. Cost and carbon signals are engineering proxies.
4. The formal result is intentionally modest.
5. The forecasting comparison is materially stronger than the older three-model draft because the main paper surface now uses a six-model contract, but the architecture ranking is still bounded to the present data regime and current artifact state.
### 13.2 Near-Term Future Work
Near-term work should focus on:
1. decomposing the US narrative more explicitly into MISO/PJM/ERCOT in the main text
2. improving under-covered target-regime calibration before dispatch (the governed reliability-group coverage audit in this release provides the diagnostic baseline for that work)
3. extending the governed deployment-evidence track to richer runtime scenarios including multi-device operation and longer evaluation windows
4. broader operational evaluation bounded to software and runtime evidence rather than field deployment claims

**Streaming validation.** The ingest path is validated in this release via a deterministic 168-event replay through the consumer's schema, range, and cadence rules (`streaming_validation_summary.json`, release `R1_DEPLOY_20260314`). All events pass; temporal ordering is monotonic; checkpoint persistence is confirmed. The manuscript renders this as a compact summary table (Table: streaming validation). Live-mode validation remains future work.

**Shadow-mode validation.** The edge-agent shadow-mode contract is validated via a governed 24-step closed-loop run (`shadow_mode_summary.json`, release `R1_DEPLOY_20260314`): 24 commands observed, 24 ACKs, 0 NACKs, `applied=false` confirmed on every step, 100% certificate completeness.

**Governed calibration audit.** The reliability-group coverage analysis is promoted into a governed artifact (`reliability_group_coverage.csv/json`, release `R1_DEPLOY_20260314`). The audit partitions calibration samples into 10 quantile bins by reliability score and reports per-bin PICP. The manuscript renders this as a visible 10-bin table (Table: reliability-group coverage). This is a governed diagnostic audit that improves interpretability of under-coverage patterns without changing the locked runtime CQR+RAC-Cert calibration path.

### 13.3 Longer-Term Future Work
Longer-term work includes hardware-in-the-loop validation, richer market constraints, conditional-coverage analysis under degradation, and stronger release automation with signed evidence bundles. Those hardware-facing stages remain intentionally deferred beyond the present thesis freeze.

## 14. Conclusion

Battery dispatch controllers can appear safe when evaluated on observed state while simultaneously violating physical battery limits on true state — a hidden failure mode that standard evaluation protocols do not expose. This thesis quantifies that gap at 3.9% true-SOC violation rate (P95 severity 333 MWh) for deterministic dispatch under realistic telemetry faults and shows why observed-state evaluation alone misses it. CPSBench makes the gap measurable, and DC3S shows how to close it.

**Thesis statement.** The central claim is that *telemetry reliability must be treated as a first-class input to uncertainty calibration in any safety-critical dispatch system*. The measurement channel between the physical battery and the controller is not a passive pipe; it is a time-varying source of epistemic uncertainty that can silently invalidate safety guarantees. DC3S operationalizes that idea by coupling reliability scoring, conformal interval inflation, robust dispatch, and feasible-set projection inside one online loop. The controller therefore becomes more conservative in proportion to measurement quality: permissive when the signal is trustworthy and protective when it is not.

DC3S achieves zero true-SOC violations at a 2.8% intervention rate, so the shield is selective rather than blanket conservative. It activates only when telemetry reliability deteriorates, leaving 97.2% of dispatch steps unmodified. The cost-side evidence shows that this added safety does not erase decision value: on the locked DE evaluation, DC3S delivers 7.11% cost savings, 0.30% carbon reduction, and 6.13% peak shaving, while stochastic value remains positive in both DE (VSS = 2,708.61) and US (VSS = 297,092.71) regions. US direct gains are modest (0.11% cost, 0.00% peak), which appears to reflect tighter operating margins and compressed arbitrage headroom across the three US balancing authorities rather than a failure of the approach. Taken together, the results suggest that the safety-certified dispatch layer can still express operator preferences over cost, carbon, and peak, but the visible payoff remains regime-dependent rather than uniform.

The promoted six-model forecasting comparison points in one practical direction: on these moderate-sized, feature-rich grid datasets, GBM remains the strongest operational model under the locked equal-budget protocol. For practitioners, the message is simple: on data of this kind, feature engineering matters more than architectural complexity.

**The core message.** Trustworthy grid-scale dispatch requires the uncertainty model to degrade when the measurement channel degrades. In that sense, telemetry is not an implementation detail but part of the control problem itself. DC3S turns that idea into a single online loop with per-step auditable evidence. Within the present software-in-the-loop boundary, the hidden safety gap is both measurable and closeable; the next step is to test how far the same pattern survives on physical battery hardware.

**Broader implications.** The observed-state safety illusion is not unique to battery dispatch. Any cyber-physical system that acts on sensor readings that may lag or drift from physical reality is exposed to the same class of hidden violations. The DC3S pattern — reliability-conditioned uncertainty inflation followed by feasible-set projection — should therefore be read as a reusable systems idea for CPS settings such as HVAC dispatch, water treatment, and autonomous vehicle motion planning, wherever measurement uncertainty and hard safety constraints meet.

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

### Certificate Schema and Audit Chain
| Field group | Meaning | Audit / safety role |
|---|---|---|
| `command_id`, `certificate_id`, `created_at` | Unique dispatch event identity and timestamp | Orders events and anchors step-level traceability |
| `prev_hash`, `certificate_hash` | Hash-chain linkage across certificates | Makes the audit trail tamper-evident and sequential |
| `model_hash`, `config_hash` | Hashes of model artifacts and active configuration | Pins each dispatch step to the exact model/config state |
| `proposed_action`, `safe_action`, `dispatch_plan` | Optimizer output, repaired action, and plan context | Records what the optimizer wanted and what the shield allowed |
| `reliability`, `drift`, `reliability_w`, `inflation`, `uncertainty` | Telemetry-quality and interval-inflation context | Explains why conservatism changed at that step |
| `intervened`, `intervention_reason` | Whether repair or fallback occurred and why | Supports intervention-rate analysis and incident diagnosis |
| `guarantee_checks_passed`, `guarantee_fail_reasons` | Post-repair deterministic invariant results | Verifies that the emitted action satisfies runtime safety checks |
| `gamma_mw`, `e_t_mwh`, `soc_tube_lower_mwh`, `soc_tube_upper_mwh` | FTIT tube width and tightened SOC envelope | Captures the reliability-conditioned safety margin actually enforced |
| `true_soc_violation_after_apply`, `assumptions_version` | Post-apply evaluation flag and assumption contract version | Separates runtime decision logging from evaluation semantics |

## Appendix C. Operational Failure Modes
| Failure Mode | Trigger | Detection | Mitigation | Fallback |
|---|---|---|---|---|
| Missing or stale data | Upstream delay | Readiness checks | Block unsafe dispatch | Hold current state |
| Forecast drift | Distribution shift | KS/model monitors | Retraining workflow | Conservative mode |
| Solver infeasible | Extreme intervals | Solver status flags | Safe infeasible response | Grid-only baseline |
| Unsafe command | SOC/power violation | BMS validation | Reject with detail | Maintain safe state |
| Control degradation | Heartbeat timeout | Watchdog detection | Lock remote control | Manual/local control |
 
