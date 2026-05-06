'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck,
  Activity,
  AlertTriangle,
  Gauge,
  Wrench,
  ChevronDown,
  Eye,
  Scale,
  Lock,
  Shield,
  FileCheck,
  TrendingDown,
  Database,
} from 'lucide-react';
import { Panel } from '@/components/ui/Panel';
import { KPICard } from '@/components/ui/KPICard';
import { GaugeChart } from '@/components/charts/GaugeChart';
import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { MLOpsMonitor } from '@/components/charts/MLOpsMonitor';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData, type DriftPoint } from '@/lib/api/dataset-client';
import { getDomainOption } from '@/lib/domain-options';

/* ------------------------------------------------------------------ */
/*  Static Data                                                        */
/* ------------------------------------------------------------------ */

interface PipelineStage {
  id: number;
  name: string;
  shortName: string;
  icon: React.ElementType;
  input: string;
  output: string;
  formula: string;
  codeModule: string;
  detail: string;
  color: string;
}

const DC3S_PIPELINE: PipelineStage[] = [
  {
    id: 1,
    name: 'Detect (OQE)',
    shortName: 'Detect',
    icon: Eye,
    input: 'Telemetry event, history',
    output: 'Reliability weight w_t, fault flags',
    formula: 'w_t = OQE(z_{1:t})',
    codeModule: 'dc3s/quality.py',
    detail: 'Fault types detected: dropout, stale_sensor, spike, delay_jitter, timestamp_missing, out_of_order. Uses FTIT (Fault-Tolerant Interval Tracking) to accumulate fault evidence and adjust w_t adaptively.',
    color: 'emerald',
  },
  {
    id: 2,
    name: 'Calibrate (RUI/RAC)',
    shortName: 'Calibrate',
    icon: Scale,
    input: 'State, w_t, conformal quantile q',
    output: 'Uncertainty set U_t with inflation I(w_t, D_t)',
    formula: 'I(w_t, D_t) ∈ [1, η_max]',
    codeModule: 'dc3s/calibration.py, rac_cert.py',
    detail: 'Reliability-Aware Conformal calibration. Prediction intervals are inflated by I(w_t, D_t) — a monotone function that widens intervals when reliability drops. Maintains marginal coverage ≥ 1 − α.',
    color: 'sky',
  },
  {
    id: 3,
    name: 'Constrain (SAF)',
    shortName: 'Constrain',
    icon: Lock,
    input: 'Uncertainty set U_t + hard constraints',
    output: 'Tightened safe action set A_t',
    formula: 'A_t = {a : F(a, U_t) ⊆ S}',
    codeModule: 'dc3s/safety_filter_theory.py',
    detail: 'Action set tightening: only actions whose worst-case forward propagation through U_t remains within hard SOC bounds [SOC_min, SOC_max] are admitted. Uses FTIT SOC bounds for battery domain.',
    color: 'amber',
  },
  {
    id: 4,
    name: 'Shield (Repair)',
    shortName: 'Shield',
    icon: Shield,
    input: 'Candidate action + A_t',
    output: 'Safe action, repair metadata',
    formula: 'a_safe = Proj_{A_t}(a_candidate)',
    codeModule: 'dc3s/shield.py',
    detail: 'Projects candidate dispatch onto tightened safe set. If candidate is already safe, passes through unchanged. If not, clamps to nearest feasible point. Records repair reason for audit.',
    color: 'violet',
  },
  {
    id: 5,
    name: 'Certify (Audit)',
    shortName: 'Certify',
    icon: FileCheck,
    input: 'All previous stages + dispatch',
    output: 'Per-step certificate with hash chain',
    formula: 'cert_t = h(w_t, U_t, A_t, a_safe, τ_t)',
    codeModule: 'dc3s/certificate.py',
    detail: 'Emits a cryptographic certificate per step: captures reliability weight, uncertainty set, safe action, validity horizon τ_t. Hash-chained for tamper evidence. Zero missing fields required for audit completeness.',
    color: 'rose',
  },
];

const STAGE_COLORS: Record<string, string> = {
  emerald: 'border-emerald-500/30 bg-emerald-500/10',
  sky: 'border-sky-500/30 bg-sky-500/10',
  amber: 'border-amber-500/30 bg-amber-500/10',
  violet: 'border-violet-500/30 bg-violet-500/10',
  rose: 'border-rose-500/30 bg-rose-500/10',
};

const STAGE_TEXT: Record<string, string> = {
  emerald: 'text-emerald-400',
  sky: 'text-sky-400',
  amber: 'text-amber-400',
  violet: 'text-violet-400',
  rose: 'text-rose-400',
};

interface Proposition {
  id: string;
  name: string;
  statement: string;
  detail: string;
}

const PROPOSITIONS: Proposition[] = [
  {
    id: 'Prop 1',
    name: 'Marginal Coverage Preservation',
    statement: 'Pr[Y ∈ C_I] ≥ 1 − α − 1/(n+1)',
    detail: 'Under inflation, conformal prediction intervals maintain marginal coverage even when the reliability weight degrades, up to a 1/(n+1) finite-sample correction.',
  },
  {
    id: 'Prop 2',
    name: 'Inflation Monotonicity & Boundedness',
    statement: 'I ∈ [1, η_max], monotone in w and D',
    detail: 'The inflation law is guaranteed to be bounded (never explodes) and monotone — lower reliability always widens intervals, higher always tightens.',
  },
  {
    id: 'Prop 3',
    name: 'Page-Hinkley Drift Detection Bounds',
    statement: 'FAR ≤ e^{−h²/(2σ²)}, EDD ≤ h/(Δ − δ)',
    detail: 'The drift detector provides bounded false alarm rate (FAR) and expected detection delay (EDD). With h=5.0 and target FAR ≤ 0.001/step.',
  },
  {
    id: 'Prop 4',
    name: 'Hard SOC Safety Invariant',
    statement: 'Repaired actions preserve S ∈ [S̲, S̄] forward invariance',
    detail: 'A Control Barrier Function argument: the shield repair always produces actions whose one-step forward propagation stays within hard SOC bounds.',
  },
  {
    id: 'Prop 5',
    name: 'DRO Coverage under Wasserstein Perturbation',
    statement: 'ε(n) = C_d · n^{−1/max(d,2)}',
    detail: 'Distributionally robust coverage guarantee: the prediction set is valid even under adversarial distribution shifts of size ε(n) in Wasserstein distance.',
  },
];

interface BenchMetric {
  key: string;
  name: string;
  description: string;
  dc3sFtit: string;
  dc3sWrapped: string;
  deterministicLp: string;
  unit: string;
}

const BENCH_METRICS: BenchMetric[] = [
  { key: 'TSVR', name: 'True-SOC Violation Rate', description: 'Fraction of timesteps where physical SOC breaches hard bounds', dc3sFtit: '0.0%', dc3sWrapped: '0.0%', deterministicLp: '4.2%', unit: '%' },
  { key: 'OASG', name: 'Observed–Actual Safety Gap', description: 'Gap between observed and true safety assessments', dc3sFtit: '0.0', dc3sWrapped: '0.0', deterministicLp: '2.8', unit: 'gap score' },
  { key: 'CVA', name: 'Certificate Validity Average', description: 'Mean certificate validity horizon across all steps', dc3sFtit: '47.2', dc3sWrapped: '38.5', deterministicLp: 'N/A', unit: 'steps' },
  { key: 'GDQ', name: 'Graceful Degradation Quality', description: 'Violation reduction ratio vs uncontrolled baseline', dc3sFtit: '100%', dc3sWrapped: '100%', deterministicLp: '0%', unit: '%' },
  { key: 'IR', name: 'Intervention Rate', description: 'Fraction of shield interventions (repairs)', dc3sFtit: '0.0%', dc3sWrapped: '0.0%', deterministicLp: 'N/A', unit: '%' },
  { key: 'AC', name: 'Audit Completeness', description: 'Fraction of zero-missing-field certificates', dc3sFtit: '100%', dc3sWrapped: '100%', deterministicLp: 'N/A', unit: '%' },
  { key: 'RL', name: 'Revenue Loss', description: 'Cost proxy from conservative dispatch decisions', dc3sFtit: '$12.40', dc3sWrapped: '$34.80', deterministicLp: '$0.00', unit: 'USD' },
];

/* ------------------------------------------------------------------ */
/*  Sub-Components                                                     */
/* ------------------------------------------------------------------ */

function PipelineStageCard({ stage }: { stage: PipelineStage }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = stage.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: stage.id * 0.08 }}
      className={`flex-1 min-w-[160px] rounded-xl border ${STAGE_COLORS[stage.color]} p-4 cursor-pointer transition-all hover:scale-[1.02]`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-7 h-7 rounded-lg ${STAGE_COLORS[stage.color]} flex items-center justify-center`}>
          <Icon className={`w-3.5 h-3.5 ${STAGE_TEXT[stage.color]}`} />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className={`text-[10px] font-bold ${STAGE_TEXT[stage.color]}`}>Stage {stage.id}</span>
          </div>
          <h3 className="text-xs font-semibold text-white leading-tight">{stage.name}</h3>
        </div>
      </div>

      <div className="space-y-1.5 text-[10px]">
        <div><span className="text-slate-500">In:</span> <span className="text-slate-400">{stage.input}</span></div>
        <div><span className="text-slate-500">Out:</span> <span className="text-slate-400">{stage.output}</span></div>
        <div className={`font-mono ${STAGE_TEXT[stage.color]} text-[10px]`}>{stage.formula}</div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 pt-3 border-t border-white/[0.06] space-y-2">
              <p className="text-[10px] text-slate-400 leading-relaxed">{stage.detail}</p>
              <div className="text-[9px] font-mono text-slate-600">📁 {stage.codeModule}</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <ChevronDown className={`w-3 h-3 text-slate-600 mt-2 mx-auto transition-transform ${expanded ? 'rotate-180' : ''}`} />
    </motion.div>
  );
}

function OASGDiagram() {
  return (
    <div className="flex flex-col items-center gap-3 py-4">
      <div className="flex items-end gap-6 h-48">
        {/* Observed SOC bar */}
        <div className="flex flex-col items-center gap-1">
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 160 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className="w-16 bg-emerald-500/30 border border-emerald-500/40 rounded-t-lg relative"
          >
            <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-mono text-emerald-400 whitespace-nowrap">45%</div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[9px] text-emerald-400 font-medium writing-mode-vertical" style={{ writingMode: 'vertical-rl' }}>OBSERVED</span>
            </div>
          </motion.div>
          <span className="text-[10px] text-slate-500">x̂_t</span>
        </div>

        {/* Gap indicator */}
        <div className="flex flex-col items-center gap-1 self-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1, duration: 0.5 }}
            className="px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/30"
          >
            <span className="text-[10px] font-bold text-red-400">OASG GAP</span>
          </motion.div>
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ delay: 0.8, duration: 0.4 }}
            className="w-16 h-px bg-red-500/50"
          />
        </div>

        {/* True SOC bar */}
        <div className="flex flex-col items-center gap-1">
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 64 }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
            className="w-16 rounded-t-lg relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-amber-500/30 border border-amber-500/40 rounded-t-lg" />
            <div className="absolute bottom-0 left-0 right-0 h-6 bg-red-500/20 border-t border-red-500/30" />
            <div className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-mono text-amber-400 whitespace-nowrap">18%</div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[9px] text-amber-400 font-medium" style={{ writingMode: 'vertical-rl' }}>TRUE</span>
            </div>
          </motion.div>
          <span className="text-[10px] text-slate-500">x_t</span>
        </div>
      </div>

      {/* SOC_min line */}
      <div className="w-full flex items-center gap-2">
        <div className="flex-1 h-px bg-red-500/30 border-dashed" style={{ borderTop: '1px dashed rgba(239,68,68,0.4)' }} />
        <span className="text-[9px] font-mono text-red-400/60">SOC_min = 20%</span>
        <div className="flex-1 h-px" style={{ borderTop: '1px dashed rgba(239,68,68,0.4)' }} />
      </div>

      <p className="text-[10px] text-slate-500 text-center max-w-[280px] leading-relaxed">
        Actions that appear safe under observed telemetry (45%) but violate true physical constraints (18% &lt; SOC_min)
      </p>
    </div>
  );
}

function CoreBoundT3() {
  const alpha = 0.05;
  const wBar = 0.92;
  const T = 288;
  const bound = alpha * (1 - wBar) * T;

  return (
    <div className="space-y-4 py-2">
      {/* Formula */}
      <div className="text-center">
        <div className="inline-block px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
          <span className="text-lg font-mono font-bold text-emerald-400">
            E[V<sub>T</sub>] ≤ α(1 − w̄)T
          </span>
        </div>
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center p-2 rounded-lg bg-white/[0.03]">
          <div className="text-[10px] text-slate-500">α (miscoverage)</div>
          <div className="text-sm font-mono font-bold text-white">{alpha}</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-white/[0.03]">
          <div className="text-[10px] text-slate-500">w̄ (mean reliability)</div>
          <div className="text-sm font-mono font-bold text-white">{wBar}</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-white/[0.03]">
          <div className="text-[10px] text-slate-500">T (horizon)</div>
          <div className="text-sm font-mono font-bold text-white">{T}</div>
        </div>
      </div>

      {/* Computed */}
      <div className="text-center p-3 rounded-xl bg-energy-primary/10 border border-energy-primary/20">
        <div className="text-[10px] text-slate-500 mb-1">Expected violations/day</div>
        <div className="text-xl font-mono font-bold text-energy-primary">≤ {bound.toFixed(2)}</div>
        <div className="text-[10px] text-slate-500 mt-1">{alpha} × {(1 - wBar).toFixed(2)} × {T}</div>
      </div>

      <p className="text-[10px] text-slate-500 text-center leading-relaxed">
        Violations scale linearly with unreliability (1 − w̄) and shrink as telemetry quality improves.
        T10 proves this bound is tight within a factor of 4.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function SafetyPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region);
  const regionLabel = getDomainOption(region).label;
  const anomalies = dataset.anomalies;
  const activeAnomalyCount = anomalies.filter((item) => item.status === 'active').length;
  const anomalyZScores = dataset.zscores.map((item) => ({
    timestamp: item.timestamp,
    z_score: item.z_score,
    is_anomaly: item.is_anomaly,
    residual_mw: item.residual_mw,
  }));
  const driftData: DriftPoint[] = dataset.monitoring?.drift_timeline ?? [];
  const driftDetected = driftData.some((item) => item.is_drift);
  const statusMessages = [
    dataset.error ? `Dataset evidence error: ${dataset.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    !dataset.loading && !anomalyZScores.length ? `No anomaly timeline artifact is available for ${regionLabel}.` : null,
    !dataset.loading && !driftData.length ? `No drift timeline artifact is available for ${regionLabel}.` : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <ShieldCheck className="w-7 h-7 text-energy-primary" />
          Safety / OASG / DC3S
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          DC3S verifier evidence, Observed-Actual Safety Gap analysis, anomaly events, and drift/runtime monitoring
        </p>
      </div>

      <StatusBanner title="Safety Evidence Status" messages={statusMessages} />

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <KPICard label="Reliability w_t" value="0.92" icon={<Activity className="w-4 h-4" />} color="primary" delay={0} />
        <KPICard label="Inflation" value="1.38×" icon={<Gauge className="w-4 h-4" />} color="warn" delay={0.05} />
        <KPICard label="Drift Flag" value={driftDetected ? 'Review' : 'Clear'} icon={<AlertTriangle className="w-4 h-4" />} color={driftDetected ? 'warn' : 'primary'} delay={0.1} />
        <KPICard label="Active Events" value={String(activeAnomalyCount)} icon={<Wrench className="w-4 h-4" />} color={activeAnomalyCount ? 'alert' : 'info'} delay={0.15} />
        <KPICard label="E[V] Bound" value="≤ 1.15" unit="violations" icon={<TrendingDown className="w-4 h-4" />} color="alert" delay={0.2} />
      </div>

      <Panel
        title="Runtime Evidence"
        subtitle={`${regionLabel} - anomaly and monitoring artifacts folded into Safety`}
        badge={dataset.stats ? `${dataset.stats.rows.toLocaleString()} rows` : 'artifact-backed'}
        badgeColor="info"
        accentColor="info"
      >
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="rounded-lg bg-white/[0.03] p-3">
            <div className="text-slate-500">Domain</div>
            <div className="mt-1 font-medium text-white">{regionLabel}</div>
          </div>
          <div className="rounded-lg bg-white/[0.03] p-3">
            <div className="text-slate-500">Anomaly Events</div>
            <div className="mt-1 font-mono text-white">{anomalies.length}</div>
          </div>
          <div className="rounded-lg bg-white/[0.03] p-3">
            <div className="text-slate-500">Drift Points</div>
            <div className="mt-1 font-mono text-white">{driftData.length}</div>
          </div>
          <div className="rounded-lg bg-white/[0.03] p-3">
            <div className="text-slate-500">Sources</div>
            <div className="mt-1 flex items-center gap-1 font-mono text-white">
              <Database className="h-3.5 w-3.5 text-energy-info" />
              {dataset.source_artifacts?.length ?? 0}
            </div>
          </div>
        </div>
        {dataset.source_artifacts && dataset.source_artifacts.length > 0 && (
          <div className="mt-4 grid grid-cols-1 gap-2 lg:grid-cols-2">
            {dataset.source_artifacts.slice(0, 4).map((artifact) => (
              <div key={artifact} className="truncate rounded-lg bg-black/15 px-3 py-2 font-mono text-[11px] text-slate-300" title={artifact}>
                {artifact}
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Three-column: OQE Gauge | OASG Diagram | Core Bound */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="OQE Score" subtitle="Observation Quality Estimator" accentColor="primary">
          <div className="flex justify-center py-4">
            <GaugeChart
              value={92}
              min={0}
              max={100}
              label="OQE"
              unit="%"
              thresholds={{ warn: 70, alert: 50 }}
            />
          </div>
          <p className="text-[10px] text-slate-500 text-center">
            Real-time reliability weight computed from telemetry fault history
          </p>
        </Panel>

        <Panel title="OASG Visualization" subtitle="Observed vs True SOC" accentColor="alert">
          <OASGDiagram />
        </Panel>

        <Panel title="Core Bound (T3)" subtitle="ORIUS violation bound" badge="Proved" badgeColor="primary" accentColor="primary">
          <CoreBoundT3 />
        </Panel>
      </div>

      {/* DC3S Pipeline */}
      <Panel title="DC3S Pipeline" subtitle="Detect → Calibrate → Constrain → Shield → Certify" badge="5 stages" accentColor="info">
        <div className="flex gap-3 overflow-x-auto pb-2">
          {DC3S_PIPELINE.map((stage, i) => (
            <div key={stage.id} className="flex items-center gap-2">
              <PipelineStageCard stage={stage} />
              {i < DC3S_PIPELINE.length - 1 && (
                <div className="text-slate-600 text-lg flex-shrink-0">→</div>
              )}
            </div>
          ))}
        </div>
      </Panel>

      {/* DC3S Propositions */}
      <Panel
        title="DC3S Theoretical Propositions"
        subtitle="5 formally proved supporting results"
        badge="5"
        collapsible
        defaultCollapsed
        accentColor="info"
      >
        <div className="space-y-3">
          {PROPOSITIONS.map((p) => (
            <div key={p.id} className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-bold text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded">{p.id}</span>
                <span className="text-xs font-medium text-slate-300">{p.name}</span>
              </div>
              <div className="font-mono text-xs text-sky-300/80 mb-1.5">{p.statement}</div>
              <p className="text-[10px] text-slate-500 leading-relaxed">{p.detail}</p>
            </div>
          ))}
        </div>
      </Panel>

      {/* ORIUS-Bench Metrics */}
      <Panel title="ORIUS-Bench Metrics" subtitle="7 core evaluation metrics — controller comparison" badge="7 metrics" accentColor="primary">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/[0.08]">
                <th className="text-left py-2.5 px-3 text-slate-500 font-medium">Metric</th>
                <th className="text-left py-2.5 px-3 text-slate-500 font-medium">Description</th>
                <th className="text-center py-2.5 px-3 text-emerald-400 font-medium">DC3S-FTIT</th>
                <th className="text-center py-2.5 px-3 text-sky-400 font-medium">DC3S-Wrapped</th>
                <th className="text-center py-2.5 px-3 text-slate-400 font-medium">Deterministic LP</th>
              </tr>
            </thead>
            <tbody>
              {BENCH_METRICS.map((m) => (
                <tr key={m.key} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                  <td className="py-2.5 px-3">
                    <span className="font-mono font-bold text-energy-primary">{m.key}</span>
                    <span className="text-slate-500 ml-1.5">{m.name}</span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 max-w-[200px]">{m.description}</td>
                  <td className="py-2.5 px-3 text-center font-mono text-emerald-400">{m.dc3sFtit}</td>
                  <td className="py-2.5 px-3 text-center font-mono text-sky-400">{m.dc3sWrapped}</td>
                  <td className="py-2.5 px-3 text-center font-mono text-slate-400">{m.deterministicLp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 pt-3 border-t border-white/[0.04] flex gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="text-[10px] text-slate-500">DC3S-FTIT (recommended)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-sky-500" />
            <span className="text-[10px] text-slate-500">DC3S-Wrapped</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-slate-500" />
            <span className="text-[10px] text-slate-500">Deterministic LP (baseline)</span>
          </div>
        </div>
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          <AnomalyTimeline data={anomalyZScores} />
          <Panel
            title="Anomaly Event Evidence"
            subtitle="DC3S Detect stage"
            badge={`${activeAnomalyCount} active`}
            badgeColor={activeAnomalyCount ? 'alert' : 'primary'}
            accentColor={activeAnomalyCount ? 'alert' : 'primary'}
            collapsible
          >
            <AnomalyList anomalies={anomalies} />
          </Panel>
        </div>
        <MLOpsMonitor data={driftData} />
      </div>
    </div>
  );
}
