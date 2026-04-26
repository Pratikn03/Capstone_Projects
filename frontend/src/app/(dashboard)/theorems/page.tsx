'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Shield,
  Filter,
  Layers,
  FileText,
  Lightbulb,
} from 'lucide-react';
import { Panel } from '@/components/ui/Panel';
import { KPICard } from '@/components/ui/KPICard';

/* ------------------------------------------------------------------ */
/*  Static Data                                                        */
/* ------------------------------------------------------------------ */

type Tier = 'proved' | 'structural' | 'constructive' | 'empirical';

interface Theorem {
  id: string;
  name: string;
  tier: Tier;
  chapter: string;
  assumptions: string[];
  statement: string;
  formal: string;
  significance: string;
  batteryInstantiation: string;
  proofMethod: string;
}

const THEOREMS: Theorem[] = [
  {
    id: 'T1',
    name: 'OASG Existence',
    tier: 'structural',
    chapter: 'Ch. 16',
    assumptions: ['A2', 'A4'],
    statement: '∃ episodes where observed-safe(a | x_obs) ≠ true-safe(a | x_true)',
    formal:
      '∃ (x_t, w_t) such that φ_obs(a | x̂_t) = 1 ∧ φ_true(a | x_t) = 0, demonstrating the Observed–Actual Safety Gap.',
    significance:
      'Establishes that telemetry faults can silently create safety violations — the foundational motivation for the entire ORIUS framework.',
    batteryInstantiation:
      'SOC sensor reads 45% (safe zone) while true SOC is 18% (below SOC_min). Dispatch action appears safe but causes deep discharge.',
    proofMethod: 'Constructive witness on battery replay with injected telemetry faults.',
  },
  {
    id: 'T2',
    name: 'One-Step Safety Preservation',
    tier: 'proved',
    chapter: 'Ch. 17',
    assumptions: ['A1', 'A2', 'A3', 'A4', 'A5', 'A7'],
    statement: 'If a_safe ∈ A_t and x_t ∈ C_RAC, then φ_true(a_safe) = 1',
    formal:
      'Given repaired action a_safe ∈ A_t (tightened safe set) and current state x_t within the RAC certificate, the true safety predicate holds at the next step.',
    significance:
      'The inductive core — proves that a single DC3S-repaired action is guaranteed safe under bounded model and telemetry error.',
    batteryInstantiation:
      'If shield clamps dispatch to tightened SOC bounds and current SOC is within the inflated certificate, next-step SOC stays in [SOC_min, SOC_max].',
    proofMethod: 'Induction on state within inflated conformal certificate + Lipschitz dynamics bound.',
  },
  {
    id: 'T3',
    name: 'ORIUS Core Bound',
    tier: 'proved',
    chapter: 'Ch. 18',
    assumptions: ['A1', 'A2', 'A3', 'A4', 'A5', 'A7'],
    statement: 'E[V_T] ≤ α(1 − w̄)T',
    formal:
      'Under the ORIUS-MDP with risk-budget contract, expected true-state violations over horizon T are bounded by α(1 − w̄)T where α is the conformal miscoverage rate and w̄ is the mean reliability weight.',
    significance:
      'The headline result — violations scale linearly with unreliability and shrink as telemetry quality improves. Connects conformal calibration to safety outcomes.',
    batteryInstantiation:
      'With α = 0.05, w̄ = 0.92, T = 288 half-hours/day: E[V] ≤ 0.05 × 0.08 × 288 = 1.15 violations/day expected.',
    proofMethod: 'Martingale decomposition on conformal coverage + inflation law monotonicity.',
  },
  {
    id: 'T4',
    name: 'No Free Safety',
    tier: 'constructive',
    chapter: 'Ch. 19',
    assumptions: ['A2', 'A3', 'A8'],
    statement: 'Quality-ignorant controllers necessarily violate safety bounds',
    formal:
      'Any controller π that does not condition on reliability weight w_t will, under fault sequences satisfying A2, accumulate violations exceeding those of reliability-aware controllers.',
    significance:
      'Proves that ignoring observation quality is not just suboptimal — it is fundamentally unsafe. Justifies the OQE stage of DC3S.',
    batteryInstantiation:
      'A fixed-interval robust controller ignores sensor degradation. When dropout rate rises, its intervals become overconfident and violations spike.',
    proofMethod: 'Constructive witness + ablation: remove OQE → show violation rate increase.',
  },
  {
    id: 'T5',
    name: 'Certificate Validity Horizon',
    tier: 'proved',
    chapter: 'Ch. 20',
    assumptions: ['A1', 'A2', 'A4', 'A5', 'A6', 'A7'],
    statement: 'τ_t = max{ Δ : F_{t+Δ} ⊆ [SOC_min, SOC_max] }',
    formal:
      'The certificate validity horizon τ_t is the largest lookahead Δ such that the forward tube F_{t+Δ} (propagated under worst-case uncertainty) remains within hard SOC bounds.',
    significance:
      'Gives runtime a measurable "time to danger" — the certificate half-life. Enables proactive fallback triggering before safety is lost.',
    batteryInstantiation:
      'If current SOC is 50% and worst-case drain is 2%/step, certificate is valid for (50−20)/2 = 15 steps before hitting SOC_min = 20%.',
    proofMethod: 'Forward reachability analysis under bounded dynamics + uncertainty inflation.',
  },
  {
    id: 'T6',
    name: 'Certificate Expiration Bound',
    tier: 'proved',
    chapter: 'Ch. 20',
    assumptions: ['A1', 'A2', 'A4', 'A5', 'A7'],
    statement: 'τ_t ≥ ⌊δ²_bnd,t / σ²_d⌋',
    formal:
      'Certificate horizon is lower-bounded by the squared distance-to-boundary divided by the drift-scale variance.',
    significance:
      'Provides a computable, conservative lower bound on how long the certificate remains valid — no simulation required.',
    batteryInstantiation:
      'Distance to nearest SOC bound = 30%, drift variance σ²_d = 4 → τ ≥ ⌊900/4⌋ = 225 steps minimum.',
    proofMethod: 'Analytical bound from dynamics propagation + variance accumulation.',
  },
  {
    id: 'T7',
    name: 'Feasible Fallback Existence',
    tier: 'proved',
    chapter: 'Ch. 20',
    assumptions: ['A3', 'A8'],
    statement: 'Zero-dispatch fallback always preserves safety from interior SOC',
    formal:
      'Under A3 (feasible repair) and A8 (admissible fallback), ∃ π_fb such that applying π_fb from any safe interior state maintains SOC ∈ [SOC_min, SOC_max].',
    significance:
      'Guarantees that the system can always "do nothing safely" — the last line of defense when all other controllers fail.',
    batteryInstantiation:
      'Setting dispatch = 0 MW. Battery SOC drifts only by self-discharge (~0.01%/step), staying safely within bounds for hundreds of steps.',
    proofMethod: 'Constructive: zero-action + self-discharge bound.',
  },
  {
    id: 'T8',
    name: 'Graceful Degradation Dominance',
    tier: 'empirical',
    chapter: 'Ch. 29',
    assumptions: ['A3', 'A4', 'A8'],
    statement: 'DC3S violations ≤ uncontrolled violations; strict improvement on fault sequences',
    formal:
      'The safe-landing controller (graceful degradation) achieves V_DC3S ≤ V_uncontrolled on all tested fault scenarios, with strict inequality on non-trivial fault sequences.',
    significance:
      'Empirical validation that the graceful degradation pathway never makes things worse — a critical monotonicity property for deployment trust.',
    batteryInstantiation:
      'Under drift_combo fault: uncontrolled has TSVR = 12.5%, DC3S safe-landing achieves TSVR = 0.0%.',
    proofMethod: 'Empirical: 6 fault scenarios × 7 controllers, paired comparison.',
  },
  {
    id: 'T9',
    name: 'Universal Impossibility',
    tier: 'proved',
    chapter: 'Ch. 37',
    assumptions: ['A2', "A6'"],
    statement: 'Any quality-ignorant π under φ-mixing faults: E[V_T] ≥ c · d · T',
    formal:
      'For any controller π that does not adapt to observation quality, under φ-mixing fault processes with dimension d, expected violations grow at least linearly: E[V_T] ≥ c · d · T.',
    significance:
      'A lower bound showing that the linear violation scaling in T3 is not an artifact of our approach — it is a fundamental limit.',
    batteryInstantiation:
      'Even with perfect battery model, if sensor quality is ignored, violations grow linearly with horizon. No fixed-interval approach can escape this.',
    proofMethod: 'Martingale argument + Azuma–Hoeffding concentration.',
  },
  {
    id: 'T10',
    name: 'Boundary-Indistinguishability',
    tier: 'proved',
    chapter: 'Ch. 37',
    assumptions: ['A1', 'A2'],
    statement: 'E[V_T] ≥ (1/4)α(1 − w̄)T — the T3 bound is tight',
    formal:
      'Under boundary-indistinguishability conditions, any controller (including reliability-aware) must incur E[V_T] ≥ (1/4)α(1−w̄)T.',
    significance:
      'Proves T3 is tight up to a constant factor of 4. The ORIUS framework is near-optimal — no fundamentally better approach exists.',
    batteryInstantiation:
      'Near SOC boundaries where safe/unsafe states are indistinguishable under noise, even optimal controllers incur ≥25% of the T3 bound.',
    proofMethod: 'Le Cam two-point method (information-theoretic).',
  },
  {
    id: 'T11',
    name: 'Typed Structural Transfer',
    tier: 'structural',
    chapter: 'Ch. 37',
    assumptions: ['A1', 'A2', 'A3', 'A7'],
    statement: 'Any domain satisfying 5 adapter obligations inherits T2–T3 safety guarantees',
    formal:
      'A domain D implementing the typed adapter contract (compute_oqe, build_uncertainty_set, tighten_action_set, repair_action, emit_certificate) inherits the T2 one-step preservation and T3 core bound without requiring battery-specific algebra.',
    significance:
      'The universality theorem — elevates ORIUS from a battery-specific framework to a cross-domain safety methodology.',
    batteryInstantiation:
      'Battery is the reference witness. AV and Healthcare close bounded runtime-contract rows; future domains remain architectural extensions until their evidence is promoted.',
    proofMethod: 'Forward proof from adapter contract + typed kernel + counterexamples for incomplete adapters.',
  },
];

interface Assumption {
  id: string;
  name: string;
  formal: string;
  domain: string;
  codeLocation: string;
}

const ASSUMPTIONS: Assumption[] = [
  { id: 'A1', name: 'Bounded Model Error', formal: '|ε_t| ≤ ε_model', domain: 'Physics / Dynamics', codeLocation: 'cpsbench_iot/plant.py' },
  { id: 'A2', name: 'Bounded Telemetry Error', formal: '|x̂_t − x_t| ≤ g(w_t)', domain: 'Sensor / Telemetry', codeLocation: 'dc3s/quality.py' },
  { id: 'A3', name: 'Feasible Safe Repair', formal: 'φ(x_t) = 1 ⟹ A_t ≠ ∅', domain: 'Controller', codeLocation: 'dc3s/shield.py' },
  { id: 'A4', name: 'Known Dynamics', formal: 'Δ(a) exact', domain: 'System Model', codeLocation: 'cpsbench_iot/plant.py' },
  { id: 'A5', name: 'Monotone Bounded Inflation', formal: '∂m_t / ∂w_t ≤ 0', domain: 'Uncertainty Scaling', codeLocation: 'dc3s/coverage_theorem.py' },
  { id: 'A6', name: 'Bounded Detector Lag', formal: 't_flag − t_onset ≤ τ_max', domain: 'Fault Detection', codeLocation: 'dc3s/guarantee_checks.py' },
  { id: 'A7', name: 'Causal Certificate', formal: 'cert_t = h(z₁, …, z_t)', domain: 'Runtime Causality', codeLocation: 'cpsbench_iot/runner.py' },
  { id: 'A8', name: 'Admissible Fallback', formal: '∃ π_fb with TSVR = 0', domain: 'Fallback Feasibility', codeLocation: 'configs/dc3s.yaml' },
];

const DEFINITIONS = [
  { id: 'D1', name: 'OASG', text: 'Observed–Actual Safety Gap: the set of state-action pairs where observed safety ≠ true safety.' },
  { id: 'D2', name: 'ORIUS-MDP', text: 'Markov Decision Process augmented with observation reliability weight w_t as a first-class state variable.' },
  { id: 'D3', name: 'Observed Safety', text: 'φ_obs(a | x̂_t) — safety predicate evaluated on observed (potentially faulty) telemetry.' },
  { id: 'D4', name: 'True Safety', text: 'φ_true(a | x_t) — safety predicate evaluated on true (hidden) physical state.' },
  { id: 'D5', name: 'OQE (Observation Quality Estimator)', text: 'Real-time estimator that maps telemetry history to reliability weight w_t ∈ [0, 1].' },
  { id: 'D6', name: 'RAC Certificate', text: 'Reliability-Aware Conformal certificate: prediction region inflated by I(w_t, D_t) to maintain coverage under faults.' },
  { id: 'D7', name: 'Forward Tube', text: 'F_{t+Δ} — reachable state set propagated Δ steps under worst-case uncertainty from current state.' },
  { id: 'D8', name: 'Quality-Ignorant Controller', text: 'Any controller π that does not condition its actions on the reliability weight w_t.' },
  { id: 'D9', name: 'Certificate Validity Horizon', text: 'τ_t — the maximum lookahead Δ for which the forward tube remains within hard constraints.' },
  { id: 'D10', name: 'Tightened Safe Set', text: 'A_t — the action set after constraint tightening accounts for state uncertainty.' },
  { id: 'D11', name: 'Inflation Law', text: 'I(w_t, D_t) — monotone function mapping reliability weight and drift state to uncertainty set scaling factor.' },
  { id: 'D12', name: 'DC3S Pipeline', text: 'Detect → Calibrate → Constrain → Shield → Certify — the 5-stage safety architecture.' },
  { id: 'D13', name: 'TSVR', text: 'True-SOC Violation Rate — fraction of timesteps where physical state breaches hard bounds.' },
  { id: 'D14', name: 'Graceful Degradation', text: 'Progressive fallback strategy: reduce aggressiveness → clamp to safe set → zero-dispatch landing.' },
  { id: 'D15', name: 'Typed Adapter Contract', text: 'The 5-method interface a domain must implement to inherit ORIUS safety guarantees (T11).' },
];

const TIER_CONFIG: Record<Tier, { label: string; bg: string; text: string; border: string }> = {
  proved: { label: 'Proved', bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  structural: { label: 'Structural', bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/30' },
  constructive: { label: 'Constructive', bg: 'bg-sky-500/15', text: 'text-sky-400', border: 'border-sky-500/30' },
  empirical: { label: 'Empirical', bg: 'bg-purple-500/15', text: 'text-purple-400', border: 'border-purple-500/30' },
};

const CHAIN_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11'];

type FilterKey = 'all' | 'core' | 'extensions' | 'proved' | 'structural' | 'constructive';

/* ------------------------------------------------------------------ */
/*  Components                                                         */
/* ------------------------------------------------------------------ */

function TheoremCard({ thm, index }: { thm: Theorem; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const tier = TIER_CONFIG[thm.tier];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
      className="glass-panel rounded-xl border border-white/[0.06] overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-5 py-4 flex items-start gap-4 hover:bg-white/[0.02] transition-colors"
      >
        {/* ID badge */}
        <span className={`flex-shrink-0 px-2.5 py-1 rounded-md text-xs font-bold ${tier.bg} ${tier.text} ${tier.border} border`}>
          {thm.id}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-white">{thm.name}</h3>
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${tier.bg} ${tier.text}`}>{tier.label}</span>
            <span className="text-[10px] text-slate-500">{thm.chapter}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5 font-mono leading-relaxed">{thm.statement}</p>
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {thm.assumptions.map((a) => (
              <span key={a} className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-slate-500 border border-white/[0.06]">
                {a}
              </span>
            ))}
          </div>
        </div>

        <ChevronDown className={`w-4 h-4 text-slate-500 flex-shrink-0 mt-1 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-2 border-t border-white/[0.04] space-y-3">
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Formal Statement</h4>
                <p className="text-xs text-slate-300 leading-relaxed">{thm.formal}</p>
              </div>
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Significance</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{thm.significance}</p>
              </div>
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Battery Instantiation</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{thm.batteryInstantiation}</p>
              </div>
              <div className="flex items-center gap-2 pt-1">
                <span className="text-[10px] text-slate-500">Proof method:</span>
                <span className="text-[10px] text-slate-400">{thm.proofMethod}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function TheoremsPage() {
  const [activeFilter, setActiveFilter] = useState<FilterKey>('all');

  const filters: { key: FilterKey; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'core', label: 'Core (T1–T8)' },
    { key: 'extensions', label: 'Extensions (T9–T11)' },
    { key: 'proved', label: 'Proved' },
    { key: 'structural', label: 'Structural' },
    { key: 'constructive', label: 'Constructive' },
  ];

  const filtered = THEOREMS.filter((t) => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'core') return parseInt(t.id.slice(1)) <= 8;
    if (activeFilter === 'extensions') return parseInt(t.id.slice(1)) >= 9;
    return t.tier === activeFilter;
  });

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <BookOpen className="w-7 h-7 text-energy-primary" />
          Theorem Ladder
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          83 formal items — 11 core theorems (T1–T11), 8 assumptions (A1–A8), 15 definitions, 9 propositions
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard label="Theorems" value="11" icon={<Shield className="w-4 h-4" />} color="primary" delay={0} />
        <KPICard label="Assumptions" value="8" icon={<AlertCircle className="w-4 h-4" />} color="warn" delay={0.05} />
        <KPICard label="Definitions" value="15" icon={<FileText className="w-4 h-4" />} color="info" delay={0.1} />
        <KPICard label="Propositions" value="9" icon={<Lightbulb className="w-4 h-4" />} color="alert" delay={0.15} />
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-2 flex-wrap">
        <Filter className="w-4 h-4 text-slate-500" />
        {filters.map((f) => (
          <button
            key={f.key}
            onClick={() => setActiveFilter(f.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeFilter === f.key
                ? 'bg-energy-primary/20 text-energy-primary border border-energy-primary/30'
                : 'bg-white/5 text-slate-400 border border-white/[0.06] hover:bg-white/10'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Proof Dependency Chain */}
      <Panel title="Proof Dependency Chain" subtitle="T1 → T11 theorem progression" badge="11 nodes" accentColor="primary">
        <div className="overflow-x-auto pb-2 -mx-1">
          <div className="flex items-center gap-1.5 min-w-max px-1">
            {CHAIN_ORDER.map((id, i) => {
              const thm = THEOREMS.find((t) => t.id === id)!;
              const tier = TIER_CONFIG[thm.tier];
              return (
                <div key={id} className="flex items-center gap-1.5">
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-12 h-12 rounded-lg ${tier.bg} border ${tier.border} flex items-center justify-center cursor-default`}
                      title={`${thm.id}: ${thm.name}`}
                    >
                      <span className={`text-xs font-bold ${tier.text}`}>{thm.id}</span>
                    </div>
                    <span className="text-[9px] text-slate-600 mt-1 max-w-[56px] text-center leading-tight truncate">
                      {thm.name}
                    </span>
                    <div className="flex gap-0.5 mt-0.5">
                      {thm.assumptions.slice(0, 3).map((a) => (
                        <span key={a} className="text-[7px] text-slate-600">{a}</span>
                      ))}
                      {thm.assumptions.length > 3 && <span className="text-[7px] text-slate-600">+{thm.assumptions.length - 3}</span>}
                    </div>
                  </div>
                  {i < CHAIN_ORDER.length - 1 && (
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 flex-shrink-0" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
        <div className="flex gap-4 mt-3 pt-3 border-t border-white/[0.04]">
          {Object.entries(TIER_CONFIG).map(([key, cfg]) => (
            <div key={key} className="flex items-center gap-1.5">
              <div className={`w-2.5 h-2.5 rounded-sm ${cfg.bg} border ${cfg.border}`} />
              <span className="text-[10px] text-slate-500">{cfg.label}</span>
            </div>
          ))}
        </div>
      </Panel>

      {/* Assumption Register */}
      <Panel
        title="Assumption Register"
        subtitle="A1–A8 formal assumptions underpinning the theorem ladder"
        badge="8"
        collapsible
        defaultCollapsed
        accentColor="warn"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/[0.06]">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">ID</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Name</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Formal Statement</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Domain</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Code</th>
              </tr>
            </thead>
            <tbody>
              {ASSUMPTIONS.map((a) => (
                <tr key={a.id} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                  <td className="py-2.5 px-3 font-mono font-bold text-amber-400">{a.id}</td>
                  <td className="py-2.5 px-3 text-slate-300">{a.name}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">{a.formal}</td>
                  <td className="py-2.5 px-3 text-slate-500">{a.domain}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-600 text-[10px]">{a.codeLocation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Theorem Cards */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
          <Layers className="w-4 h-4" />
          {activeFilter === 'all' ? 'All Theorems' : `Filtered: ${filters.find(f => f.key === activeFilter)?.label}`}
          <span className="text-slate-600">({filtered.length})</span>
        </h2>
        {filtered.map((thm, i) => (
          <TheoremCard key={thm.id} thm={thm} index={i} />
        ))}
      </div>

      {/* Core Definitions */}
      <Panel
        title="Core Definitions"
        subtitle="D1–D15 foundational definitions"
        badge="15"
        collapsible
        defaultCollapsed
        accentColor="info"
      >
        <div className="space-y-2">
          {DEFINITIONS.map((d) => (
            <div key={d.id} className="flex gap-3 py-2 border-b border-white/[0.03] last:border-0">
              <span className="text-[10px] font-bold text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded h-fit flex-shrink-0">{d.id}</span>
              <div>
                <span className="text-xs font-medium text-slate-300">{d.name}</span>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{d.text}</p>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
