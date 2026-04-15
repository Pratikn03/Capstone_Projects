'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Globe2,
  Battery,
  Car,
  Factory,
  HeartPulse,
  Plane,
  Navigation,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ChevronDown,
  Layers,
  GitBranch,
  FileCheck,
  BookOpen,
} from 'lucide-react';
import { Panel } from '@/components/ui/Panel';
import { KPICard } from '@/components/ui/KPICard';

/* ------------------------------------------------------------------ */
/*  Static Data                                                        */
/* ------------------------------------------------------------------ */

type EvidenceTier = 'reference' | 'proof-validated' | 'proof-candidate' | 'experimental' | 'shadow-synthetic';

const TIER_CONFIG: Record<EvidenceTier, { label: string; bg: string; text: string; border: string; description: string }> = {
  reference: {
    label: 'Reference',
    bg: 'bg-emerald-500/15',
    text: 'text-emerald-400',
    border: 'border-emerald-500/30',
    description: 'Locked baseline witness — full proof spine validated on real operational data',
  },
  'proof-validated': {
    label: 'Proof-Validated',
    bg: 'bg-sky-500/15',
    text: 'text-sky-400',
    border: 'border-sky-500/30',
    description: 'Soundness + contingency logic verified — adapter contract fully exercised',
  },
  'proof-candidate': {
    label: 'Proof-Candidate',
    bg: 'bg-cyan-500/15',
    text: 'text-cyan-400',
    border: 'border-cyan-500/30',
    description: 'Forward proof exercise complete — pending real-data validation stage',
  },
  experimental: {
    label: 'Experimental',
    bg: 'bg-amber-500/15',
    text: 'text-amber-400',
    border: 'border-amber-500/30',
    description: 'Public data lane exercised — promotion contract not yet closed',
  },
  'shadow-synthetic': {
    label: 'Shadow-Synthetic',
    bg: 'bg-slate-500/15',
    text: 'text-slate-400',
    border: 'border-slate-500/30',
    description: 'Synthetic data only — real-data gap blocks promotion',
  },
};

interface Domain {
  key: string;
  name: string;
  icon: React.ElementType;
  tier: EvidenceTier;
  dataset: string;
  source: string;
  tsvrBefore: string;
  tsvrAfter: string;
  picp: string;
  ir: string;
  status: 'verified' | 'warning' | 'blocked';
  statusLabel: string;
  blocker?: string;
  description: string;
}

const DOMAINS: Domain[] = [
  {
    key: 'battery',
    name: 'Battery / Energy Storage',
    icon: Battery,
    tier: 'reference',
    dataset: 'OPSD (Open Power System Data)',
    source: 'Germany + EIA-930 US',
    tsvrBefore: '0.75%',
    tsvrAfter: '0.0%',
    picp: '100%',
    ir: '0.0%',
    status: 'verified',
    statusLabel: 'Locked artifact',
    description: 'Reference witness row for the full theorem ladder. German grid (OPSD) + US balancing authorities (EIA-930). All T1–T11 validated.',
  },
  {
    key: 'av',
    name: 'Autonomous Vehicles',
    icon: Car,
    tier: 'proof-candidate',
    dataset: 'Synthetic AV traces',
    source: 'Generated via adapter',
    tsvrBefore: '0.083%',
    tsvrAfter: '0.0%',
    picp: '100%',
    ir: '2.1%',
    status: 'warning',
    statusLabel: 'Constraint-class mismatch',
    blocker: '1-D acceleration repair is insufficient for multi-D headway constraints. Forward proof exercised but real-data validation blocked.',
    description: 'Proof-candidate tier: typed kernel exercised on synthetic longitudinal control. Transfer obligations met for 1-D accel repair.',
  },
  {
    key: 'industrial',
    name: 'Industrial Process',
    icon: Factory,
    tier: 'proof-validated',
    dataset: 'UCI CCPP',
    source: 'Combined Cycle Power Plant',
    tsvrBefore: '1.0%',
    tsvrAfter: '0.0%',
    picp: '100%',
    ir: '0.8%',
    status: 'verified',
    statusLabel: 'Train-validation chain complete',
    description: 'Defended-bounded domain. Plant-family replay on UCI Combined Cycle Power Plant data. Full soundness and contingency logic verified.',
  },
  {
    key: 'healthcare',
    name: 'Healthcare Monitoring',
    icon: HeartPulse,
    tier: 'proof-validated',
    dataset: 'MIMIC-III/IV (proxy)',
    source: 'Credentialed ICU data',
    tsvrBefore: '1.0%',
    tsvrAfter: '0.583%',
    picp: '97.2%',
    ir: '4.3%',
    status: 'verified',
    statusLabel: 'Accessible proxy verified',
    description: 'Defended-bounded domain. Long-term ICU validation target. Accessible proxy row used and verified.',
  },
  {
    key: 'aerospace',
    name: 'Aerospace Control',
    icon: Plane,
    tier: 'experimental',
    dataset: 'NASA C-MAPSS',
    source: 'Engine simulation FD001-FD004',
    tsvrBefore: 'N/A',
    tsvrAfter: 'N/A',
    picp: 'N/A',
    ir: 'N/A',
    status: 'warning',
    statusLabel: 'Promotion not closed',
    blocker: 'Public ADS-B evidence lane only. Adapter executes but promotion contract not yet closed for runtime certification.',
    description: 'Experimental tier: trainable aerospace companion surface. Public flight calibration used, not sufficient for runtime promotion alone.',
  },
  {
    key: 'navigation',
    name: 'Navigation & Guidance',
    icon: Navigation,
    tier: 'shadow-synthetic',
    dataset: 'KITTI Odometry',
    source: 'Benchmark (synthetic staging)',
    tsvrBefore: 'N/A',
    tsvrAfter: 'N/A',
    picp: 'N/A',
    ir: 'N/A',
    status: 'blocked',
    statusLabel: 'Real-data gap',
    blocker: 'KITTI real-data staging blocked (navigation_real_data_gap). Typed kernel exercised only on bounded synthetic surface — portability evidence only.',
    description: 'Shadow-synthetic tier: real-data closure source for autonomous navigation. Currently limited to synthetic portability evidence.',
  },
];

// Theorem-Domain matrix: true = full, 'partial' = partial, false = N/A
type Applicability = true | 'partial' | false;
const THEOREM_IDS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11'];

const DOMAIN_THEOREM_MATRIX: Record<string, Applicability[]> = {
  battery:    [true, true, true, true, true, true, true, true, true, true, true],
  av:         [true, true, true, true, 'partial', 'partial', 'partial', 'partial', true, 'partial', true],
  industrial: [true, true, true, true, 'partial', 'partial', true, true, 'partial', 'partial', true],
  healthcare: [true, true, true, true, 'partial', 'partial', true, true, 'partial', 'partial', true],
  aerospace:  [true, true, true, true, 'partial', 'partial', 'partial', 'partial', 'partial', 'partial', true],
  navigation: [true, 'partial', 'partial', 'partial', false, false, 'partial', 'partial', 'partial', false, true],
};

interface Claim {
  id: string;
  claim: string;
  theorems: string;
  evidence: string;
}

const CLAIMS: Claim[] = [
  { id: 'C001', claim: 'Telemetry faults create hidden safety violations (OASG exists)', theorems: 'T1', evidence: 'Battery replay witness' },
  { id: 'C002', claim: 'DC3S one-step repair preserves true safety', theorems: 'T2', evidence: 'Inductive proof + empirical' },
  { id: 'C003', claim: 'Expected violations bounded by α(1−w̄)T', theorems: 'T3', evidence: 'Martingale proof + battery data' },
  { id: 'C004', claim: 'Quality-ignorant controllers are fundamentally unsafe', theorems: 'T4', evidence: 'Constructive witness + ablation' },
  { id: 'C005', claim: 'Certificate validity horizon is computable', theorems: 'T5, T6', evidence: 'Forward tube analysis' },
  { id: 'C006', claim: 'Feasible fallback always exists from safe interior', theorems: 'T7', evidence: 'Zero-dispatch construction' },
  { id: 'C007', claim: 'Graceful degradation never worsens outcomes', theorems: 'T8', evidence: '6 fault × 7 controller comparison' },
  { id: 'C008', claim: 'Quality-ignorance has fundamental lower bound', theorems: 'T9', evidence: 'Martingale + Azuma–Hoeffding' },
  { id: 'C009', claim: 'T3 bound is tight within factor 4', theorems: 'T10', evidence: 'Le Cam two-point method' },
  { id: 'C010', claim: 'ORIUS transfers to any compliant domain', theorems: 'T11', evidence: 'Typed adapter + 6 domain exercise' },
  { id: 'C011', claim: 'DC3S achieves zero TSVR under all nominal + fault scenarios', theorems: 'T2, T3, T7', evidence: 'ORIUS-Bench full suite' },
];

interface DesignPrinciple {
  id: string;
  name: string;
  detail: string;
}

const DESIGN_PRINCIPLES: DesignPrinciple[] = [
  { id: 'DP1', name: 'Separate shared safety logic from plant-specific semantics', detail: 'Universal kernel enables portability across domains.' },
  { id: 'DP2', name: 'Make observation reliability a first-class runtime input', detail: 'OQE + conformal calibration: w_t drives safety conservatism.' },
  { id: 'DP3', name: 'Express uncertainty as observation-consistent state set', detail: 'Conformal + repair literature enables gap reasoning.' },
  { id: 'DP4', name: 'Use repair instead of assuming upstream controller perfection', detail: 'Safe correction without replacing the underlying controller.' },
  { id: 'DP5', name: 'Emit auditable runtime certificates', detail: 'Governed evidence trail vs hidden logic — CertOS framing.' },
  { id: 'DP6', name: 'Standardize the replay schema before comparing domains', detail: 'Prevents domain-shifting success definitions.' },
  { id: 'DP7', name: 'Keep evidence tiers explicit and governed', detail: 'Prevents overclaiming universality beyond actual evidence.' },
  { id: 'DP8', name: 'Serve tracked artifacts through one backend truth path', detail: 'Removes dashboard cache dependence — research router pattern.' },
];

const TRANSFER_OBLIGATIONS = [
  { method: 'compute_oqe', description: 'Estimate observation quality w_t from domain telemetry' },
  { method: 'build_uncertainty_set', description: 'Construct calibrated uncertainty set U_t with inflation' },
  { method: 'tighten_action_set', description: 'Compute tightened safe action set A_t from U_t + constraints' },
  { method: 'repair_action', description: 'Project candidate action onto A_t to produce safe action' },
  { method: 'emit_certificate', description: 'Emit per-step auditable certificate with hash chain' },
];

/* ------------------------------------------------------------------ */
/*  Sub-Components                                                     */
/* ------------------------------------------------------------------ */

function StatusIcon({ status }: { status: Domain['status'] }) {
  switch (status) {
    case 'verified':
      return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
    case 'warning':
      return <AlertTriangle className="w-4 h-4 text-amber-400" />;
    case 'blocked':
      return <XCircle className="w-4 h-4 text-red-400" />;
  }
}

function DomainCard({ domain, index }: { domain: Domain; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const tierCfg = TIER_CONFIG[domain.tier];
  const Icon = domain.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.3 }}
      className="glass-panel rounded-xl border border-white/[0.06] overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-4 hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-start gap-3">
          <div className={`w-10 h-10 rounded-lg ${tierCfg.bg} border ${tierCfg.border} flex items-center justify-center flex-shrink-0`}>
            <Icon className={`w-5 h-5 ${tierCfg.text}`} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-sm font-semibold text-white">{domain.name}</h3>
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${tierCfg.bg} ${tierCfg.text} border ${tierCfg.border}`}>
                {tierCfg.label}
              </span>
              <StatusIcon status={domain.status} />
            </div>
            <p className="text-[10px] text-slate-500 mt-0.5">{domain.dataset} — {domain.source}</p>

            {/* Metrics */}
            <div className="grid grid-cols-4 gap-2 mt-3">
              <div>
                <div className="text-[9px] text-slate-600">TSVR (before)</div>
                <div className="text-xs font-mono text-slate-400">{domain.tsvrBefore}</div>
              </div>
              <div>
                <div className="text-[9px] text-slate-600">TSVR (after)</div>
                <div className="text-xs font-mono text-emerald-400">{domain.tsvrAfter}</div>
              </div>
              <div>
                <div className="text-[9px] text-slate-600">PICP</div>
                <div className="text-xs font-mono text-slate-400">{domain.picp}</div>
              </div>
              <div>
                <div className="text-[9px] text-slate-600">IR</div>
                <div className="text-xs font-mono text-slate-400">{domain.ir}</div>
              </div>
            </div>

            {domain.blocker && (
              <div className="mt-2 flex items-start gap-1.5">
                <AlertTriangle className="w-3 h-3 text-amber-400 flex-shrink-0 mt-0.5" />
                <span className="text-[10px] text-amber-400/80">{domain.statusLabel}</span>
              </div>
            )}
          </div>
          <ChevronDown className={`w-4 h-4 text-slate-500 flex-shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`} />
        </div>
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
            <div className="px-4 pb-4 pt-2 border-t border-white/[0.04] space-y-2">
              <p className="text-xs text-slate-400 leading-relaxed">{domain.description}</p>
              {domain.blocker && (
                <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <div className="text-[10px] font-semibold text-amber-400 mb-0.5">Blocker</div>
                  <p className="text-[10px] text-amber-400/70 leading-relaxed">{domain.blocker}</p>
                </div>
              )}
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

export default function DomainsPage() {
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Globe2 className="w-7 h-7 text-energy-primary" />
          Universal Domain Coverage
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          6 domains × 11 theorems — evidence tiers, transfer obligations, and claims register
        </p>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard label="Domains" value="6" icon={<Globe2 className="w-4 h-4" />} color="primary" delay={0} />
        <KPICard label="Evidence Tiers" value="5" icon={<Layers className="w-4 h-4" />} color="info" delay={0.05} />
        <KPICard label="Transfer Theorem" value="T11" icon={<GitBranch className="w-4 h-4" />} color="warn" delay={0.1} />
        <KPICard label="Claims" value="11" icon={<FileCheck className="w-4 h-4" />} color="alert" delay={0.15} />
      </div>

      {/* Evidence Tier Framework */}
      <Panel title="Evidence Tier Framework" subtitle="5 tiers of domain evidence maturity" accentColor="info">
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
          {(Object.entries(TIER_CONFIG) as [EvidenceTier, typeof TIER_CONFIG[EvidenceTier]][]).map(([key, cfg], i) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              className={`p-3 rounded-lg border ${cfg.border} ${cfg.bg}`}
            >
              <div className={`text-xs font-bold ${cfg.text} mb-1`}>{cfg.label}</div>
              <p className="text-[10px] text-slate-500 leading-relaxed">{cfg.description}</p>
            </motion.div>
          ))}
        </div>
      </Panel>

      {/* T11 Transfer Theorem */}
      <Panel
        title="T11: Typed Structural Transfer"
        subtitle="Universal adapter contract — any compliant domain inherits T2–T3 safety guarantees"
        badge="Structural"
        badgeColor="warn"
        accentColor="warn"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400 leading-relaxed">
            A domain D implementing the typed adapter contract inherits the T2 one-step preservation and T3 core bound
            without requiring battery-specific algebra. The 5 transfer obligations define the contract:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {TRANSFER_OBLIGATIONS.map((ob, i) => (
              <motion.div
                key={ob.method}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.06 }}
                className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20"
              >
                <div className="text-[10px] font-mono font-bold text-amber-400 mb-1">{ob.method}</div>
                <p className="text-[10px] text-slate-500 leading-relaxed">{ob.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </Panel>

      {/* Theorem-Domain Matrix */}
      <Panel title="Theorem–Domain Applicability Matrix" subtitle="6 domains × 11 theorems" accentColor="primary">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/[0.08]">
                <th className="text-left py-2.5 px-3 text-slate-500 font-medium">Domain</th>
                {THEOREM_IDS.map((id) => (
                  <th key={id} className="text-center py-2.5 px-2 text-slate-500 font-medium font-mono">{id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DOMAINS.map((d) => {
                const row = DOMAIN_THEOREM_MATRIX[d.key];
                const Icon = d.icon;
                return (
                  <tr key={d.key} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        <Icon className="w-3.5 h-3.5 text-slate-500" />
                        <span className="text-slate-300">{d.name}</span>
                      </div>
                    </td>
                    {row.map((val, i) => (
                      <td key={i} className="text-center py-2.5 px-2">
                        {val === true && <span className="text-emerald-400 text-sm">●</span>}
                        {val === 'partial' && <span className="text-amber-400 text-sm">○</span>}
                        {val === false && <span className="text-slate-600">—</span>}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex gap-4 mt-3 pt-3 border-t border-white/[0.04]">
          <div className="flex items-center gap-1.5">
            <span className="text-emerald-400 text-sm">●</span>
            <span className="text-[10px] text-slate-500">Fully applicable</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-amber-400 text-sm">○</span>
            <span className="text-[10px] text-slate-500">Partial / pending</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-600 text-sm">—</span>
            <span className="text-[10px] text-slate-500">Not applicable</span>
          </div>
        </div>
      </Panel>

      {/* Domain Cards */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
          <Globe2 className="w-4 h-4" />
          Domain Details
          <span className="text-slate-600">(6)</span>
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {DOMAINS.map((d, i) => (
            <DomainCard key={d.key} domain={d} index={i} />
          ))}
        </div>
      </div>

      {/* Claims Register */}
      <Panel
        title="Claims Register"
        subtitle="C001–C011 theorem-to-evidence linkage"
        badge="11"
        collapsible
        defaultCollapsed
        accentColor="primary"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/[0.06]">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">ID</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Claim</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Theorems</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {CLAIMS.map((c) => (
                <tr key={c.id} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                  <td className="py-2.5 px-3 font-mono font-bold text-energy-primary">{c.id}</td>
                  <td className="py-2.5 px-3 text-slate-300">{c.claim}</td>
                  <td className="py-2.5 px-3 font-mono text-sky-400">{c.theorems}</td>
                  <td className="py-2.5 px-3 text-slate-500">{c.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Design Principles */}
      <Panel
        title="Design Principles"
        subtitle="DP1–DP8 cross-domain design guidelines"
        badge="8"
        collapsible
        defaultCollapsed
        accentColor="info"
      >
        <div className="space-y-2">
          {DESIGN_PRINCIPLES.map((dp) => (
            <div key={dp.id} className="flex gap-3 py-2 border-b border-white/[0.03] last:border-0">
              <span className="text-[10px] font-bold text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded h-fit flex-shrink-0">{dp.id}</span>
              <div>
                <span className="text-xs font-medium text-slate-300">{dp.name}</span>
                <p className="text-xs text-slate-500 mt-0.5">{dp.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
