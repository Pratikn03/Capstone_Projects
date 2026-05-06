'use client';

import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FileText,
  Filter,
  Layers,
  Lightbulb,
  Shield,
} from 'lucide-react';

import { KPICard } from '@/components/ui/KPICard';
import { Panel } from '@/components/ui/Panel';
import type { TheoremDashboardData, TheoremDashboardRow, TheoremTier } from '@/lib/theorem-dashboard-types';

type FilterKey = 'all' | 'core' | 'extensions' | TheoremTier;

const TIER_CONFIG: Record<TheoremTier, { label: string; bg: string; text: string; border: string }> = {
  flagship: { label: 'Flagship Defended', bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  supporting: { label: 'Supporting Defended', bg: 'bg-sky-500/15', text: 'text-sky-400', border: 'border-sky-500/30' },
  structural: { label: 'Structural', bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/30' },
  definition: { label: 'Definition', bg: 'bg-violet-500/15', text: 'text-violet-300', border: 'border-violet-500/30' },
  draft: { label: 'Draft / Non-Defended', bg: 'bg-slate-500/15', text: 'text-slate-400', border: 'border-slate-500/30' },
};

const DEFINITIONS = [
  { id: 'D1', name: 'OASG', text: 'Observed-Actual Safety Gap: the set of state-action pairs where observed safety differs from true safety.' },
  { id: 'D2', name: 'ORIUS-MDP', text: 'A Markov decision process augmented with observation reliability weight as a first-class runtime input.' },
  { id: 'D3', name: 'Observed Safety', text: 'Safety evaluated on observed and possibly faulty telemetry.' },
  { id: 'D4', name: 'True Safety', text: 'Safety evaluated on the hidden physical state.' },
  { id: 'D5', name: 'OQE', text: 'Observation quality estimator that maps telemetry history to a reliability weight.' },
  { id: 'D6', name: 'RAC Certificate', text: 'Reliability-aware conformal certificate inflated under degraded observation quality.' },
  { id: 'D7', name: 'Forward Tube', text: 'Reachable state set propagated under worst-case uncertainty from the current state.' },
  { id: 'D8', name: 'Quality-Ignorant Controller', text: 'Controller that does not condition actions on observation reliability.' },
  { id: 'D9', name: 'Certificate Validity Horizon', text: 'Maximum lookahead for which the forward tube remains inside hard constraints.' },
  { id: 'D10', name: 'Tightened Safe Set', text: 'Action set after constraint tightening accounts for observation-consistent uncertainty.' },
  { id: 'D11', name: 'Inflation Law', text: 'Monotone map from reliability and drift state to uncertainty-set scale.' },
  { id: 'D12', name: 'DC3S Pipeline', text: 'Detect, Calibrate, Constrain, Shield, and Certify runtime safety loop.' },
  { id: 'D13', name: 'TSVR', text: 'True-state violation rate over a replay denominator.' },
  { id: 'D14', name: 'Graceful Degradation', text: 'Fallback strategy that reduces aggressiveness, clamps to the safe set, then fails closed.' },
  { id: 'D15', name: 'Typed Adapter Contract', text: 'The interface a domain must implement to inherit the T11 runtime-assurance contract.' },
];

function theoremNumber(id: string): number {
  const match = id.match(/^T(\d+)/u);
  return match ? Number(match[1]) : 999;
}

function theoremAnchorId(displayId: string): string {
  return `theorem-${displayId.toLowerCase().replace(/[^a-z0-9_-]+/gu, '-')}`;
}

function TheoremCard({ thm, index }: { thm: TheoremDashboardRow; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const tier = TIER_CONFIG[thm.tier];

  return (
    <motion.div
      id={theoremAnchorId(thm.displayId)}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
      className="glass-panel scroll-mt-24 rounded-xl border border-white/[0.06] overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-5 py-4 flex items-start gap-4 hover:bg-white/[0.02] transition-colors"
      >
        <span className={`flex-shrink-0 px-2.5 py-1 rounded-md text-xs font-bold ${tier.bg} ${tier.text} ${tier.border} border`}>
          {thm.displayId}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-white">{thm.name}</h3>
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${tier.bg} ${tier.text}`}>{tier.label}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/[0.06]">
              {thm.roleLabel}
            </span>
            <span className="text-[10px] text-slate-500">{thm.chapter}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{thm.statement}</p>
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {thm.assumptions.slice(0, 8).map((item) => (
              <span key={item} className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-slate-500 border border-white/[0.06]">
                {item.length > 34 ? `${item.slice(0, 31)}...` : item}
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
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Audit Anchors</h4>
                <p className="text-xs text-slate-300 leading-relaxed">{thm.formal}</p>
              </div>
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Domain Scope</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{thm.domainApplicability}</p>
              </div>
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Instantiation / Evidence Scope</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{thm.batteryInstantiation}</p>
              </div>
              <div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Promotion Gates</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                  {thm.promotionGates.map((item) => (
                    <div key={item.key} className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2">
                      <div className="flex items-center gap-1.5">
                        {item.status === 'pass' ? (
                          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <AlertCircle className="w-3 h-3 text-amber-400" />
                        )}
                        <span className="text-[10px] font-semibold text-slate-300">{item.label}</span>
                      </div>
                      <p className="mt-1 text-[9px] text-slate-500 leading-relaxed">{item.detail}</p>
                    </div>
                  ))}
                </div>
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

function AssumptionRegister({ theorems }: { theorems: TheoremDashboardRow[] }) {
  const rows = useMemo(() => {
    const usage = new Map<string, string[]>();
    for (const theorem of theorems) {
      for (const assumption of theorem.assumptions) {
        const bucket = usage.get(assumption) ?? [];
        bucket.push(theorem.displayId);
        usage.set(assumption, bucket);
      }
    }
    return Array.from(usage.entries()).sort(([left], [right]) => left.localeCompare(right));
  }, [theorems]);

  return (
    <Panel
      title="Assumption And Obligation Register"
      subtitle="Loaded from active theorem audit rows"
      badge={String(rows.length)}
      collapsible
      defaultCollapsed
      accentColor="warn"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-white/[0.06]">
              <th className="text-left py-2 px-3 text-slate-500 font-medium">Item</th>
              <th className="text-left py-2 px-3 text-slate-500 font-medium">Used By</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([item, usedBy]) => (
              <tr key={item} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                <td className="py-2.5 px-3 text-slate-300">{item}</td>
                <td className="py-2.5 px-3 font-mono text-sky-400">{usedBy.join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export function TheoremsClient({ data }: { data: TheoremDashboardData }) {
  const [activeFilter, setActiveFilter] = useState<FilterKey>('all');
  const filters: { key: FilterKey; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'core', label: 'Core' },
    { key: 'extensions', label: 'Extensions' },
    { key: 'flagship', label: 'Flagship' },
    { key: 'supporting', label: 'Supporting' },
    { key: 'definition', label: 'Definition' },
    { key: 'draft', label: 'Draft' },
  ];

  const filtered = data.theorems.filter((theorem) => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'core') return theoremNumber(theorem.id) <= 8;
    if (activeFilter === 'extensions') return theoremNumber(theorem.id) >= 9;
    return theorem.tier === activeFilter;
  });

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <BookOpen className="w-7 h-7 text-energy-primary" />
          Theorem Ledger
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Theorem metadata from {data.summary.sourcePath}; T5 is a Definition, T7_Battery is a Battery Instantiation, and T9/T10 are assumption-qualified Supporting Defended rows.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard label="Flagship" value={String(data.summary.flagshipDefended)} icon={<Shield className="w-4 h-4" />} color="primary" delay={0} />
        <KPICard label="Supporting" value={String(data.summary.supportingDefended)} icon={<AlertCircle className="w-4 h-4" />} color="warn" delay={0.05} />
        <KPICard label="Definition" value={data.summary.definitionIds.join(', ') || '0'} icon={<FileText className="w-4 h-4" />} color="info" delay={0.1} />
        <KPICard label="Draft" value={String(data.summary.draftNonDefended)} icon={<Lightbulb className="w-4 h-4" />} color="alert" delay={0.15} />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Filter className="w-4 h-4 text-slate-500" />
        {filters.map((filter) => (
          <button
            key={filter.key}
            onClick={() => setActiveFilter(filter.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeFilter === filter.key
                ? 'bg-energy-primary/20 text-energy-primary border border-energy-primary/30'
                : 'bg-white/5 text-slate-400 border border-white/[0.06] hover:bg-white/10'
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>

      <Panel title="Proof Dependency Chain" subtitle="Generated theorem audit progression" badge={`${data.theorems.length} rows`} accentColor="primary">
        <div className="overflow-x-auto pb-2 -mx-1">
          <div className="flex items-center gap-1.5 min-w-max px-1">
            {data.theorems.map((theorem, index) => {
              const tier = TIER_CONFIG[theorem.tier];
              return (
                <div key={theorem.id} className="flex items-center gap-1.5">
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-20 h-12 rounded-lg ${tier.bg} border ${tier.border} flex items-center justify-center cursor-default`}
                      title={`${theorem.displayId}: ${theorem.name}`}
                    >
                      <span className={`text-[10px] font-bold ${tier.text}`}>{theorem.displayId}</span>
                    </div>
                    <span className="text-[9px] text-slate-600 mt-1 max-w-[72px] text-center leading-tight truncate">
                      {theorem.roleLabel}
                    </span>
                  </div>
                  {index < data.theorems.length - 1 && <ChevronRight className="w-3.5 h-3.5 text-slate-600 flex-shrink-0" />}
                </div>
              );
            })}
          </div>
        </div>
        <div className="flex gap-4 mt-3 pt-3 border-t border-white/[0.04] flex-wrap">
          {Object.entries(TIER_CONFIG).map(([key, cfg]) => (
            <div key={key} className="flex items-center gap-1.5">
              <div className={`w-2.5 h-2.5 rounded-sm ${cfg.bg} border ${cfg.border}`} />
              <span className="text-[10px] text-slate-500">{cfg.label}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Promotion Gates" subtitle="Required before any theorem becomes flagship" badge="12 gates" accentColor="warn">
        <p className="text-xs text-slate-400 leading-relaxed">{data.promotionPolicy}</p>
        <p className="text-xs text-slate-500 leading-relaxed mt-2">
          T11 remains the main universal theorem: any compliant domain inherits the ORIUS runtime-assurance contract. T7_Battery stays a battery instantiation unless a separate cross-domain fallback theorem is proved.
        </p>
      </Panel>

      <AssumptionRegister theorems={data.theorems} />

      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
          <Layers className="w-4 h-4" />
          {activeFilter === 'all' ? 'All Theorem Rows' : `Filtered: ${filters.find((filter) => filter.key === activeFilter)?.label}`}
          <span className="text-slate-600">({filtered.length})</span>
        </h2>
        {filtered.map((theorem, index) => (
          <TheoremCard key={theorem.id} thm={theorem} index={index} />
        ))}
      </div>

      <Panel title="Core Definitions" subtitle="D1-D15 runtime vocabulary" badge="15" collapsible defaultCollapsed accentColor="info">
        <div className="space-y-2">
          {DEFINITIONS.map((definition) => (
            <div key={definition.id} className="flex gap-3 py-2 border-b border-white/[0.03] last:border-0">
              <span className="text-[10px] font-bold text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded h-fit flex-shrink-0">{definition.id}</span>
              <div>
                <span className="text-xs font-medium text-slate-300">{definition.name}</span>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{definition.text}</p>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
