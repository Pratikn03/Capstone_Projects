'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import {
  AlertCircle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Database,
  ExternalLink,
  FileText,
  GitBranch,
  Globe2,
  LineChart,
  Scale,
  ShieldCheck,
  TriangleAlert,
  Zap,
} from 'lucide-react';

import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useReportsData } from '@/lib/api/reports-client';
import { DOMAIN_OPTIONS, getDomainOption, isBatteryDomain } from '@/lib/domain-options';
import type { TheoremDashboardData, TheoremDashboardRow } from '@/lib/theorem-dashboard-types';
import { formatCurrency, formatPercent } from '@/lib/utils';

type VerdictTone = 'pass' | 'warn' | 'block' | 'info';
type MetricEvidence = {
  target: string;
  model: string;
  rmse: number;
  coverage_90?: number | null;
};

type PromotionBlocker = {
  theorem: TheoremDashboardRow;
  pendingGates: TheoremDashboardRow['promotionGates'];
  reason: string;
};

const verdictToneClasses: Record<VerdictTone, string> = {
  pass: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
  warn: 'border-amber-500/35 bg-amber-500/10 text-amber-300',
  block: 'border-red-500/35 bg-red-500/10 text-red-300',
  info: 'border-sky-500/35 bg-sky-500/10 text-sky-300',
};

const compactToneText: Record<VerdictTone, string> = {
  pass: 'text-emerald-300',
  warn: 'text-amber-300',
  block: 'text-red-300',
  info: 'text-sky-300',
};

function toFiniteNumber(row: Record<string, unknown> | undefined, key: string): number | null {
  const value = row?.[key];
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function formatFractionPercent(value: number | null | undefined): string {
  return value === null || value === undefined ? 'Pending' : formatPercent(value * 100);
}

function formatMaybePercent(value: number | null | undefined): string {
  return value === null || value === undefined ? 'Pending' : formatPercent(value);
}

function formatMaybeCurrency(value: number | null | undefined): string {
  return value === null || value === undefined ? 'Pending' : formatCurrency(value, 'USD');
}

function formatMaybeDate(value: string | null | undefined): string {
  if (!value) return 'No artifact timestamp';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 16);
  return date.toISOString().slice(0, 16).replace('T', ' ');
}

function theoremAnchorId(displayId: string): string {
  return `theorem-${displayId.toLowerCase().replace(/[^a-z0-9_-]+/gu, '-')}`;
}

function artifactPathsFromText(value: string): string[] {
  const matches = value.match(/\b(?:reports|data)\/[^\s;,]+/gu) ?? [];
  return Array.from(new Set(matches.map((match) => match.replace(/[.)\]]+$/u, ''))));
}

function artifactHrefFor(artifact: string): string | null {
  const normalized = artifact.trim().replace(/^["'`]+|["'`]+$/gu, '');
  if (normalized.startsWith('reports/')) {
    const reportsRelativePath = normalized.replace(/^reports\//u, '');
    return `/api/reports/file?path=${encodeURIComponent(reportsRelativePath)}`;
  }
  if (normalized.startsWith('data/')) {
    return `/data?artifact=${encodeURIComponent(normalized)}`;
  }
  return null;
}

function EvidenceChip({ tone, children }: { tone: VerdictTone; children: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${verdictToneClasses[tone]}`}>
      {children}
    </span>
  );
}

function CompactMetric({
  label,
  value,
  tone = 'info',
  detail,
}: {
  label: string;
  value: string;
  tone?: VerdictTone;
  detail?: string;
}) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.025] px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 font-mono text-sm font-semibold ${compactToneText[tone]}`}>{value}</div>
      {detail && <div className="mt-1 text-[10px] leading-snug text-slate-500">{detail}</div>}
    </div>
  );
}

function ArtifactLink({ artifact, label = artifact }: { artifact: string; label?: string }) {
  const href = artifactHrefFor(artifact);
  const content = (
    <>
      <span className="min-w-0 truncate">{label}</span>
      {href && <ExternalLink className="h-3 w-3 shrink-0 opacity-70" />}
    </>
  );

  if (!href) {
    return (
      <span className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md bg-white/[0.035] px-2 py-1 font-mono text-[11px] text-slate-300" title={artifact}>
        {content}
      </span>
    );
  }

  return (
    <Link
      href={href}
      target={href.startsWith('/api/reports/file') ? '_blank' : undefined}
      rel={href.startsWith('/api/reports/file') ? 'noreferrer' : undefined}
      className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md border border-white/[0.06] bg-white/[0.035] px-2 py-1 font-mono text-[11px] text-slate-300 transition-colors hover:border-energy-primary/35 hover:text-white"
      title={artifact}
    >
      {content}
    </Link>
  );
}

function SourceRow({
  label,
  value,
  tone = 'info',
  href,
  artifacts,
}: {
  label: string;
  value: string;
  tone?: VerdictTone;
  href?: string;
  artifacts?: string[];
}) {
  const linkedArtifacts = artifacts ?? artifactPathsFromText(value);
  return (
    <div className="grid grid-cols-1 gap-1 border-b border-white/[0.04] py-2.5 text-xs last:border-0 sm:grid-cols-[140px_1fr_auto] sm:gap-3">
      <div className="text-slate-500">{label}</div>
      {linkedArtifacts.length > 0 ? (
        <div className="flex min-w-0 flex-wrap gap-1.5">
          {linkedArtifacts.map((artifact) => (
            <ArtifactLink key={artifact} artifact={artifact} />
          ))}
        </div>
      ) : href ? (
        <Link href={href} className="min-w-0 truncate font-mono text-[11px] text-slate-300 hover:text-white" title={value}>
          {value}
        </Link>
      ) : (
        <div className="min-w-0 truncate font-mono text-[11px] text-slate-300" title={value}>
          {value}
        </div>
      )}
      <EvidenceChip tone={tone}>{tone === 'pass' ? 'loaded' : tone === 'warn' ? 'check' : tone === 'block' ? 'blocked' : 'tracked'}</EvidenceChip>
    </div>
  );
}

export function ResearchHubClient({ theoremData }: { theoremData: TheoremDashboardData }) {
  const { region } = useRegion();
  const currentDomain = getDomainOption(region);
  const batteryDomain = isBatteryDomain(region);
  const dataset = useDatasetData(region);
  const reports = useReportsData();
  const dispatch = useDispatchCompare(region, 24);

  const pendingGateCount = useMemo(
    () => theoremData.theorems.reduce(
      (count, theorem) => count + theorem.promotionGates.filter((gate) => gate.status === 'pending').length,
      0
    ),
    [theoremData.theorems],
  );
  const totalGateCount = useMemo(
    () => theoremData.theorems.reduce((count, theorem) => count + theorem.promotionGates.length, 0),
    [theoremData.theorems],
  );
  const passedGateCount = totalGateCount - pendingGateCount;

  const runtimeRows = (dataset.runtime_summary ?? []) as Array<Record<string, unknown>>;
  const runtimeByController = (controller: string) => runtimeRows.find((row) => row.controller === controller);
  const oriusRuntime = runtimeByController('orius');
  const baselineRuntime = runtimeByController('baseline');
  const baselineTsvr = toFiniteNumber(baselineRuntime, 'tsvr');
  const oriusTsvr = toFiniteNumber(oriusRuntime, 'tsvr');
  const certificateRate = toFiniteNumber(oriusRuntime, 'cva');
  const auditCompleteness = toFiniteNumber(oriusRuntime, 'audit_completeness');
  const oriusOasg = toFiniteNumber(oriusRuntime, 'oasg');
  const interventionRate = toFiniteNumber(oriusRuntime, 'intervention_rate');
  const runtimeStepCount = toFiniteNumber(oriusRuntime, 'n_steps');

  const reportBundle = reports.regions[region];
  const metricsActive = reportBundle?.metrics?.length ? reportBundle.metrics : reports.metrics;
  const impactActive = reportBundle?.impact ?? reports.impact;
  const robustnessActive = reportBundle?.robustness ?? reports.robustness;
  const forecastTarget = currentDomain.primaryTarget;
  const forecastData = dataset.forecast?.[forecastTarget] ?? [];
  const targetMetrics = dataset.metrics.filter((metric) => metric.target === forecastTarget);
  const fallbackMetrics = metricsActive.filter((metric) => metric.target === forecastTarget || metric.target === 'runtime_tsvr');
  const candidateMetrics: MetricEvidence[] = targetMetrics.length ? targetMetrics : fallbackMetrics;
  const bestMetric = candidateMetrics.reduce<MetricEvidence | null>(
    (best, metric) => (!best || metric.rmse < best.rmse ? metric : best),
    null,
  );

  const realImpact = dataset.impact;
  const costSavingsPct = realImpact?.cost_savings_pct ?? impactActive?.cost_savings_pct ?? null;
  const costSavingsUsd =
    realImpact
      ? (realImpact.baseline_cost_usd ?? 0) - (realImpact.orius_cost_usd ?? 0)
      : impactActive?.cost_savings_usd ?? null;
  const carbonReductionPct = realImpact?.carbon_reduction_pct ?? impactActive?.carbon_reduction_pct ?? null;
  const carbonReductionKg =
    realImpact?.baseline_carbon_kg != null && realImpact?.orius_carbon_kg != null
      ? realImpact.baseline_carbon_kg - realImpact.orius_carbon_kg
      : impactActive?.carbon_reduction_kg ?? null;
  const peakShavingPct = realImpact?.peak_shaving_pct ?? impactActive?.peak_shaving_pct ?? null;
  const peakShavingMw =
    realImpact?.baseline_peak_mw != null && realImpact?.orius_peak_mw != null
      ? realImpact.baseline_peak_mw - realImpact.orius_peak_mw
      : impactActive?.peak_shaving_mw ?? null;

  const optimizationSummary = {
    cost_savings_pct: costSavingsPct,
    cost_savings_usd: costSavingsUsd,
    carbon_reduction_pct: carbonReductionPct,
    carbon_reduction_kg: carbonReductionKg,
    peak_shaving_pct: peakShavingPct,
    peak_shaving_mw: peakShavingMw,
  };

  const sourceArtifacts = dataset.source_artifacts ?? [];
  const warningMessages = [
    dataset.error ? `Dataset evidence error: ${dataset.error}` : null,
    reports.error ? `Report evidence error: ${reports.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    ...(reports.meta.warnings ?? []),
    !dataset.loading && !dataset.stats ? `No dataset statistics artifact is available for ${currentDomain.label}.` : null,
  ].filter((message): message is string => Boolean(message));
  const theoremSourceArtifacts = artifactPathsFromText(theoremData.summary.sourcePath);
  const sourceLineageArtifacts = Array.from(new Set([...theoremSourceArtifacts, ...sourceArtifacts])).slice(0, 5);
  const blockingTheorems = useMemo<PromotionBlocker[]>(
    () =>
      theoremData.theorems
        .map((theorem) => {
          const pendingGates = theorem.promotionGates.filter((gate) => gate.status === 'pending');
          if (!pendingGates.length && theorem.tier !== 'draft') return null;
          return {
            theorem,
            pendingGates,
            reason: pendingGates.length
              ? pendingGates.map((gate) => gate.label).join(', ')
              : 'Draft / non-defended theorem row',
          };
        })
        .filter((row): row is PromotionBlocker => row !== null)
        .sort((left, right) => {
          const leftPriority = left.theorem.id === 'T9' || left.theorem.id === 'T10' ? 0 : 1;
          const rightPriority = right.theorem.id === 'T9' || right.theorem.id === 'T10' ? 0 : 1;
          return leftPriority - rightPriority || left.theorem.sortOrder - right.theorem.sortOrder;
        }),
    [theoremData.theorems],
  );

  const theoremEvidencePresent =
    theoremData.theorems.length > 0 &&
    theoremData.summary.flagshipDefended + theoremData.summary.supportingDefended > 0;
  const reproducible =
    !reports.loading &&
    !dataset.loading &&
    reports.meta.source === 'reports' &&
    sourceArtifacts.length > 0 &&
    warningMessages.length === 0;
  const promotionReady = pendingGateCount === 0 && theoremData.summary.draftNonDefended === 0 && reproducible;

  const verdicts = [
    {
      label: 'Valid',
      status: theoremEvidencePresent ? 'Evidence present' : 'Evidence missing',
      tone: theoremEvidencePresent ? 'pass' as const : 'block' as const,
      detail: `${theoremData.summary.flagshipDefended} flagship, ${theoremData.summary.supportingDefended} supporting theorem rows`,
      icon: theoremEvidencePresent ? CheckCircle2 : AlertCircle,
    },
    {
      label: 'Reproducible',
      status: reports.loading || dataset.loading ? 'Checking artifacts' : reproducible ? 'Artifact-backed' : 'Warnings pending',
      tone: reports.loading || dataset.loading ? 'info' as const : reproducible ? 'pass' as const : 'warn' as const,
      detail: `${sourceArtifacts.length} source artifact(s), ${reports.reports.length} report artifact(s)`,
      icon: reproducible ? CheckCircle2 : TriangleAlert,
    },
    {
      label: 'Promotion readiness',
      status: promotionReady ? 'Ready' : 'Blocked / partial',
      tone: promotionReady ? 'pass' as const : pendingGateCount > 0 ? 'block' as const : 'warn' as const,
      detail: `${pendingGateCount} pending gate(s), ${theoremData.summary.draftNonDefended} draft row(s)`,
      icon: promotionReady ? CheckCircle2 : AlertCircle,
    },
  ];

  const domainRows = [
    {
      domain: 'Battery',
      tier: 'Reference witness',
      closure: 'Closed evidence tier',
      status: batteryDomain && !dataset.loading && dataset.stats ? 'Active view' : 'Artifact-backed',
      tone: 'pass' as const,
      evidence: 'T7_Battery, DC3S shield, cost-safety frontier',
    },
    {
      domain: 'AV',
      tier: 'Runtime contract',
      closure: 'Closed with T11 witness',
      status: region === 'AV' ? 'Active view' : 'Tracked',
      tone: pendingGateCount > 0 ? 'warn' as const : 'pass' as const,
      evidence: 'Typed adapter, TSVR replay, runtime comparator summary',
    },
    {
      domain: 'Healthcare',
      tier: 'Runtime contract',
      closure: 'Closed with T11 witness',
      status: region === 'HEALTHCARE' ? 'Active view' : 'Tracked',
      tone: pendingGateCount > 0 ? 'warn' as const : 'pass' as const,
      evidence: 'Source holdout, SpO2 proxy replay, runtime comparator summary',
    },
  ];

  const dispatchOptimized = dataset.dispatch.length
    ? dataset.dispatch.map((point) => ({
        ...point,
        battery_dispatch: 0,
        price_eur_mwh: point.price_eur_mwh ?? undefined,
      }))
    : dispatch.optimized;

  return (
    <div className="mx-auto max-w-[1500px] space-y-6 p-5 md:p-6">
      <section className="glass-panel rounded-xl border border-white/[0.08] bg-grid-panel/80 p-4 md:p-5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <EvidenceChip tone="info">Research Hub</EvidenceChip>
              <EvidenceChip tone={promotionReady ? 'pass' : 'warn'}>{currentDomain.shortLabel} evidence view</EvidenceChip>
            </div>
            <h1 className="mt-3 text-2xl font-semibold tracking-normal text-white md:text-3xl">
              Is ORIUS evidence valid, reproducible, and promotion-ready?
            </h1>
            <p className="mt-2 max-w-4xl text-sm leading-relaxed text-slate-400">
              The console prioritizes theorem gates, domain closure, runtime safety, and artifact lineage before operational grid metrics.
            </p>
          </div>

          <div className="grid min-w-[280px] grid-cols-2 gap-2 text-xs sm:grid-cols-4 xl:min-w-[520px]">
            <CompactMetric label="Gate pass" value={`${passedGateCount}/${totalGateCount}`} tone={pendingGateCount ? 'warn' : 'pass'} />
            <CompactMetric label="Pending" value={String(pendingGateCount)} tone={pendingGateCount ? 'block' : 'pass'} />
            <CompactMetric label="Warnings" value={String(warningMessages.length)} tone={warningMessages.length ? 'warn' : 'pass'} />
            <CompactMetric label="Reports" value={String(reports.reports.length)} tone={reports.meta.source === 'reports' ? 'pass' : 'warn'} />
          </div>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-3 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-xl border border-red-500/20 bg-red-500/[0.045] p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <AlertCircle className="h-4 w-4 text-red-300" />
                  Promotion Blockers
                </div>
                <div className="mt-1 text-xs text-slate-400">
                  {promotionReady
                    ? 'No theorem gate, draft-row, or artifact-warning blocker is active.'
                    : 'Promotion remains blocked until these rows and artifact warnings are cleared.'}
                </div>
              </div>
              <EvidenceChip tone={promotionReady ? 'pass' : 'block'}>
                {promotionReady ? 'promotion-ready' : `${blockingTheorems.length + warningMessages.length} blocker(s)`}
              </EvidenceChip>
            </div>

            <div className="mt-3 space-y-2">
              {blockingTheorems.length === 0 && warningMessages.length === 0 ? (
                <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/[0.04] px-3 py-2 text-xs text-emerald-200">
                  Gate ledger has no blocking theorem rows.
                </div>
              ) : (
                <>
                  {blockingTheorems.slice(0, 4).map(({ theorem, pendingGates, reason }) => (
                    <Link
                      key={theorem.id}
                      href={`/theorems#${theoremAnchorId(theorem.displayId)}`}
                      className="block rounded-lg border border-white/[0.06] bg-black/20 px-3 py-2 transition-colors hover:border-red-300/30 hover:bg-red-500/[0.06]"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="font-mono text-[11px] font-semibold text-red-200">{theorem.displayId}</span>
                        <EvidenceChip tone="block">{pendingGates.length ? `${pendingGates.length} gate(s)` : 'draft'}</EvidenceChip>
                      </div>
                      <div className="mt-1 truncate text-xs font-medium text-white" title={theorem.name}>{theorem.name}</div>
                      <div className="mt-1 text-[11px] leading-relaxed text-slate-400">{reason}</div>
                    </Link>
                  ))}
                  {warningMessages.slice(0, 2).map((message) => (
                    <div key={message} className="rounded-lg border border-amber-500/15 bg-amber-500/[0.04] px-3 py-2 text-[11px] leading-relaxed text-amber-100/90">
                      {message}
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <GitBranch className="h-4 w-4 text-violet-300" />
                Source Lineage Jump Points
              </div>
              <Link href="/reports" className="text-[11px] text-violet-300 hover:text-violet-200">
                Reports
              </Link>
            </div>
            <div className="mt-3 space-y-2">
              {sourceLineageArtifacts.length ? (
                sourceLineageArtifacts.map((artifact) => (
                  <ArtifactLink key={artifact} artifact={artifact} />
                ))
              ) : (
                <div className="rounded-lg bg-black/15 px-3 py-2 text-xs text-slate-500">
                  Source paths will appear after theorem and dataset artifacts load.
                </div>
              )}
            </div>
            <div className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Report artifacts open directly; dataset artifacts resolve to the exact Data Explorer artifact query.
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
          {verdicts.map((verdict) => {
            const Icon = verdict.icon;
            return (
              <div key={verdict.label} className={`rounded-lg border p-4 ${verdictToneClasses[verdict.tone]}`}>
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs font-medium uppercase tracking-wide text-slate-300">{verdict.label}</div>
                  <Icon className="h-4 w-4 shrink-0" />
                </div>
                <div className="mt-2 text-lg font-semibold text-white">{verdict.status}</div>
                <div className="mt-1 text-xs leading-relaxed text-slate-300">{verdict.detail}</div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[1.05fr_1fr_1.1fr]">
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <BookOpen className="h-4 w-4 text-emerald-300" />
                Theorem Status
              </div>
              <Link href="/theorems" className="text-[11px] text-emerald-300 hover:text-emerald-200">
                Theorem Ledger
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <CompactMetric label="Flagship" value={String(theoremData.summary.flagshipDefended)} tone="pass" />
              <CompactMetric label="Supporting" value={String(theoremData.summary.supportingDefended)} tone="info" />
              <CompactMetric label="Draft" value={String(theoremData.summary.draftNonDefended)} tone={theoremData.summary.draftNonDefended ? 'warn' : 'pass'} />
              <CompactMetric label="Pending gates" value={String(pendingGateCount)} tone={pendingGateCount ? 'block' : 'pass'} />
            </div>
            <div className="mt-3 rounded-lg bg-black/15 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
              <span className="text-slate-300">Policy:</span> T9/T10 stay supporting or draft until three-domain constants and assumptions are discharged.
            </div>
          </div>

          <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Globe2 className="h-4 w-4 text-sky-300" />
                Domain Gates
              </div>
              <Link href="/domains" className="text-[11px] text-sky-300 hover:text-sky-200">
                Domain Evidence
              </Link>
            </div>
            <div className="space-y-2">
              {domainRows.map((row) => (
                <div key={row.domain} className="rounded-lg border border-white/[0.05] bg-black/10 px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-medium text-white">{row.domain}</div>
                    <EvidenceChip tone={row.tone}>{row.status}</EvidenceChip>
                  </div>
                  <div className="mt-1 grid grid-cols-1 gap-1 text-[11px] text-slate-500 sm:grid-cols-2">
                    <span>{row.tier}</span>
                    <span>{row.closure}</span>
                  </div>
                  <div className="mt-1 truncate text-[11px] text-slate-400" title={row.evidence}>{row.evidence}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <ShieldCheck className="h-4 w-4 text-amber-300" />
                Runtime Evidence
              </div>
              <Link href="/safety" className="text-[11px] text-amber-300 hover:text-amber-200">
                Safety / OASG / DC3S
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
              <CompactMetric label="ORIUS TSVR" value={formatFractionPercent(oriusTsvr)} tone={oriusTsvr === 0 ? 'pass' : 'warn'} />
              <CompactMetric label="Baseline TSVR" value={formatFractionPercent(baselineTsvr)} tone="info" />
              <CompactMetric label="CVA" value={formatFractionPercent(certificateRate)} tone={certificateRate === null ? 'warn' : 'pass'} />
              <CompactMetric
                label="Audit"
                value={formatFractionPercent(auditCompleteness)}
                tone={auditCompleteness === null ? 'warn' : 'pass'}
                detail={runtimeStepCount === null ? undefined : `${runtimeStepCount.toLocaleString()} runtime rows`}
              />
              <CompactMetric label="OASG" value={oriusOasg === null ? 'Pending' : oriusOasg.toFixed(4)} tone={oriusOasg === 0 ? 'pass' : 'info'} />
              <CompactMetric label="Intervene" value={formatFractionPercent(interventionRate)} tone="info" />
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-white/[0.06] bg-black/10 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-white">
            <GitBranch className="h-4 w-4 text-violet-300" />
            Artifact Lineage
          </div>
          <div className="grid grid-cols-1 gap-x-6 xl:grid-cols-2">
            <SourceRow label="Theorem source" value={theoremData.summary.sourcePath} tone={theoremData.summary.sourcePath ? 'pass' : 'warn'} />
            <SourceRow label="Theorem updated" value={formatMaybeDate(theoremData.summary.sourceUpdatedAt)} tone={theoremData.summary.sourceUpdatedAt ? 'pass' : 'warn'} />
            <SourceRow label="Reports index" value={reports.meta.source === 'reports' ? `${reports.reports.length} reports loaded` : 'reports missing'} tone={reports.meta.source === 'reports' ? 'pass' : 'warn'} href="/reports" />
            <SourceRow label="Reports updated" value={formatMaybeDate(reports.meta.last_updated)} tone={reports.meta.last_updated ? 'pass' : 'warn'} />
            <SourceRow label="Data freshness" value={dataset.stats?.date_range?.end?.slice(0, 10) ?? 'No artifact'} tone={dataset.stats ? 'pass' : 'warn'} href="/data" />
            <SourceRow label="Warning count" value={String(warningMessages.length)} tone={warningMessages.length ? 'warn' : 'pass'} />
          </div>
          {sourceArtifacts.length > 0 && (
            <div className="mt-3 grid grid-cols-1 gap-2 lg:grid-cols-2">
              {sourceArtifacts.slice(0, 4).map((artifact) => (
                <ArtifactLink key={artifact} artifact={artifact} />
              ))}
            </div>
          )}
        </div>
      </section>

      <StatusBanner title="Evidence Status" messages={warningMessages} />

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <CompactMetric
          label="Forecast error witness"
          value={bestMetric ? bestMetric.rmse.toFixed(bestMetric.rmse > 100 ? 0 : 3) : 'Pending'}
          tone={bestMetric ? 'info' : 'warn'}
          detail={`${forecastTarget} / ${currentDomain.primaryUnit}`}
        />
        <CompactMetric
          label="Uncertainty coverage"
          value={typeof bestMetric?.coverage_90 === 'number' ? formatPercent(bestMetric.coverage_90) : 'Pending'}
          tone={typeof bestMetric?.coverage_90 === 'number' ? 'pass' : 'warn'}
          detail={bestMetric?.model ?? 'Artifact metric row'}
        />
        <CompactMetric
          label="Tradeoff delta"
          value={formatMaybePercent(costSavingsPct)}
          tone={costSavingsPct === null ? 'warn' : 'pass'}
          detail={formatMaybeCurrency(costSavingsUsd)}
        />
        <CompactMetric
          label="Stress regret p95"
          value={formatMaybeCurrency(robustnessActive?.p95_regret)}
          tone={robustnessActive?.p95_regret === null || robustnessActive?.p95_regret === undefined ? 'warn' : 'info'}
          detail={robustnessActive ? `Infeasible ${formatFractionPercent(robustnessActive.infeasible_rate)}` : 'Stress artifact pending'}
        />
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <LineChart className="h-4 w-4 text-energy-info" />
            Supporting Evidence Traces
          </div>
          <EvidenceChip tone="info">below promotion gates</EvidenceChip>
        </div>
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <ForecastChart
            data={forecastData.length ? forecastData : undefined}
            target={forecastTarget}
            zoneId={region}
            unit={currentDomain.primaryUnit}
            metrics={
              bestMetric
                ? { rmse: bestMetric.rmse, coverage_90: bestMetric.coverage_90 ?? undefined, model: bestMetric.model }
                : undefined
            }
            loading={dataset.loading}
          />
          <DispatchChart
            optimized={dispatchOptimized}
            baseline={dispatch.baseline}
            title={`Runtime Tradeoff Trace - ${currentDomain.label}`}
            showBaseline={!!dispatch.baseline}
          />
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <CarbonCostPanel
          data={dataset.pareto.length ? dataset.pareto : undefined}
          zoneId={region}
          summary={optimizationSummary}
        />
        <Panel
          title="Evidence Routing"
          subtitle="Primary ORIUS research structure"
          badge="8 surfaces"
          accentColor="info"
        >
          <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
            {[
              ['Research Hub', '/', Database],
              ['Theorem Ledger', '/theorems', BookOpen],
              ['Safety / OASG / DC3S', '/safety', ShieldCheck],
              ['Domain Evidence', '/domains', Globe2],
              ['Forecasting Evidence', '/forecasting', LineChart],
              ['Optimization / Cost-Safety Tradeoff', '/optimization', Scale],
              ['Reports', '/reports', FileText],
              ['Data Explorer', '/data', BarChart3],
            ].map(([label, href, Icon]) => {
              const RouteIcon = Icon as typeof Database;
              return (
                <Link
                  key={label as string}
                  href={href as string}
                  className="flex items-center justify-between gap-3 rounded-lg border border-white/[0.05] bg-white/[0.025] px-3 py-2 text-slate-300 transition-colors hover:border-energy-primary/30 hover:text-white"
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <RouteIcon className="h-3.5 w-3.5 shrink-0 text-energy-info" />
                    <span className="truncate">{label as string}</span>
                  </span>
                  <Zap className="h-3 w-3 shrink-0 text-slate-600" />
                </Link>
              );
            })}
          </div>
          <div className="mt-4 rounded-lg bg-black/15 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
            Legacy carbon, anomaly, and monitoring views now resolve into Optimization and Safety evidence pages while their artifacts remain available to the research console.
          </div>
          <div className="mt-4 grid grid-cols-1 gap-2">
            {DOMAIN_OPTIONS.map((option) => (
              <div key={option.id} className="flex items-center justify-between rounded-lg bg-white/[0.025] px-3 py-2 text-xs">
                <span className="text-slate-300">{option.label}</span>
                <EvidenceChip tone={option.id === region ? 'pass' : 'info'}>{option.family}</EvidenceChip>
              </div>
            ))}
          </div>
        </Panel>
      </section>
    </div>
  );
}
