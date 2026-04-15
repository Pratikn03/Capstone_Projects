'use client';

import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import Link from 'next/link';
import { ShieldCheck, ChevronRight } from 'lucide-react';

export default function AnomaliesPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region as 'DE' | 'US');
  
  // Use real extracted data
  const anomalies = dataset.anomalies;
  const zScores = dataset.zscores.map((z) => ({
    timestamp: z.timestamp,
    z_score: z.z_score,
    is_anomaly: z.is_anomaly,
    residual_mw: z.residual_mw,
  }));
  const regionLabel = region === 'US' ? 'USA' : 'Germany';
  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    !zScores.length ? 'No anomaly time-series artifact is available for this region.' : null,
    !anomalies.length ? 'No anomaly event log entries were found for this region.' : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Anomaly Detection</h1>
        {dataset.stats && (
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
              {regionLabel}
            </span>
            <span className="text-slate-500">
              {dataset.stats.rows.toLocaleString()} observations
            </span>
          </div>
        )}
      </div>

      {/* ORIUS DC3S Detection Context */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-500/5 via-transparent to-transparent border border-amber-500/10">
        <ShieldCheck className="w-4 h-4 text-amber-400 shrink-0" />
        <span className="text-[11px] text-slate-400">
          <span className="text-amber-400 font-semibold">DC3S Stage 1 — Detect</span> · OQE fault taxonomy: dropout, stale, spike, delay, drift · <span className="text-white/70">T6 (conformal → anomaly)</span> flags out-of-distribution inputs · z-score residual analysis
        </span>
        <Link href="/safety" className="ml-auto flex items-center gap-1 text-[10px] text-amber-400/70 hover:text-amber-400 transition-colors shrink-0">
          DC3S pipeline <ChevronRight className="w-3 h-3" />
        </Link>
      </div>

      <StatusBanner title="Anomaly View Status" messages={statusMessages} />

      <AnomalyTimeline data={zScores} />

      <Panel title="Event Log" badge={`${anomalies.length} events`} badgeColor="warn">
        <AnomalyList anomalies={anomalies} />
      </Panel>

      {/* Dataset Quality Overview */}
      {dataset.stats && (
        <Panel title="Data Quality" subtitle={`${regionLabel} dataset statistics`}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Total Rows</div>
              <div className="text-white font-mono font-medium">{dataset.stats.rows.toLocaleString()}</div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Features</div>
              <div className="text-white font-mono font-medium">{dataset.stats.columns}</div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Non-zero %</div>
              <div className="text-energy-primary font-mono font-medium">
                {(() => {
                  const tgt = dataset.stats?.targets_summary ?? dataset.stats?.targets;
                  const loadKey = tgt ? Object.keys(tgt).find(k => k.includes('load')) : undefined;
                  const val = loadKey ? tgt?.[loadKey]?.non_zero_pct : undefined;
                  return val !== undefined ? `${val.toFixed(1)}%` : 'N/A';
                })()}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Date Range</div>
              <div className="text-white font-mono font-medium text-[10px]">
                {dataset.stats.date_range?.start?.slice(0, 10) ?? 'N/A'} → {dataset.stats.date_range?.end?.slice(0, 10) ?? 'N/A'}
              </div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
