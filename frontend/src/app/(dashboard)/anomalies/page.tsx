'use client';

import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { Panel } from '@/components/ui/Panel';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import { mockAnomalies, mockAnomalyZScores } from '@/lib/api/mock-data';

export default function AnomaliesPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region as 'DE' | 'US');
  const anomalies = mockAnomalies();
  const zScores = mockAnomalyZScores(72);
  const regionLabel = region === 'US' ? 'USA' : 'Germany';

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
