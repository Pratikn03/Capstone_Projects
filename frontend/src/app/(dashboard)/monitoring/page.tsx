'use client';

import { MLOpsMonitor } from '@/components/charts/MLOpsMonitor';
import { ArchitectureDiagram } from '@/components/charts/ArchitectureDiagram';
import { Panel } from '@/components/ui/Panel';
import { mockDriftData } from '@/lib/api/mock-data';
import { useReportsData } from '@/lib/api/reports-client';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData, type DriftPoint } from '@/lib/api/dataset-client';

export default function MonitoringPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region as 'DE' | 'US');
  const { metrics: reportsMetrics, regions } = useReportsData();
  const formatMaybe = (value: number | undefined, digits: number) => (value === undefined ? 'N/A' : value.toFixed(digits));
  const regionLabel = region === 'US' ? 'USA' : 'Germany';

  // Use real monitoring data from extracted JSON, fallback to mock
  const realMonitoring = dataset.monitoring;
  const drift: DriftPoint[] = realMonitoring?.drift_timeline ?? mockDriftData(30);

  // Prefer real dataset metrics, then reports metrics
  const realMetrics = dataset.metrics.length ? dataset.metrics : [];
  const regionReportsMetrics = regions[region]?.metrics ?? reportsMetrics;
  const displayMetrics: Array<{ target: string; model: string; rmse: number; mae: number; mape?: number | null; r2?: number; coverage_90?: number }> =
    realMetrics.length ? realMetrics : regionReportsMetrics;

  // Real model registry
  const registry = dataset.registry;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Model Monitoring & MLOps</h1>
        {dataset.stats && (
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
              {regionLabel}
            </span>
            <span className="text-slate-500">
              {registry.length} models registered
            </span>
          </div>
        )}
      </div>

      <MLOpsMonitor data={drift} />

      <Panel title="Active Model Versions" subtitle={`Production Registry — ${regionLabel}`}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Target</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">R²</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">Coverage</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {displayMetrics.map((m, i) => (
                <tr key={i} className="border-b border-white/5 hover:bg-white/3 transition-colors">
                  <td className="py-2.5 px-3 text-white">{m.target}</td>
                  <td className="py-2.5 px-3 text-slate-300">{m.model}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-white">{m.rmse.toFixed(1)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-energy-primary">{formatMaybe(m.r2, 3)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                    {m.coverage_90 === undefined ? 'N/A' : `${m.coverage_90.toFixed(1)}%`}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
                      Active
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Model Registry with file sizes */}
      {registry.length > 0 && (
        <Panel title="Model Artifacts" subtitle="File sizes & training dates">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {registry.map((r, i) => (
              <div key={i} className="px-3 py-2.5 rounded-lg bg-white/3 flex items-center justify-between">
                <div>
                  <div className="text-white text-xs font-medium">{r.target} — {r.model}</div>
                  <div className="text-slate-500 text-[10px] mt-0.5">{r.path ?? r.file}</div>
                </div>
                <div className="text-right">
                  <div className="text-energy-primary font-mono text-xs">
                    {(r.size_bytes / 1024 / 1024).toFixed(1)} MB
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* System Architecture */}
      <Panel title="System Architecture" subtitle="End-to-end ML pipeline">
        <ArchitectureDiagram />
      </Panel>
    </div>
  );
}
