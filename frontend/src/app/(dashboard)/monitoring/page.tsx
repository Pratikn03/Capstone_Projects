'use client';

import { MLOpsMonitor } from '@/components/charts/MLOpsMonitor';
import { Panel } from '@/components/ui/Panel';
import { mockDriftData } from '@/lib/api/mock-data';
import { useReportsData } from '@/lib/api/reports-client';

export default function MonitoringPage() {
  const drift = mockDriftData(30);
  const { metrics } = useReportsData();
  const formatMaybe = (value: number | undefined, digits: number) => (value === undefined ? 'N/A' : value.toFixed(digits));

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-white">Model Monitoring & MLOps</h1>

      <MLOpsMonitor data={drift} />

      <Panel title="Active Model Versions" subtitle="Production Registry">
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
              {metrics.map((m, i) => (
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
    </div>
  );
}
