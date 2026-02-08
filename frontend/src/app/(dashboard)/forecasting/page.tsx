'use client';

import { useState } from 'react';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { Panel } from '@/components/ui/Panel';
import { mockForecastWithPI } from '@/lib/api/mock-data';
import { useReportsData } from '@/lib/api/reports-client';

const targets = [
  { id: 'load_mw', label: 'Load', icon: '⚡' },
  { id: 'wind_mw', label: 'Wind', icon: '💨' },
  { id: 'solar_mw', label: 'Solar', icon: '☀️' },
] as const;

export default function ForecastingPage() {
  const [selectedTarget, setSelectedTarget] = useState<string>('load_mw');
  const [selectedRegion, setSelectedRegion] = useState<string>('DE');
  const { metrics } = useReportsData();
  const data = mockForecastWithPI(selectedTarget, 48);
  const formatMaybe = (value: number | undefined, digits: number) => (value === undefined ? 'N/A' : value.toFixed(digits));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Probabilistic Forecasting</h1>

        <div className="flex items-center gap-3">
          {/* Target toggle */}
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10">
            {targets.map((t) => (
              <button
                key={t.id}
                onClick={() => setSelectedTarget(t.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  selectedTarget === t.id
                    ? 'bg-energy-primary/20 text-energy-primary'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          {/* Region toggle */}
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10">
            {[
              { id: 'DE', label: '🇩🇪 Germany' },
              { id: 'US', label: '🇺🇸 USA' },
            ].map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedRegion(r.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  selectedRegion === r.id
                    ? 'bg-energy-info/20 text-energy-info'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <ForecastChart data={data} target={selectedTarget} zoneId={selectedRegion} />

      {/* Metrics table */}
      <Panel title="Model Comparison" subtitle="All models for selected target" badge="Conformal PI" badgeColor="info">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE (MW)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAE (MW)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAPE (%)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">R²</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">90% PICP</th>
              </tr>
            </thead>
            <tbody>
              {metrics
                .filter((m) => m.target === selectedTarget)
                .map((m, i) => {
                  const isBest = m.rmse === Math.min(
                    ...metrics.filter((x) => x.target === selectedTarget).map((x) => x.rmse)
                  );
                  return (
                    <tr key={i} className={`border-b border-white/5 ${isBest ? 'bg-energy-primary/5' : ''}`}>
                      <td className="py-2.5 px-3 text-white font-medium">
                        {isBest && <span className="text-energy-primary mr-1">★</span>}
                        {m.model}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-white">{m.rmse.toFixed(1)}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">{m.mae.toFixed(1)}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">{formatMaybe(m.mape, 1)}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-energy-primary">{formatMaybe(m.r2, 3)}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                        {m.coverage_90 === undefined ? 'N/A' : `${m.coverage_90.toFixed(1)}%`}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
