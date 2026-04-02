'use client';

import { Panel } from '@/components/ui/Panel';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, AreaChart, Area,
} from 'recharts';

export default function DataExplorerPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region as 'DE' | 'US');
  const regionLabel = region === 'US' ? 'USA (EIA-930)' : 'Germany (OPSD)';

  const stats = dataset.stats;
  const timeseries = dataset.timeseries;
  const profilesMap = dataset.profiles;
  const registry = dataset.registry;
  const forecastMap = dataset.forecast;

  const targets = stats?.targets_summary ?? stats?.targets ?? {};
  const targetNames = Object.keys(targets);

  // Flatten profiles: pick first target that has data
  const profileKey = Object.keys(profilesMap)[0] ?? '';
  const profileData = profilesMap[profileKey] ?? [];

  // Flatten forecast: pick first target
  const forecastKey = Object.keys(forecastMap)[0] ?? '';
  const forecastData = forecastMap[forecastKey] ?? [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Data Explorer</h1>
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
            {regionLabel}
          </span>
          {stats && (
            <span className="text-slate-500">
              {stats.rows.toLocaleString()} rows × {stats.columns} features
            </span>
          )}
        </div>
      </div>

      {dataset.loading && (
        <div className="text-sm text-slate-400">Loading dataset…</div>
      )}

      {/* Dataset Overview Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Observations" value={stats.rows.toLocaleString()} />
          <StatCard label="Features" value={stats.columns.toString()} />
          <StatCard label="Start Date" value={stats.date_range?.start?.slice(0, 10) ?? 'N/A'} />
          <StatCard label="End Date" value={stats.date_range?.end?.slice(0, 10) ?? 'N/A'} />
        </div>
      )}

      {/* Target Variable Statistics */}
      {targetNames.length > 0 && (
        <Panel title="Target Variables" subtitle="Summary statistics for forecasting targets">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Target</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Mean</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Std Dev</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Min</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Max</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Non-zero %</th>
                </tr>
              </thead>
              <tbody>
                {targetNames.map((name) => {
                  const t = targets[name];
                  return (
                    <tr key={name} className="border-b border-white/5 hover:bg-white/3">
                      <td className="py-2.5 px-3 text-white font-medium">{name}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-white">
                        {t.mean?.toFixed(1) ?? 'N/A'}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                        {t.std?.toFixed(1) ?? 'N/A'}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                        {t.min?.toFixed(1) ?? 'N/A'}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-energy-primary">
                        {t.max?.toFixed(1) ?? 'N/A'}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                        {t.non_zero_pct !== undefined ? `${t.non_zero_pct.toFixed(1)}%` : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {/* Time Series Preview */}
      {timeseries.length > 0 && (
        <Panel title="Time Series Preview" subtitle="Last 168 hours of actual generation">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeseries}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(v: string) => v?.slice(11, 16) ?? ''}
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fontSize: 10 }}
                  interval={23}
                />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(10,15,30,0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                {timeseries[0]?.load_mw !== undefined && (
                  <Area
                    type="monotone" dataKey="load_mw" name="Load"
                    stroke="#10b981" fill="#10b98120" strokeWidth={1.5}
                  />
                )}
                {timeseries[0]?.solar_mw !== undefined && (
                  <Area
                    type="monotone" dataKey="solar_mw" name="Solar"
                    stroke="#f59e0b" fill="#f59e0b15" strokeWidth={1.5}
                  />
                )}
                {timeseries[0]?.wind_mw !== undefined && (
                  <Area
                    type="monotone" dataKey="wind_mw" name="Wind"
                    stroke="#3b82f6" fill="#3b82f620" strokeWidth={1.5}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {/* Hourly Profiles */}
      {profileData.length > 0 && (
        <Panel title="Hourly Generation Profiles" subtitle="Average MW by hour of day">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={profileData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="hour"
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v: number) => `${v}:00`}
                />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(10,15,30,0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="mean" name={profileKey || 'Mean'} fill="#10b981" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {/* Forecast vs Actual */}
      {forecastData.length > 0 && (
        <Panel title="Forecast vs Actual" subtitle="72-hour comparison from latest model run">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(v: string) => v?.slice(11, 16) ?? ''}
                  stroke="rgba(255,255,255,0.3)"
                  tick={{ fontSize: 10 }}
                  interval={11}
                />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(10,15,30,0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="actual" name="Actual" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="forecast" name="Predicted" stroke="#a78bfa" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {/* Model Registry */}
      {registry.length > 0 && (
        <Panel title="Trained Models" subtitle={`${registry.length} models in registry`}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Target</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Path</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">Size</th>
                </tr>
              </thead>
              <tbody>
                {registry.map((r, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/3">
                    <td className="py-2.5 px-3 text-white">{r.target}</td>
                    <td className="py-2.5 px-3 text-slate-300">{r.model}</td>
                    <td className="py-2.5 px-3 text-slate-500 font-mono text-[10px]">{r.path ?? r.file}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-energy-primary">
                      {(r.size_bytes / 1024 / 1024).toFixed(1)} MB
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-4 py-3 rounded-xl bg-white/3 border border-white/6">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-white font-mono font-semibold text-sm">{value}</div>
    </div>
  );
}
