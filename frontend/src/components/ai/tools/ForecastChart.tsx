'use client';

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { motion } from 'framer-motion';
import type { ForecastWithPI } from '@/lib/api/mock-data';

interface ForecastChartProps {
  data?: ForecastWithPI[];
  target: string;
  zoneId: string;
  unit?: string;
  metrics?: {
    rmse?: number;
    coverage_90?: number;
    model?: string;
  };
  loading?: boolean;
}

const targetNames: Record<string, string> = {
  load_mw: 'Load',
  wind_mw: 'Wind Generation',
  solar_mw: 'Solar Generation',
  true_margin: 'True Safety Margin',
  safe_acceleration_mps2: 'Safe Acceleration',
  reliability_w: 'Reliability',
  spo2_proxy: 'SpO₂ Proxy',
  forecast: 'SpO₂ Prediction',
  reliability: 'Reliability',
};

const targetColors: Record<string, string> = {
  load_mw: '#f1f5f9',
  wind_mw: '#3b82f6',
  solar_mw: '#fbbf24',
  true_margin: '#38bdf8',
  safe_acceleration_mps2: '#f97316',
  reliability_w: '#10b981',
  spo2_proxy: '#ec4899',
  forecast: '#a78bfa',
  reliability: '#10b981',
};

function formatTime(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return `${d.getUTCDate()}/${d.getUTCMonth() + 1} ${d.getUTCHours()}:00`;
}

function formatValue(value: unknown, unit: string): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
  if (unit === 'MW') return `${Math.round(value).toLocaleString()} MW`;
  return `${value.toFixed(Math.abs(value) < 10 ? 3 : 2)}${unit ? ` ${unit}` : ''}`;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-panel rounded-lg p-3 text-xs min-w-[200px]">
      <div className="font-semibold text-white mb-2">{formatTime(label)}</div>
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.stroke }} />
            {p.name}
          </span>
          <span className="font-mono text-white">{formatValue(p.value, p.payload?.unit ?? '')}</span>
        </div>
      ))}
    </div>
  );
}

export function ForecastChart({ data: dataProp, target, zoneId, unit = target.endsWith('_mw') ? 'MW' : '', metrics, loading = false }: ForecastChartProps) {
  const data = dataProp?.length ? dataProp.map((row) => ({ ...row, unit })) : [];
  const color = targetColors[target] || '#10b981';
  const name = targetNames[target] || target;
  const title = /\b(forecast|prediction)\b/i.test(name) ? `${name} — ${zoneId}` : `${name} Forecast — ${zoneId}`;
  const horizonHours = data.length;
  const coverage = metrics?.coverage_90;
  const rmse = metrics?.rmse;
  const modelLabel = metrics?.model;
  const hasPredictionIntervals = data.some(
    (row) =>
      typeof row.upper_90 === 'number' &&
      typeof row.lower_90 === 'number' &&
      typeof row.upper_50 === 'number' &&
      typeof row.lower_50 === 'number'
  );

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">
            {title}
          </h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-info/10 text-energy-info border border-energy-info/20">
            90% PI
          </span>
          <span className="text-[10px] text-slate-500">Horizon: {horizonHours || 0}h</span>
        </div>

        <div className="text-[10px] text-slate-500">
          {modelLabel ? `Active metric row: ${modelLabel}` : 'Artifact forecast trace'}
        </div>
      </div>

      <div className="px-5 py-4 h-[320px]">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id={`grad90_${target}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.15} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id={`grad50_${target}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.05} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={formatTime} />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 10 }}
                tickFormatter={(v: number) => (unit === 'MW' ? `${(v / 1000).toFixed(1)}k` : v.toFixed(Math.abs(v) < 10 ? 2 : 0))}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />

              {hasPredictionIntervals && (
                <>
                  <Area
                    type="monotone" dataKey="upper_90" stroke="none"
                    fill={`url(#grad90_${target})`} name="90% PI Upper"
                    legendType="none"
                  />
                  <Area
                    type="monotone" dataKey="lower_90" stroke="none"
                    fill={`url(#grad90_${target})`} name="90% PI Lower"
                    legendType="none"
                  />
                  <Area
                    type="monotone" dataKey="upper_50" stroke="none"
                    fill={`url(#grad50_${target})`} name="50% PI"
                    legendType="none"
                  />
                  <Area
                    type="monotone" dataKey="lower_50" stroke="none"
                    fill="transparent" name=""
                    legendType="none"
                  />
                </>
              )}
              <Area
                type="monotone" dataKey="actual" stroke="#94a3b8" fill="none"
                strokeWidth={1} strokeDasharray="4 3" name="Actual"
              />
              <Area
                type="monotone" dataKey="forecast" stroke={color} fill="none"
                strokeWidth={2} name={`${name} Forecast`}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : loading ? (
          <div className="flex h-full items-center justify-center text-xs text-slate-500">
            Loading artifact forecast trace...
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500">
            No forecast trace available for this view.
          </div>
        )}
      </div>

      {/* Coverage badge */}
      <div className="px-5 pb-3 flex items-center gap-3 text-xs text-slate-500">
        <span>
          PICP:{' '}
          <span className="text-energy-primary font-mono">
            {coverage === undefined ? 'N/A' : `${coverage.toFixed(1)}%`}
          </span>
        </span>
        <span>•</span>
        <span>
          RMSE:{' '}
          <span className="text-white font-mono">
            {rmse === undefined ? 'N/A' : formatValue(rmse, unit)}
          </span>
        </span>
        <span>•</span>
        <span>{data.length ? (hasPredictionIntervals ? 'Coverage verified with conformal prediction' : 'Real artifact actual-vs-forecast trace') : loading ? 'Loading artifact data' : 'Chart hidden until artifact data is available'}</span>
      </div>
    </motion.div>
  );
}
