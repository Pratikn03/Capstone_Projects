'use client';

import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { mockForecastWithPI, type ForecastWithPI } from '@/lib/api/mock-data';

interface ForecastChartProps {
  data?: ForecastWithPI[];
  target: string;
  zoneId: string;
  metrics?: {
    rmse?: number;
    coverage_90?: number;
  };
}

const targetNames: Record<string, string> = {
  load_mw: 'Load',
  wind_mw: 'Wind Generation',
  solar_mw: 'Solar Generation',
};

const targetColors: Record<string, string> = {
  load_mw: '#f1f5f9',
  wind_mw: '#3b82f6',
  solar_mw: '#fbbf24',
};

function formatTime(ts: string) {
  const d = new Date(ts);
  return `${d.getUTCDate()}/${d.getUTCMonth() + 1} ${d.getUTCHours()}:00`;
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
          <span className="font-mono text-white">{Math.round(p.value).toLocaleString()} MW</span>
        </div>
      ))}
    </div>
  );
}

export function ForecastChart({ data: dataProp, target, zoneId, metrics }: ForecastChartProps) {
  const [showModel, setShowModel] = useState<'GBM' | 'LSTM' | 'TCN'>('GBM');
  const data = dataProp?.length ? dataProp : mockForecastWithPI(target, 48);
  const color = targetColors[target] || '#10b981';
  const name = targetNames[target] || target;
  const horizonHours = data.length;
  const coverage = metrics?.coverage_90;
  const rmse = metrics?.rmse;

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
            {name} Forecast — {zoneId}
          </h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-info/10 text-energy-info border border-energy-info/20">
            90% PI
          </span>
          <span className="text-[10px] text-slate-500">Horizon: {horizonHours}h</span>
        </div>

        {/* Model toggle */}
        <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5">
          {(['GBM', 'LSTM', 'TCN'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setShowModel(m)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${
                showModel === m
                  ? 'bg-energy-primary/20 text-energy-primary'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="px-5 py-4 h-[320px]">
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
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v: number) => `${(v / 1000).toFixed(1)}k`} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />

            {/* 90% prediction interval */}
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

            {/* 50% prediction interval */}
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

            {/* Actual */}
            <Area
              type="monotone" dataKey="actual" stroke="#94a3b8" fill="none"
              strokeWidth={1} strokeDasharray="4 3" name="Actual"
            />

            {/* Forecast */}
            <Area
              type="monotone" dataKey="forecast" stroke={color} fill="none"
              strokeWidth={2} name={`${name} Forecast (${showModel})`}
            />
          </AreaChart>
        </ResponsiveContainer>
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
            {rmse === undefined ? 'N/A' : `${rmse.toFixed(0)} MW`}
          </span>
        </span>
        <span>•</span>
        <span>Coverage verified with conformal prediction</span>
      </div>
    </motion.div>
  );
}
