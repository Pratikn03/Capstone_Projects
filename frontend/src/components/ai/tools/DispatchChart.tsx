'use client';

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ReferenceLine,
} from 'recharts';
import { motion } from 'framer-motion';
import { useState } from 'react';

interface DispatchData {
  timestamp: string;
  load_mw: number;
  generation_solar: number;
  generation_wind: number;
  generation_gas: number;
  battery_dispatch: number;
  price_eur_mwh?: number;
  carbon_kg_mwh?: number;
}

interface DispatchChartProps {
  optimized: DispatchData[];
  baseline?: DispatchData[];
  title: string;
  showBaseline?: boolean;
  defaultMode?: 'optimized' | 'baseline';
}

function formatTime(ts: string) {
  const d = new Date(ts);
  const h = d.getUTCHours();
  return `${h.toString().padStart(2, '0')}:00`;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-panel rounded-lg p-3 text-xs min-w-[180px]">
      <div className="font-semibold text-white mb-2">{formatTime(label)}</div>
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="font-mono text-white">{Math.round(p.value).toLocaleString()} MW</span>
        </div>
      ))}
    </div>
  );
}

export function DispatchChart({
  optimized,
  baseline,
  title,
  showBaseline = false,
  defaultMode = 'optimized',
}: DispatchChartProps) {
  const allowBaseline = Boolean(baseline && baseline.length) && showBaseline;
  const [mode, setMode] = useState<'optimized' | 'baseline'>(allowBaseline ? defaultMode : 'optimized');
  const chartData = mode === 'baseline' && baseline ? baseline : optimized;
  const peakLoad = Math.max(...chartData.map((d) => d.load_mw));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <div className="flex items-center gap-2">
          {allowBaseline && (
            <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5">
              {(['optimized', 'baseline'] as const).map((view) => (
                <button
                  key={view}
                  onClick={() => setMode(view)}
                  className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${
                    mode === view
                      ? 'bg-energy-primary/20 text-energy-primary'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  {view === 'optimized' ? 'GridPulse Optimized' : 'Baseline'}
                </button>
              ))}
            </div>
          )}
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
            {mode === 'optimized' ? 'Live Telemetry' : 'Baseline Mode'}
          </span>
          <span className="text-[10px] text-slate-500">
            Peak: {peakLoad.toLocaleString()} MW
          </span>
        </div>
      </div>

      <div className="px-5 py-4 h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="gradSolar" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="gradWind" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="gradGas" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="gradBattery" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#a855f7" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={formatTime} />
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
              formatter={(v: string) => <span className="text-slate-300">{v}</span>}
            />

            {/* Stacked generation */}
            <Area type="monotone" dataKey="generation_gas" stackId="gen" stroke="#ef4444" fill="url(#gradGas)" name="Gas / Thermal" />
            <Area type="monotone" dataKey="generation_wind" stackId="gen" stroke="#3b82f6" fill="url(#gradWind)" name="Wind" />
            <Area type="monotone" dataKey="generation_solar" stackId="gen" stroke="#fbbf24" fill="url(#gradSolar)" name="Solar" />
            <Area type="monotone" dataKey="battery_dispatch" stackId="gen" stroke="#a855f7" fill="url(#gradBattery)" name="Battery" />

            {/* Load line overlay */}
            <Area
              type="monotone" dataKey="load_mw" stroke="#f1f5f9" fill="none"
              strokeWidth={2} strokeDasharray="6 3" name="Net Load"
            />

            {/* Peak reference */}
            <ReferenceLine y={peakLoad * 1.05} stroke="#ef444480" strokeDasharray="4 4" label="" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="px-5 pb-3 text-[11px] text-slate-500">
        {mode === 'optimized'
          ? 'Optimization shifts charging to solar peaks and discharges during evening ramps to reduce thermal output.'
          : 'Baseline dispatch meets net load without battery support or carbon-aware scheduling.'}
      </div>
    </motion.div>
  );
}
