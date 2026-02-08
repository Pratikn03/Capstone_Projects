'use client';

import {
  ComposedChart, Area, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';
import { motion } from 'framer-motion';
import type { BatteryState } from '@/lib/api/schema';

interface BatterySOCChartProps {
  schedule: BatteryState[];
  metrics: {
    cost_savings_eur: number;
    carbon_reduction_kg: number;
    peak_shaving_pct: number;
    avg_efficiency: number;
  };
}

function formatTime(ts: string) {
  const d = new Date(ts);
  return `${d.getUTCHours().toString().padStart(2, '0')}:00`;
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
          <span className="font-mono text-white">
            {p.name === 'SOC' ? `${p.value.toFixed(1)}%` : `${Math.round(p.value).toLocaleString()} MW`}
          </span>
        </div>
      ))}
    </div>
  );
}

export function BatterySOCChart({ schedule, metrics }: BatterySOCChartProps) {
  const chartData = schedule.map((s) => ({
    ...s,
    charge: s.power_mw < 0 ? Math.abs(s.power_mw) : 0,
    discharge: s.power_mw > 0 ? s.power_mw : 0,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">Battery State of Charge</h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-battery/10 text-battery border border-battery/20">
            {schedule[0]?.capacity_mwh?.toLocaleString()} MWh
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-slate-500">Efficiency: <span className="text-white font-mono">{metrics.avg_efficiency}%</span></span>
          <span className="text-slate-500">Peak Shaving: <span className="text-energy-primary font-mono">{metrics.peak_shaving_pct}%</span></span>
        </div>
      </div>

      <div className="px-5 py-4 h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="gradSOC" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#a855f7" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={formatTime} />
            <YAxis yAxisId="soc" stroke="#a855f7" tick={{ fontSize: 10 }} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
            <YAxis yAxisId="power" orientation="right" stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v: number) => `${(v/1000).toFixed(1)}k`} />

            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />

            {/* SOC constraint lines */}
            <ReferenceLine yAxisId="soc" y={90} stroke="#a855f780" strokeDasharray="4 4" />
            <ReferenceLine yAxisId="soc" y={15} stroke="#ef444480" strokeDasharray="4 4" />

            {/* Charge/Discharge bars */}
            <Bar yAxisId="power" dataKey="charge" fill="#10b981" opacity={0.6} name="Charging" radius={[2, 2, 0, 0]} />
            <Bar yAxisId="power" dataKey="discharge" fill="#f59e0b" opacity={0.6} name="Discharging" radius={[2, 2, 0, 0]} />

            {/* SOC line */}
            <Area
              yAxisId="soc" type="monotone" dataKey="soc_percent"
              stroke="#a855f7" fill="url(#gradSOC)" strokeWidth={2.5}
              name="SOC"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics bar */}
      <div className="px-5 pb-3 flex items-center gap-6 text-xs">
        <div>
          <span className="text-slate-500">Cost Savings</span>
          <span className="ml-2 text-energy-primary font-semibold">€{metrics.cost_savings_eur.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-slate-500">Carbon Reduction</span>
          <span className="ml-2 text-energy-primary font-semibold">{(metrics.carbon_reduction_kg / 1000).toFixed(1)} tCO₂</span>
        </div>
      </div>
    </motion.div>
  );
}
