'use client';

import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Line, ComposedChart,
} from 'recharts';
import { motion } from 'framer-motion';
import type { ParetoPoint } from '@/lib/api/mock-data';

interface CarbonCostPanelProps {
  data: ParetoPoint[];
  zoneId: string;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ParetoPoint;
  return (
    <div className="glass-panel rounded-lg p-3 text-xs min-w-[200px]">
      <div className="font-semibold text-white mb-2">Carbon Weight: {d.carbon_weight}</div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Total Cost</span>
        <span className="font-mono text-white">€{d.total_cost_eur.toLocaleString()}</span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Total Carbon</span>
        <span className="font-mono text-white">{(d.total_carbon_kg / 1000).toFixed(1)} tCO₂</span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Cost Savings</span>
        <span className={`font-mono ${d.cost_savings_pct >= 0 ? 'text-energy-primary' : 'text-energy-alert'}`}>{d.cost_savings_pct}%</span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Carbon Reduction</span>
        <span className="font-mono text-energy-primary">{d.carbon_reduction_pct}%</span>
      </div>
    </div>
  );
}

export function CarbonCostPanel({ data, zoneId }: CarbonCostPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">Cost–Carbon Trade-off</h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
            Pareto Frontier
          </span>
        </div>
        <span className="text-xs text-slate-500">{zoneId}</span>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-3 gap-4 px-5 pt-4">
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-primary stat-value">17.4%</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Cost Savings</div>
          <div className="text-xs text-slate-400 font-mono">€23,500</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-primary stat-value">32.6%</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Carbon Reduction</div>
          <div className="text-xs text-slate-400 font-mono">47.8 tCO₂</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-info stat-value">19.5%</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Peak Shaving</div>
          <div className="text-xs text-slate-400 font-mono">5,000 MW</div>
        </div>
      </div>

      {/* Pareto chart */}
      <div className="px-5 py-4 h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="total_carbon_kg"
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}t`}
              label={{ value: 'Carbon (tCO₂)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              dataKey="total_cost_eur"
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => `€${(v / 1000).toFixed(0)}k`}
              label={{ value: 'Cost (€)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />

            <Line
              type="monotone" dataKey="total_cost_eur"
              stroke="#10b981" strokeWidth={2} dot={{ r: 5, fill: '#10b981', stroke: '#0f172a', strokeWidth: 2 }}
              name="Pareto Front"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
