'use client';

import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Line, ComposedChart, ReferenceLine,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useCallback } from 'react';
import type { ParetoPoint } from '@/lib/api/mock-data';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';

interface CarbonCostPanelProps {
  data?: ParetoPoint[];
  zoneId: string;
  summary?: {
    cost_savings_pct?: number | null;
    cost_savings_usd?: number | null;
    carbon_reduction_pct?: number | null;
    carbon_reduction_kg?: number | null;
    peak_shaving_pct?: number | null;
    peak_shaving_mw?: number | null;
  };
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

export function CarbonCostPanel({ data, zoneId, summary }: CarbonCostPanelProps) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const costSavingsPct = summary?.cost_savings_pct ?? null;
  const costSavingsUsd = summary?.cost_savings_usd ?? null;
  const carbonReductionPct = summary?.carbon_reduction_pct ?? null;
  const carbonTons = summary?.carbon_reduction_kg ? summary.carbon_reduction_kg / 1000 : null;
  const peakShavingPct = summary?.peak_shaving_pct ?? null;
  const peakShavingMw = summary?.peak_shaving_mw ?? null;
  const formatMaybe = (value: number | null | undefined, formatter: (value: number) => string) =>
    typeof value === 'number' && Number.isFinite(value) ? formatter(value) : 'Pending';

  const selectedPoint = data && selectedIdx !== null ? data[selectedIdx] : null;
  const bestCostPoint = data?.length ? data.reduce((a, b) => (a.total_cost_eur < b.total_cost_eur ? a : b)) : null;
  const bestCarbonPoint = data?.length ? data.reduce((a, b) => (a.total_carbon_kg < b.total_carbon_kg ? a : b)) : null;

  const handleChartClick = useCallback(
    (state: any) => {
      if (!data || !state?.activePayload?.length) return;
      const clicked = state.activePayload[0].payload as ParetoPoint;
      const idx = data.findIndex((d) => d.carbon_weight === clicked.carbon_weight);
      setSelectedIdx((prev) => (prev === idx ? null : idx));
    },
    [data],
  );

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 px-5 py-3.5 border-b border-white/6">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <h3 className="text-sm font-semibold text-white">Publication Trade-off Frontier</h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
            Cost / Carbon Evidence
          </span>
        </div>
        <span className="text-xs text-slate-500">{zoneId}</span>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-1 gap-3 px-5 pt-4 sm:grid-cols-3 sm:gap-4">
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-primary stat-value">{formatMaybe(costSavingsPct, formatPercent)}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Cost Delta</div>
          <div className="text-xs text-slate-400 font-mono">
            {costSavingsUsd !== null ? formatCurrency(costSavingsUsd, 'USD') : 'Pending'}
          </div>
        </div>
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-primary stat-value">{formatMaybe(carbonReductionPct, formatPercent)}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Carbon Delta</div>
          <div className="text-xs text-slate-400 font-mono">{formatMaybe(carbonTons, (value) => `${value.toFixed(1)} tCO₂`)}</div>
        </div>
        <div className="text-center p-3 rounded-lg bg-white/3">
          <div className="text-lg font-bold text-energy-info stat-value">{formatMaybe(peakShavingPct, formatPercent)}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Peak Constraint Relief</div>
          <div className="text-xs text-slate-400 font-mono">{formatMaybe(peakShavingMw, formatMW)}</div>
        </div>
      </div>

      {/* Pareto chart */}
      <div className="px-5 py-4 h-[260px]">
        {data && data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }} onClick={handleChartClick}>
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

              {selectedPoint && (
                <>
                  <ReferenceLine x={selectedPoint.total_carbon_kg} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth={1} />
                  <ReferenceLine y={selectedPoint.total_cost_eur} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth={1} />
                </>
              )}

              <Line
                type="monotone" dataKey="total_cost_eur"
                stroke="#10b981" strokeWidth={2}
                dot={(props: any) => {
                  const { cx, cy, index } = props;
                  const isSelected = index === selectedIdx;
                  const isBestCost = bestCostPoint && data[index]?.carbon_weight === bestCostPoint.carbon_weight;
                  const isBestCarbon = bestCarbonPoint && data[index]?.carbon_weight === bestCarbonPoint.carbon_weight;
                  return (
                    <circle
                      key={index}
                      cx={cx} cy={cy}
                      r={isSelected ? 8 : isBestCost || isBestCarbon ? 6 : 5}
                      fill={isSelected ? '#f59e0b' : '#10b981'}
                      stroke={isSelected ? '#fbbf24' : '#0f172a'}
                      strokeWidth={isSelected ? 3 : 2}
                      style={{ cursor: 'pointer', transition: 'r 0.2s, fill 0.2s' }}
                    />
                  );
                }}
                activeDot={{ r: 7, fill: '#34d399', stroke: '#10b981', strokeWidth: 2 }}
                name="Pareto Front"
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-slate-500 text-xs">
            No trade-off artifact available
          </div>
        )}
      </div>

      {/* Selected operating point detail */}
      <AnimatePresence>
        {selectedPoint && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mx-5 mb-4 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-amber-400">Selected Operating Point</span>
                <button
                  onClick={() => setSelectedIdx(null)}
                  className="text-[10px] text-slate-500 hover:text-white transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="grid grid-cols-4 gap-3 text-xs">
                <div>
                  <div className="text-slate-500">Weight</div>
                  <div className="text-white font-mono">{selectedPoint.carbon_weight}</div>
                </div>
                <div>
                  <div className="text-slate-500">Cost</div>
                  <div className="text-white font-mono">€{(selectedPoint.total_cost_eur / 1000).toFixed(0)}k</div>
                </div>
                <div>
                  <div className="text-slate-500">Carbon</div>
                  <div className="text-white font-mono">{(selectedPoint.total_carbon_kg / 1000).toFixed(1)}t</div>
                </div>
                <div>
                  <div className="text-slate-500">vs Best Cost</div>
                  <div className="text-amber-400 font-mono">
                    {bestCostPoint
                      ? `+€${((selectedPoint.total_cost_eur - bestCostPoint.total_cost_eur) / 1000).toFixed(0)}k`
                      : 'Pending'}
                    {bestCostPoint
                      ? ` / −${((bestCostPoint.total_carbon_kg - selectedPoint.total_carbon_kg) / 1000).toFixed(1)}t`
                      : ''}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
