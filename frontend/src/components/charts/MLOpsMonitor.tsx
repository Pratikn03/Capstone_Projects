'use client';

import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';
import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import type { DriftPoint } from '@/lib/api/mock-data';

interface MLOpsMonitorProps {
  data: DriftPoint[];
  lastTrained?: string;
  modelVersion?: string;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as DriftPoint;
  return (
    <div className="glass-panel rounded-lg p-3 text-xs min-w-[200px]">
      <div className="font-semibold text-white mb-2">{d.date}</div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">KS Statistic</span>
        <span className={`font-mono ${d.is_drift ? 'text-energy-alert' : 'text-white'}`}>{d.ks_statistic}</span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Rolling RMSE</span>
        <span className="font-mono text-white">{d.rolling_rmse} MW</span>
      </div>
      {d.is_drift && (
        <div className="mt-1 text-energy-warn font-medium">⚠ Drift Detected</div>
      )}
    </div>
  );
}

export function MLOpsMonitor({
  data,
  lastTrained = '2026-02-07',
  modelVersion = 'v2.3.0-50ep',
}: MLOpsMonitorProps) {
  const hasDrift = data.some((d) => d.is_drift);
  const latestKS = data[data.length - 1]?.ks_statistic ?? 0;
  const latestRMSE = data[data.length - 1]?.rolling_rmse ?? 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">MLOps & Model Monitoring</h3>
          {hasDrift ? (
            <div className="flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-warn/10 text-energy-warn border border-energy-warn/20">
              <AlertTriangle className="w-3 h-3" />
              Drift Detected
            </div>
          ) : (
            <div className="flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-primary/10 text-energy-primary border border-energy-primary/20">
              <CheckCircle className="w-3 h-3" />
              Stable
            </div>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span>Model: <span className="text-white font-mono">{modelVersion}</span></span>
          <span>Trained: <span className="text-white font-mono">{lastTrained}</span></span>
        </div>
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-3 gap-3 px-5 pt-4">
        <div className="p-2.5 rounded-lg bg-white/3 text-center">
          <div className="text-[10px] text-slate-500 mb-1">Data Drift (KS)</div>
          <div className={`text-sm font-bold font-mono ${latestKS > 0.08 ? 'text-energy-alert' : 'text-energy-primary'}`}>
            {latestKS.toFixed(3)}
          </div>
        </div>
        <div className="p-2.5 rounded-lg bg-white/3 text-center">
          <div className="text-[10px] text-slate-500 mb-1">Rolling RMSE</div>
          <div className="text-sm font-bold font-mono text-white">{latestRMSE} MW</div>
        </div>
        <div className="p-2.5 rounded-lg bg-white/3 text-center">
          <div className="text-[10px] text-slate-500 mb-1">Retraining</div>
          <div className="flex items-center justify-center gap-1">
            <RefreshCw className={`w-3.5 h-3.5 ${hasDrift ? 'text-energy-warn animate-spin' : 'text-energy-primary'}`} />
            <span className={`text-sm font-bold ${hasDrift ? 'text-energy-warn' : 'text-energy-primary'}`}>
              {hasDrift ? 'Needed' : 'OK'}
            </span>
          </div>
        </div>
      </div>

      {/* Dual-axis chart */}
      <div className="px-5 py-4 h-[220px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 9 }} interval={4} />
            <YAxis yAxisId="ks" stroke="#3b82f6" tick={{ fontSize: 10 }} domain={[0, 0.15]} />
            <YAxis yAxisId="rmse" orientation="right" stroke="#f59e0b" tick={{ fontSize: 10 }} />

            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />

            <ReferenceLine yAxisId="ks" y={0.08} stroke="#ef444460" strokeDasharray="4 4" label={{ value: 'Drift Threshold', fill: '#ef444480', fontSize: 9, position: 'insideTopRight' }} />

            <Area
              yAxisId="ks" type="monotone" dataKey="ks_statistic"
              stroke="#3b82f6" fill="#3b82f620" strokeWidth={1.5}
              name="KS Statistic" dot={false}
            />
            <Line
              yAxisId="rmse" type="monotone" dataKey="rolling_rmse"
              stroke="#f59e0b" strokeWidth={1.5} name="Rolling RMSE (MW)"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Dataset lineage */}
      <div className="px-5 pb-3 flex items-center gap-4 text-[10px] text-slate-600">
        <span>Datasets: OPSD Germany, EIA-930 USA</span>
        <span>•</span>
        <span>Models: GBM • LSTM • TCN (50 epochs, CosineAnnealingLR)</span>
      </div>
    </motion.div>
  );
}
