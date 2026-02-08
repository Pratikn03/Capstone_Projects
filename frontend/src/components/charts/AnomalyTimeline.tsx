'use client';

import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import type { AnomalyZScore } from '@/lib/api/mock-data';

interface AnomalyTimelineProps {
  data: AnomalyZScore[];
}

function formatTime(ts: string) {
  const d = new Date(ts);
  return `${d.getUTCDate()}/${d.getUTCMonth() + 1} ${d.getUTCHours()}h`;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as AnomalyZScore;
  return (
    <div className="glass-panel rounded-lg p-3 text-xs min-w-[180px]">
      <div className="font-semibold text-white mb-2">{formatTime(label)}</div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Z-Score</span>
        <span className={`font-mono ${d.is_anomaly ? 'text-energy-alert' : 'text-white'}`}>{d.z_score.toFixed(2)}</span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="text-slate-400">Residual</span>
        <span className="font-mono text-white">{d.residual_mw.toLocaleString()} MW</span>
      </div>
      {d.is_anomaly && (
        <div className="mt-1 text-energy-alert font-medium">⚠ Anomaly Detected</div>
      )}
    </div>
  );
}

export function AnomalyTimeline({ data }: AnomalyTimelineProps) {
  const anomalyCount = data.filter((d) => d.is_anomaly).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full glass-panel rounded-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">Anomaly Detection Timeline</h3>
          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-energy-alert/10 text-energy-alert border border-energy-alert/20">
            {anomalyCount} detected
          </span>
        </div>
        <span className="text-xs text-slate-500">Isolation Forest + Z-Score</span>
      </div>

      <div className="px-5 py-4 h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="timestamp" stroke="#64748b" tick={{ fontSize: 9 }} tickFormatter={formatTime} interval={Math.floor(data.length / 8)} />
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[-4, 4]} />

            <Tooltip content={<CustomTooltip />} />

            {/* Threshold lines */}
            <ReferenceLine y={2} stroke="#ef444460" strokeDasharray="4 4" label={{ value: '+2σ', position: 'right', fill: '#ef4444', fontSize: 9 }} />
            <ReferenceLine y={-2} stroke="#ef444460" strokeDasharray="4 4" label={{ value: '-2σ', position: 'right', fill: '#ef4444', fontSize: 9 }} />
            <ReferenceLine y={0} stroke="#64748b40" />

            {/* Z-score bars with conditional coloring */}
            <Bar dataKey="z_score" name="Z-Score" radius={[1, 1, 0, 0]}>
              {data.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.is_anomaly ? '#ef4444' : '#64748b40'}
                  stroke={entry.is_anomaly ? '#ef4444' : 'transparent'}
                />
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
