'use client';

import { motion } from 'framer-motion';
import { Zap, Wind, Sun, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import type { ZoneSummary } from '@/lib/api/schema';

const statusConfig = {
  normal: { color: 'text-energy-primary', bg: 'bg-energy-primary/10', icon: CheckCircle, label: 'Normal' },
  warning: { color: 'text-energy-warn', bg: 'bg-energy-warn/10', icon: AlertTriangle, label: 'Warning' },
  critical: { color: 'text-energy-alert', bg: 'bg-energy-alert/10', icon: AlertCircle, label: 'Critical' },
};

export function GridStatusCard({ zone }: { zone: ZoneSummary }) {
  const status = statusConfig[zone.status];
  const StatusIcon = status.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel-hover rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold text-white">{zone.name}</span>
          <span className="text-xs text-slate-500 font-mono">{zone.zone_id}</span>
        </div>
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full ${status.bg}`}>
          <StatusIcon className={`w-3.5 h-3.5 ${status.color}`} />
          <span className={`text-xs font-medium ${status.color}`}>{status.label}</span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 text-center">
        <div>
          <Zap className="w-4 h-4 text-slate-400 mx-auto mb-1" />
          <div className="text-sm font-bold text-white stat-value">{(zone.load_mw / 1000).toFixed(1)}</div>
          <div className="text-[10px] text-slate-500">GW Load</div>
        </div>
        <div>
          <Wind className="w-4 h-4 text-energy-info mx-auto mb-1" />
          <div className="text-sm font-bold text-white stat-value">{zone.renewable_pct.toFixed(1)}%</div>
          <div className="text-[10px] text-slate-500">Renewable</div>
        </div>
        <div>
          <Sun className="w-4 h-4 text-solar mx-auto mb-1" />
          <div className="text-sm font-bold text-white stat-value">{zone.frequency_hz.toFixed(2)}</div>
          <div className="text-[10px] text-slate-500">Hz</div>
        </div>
        <div>
          <AlertTriangle className={`w-4 h-4 mx-auto mb-1 ${zone.anomaly_count > 0 ? 'text-energy-warn' : 'text-slate-600'}`} />
          <div className="text-sm font-bold text-white stat-value">{zone.anomaly_count}</div>
          <div className="text-[10px] text-slate-500">Anomalies</div>
        </div>
      </div>
    </motion.div>
  );
}
