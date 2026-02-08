'use client';

import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Search, Clock } from 'lucide-react';
import type { Anomaly } from '@/lib/api/schema';

const severityColors = {
  low: 'text-slate-400 bg-slate-400/10 border-slate-400/20',
  medium: 'text-energy-warn bg-energy-warn/10 border-energy-warn/20',
  high: 'text-energy-alert bg-energy-alert/10 border-energy-alert/20',
  critical: 'text-red-400 bg-red-400/15 border-red-400/30',
};

const statusIcons = {
  active: { icon: AlertTriangle, color: 'text-energy-alert' },
  investigating: { icon: Search, color: 'text-energy-warn' },
  resolved: { icon: CheckCircle, color: 'text-energy-primary' },
};

interface AnomalyListProps {
  anomalies: Anomaly[];
}

export function AnomalyList({ anomalies }: AnomalyListProps) {
  return (
    <div className="space-y-2">
      {anomalies.map((a, i) => {
        const StatusIcon = statusIcons[a.status].icon;
        const statusColor = statusIcons[a.status].color;
        const time = new Date(a.timestamp);
        const timeStr = `${time.getUTCHours().toString().padStart(2, '0')}:${time.getUTCMinutes().toString().padStart(2, '0')}`;

        return (
          <motion.div
            key={a.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-white/3 hover:bg-white/5 transition-colors cursor-pointer group"
          >
            <div className="flex items-center gap-2 min-w-[70px]">
              <Clock className="w-3 h-3 text-slate-600" />
              <span className="text-xs text-slate-400 font-mono">{timeStr}</span>
            </div>

            <div className="flex-1 min-w-0">
              <div className="text-xs text-slate-200 truncate group-hover:text-white transition-colors">
                {a.description}
              </div>
            </div>

            <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${severityColors[a.severity]}`}>
              {a.severity}
            </span>

            <div className="flex items-center gap-1">
              <StatusIcon className={`w-3.5 h-3.5 ${statusColor}`} />
              <span className={`text-[10px] ${statusColor} capitalize`}>{a.status}</span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
