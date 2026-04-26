'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Activity, Shield, X, Clock } from 'lucide-react';

export interface Notification {
  id: string;
  type: 'anomaly' | 'drift' | 'safety';
  title: string;
  message: string;
  timestamp: string;
  severity: 'info' | 'warn' | 'critical';
}

const typeIcons = {
  anomaly: AlertTriangle,
  drift: Activity,
  safety: Shield,
};

const severityColors = {
  info: 'text-energy-info bg-energy-info/10 border-energy-info/20',
  warn: 'text-energy-warn bg-energy-warn/10 border-energy-warn/20',
  critical: 'text-energy-alert bg-energy-alert/10 border-energy-alert/20',
};

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

interface NotificationPanelProps {
  notifications: Notification[];
  open: boolean;
  onClose: () => void;
  onDismiss: (id: string) => void;
}

export function NotificationPanel({ notifications, open, onClose, onDismiss }: NotificationPanelProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -8, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.95 }}
          transition={{ duration: 0.15 }}
          className="absolute top-full right-0 mt-2 w-80 glass-panel-elevated rounded-xl overflow-hidden z-50"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/6">
            <span className="text-xs font-semibold text-white">Notifications</span>
            <button onClick={onClose} className="p-1 rounded-md hover:bg-white/5 transition-colors">
              <X className="w-3.5 h-3.5 text-slate-500" />
            </button>
          </div>

          {/* List */}
          <div className="max-h-80 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="px-4 py-8 text-center text-xs text-slate-500">No notifications</div>
            ) : (
              notifications.map((n, i) => {
                const Icon = typeIcons[n.type];
                return (
                  <motion.div
                    key={n.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="flex items-start gap-3 px-4 py-3 border-b border-white/4 hover:bg-white/3 transition-colors group"
                  >
                    <div className={`p-1.5 rounded-md border flex-shrink-0 mt-0.5 ${severityColors[n.severity]}`}>
                      <Icon className="w-3 h-3" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-medium text-white truncate">{n.title}</span>
                        <button
                          onClick={() => onDismiss(n.id)}
                          className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-white/10 transition-all"
                        >
                          <X className="w-3 h-3 text-slate-500" />
                        </button>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5 line-clamp-2">{n.message}</p>
                      <div className="flex items-center gap-1 mt-1 text-[10px] text-slate-600">
                        <Clock className="w-2.5 h-2.5" />
                        {timeAgo(n.timestamp)}
                      </div>
                    </div>
                  </motion.div>
                );
              })
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
