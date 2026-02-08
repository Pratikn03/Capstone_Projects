'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { ReactNode } from 'react';

interface KPICardProps {
  label: string;
  value: string;
  unit?: string;
  change?: number;
  changeLabel?: string;
  icon?: ReactNode;
  color?: 'primary' | 'alert' | 'warn' | 'info';
  delay?: number;
}

const colorMap = {
  primary: {
    bg: 'bg-energy-primary-dim',
    border: 'border-energy-primary/20',
    text: 'text-energy-primary',
    glow: 'shadow-energy-primary/10',
  },
  alert: {
    bg: 'bg-energy-alert-dim',
    border: 'border-energy-alert/20',
    text: 'text-energy-alert',
    glow: 'shadow-energy-alert/10',
  },
  warn: {
    bg: 'bg-energy-warn-dim',
    border: 'border-energy-warn/20',
    text: 'text-energy-warn',
    glow: 'shadow-energy-warn/10',
  },
  info: {
    bg: 'bg-energy-info-dim',
    border: 'border-energy-info/20',
    text: 'text-energy-info',
    glow: 'shadow-energy-info/10',
  },
};

export function KPICard({ label, value, unit, change, changeLabel, icon, color = 'primary', delay = 0 }: KPICardProps) {
  const c = colorMap[color];
  const isPositive = change !== undefined && change > 0;
  const isNegative = change !== undefined && change < 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="glass-panel-hover rounded-xl p-4"
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        {icon && (
          <div className={`p-1.5 rounded-lg ${c.bg} ${c.border} border`}>
            {icon}
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-white stat-value">{value}</span>
        {unit && <span className="text-sm text-slate-400">{unit}</span>}
      </div>

      {change !== undefined && (
        <div className="flex items-center gap-1.5 mt-2">
          {isPositive && <TrendingUp className="w-3.5 h-3.5 text-energy-primary" />}
          {isNegative && <TrendingDown className="w-3.5 h-3.5 text-energy-alert" />}
          {!isPositive && !isNegative && <Minus className="w-3.5 h-3.5 text-slate-500" />}
          <span
            className={`text-xs font-medium ${
              isPositive ? 'text-energy-primary' : isNegative ? 'text-energy-alert' : 'text-slate-500'
            }`}
          >
            {isPositive ? '+' : ''}
            {change.toFixed(1)}%
          </span>
          {changeLabel && <span className="text-xs text-slate-500">{changeLabel}</span>}
        </div>
      )}
    </motion.div>
  );
}
