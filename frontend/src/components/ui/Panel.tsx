'use client';

import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  subtitle?: string;
  badge?: string;
  badgeColor?: 'primary' | 'alert' | 'warn' | 'info';
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
  delay?: number;
}

const badgeColors = {
  primary: 'text-energy-primary bg-energy-primary/10 border-energy-primary/20',
  alert: 'text-energy-alert bg-energy-alert/10 border-energy-alert/20',
  warn: 'text-energy-warn bg-energy-warn/10 border-energy-warn/20',
  info: 'text-energy-info bg-energy-info/10 border-energy-info/20',
};

export function Panel({ title, subtitle, badge, badgeColor = 'primary', children, className = '', actions, delay = 0 }: PanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={`glass-panel rounded-xl overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {subtitle && <span className="text-xs text-slate-500">{subtitle}</span>}
          {badge && (
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${badgeColors[badgeColor]}`}>
              {badge}
            </span>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>

      {/* Content */}
      <div className="p-5">{children}</div>
    </motion.div>
  );
}
