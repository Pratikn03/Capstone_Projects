'use client';

import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Sparkline } from '@/components/charts/Sparkline';

interface KPICardProps {
  label: string;
  value: string;
  unit?: string;
  change?: number;
  changeLabel?: string;
  icon?: ReactNode;
  color?: 'primary' | 'alert' | 'warn' | 'info';
  delay?: number;
  sparkData?: number[];
  live?: boolean;
}

const colorMap = {
  primary: {
    bg: 'bg-energy-primary-dim',
    border: 'border-energy-primary/20',
    text: 'text-energy-primary',
    glow: 'shadow-energy-primary/10',
    hex: '#10b981',
    accent: 'from-energy-primary/5 to-transparent',
  },
  alert: {
    bg: 'bg-energy-alert-dim',
    border: 'border-energy-alert/20',
    text: 'text-energy-alert',
    glow: 'shadow-energy-alert/10',
    hex: '#ef4444',
    accent: 'from-energy-alert/5 to-transparent',
  },
  warn: {
    bg: 'bg-energy-warn-dim',
    border: 'border-energy-warn/20',
    text: 'text-energy-warn',
    glow: 'shadow-energy-warn/10',
    hex: '#f59e0b',
    accent: 'from-energy-warn/5 to-transparent',
  },
  info: {
    bg: 'bg-energy-info-dim',
    border: 'border-energy-info/20',
    text: 'text-energy-info',
    glow: 'shadow-energy-info/10',
    hex: '#3b82f6',
    accent: 'from-energy-info/5 to-transparent',
  },
};

function AnimatedNumber({ target, format }: { target: number; format?: (v: number) => string }) {
  const motionValue = useMotionValue(0);
  const display = useTransform(motionValue, (v) => (format ? format(v) : v.toFixed(0)));
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const controls = animate(motionValue, target, { duration: 1.2, ease: 'easeOut' });
    return controls.stop;
  }, [target, motionValue]);

  useEffect(() => {
    const unsubscribe = display.on('change', (v) => {
      if (ref.current) ref.current.textContent = v;
    });
    return unsubscribe;
  }, [display]);

  return <span ref={ref}>{format ? format(0) : '0'}</span>;
}

export function KPICard({
  label,
  value,
  unit,
  change,
  changeLabel,
  icon,
  color = 'primary',
  delay = 0,
  sparkData,
  live,
}: KPICardProps) {
  const c = colorMap[color];
  const isPositive = change !== undefined && change > 0;
  const isNegative = change !== undefined && change < 0;
  const numericValue = parseFloat(value);
  const isNumeric = !isNaN(numericValue) && value !== 'N/A';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className={`glass-panel-hover rounded-xl p-4 relative overflow-hidden ${live ? 'live-border' : ''}`}
    >
      {/* Subtle gradient accent */}
      <div className={`absolute inset-0 bg-gradient-to-br ${c.accent} pointer-events-none`} />

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
            {live && (
              <span className="w-1.5 h-1.5 rounded-full bg-energy-primary live-dot" />
            )}
          </div>
          {icon && (
            <div className={`p-1.5 rounded-lg ${c.bg} ${c.border} border`}>
              {icon}
            </div>
          )}
        </div>

        <div className="flex items-end justify-between gap-2">
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white stat-value">
                {isNumeric ? (
                  <AnimatedNumber target={numericValue} format={(v) => v.toFixed(value.includes('.') ? 1 : 0)} />
                ) : (
                  value
                )}
              </span>
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
          </div>

          {sparkData && sparkData.length > 1 && (
            <Sparkline data={sparkData} color={c.hex} height={32} width={72} className="opacity-60" />
          )}
        </div>
      </div>
    </motion.div>
  );
}
