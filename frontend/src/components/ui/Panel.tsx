'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Maximize2, Minimize2 } from 'lucide-react';
import { useState } from 'react';
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
  accentColor?: 'primary' | 'alert' | 'warn' | 'info';
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  live?: boolean;
}

const badgeColors = {
  primary: 'text-energy-primary bg-energy-primary/10 border-energy-primary/20',
  alert: 'text-energy-alert bg-energy-alert/10 border-energy-alert/20',
  warn: 'text-energy-warn bg-energy-warn/10 border-energy-warn/20',
  info: 'text-energy-info bg-energy-info/10 border-energy-info/20',
};

const accentHex = {
  primary: '#10b981',
  alert: '#ef4444',
  warn: '#f59e0b',
  info: '#3b82f6',
};

export function Panel({
  title,
  subtitle,
  badge,
  badgeColor = 'primary',
  children,
  className = '',
  actions,
  delay = 0,
  accentColor,
  collapsible = false,
  defaultCollapsed = false,
  live = false,
}: PanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [fullscreen, setFullscreen] = useState(false);

  const wrapperClass = fullscreen
    ? 'fixed inset-4 z-50 glass-panel-elevated rounded-xl overflow-hidden flex flex-col'
    : `glass-panel rounded-xl overflow-hidden ${live ? 'live-border' : ''} ${className}`;

  return (
    <>
      {fullscreen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          onClick={() => setFullscreen(false)}
        />
      )}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay }}
        className={wrapperClass}
        layout
      >
        {/* Accent bar */}
        {accentColor && !fullscreen && (
          <div
            className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl"
            style={{ background: `linear-gradient(to bottom, ${accentHex[accentColor]}, transparent)` }}
          />
        )}

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
          <div className="flex items-center gap-3">
            {collapsible && (
              <button
                onClick={() => setCollapsed(!collapsed)}
                className="p-0.5 rounded hover:bg-white/5 transition-colors"
              >
                <motion.div animate={{ rotate: collapsed ? -90 : 0 }} transition={{ duration: 0.2 }}>
                  <ChevronDown className="w-4 h-4 text-slate-500" />
                </motion.div>
              </button>
            )}
            <h3 className="text-sm font-semibold text-white">{title}</h3>
            {subtitle && <span className="text-xs text-slate-500">{subtitle}</span>}
            {badge && (
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${badgeColors[badgeColor]}`}>
                {badge}
              </span>
            )}
            {live && (
              <span className="w-1.5 h-1.5 rounded-full bg-energy-primary live-dot" />
            )}
          </div>
          <div className="flex items-center gap-2">
            {actions}
            <button
              onClick={() => setFullscreen(!fullscreen)}
              className="p-1.5 rounded-md hover:bg-white/5 transition-colors text-slate-500 hover:text-slate-300"
            >
              {fullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Content */}
        <AnimatePresence initial={false}>
          {!collapsed && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              className={fullscreen ? 'flex-1 overflow-auto' : ''}
            >
              <div className="p-5">{children}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </>
  );
}
