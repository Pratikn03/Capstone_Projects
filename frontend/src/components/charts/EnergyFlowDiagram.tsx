'use client';

import { motion } from 'framer-motion';
import { Sun, Wind, Flame, Battery, Zap } from 'lucide-react';

interface EnergyFlowProps {
  solar: number;
  wind: number;
  gas: number;
  battery: number; // positive = discharge, negative = charge
  load: number;
  className?: string;
}

function FlowArrow({ delay, color, active }: { delay: number; color: string; active: boolean }) {
  return (
    <svg width="40" height="20" viewBox="0 0 40 20" className="flex-shrink-0">
      <motion.line
        x1="0" y1="10" x2="32" y2="10"
        stroke={active ? color : 'rgba(255,255,255,0.06)'}
        strokeWidth="2"
        strokeDasharray="4 3"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.8, delay }}
      />
      {active && (
        <motion.polygon
          points="32,5 40,10 32,15"
          fill={color}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.6 }}
        />
      )}
    </svg>
  );
}

function SourceNode({
  icon: Icon,
  label,
  value,
  unit,
  color,
  bgColor,
  delay,
}: {
  icon: typeof Sun;
  label: string;
  value: number;
  unit: string;
  color: string;
  bgColor: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay }}
      className="flex flex-col items-center gap-1.5"
    >
      <div
        className="w-12 h-12 rounded-xl flex items-center justify-center border"
        style={{ backgroundColor: bgColor, borderColor: `${color}30` }}
      >
        <Icon className="w-5 h-5" style={{ color }} />
      </div>
      <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className="text-xs font-bold text-white font-mono stat-value">
        {value >= 1000 ? `${(value / 1000).toFixed(1)}` : value.toFixed(0)}
      </span>
      <span className="text-[9px] text-slate-600">{value >= 1000 ? 'GW' : unit}</span>
    </motion.div>
  );
}

export function EnergyFlowDiagram({ solar, wind, gas, battery, load, className = '' }: EnergyFlowProps) {
  const total = solar + wind + gas + Math.max(0, battery);
  const renewablePct = total > 0 ? ((solar + wind) / total) * 100 : 0;
  const batteryCharging = battery < 0;

  return (
    <div className={`${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs text-slate-500 uppercase tracking-wider">Energy Flow</div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500">Renewable</span>
          <span className="text-xs font-bold text-energy-primary font-mono">{renewablePct.toFixed(0)}%</span>
        </div>
      </div>

      {/* Flow diagram */}
      <div className="flex items-center justify-between gap-1">
        {/* Sources column */}
        <div className="flex flex-col gap-4">
          <SourceNode icon={Sun} label="Solar" value={solar} unit="MW" color="#fbbf24" bgColor="rgba(251,191,36,0.1)" delay={0} />
          <SourceNode icon={Wind} label="Wind" value={wind} unit="MW" color="#3b82f6" bgColor="rgba(59,130,246,0.1)" delay={0.1} />
          <SourceNode icon={Flame} label="Gas" value={gas} unit="MW" color="#ef4444" bgColor="rgba(239,68,68,0.1)" delay={0.2} />
        </div>

        {/* Arrows to hub */}
        <div className="flex flex-col gap-8 py-4">
          <FlowArrow delay={0.3} color="#fbbf24" active={solar > 0} />
          <FlowArrow delay={0.4} color="#3b82f6" active={wind > 0} />
          <FlowArrow delay={0.5} color="#ef4444" active={gas > 0} />
        </div>

        {/* Central hub */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="flex flex-col items-center gap-2"
        >
          <div className="w-16 h-16 rounded-2xl glass-panel-elevated flex items-center justify-center relative">
            <Zap className="w-6 h-6 text-energy-primary" />
            <motion.div
              className="absolute inset-0 rounded-2xl"
              animate={{ boxShadow: ['0 0 10px rgba(16,185,129,0.1)', '0 0 20px rgba(16,185,129,0.2)', '0 0 10px rgba(16,185,129,0.1)'] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
          <span className="text-[9px] text-slate-600">GRID HUB</span>
        </motion.div>

        {/* Arrows from hub */}
        <div className="flex flex-col gap-10 py-4">
          <FlowArrow delay={0.6} color="#a855f7" active={Math.abs(battery) > 0} />
          <FlowArrow delay={0.7} color="#f1f5f9" active={load > 0} />
        </div>

        {/* Sinks column */}
        <div className="flex flex-col gap-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.6 }}
            className="flex flex-col items-center gap-1.5"
          >
            <div className="w-12 h-12 rounded-xl flex items-center justify-center border" style={{ backgroundColor: 'rgba(168,85,247,0.1)', borderColor: 'rgba(168,85,247,0.3)' }}>
              <Battery className="w-5 h-5 text-battery" />
            </div>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">Battery</span>
            <span className={`text-xs font-bold font-mono stat-value ${batteryCharging ? 'text-energy-info' : 'text-battery'}`}>
              {batteryCharging ? '−' : '+'}{Math.abs(battery).toFixed(0)}
            </span>
            <span className="text-[9px] text-slate-600">{batteryCharging ? 'Charging' : 'Discharge'}</span>
          </motion.div>

          <SourceNode icon={Zap} label="Load" value={load} unit="MW" color="#f1f5f9" bgColor="rgba(241,245,249,0.06)" delay={0.7} />
        </div>
      </div>

      {/* Generation bars */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-4 pt-3 border-t border-white/4"
      >
        <div className="flex items-center gap-1 h-3 rounded-full overflow-hidden bg-white/3">
          {total > 0 && (
            <>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(solar / total) * 100}%` }}
                transition={{ duration: 0.8, delay: 0.9 }}
                className="h-full rounded-full"
                style={{ backgroundColor: '#fbbf24' }}
              />
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(wind / total) * 100}%` }}
                transition={{ duration: 0.8, delay: 1.0 }}
                className="h-full rounded-full"
                style={{ backgroundColor: '#3b82f6' }}
              />
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(gas / total) * 100}%` }}
                transition={{ duration: 0.8, delay: 1.1 }}
                className="h-full rounded-full"
                style={{ backgroundColor: '#ef4444' }}
              />
              {battery > 0 && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(battery / total) * 100}%` }}
                  transition={{ duration: 0.8, delay: 1.2 }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: '#a855f7' }}
                />
              )}
            </>
          )}
        </div>
        <div className="flex items-center justify-between mt-1.5 text-[9px] text-slate-600">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-solar" />Solar</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-wind" />Wind</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-gas" />Gas</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-battery" />Battery</span>
          </div>
          <span className="font-mono">{total >= 1000 ? `${(total / 1000).toFixed(1)} GW` : `${total.toFixed(0)} MW`} total</span>
        </div>
      </motion.div>
    </div>
  );
}
