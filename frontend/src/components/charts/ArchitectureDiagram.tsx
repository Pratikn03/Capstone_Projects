'use client';

import { motion } from 'framer-motion';
import { Database, Cpu, Brain, Activity, Server, Monitor, Zap, Wind, Sun, Cloud, MapPin, BarChart3, Shield, TrendingUp } from 'lucide-react';

interface LayerProps {
  title: string;
  icon: React.ReactNode;
  items: { icon: React.ReactNode; label: string; sublabel: string }[];
  color: string;
  delay: number;
}

const Layer = ({ title, icon, items, color, delay }: LayerProps) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5, ease: 'easeOut' }}
    className="relative"
  >
    {/* Layer Header */}
    <div className={`flex items-center gap-2 mb-3 px-3 py-1.5 rounded-lg bg-gradient-to-r ${color} w-fit`}>
      {icon}
      <span className="text-xs font-semibold tracking-wide uppercase text-white">{title}</span>
    </div>

    {/* Layer Items */}
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {items.map((item, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: delay + 0.1 * (idx + 1), duration: 0.3 }}
          className="group relative"
        >
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative p-4 rounded-xl bg-white/[0.03] border border-white/10 hover:border-white/20 hover:bg-white/[0.05] transition-all cursor-default">
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-lg bg-gradient-to-br ${color} opacity-80`}>
                {item.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-white truncate">{item.label}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{item.sublabel}</div>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  </motion.div>
);

const FlowArrow = ({ delay }: { delay: number }) => (
  <motion.div
    initial={{ opacity: 0, scaleY: 0 }}
    animate={{ opacity: 1, scaleY: 1 }}
    transition={{ delay, duration: 0.3 }}
    className="flex justify-center py-2"
  >
    <div className="flex flex-col items-center gap-1">
      <div className="w-px h-6 bg-gradient-to-b from-slate-600 to-slate-500" />
      <svg width="12" height="8" viewBox="0 0 12 8" fill="none" className="text-slate-500">
        <path d="M6 8L0 0h12L6 8z" fill="currentColor" />
      </svg>
    </div>
  </motion.div>
);

export function ArchitectureDiagram() {
  const layers = [
    {
      title: 'Data Sources',
      icon: <Database className="w-3.5 h-3.5 text-white" />,
      color: 'from-blue-600/80 to-blue-500/60',
      items: [
        { icon: <Zap className="w-4 h-4 text-white" />, label: 'OPSD Germany', sublabel: 'Load · Wind · Solar · Price' },
        { icon: <MapPin className="w-4 h-4 text-white" />, label: 'EIA-930 USA', sublabel: 'MISO Demand · Generation' },
        { icon: <Cloud className="w-4 h-4 text-white" />, label: 'Open-Meteo', sublabel: 'Berlin · Chicago Weather' },
      ],
    },
    {
      title: 'Data Pipeline',
      icon: <Cpu className="w-3.5 h-3.5 text-white" />,
      color: 'from-emerald-600/80 to-emerald-500/60',
      items: [
        { icon: <Database className="w-4 h-4 text-white" />, label: 'Ingestion', sublabel: 'Validation & Cleaning' },
        { icon: <BarChart3 className="w-4 h-4 text-white" />, label: 'Feature Engineering', sublabel: 'Lags · Calendar · Weather' },
        { icon: <TrendingUp className="w-4 h-4 text-white" />, label: 'Time-Series Splits', sublabel: 'Train · Val · Test' },
      ],
    },
    {
      title: 'ML Engine',
      icon: <Brain className="w-3.5 h-3.5 text-white" />,
      color: 'from-purple-600/80 to-purple-500/60',
      items: [
        { icon: <BarChart3 className="w-4 h-4 text-white" />, label: 'LightGBM', sublabel: '21 Ensemble Models' },
        { icon: <Brain className="w-4 h-4 text-white" />, label: 'Deep Learning', sublabel: 'LSTM · TCN Networks' },
        { icon: <Shield className="w-4 h-4 text-white" />, label: 'Conformal Prediction', sublabel: '90% Coverage Intervals' },
      ],
    },
    {
      title: 'Operations',
      icon: <Activity className="w-3.5 h-3.5 text-white" />,
      color: 'from-rose-600/80 to-rose-500/60',
      items: [
        { icon: <Activity className="w-4 h-4 text-white" />, label: 'Anomaly Detection', sublabel: 'Z-Score · IsolationForest' },
        { icon: <Zap className="w-4 h-4 text-white" />, label: 'LP Optimizer', sublabel: 'Cost · Carbon · Battery' },
        { icon: <TrendingUp className="w-4 h-4 text-white" />, label: 'MLOps Monitor', sublabel: 'Drift · Retraining' },
      ],
    },
    {
      title: 'Serving',
      icon: <Server className="w-3.5 h-3.5 text-white" />,
      color: 'from-cyan-600/80 to-cyan-500/60',
      items: [
        { icon: <Server className="w-4 h-4 text-white" />, label: 'FastAPI Backend', sublabel: '/forecast · /optimize' },
        { icon: <Monitor className="w-4 h-4 text-white" />, label: 'Next.js 15', sublabel: '8 Pages · Real-time' },
        { icon: <Activity className="w-4 h-4 text-white" />, label: 'Region Toggle', sublabel: 'Germany · USA' },
      ],
    },
  ];

  return (
    <div className="p-6 space-y-2">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-6"
      >
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-gradient-to-br from-energy-primary/20 to-energy-info/20 border border-energy-primary/30">
              <Cpu className="w-4 h-4 text-energy-primary" />
            </div>
            System Architecture
          </h2>
          <p className="text-xs text-slate-500 mt-1">End-to-end ML pipeline for energy forecasting & optimization</p>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <span className="px-2 py-1 rounded-full bg-white/5 border border-white/10">v2.0</span>
          <span className="px-2 py-1 rounded-full bg-energy-primary/10 border border-energy-primary/20 text-energy-primary">Production</span>
        </div>
      </motion.div>

      {/* Architecture Layers */}
      <div className="space-y-1">
        {layers.map((layer, idx) => (
          <div key={layer.title}>
            <Layer {...layer} delay={idx * 0.15} />
            {idx < layers.length - 1 && <FlowArrow delay={idx * 0.15 + 0.3} />}
          </div>
        ))}
      </div>

      {/* Footer Stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="flex items-center justify-center gap-6 pt-6 text-[10px] text-slate-600"
      >
        <span className="flex items-center gap-1.5">
          <Wind className="w-3 h-3" /> 3 Targets
        </span>
        <span>•</span>
        <span className="flex items-center gap-1.5">
          <Brain className="w-3 h-3" /> 21+ Models
        </span>
        <span>•</span>
        <span className="flex items-center gap-1.5">
          <Sun className="w-3 h-3" /> 2 Regions
        </span>
        <span>•</span>
        <span className="flex items-center gap-1.5">
          <Activity className="w-3 h-3" /> Real-time
        </span>
      </motion.div>
    </div>
  );
}
