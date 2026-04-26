'use client';

import { motion } from 'framer-motion';

interface GaugeChartProps {
  value: number | null | undefined;
  min?: number;
  max?: number;
  label: string;
  unit?: string;
  color?: string;
  size?: number;
  thresholds?: { warn: number; alert: number };
}

export function GaugeChart({
  value,
  min = 0,
  max = 100,
  label,
  unit = '',
  color = '#10b981',
  size = 120,
  thresholds,
}: GaugeChartProps) {
  const hasValue = typeof value === 'number' && Number.isFinite(value);
  const numericValue = hasValue ? value : min;
  const clamped = Math.min(Math.max(numericValue, min), max);
  const ratio = (clamped - min) / (max - min);
  const sweepAngle = 240;
  const startAngle = (360 - sweepAngle) / 2 + 90;

  const radius = (size - 16) / 2;
  const cx = size / 2;
  const cy = size / 2 + 4;
  const strokeWidth = 8;

  const resolvedColor = thresholds
    ? numericValue >= thresholds.alert
      ? '#ef4444'
      : numericValue >= thresholds.warn
        ? '#f59e0b'
        : color
    : color;

  function polarToCartesian(angle: number) {
    const rad = ((angle - 90) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  function arcPath(startA: number, endA: number) {
    const s = polarToCartesian(startA);
    const e = polarToCartesian(endA);
    const large = endA - startA > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const bgPath = arcPath(startAngle, startAngle + sweepAngle);
  const endAngle = startAngle + sweepAngle * ratio;

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <svg width={size} height={size * 0.75} viewBox={`0 0 ${size} ${size * 0.85}`}>
        {/* Background arc */}
        <path
          d={bgPath}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Value arc */}
        <motion.path
          d={arcPath(startAngle, endAngle)}
          fill="none"
          stroke={resolvedColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{ filter: `drop-shadow(0 0 6px ${resolvedColor}40)` }}
        />
        {/* Center text */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          className="fill-white text-lg font-bold stat-value"
          style={{ fontSize: size * 0.18, fontVariantNumeric: 'tabular-nums' }}
        >
          {hasValue ? numericValue.toFixed(numericValue % 1 ? 1 : 0) : 'N/A'}
        </text>
        <text
          x={cx}
          y={cy + size * 0.12}
          textAnchor="middle"
          className="fill-slate-500"
          style={{ fontSize: size * 0.09 }}
        >
          {unit}
        </text>
      </svg>
      <span className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">{label}</span>
    </div>
  );
}
