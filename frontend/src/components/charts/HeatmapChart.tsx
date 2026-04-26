'use client';

import { motion } from 'framer-motion';
import { useMemo } from 'react';

export interface HeatmapData {
  hour: number;
  day: number; // 0=Mon … 6=Sun
  value: number;
}

interface HeatmapChartProps {
  data: HeatmapData[];
  label?: string;
  unit?: string;
  colorRange?: [string, string, string]; // [low, mid, high]
  className?: string;
}

const dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function interpolateColor(t: number, low: string, mid: string, high: string): string {
  // t in [0,1] — parse hex to rgb and lerp
  function hex2rgb(hex: string) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const [r1, g1, b1] = t < 0.5 ? hex2rgb(low) : hex2rgb(mid);
  const [r2, g2, b2] = t < 0.5 ? hex2rgb(mid) : hex2rgb(high);
  const p = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const r = Math.round(r1 + (r2 - r1) * p);
  const g = Math.round(g1 + (g2 - g1) * p);
  const b = Math.round(b1 + (b2 - b1) * p);
  return `rgb(${r},${g},${b})`;
}

export function HeatmapChart({
  data,
  label = 'Load Pattern',
  unit = 'MW',
  colorRange = ['#0f172a', '#10b981', '#fbbf24'],
  className = '',
}: HeatmapChartProps) {
  const { min, max, grid } = useMemo(() => {
    const values = data.map((d) => d.value);
    const mn = Math.min(...values);
    const mx = Math.max(...values);
    // Build 7×24 grid
    const g: (number | null)[][] = Array.from({ length: 7 }, () => Array(24).fill(null));
    for (const d of data) {
      if (d.day >= 0 && d.day < 7 && d.hour >= 0 && d.hour < 24) {
        g[d.day][d.hour] = d.value;
      }
    }
    return { min: mn, max: mx, grid: g };
  }, [data]);

  const cellW = 20;
  const cellH = 24;
  const gapX = 84;
  const gapY = 22;
  const svgW = gapX + 24 * cellW + 8;
  const svgH = gapY + 7 * cellH + 40;

  if (!data.length) {
    return (
      <div className={`${className} flex min-h-[220px] items-center justify-center rounded-lg border border-white/6 bg-white/[0.02] text-xs text-slate-500`}>
        No hour-by-day heatmap is available for this artifact timestamp format.
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-2 text-[9px] text-slate-600">
          <span>Low</span>
          <div className="flex gap-px">
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <div
                key={t}
                className="w-4 h-2 first:rounded-l last:rounded-r"
                style={{ background: interpolateColor(t, ...colorRange) }}
              />
            ))}
          </div>
          <span>High</span>
        </div>
      </div>

      <svg width="100%" viewBox={`0 0 ${svgW} ${svgH}`} className="block">
        {/* Hour labels */}
        {Array.from({ length: 24 }, (_, h) => (
          <text
            key={`h${h}`}
            x={gapX + h * cellW + cellW / 2}
            y={14}
            textAnchor="middle"
            className="fill-slate-600"
            style={{ fontSize: 8 }}
          >
            {h % 3 === 0 ? `${h}` : ''}
          </text>
        ))}

        {/* Day labels + cells */}
        {grid.map((row, day) => (
          <g key={day}>
            <text
              x={gapX - 8}
              y={gapY + day * cellH + cellH / 2 + 3}
              textAnchor="end"
              className="fill-slate-500"
              style={{ fontSize: 9 }}
            >
              {dayLabels[day]}
            </text>
            {row.map((val, hour) => {
              const t = val !== null && max > min ? (val - min) / (max - min) : 0;
              return (
                <motion.rect
                  key={`${day}-${hour}`}
                  x={gapX + hour * cellW}
                  y={gapY + day * cellH}
                  width={cellW - 1}
                  height={cellH - 1}
                  rx={3}
                  fill={val !== null ? interpolateColor(t, ...colorRange) : 'rgba(255,255,255,0.02)'}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: (day * 24 + hour) * 0.002 }}
                >
                  {val !== null && (
                    <title>{`${dayLabels[day]} ${hour}:00 — ${val.toFixed(0)} ${unit}`}</title>
                  )}
                </motion.rect>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
