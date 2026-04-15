'use client';

import type { TooltipProps } from 'recharts';

interface ChartTooltipRow {
  key: string;
  label: string;
  color: string;
  format?: (v: number) => string;
}

interface ChartTooltipConfig {
  rows: ChartTooltipRow[];
  timeKey?: string;
}

function defaultFormat(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}

export function createChartTooltip(config: ChartTooltipConfig) {
  return function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
    if (!active || !payload?.length) return null;

    const timeLabel = config.timeKey
      ? (payload[0]?.payload?.[config.timeKey] as string)
      : (label as string);

    const displayTime = timeLabel
      ? timeLabel.includes('T')
        ? `${new Date(timeLabel).getUTCHours().toString().padStart(2, '0')}:00`
        : timeLabel
      : '';

    return (
      <div className="glass-panel-elevated rounded-lg px-3 py-2.5 text-xs min-w-[140px]">
        {displayTime && (
          <div className="text-[10px] text-slate-500 font-mono mb-1.5 pb-1.5 border-b border-white/6">
            {displayTime}
          </div>
        )}
        <div className="space-y-1">
          {config.rows.map((row) => {
            const entry = payload.find((p) => p.dataKey === row.key);
            if (!entry || entry.value == null) return null;
            const fmt = row.format ?? defaultFormat;
            return (
              <div key={row.key} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: row.color }}
                  />
                  <span className="text-slate-400">{row.label}</span>
                </div>
                <span className="font-mono text-white font-medium">{fmt(entry.value as number)}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };
}
