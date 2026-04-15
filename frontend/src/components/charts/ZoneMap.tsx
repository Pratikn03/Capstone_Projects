'use client';

import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useState, useMemo } from 'react';

export interface ZoneData {
  id: string;
  label: string;
  renewablePct: number;
  loadMw?: number;
}

interface ZoneMapProps {
  region: 'DE' | 'US';
  zones: ZoneData[];
}

/* -------- simplified SVG path data for DE Bundesländer -------- */
const DE_ZONES: Record<string, { path: string; cx: number; cy: number }> = {
  'DE-SH': { path: 'M145,30 L170,15 L195,25 L200,55 L175,72 L150,65 L140,45Z', cx: 168, cy: 42 },
  'DE-HH': { path: 'M175,72 L190,68 L195,78 L185,85 L172,80Z', cx: 183, cy: 76 },
  'DE-NI': { path: 'M100,65 L150,65 L175,80 L185,85 L195,78 L205,90 L195,130 L150,140 L110,130 L80,105 L85,80Z', cx: 145, cy: 105 },
  'DE-HB': { path: 'M140,88 L155,85 L158,95 L145,98Z', cx: 149, cy: 91 },
  'DE-MV': { path: 'M200,25 L260,20 L280,40 L275,70 L240,80 L200,75 L195,55Z', cx: 238, cy: 50 },
  'DE-BB': { path: 'M240,80 L275,70 L290,95 L295,140 L280,170 L240,175 L220,155 L215,120 L220,90Z', cx: 255, cy: 125 },
  'DE-BE': { path: 'M248,120 L262,115 L268,128 L260,138 L248,132Z', cx: 256, cy: 127 },
  'DE-ST': { path: 'M195,130 L220,90 L240,95 L240,140 L230,165 L200,165 L185,150Z', cx: 215, cy: 135 },
  'DE-NW': { path: 'M60,120 L110,130 L130,150 L120,185 L90,195 L55,180 L45,150Z', cx: 87, cy: 160 },
  'DE-HE': { path: 'M120,185 L130,150 L160,155 L175,170 L170,210 L140,220 L120,205Z', cx: 147, cy: 185 },
  'DE-TH': { path: 'M175,170 L200,165 L230,165 L235,185 L220,210 L190,210 L170,195Z', cx: 203, cy: 188 },
  'DE-SN': { path: 'M235,185 L240,175 L280,170 L295,185 L290,215 L255,220 L240,210Z', cx: 265, cy: 198 },
  'DE-RP': { path: 'M55,180 L90,195 L100,215 L90,250 L60,255 L40,230 L38,200Z', cx: 68, cy: 220 },
  'DE-SL': { path: 'M40,250 L60,255 L62,272 L48,280 L35,268Z', cx: 49, cy: 265 },
  'DE-BW': { path: 'M90,250 L100,215 L140,220 L155,240 L150,285 L120,305 L85,300 L75,275Z', cx: 115, cy: 270 },
  'DE-BY': { path: 'M155,240 L170,210 L190,210 L220,210 L255,220 L270,245 L265,290 L240,320 L200,330 L165,310 L150,285Z', cx: 210, cy: 275 },
};

/* -------- simplified SVG regions for US balancing authorities -------- */
const US_ZONES: Record<string, { path: string; cx: number; cy: number }> = {
  'US-MISO': { path: 'M160,60 L230,50 L250,80 L240,140 L200,160 L170,150 L150,110 L145,80Z', cx: 197, cy: 105 },
  'US-PJM': { path: 'M250,80 L300,70 L320,95 L310,140 L280,150 L240,140Z', cx: 280, cy: 110 },
  'US-NYISO': { path: 'M300,50 L330,40 L345,60 L335,85 L320,95 L300,70Z', cx: 318, cy: 68 },
  'US-ISONE': { path: 'M335,35 L360,30 L370,50 L355,65 L345,60 L335,45Z', cx: 352, cy: 48 },
  'US-SPP': { path: 'M130,120 L170,110 L200,160 L190,200 L150,210 L120,185 L115,150Z', cx: 158, cy: 165 },
  'US-ERCOT': { path: 'M120,200 L155,210 L175,240 L160,290 L120,300 L95,270 L100,230Z', cx: 133, cy: 255 },
  'US-CAISO': { path: 'M30,100 L55,90 L65,120 L60,180 L45,210 L25,200 L20,150Z', cx: 42, cy: 150 },
  'US-SOCO': { path: 'M230,170 L280,160 L300,190 L290,230 L250,240 L225,215 L220,190Z', cx: 258, cy: 200 },
};

function interpolateColor(pct: number): string {
  // 0% → red, 50% → amber, 100% → green
  const stops = [
    { at: 0, r: 239, g: 68, b: 68 },
    { at: 50, r: 245, g: 158, b: 11 },
    { at: 100, r: 16, g: 185, b: 129 },
  ];
  const clamped = Math.max(0, Math.min(100, pct));
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped >= stops[i].at && clamped <= stops[i + 1].at) {
      lo = stops[i]; hi = stops[i + 1]; break;
    }
  }
  const t = (clamped - lo.at) / (hi.at - lo.at || 1);
  const r = Math.round(lo.r + t * (hi.r - lo.r));
  const g = Math.round(lo.g + t * (hi.g - lo.g));
  const b = Math.round(lo.b + t * (hi.b - lo.b));
  return `rgb(${r},${g},${b})`;
}

export function ZoneMap({ region, zones }: ZoneMapProps) {
  const router = useRouter();
  const [hovered, setHovered] = useState<string | null>(null);
  const paths = region === 'DE' ? DE_ZONES : US_ZONES;

  const zoneMap = useMemo(() => {
    const m = new Map<string, ZoneData>();
    zones.forEach((z) => m.set(z.id, z));
    return m;
  }, [zones]);

  const viewBox = region === 'DE' ? '20 5 300 340' : '10 20 380 300';

  return (
    <div className="relative">
      <svg viewBox={viewBox} className="w-full h-auto max-h-[320px]" role="img" aria-label={`${region} zone map`}>
        {Object.entries(paths).map(([id, { path, cx, cy }]) => {
          const zone = zoneMap.get(id);
          const pct = zone?.renewablePct ?? 0;
          const fill = interpolateColor(pct);
          const isHovered = hovered === id;

          return (
            <motion.g
              key={id}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => router.push(`/zone/${id}`)}
              style={{ cursor: 'pointer' }}
              whileHover={{ scale: 1.03 }}
            >
              <motion.path
                d={path}
                fill={fill}
                fillOpacity={isHovered ? 0.9 : 0.55}
                stroke={isHovered ? '#fff' : '#334155'}
                strokeWidth={isHovered ? 2 : 1}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: Math.random() * 0.3 }}
              />
              <text
                x={cx} y={cy}
                textAnchor="middle"
                dominantBaseline="central"
                className="pointer-events-none select-none"
                fill="#fff"
                fontSize={region === 'DE' ? 8 : 9}
                fontWeight={600}
                opacity={isHovered ? 1 : 0.7}
              >
                {zone?.label ?? id.split('-')[1]}
              </text>
              {isHovered && zone && (
                <text
                  x={cx} y={cy + (region === 'DE' ? 12 : 14)}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="pointer-events-none select-none"
                  fill="#94a3b8"
                  fontSize={7}
                >
                  {pct.toFixed(0)}% renewable
                </text>
              )}
            </motion.g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-2 mt-2">
        <span className="text-[10px] text-slate-500">0%</span>
        <div className="h-2 w-24 rounded-full" style={{
          background: 'linear-gradient(to right, rgb(239,68,68), rgb(245,158,11), rgb(16,185,129))',
        }} />
        <span className="text-[10px] text-slate-500">100% renewable</span>
      </div>
    </div>
  );
}
