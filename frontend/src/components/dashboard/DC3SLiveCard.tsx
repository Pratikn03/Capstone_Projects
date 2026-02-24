'use client';

import { ExternalLink, RefreshCcw, ShieldCheck } from 'lucide-react';

import { Panel } from '@/components/ui/Panel';
import type { Dc3sLivePayload } from '@/lib/api/dc3s-client';

interface DC3SLiveCardProps {
  region: 'DE' | 'US';
  data: Dc3sLivePayload;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  autoRefreshSeconds: number;
  onAutoRefreshSecondsChange: (seconds: number) => void;
  auditHref?: string | null;
}

function fmt(value: number | null | undefined, digits = 2): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
  return value.toFixed(digits);
}

function actionFmt(action: { charge_mw?: number; discharge_mw?: number } | undefined): string {
  if (!action) return 'N/A';
  const c = typeof action.charge_mw === 'number' ? action.charge_mw.toFixed(2) : '0.00';
  const d = typeof action.discharge_mw === 'number' ? action.discharge_mw.toFixed(2) : '0.00';
  return `C ${c} / D ${d} MW`;
}

export function DC3SLiveCard({
  region,
  data,
  loading,
  error,
  onRefresh,
  autoRefreshSeconds,
  onAutoRefreshSecondsChange,
  auditHref,
}: DC3SLiveCardProps) {
  const rows = data.uncertainty_preview ?? [];

  return (
    <Panel
      title="DC3S Live Shield"
      subtitle={`${region} • ${data.controller ?? 'deterministic'}`}
      badge={
        loading
          ? 'Loading'
          : error
          ? 'Error'
          : data.repaired
          ? 'Intervened'
          : 'Pass-through'
      }
      badgeColor={error ? 'alert' : data.repaired ? 'warn' : 'primary'}
      delay={0.2}
      actions={
        <>
          <div className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs text-slate-300">
            <span className="text-slate-400">Auto</span>
            <select
              value={autoRefreshSeconds}
              onChange={(e) => onAutoRefreshSecondsChange(Number(e.target.value))}
              className="bg-transparent text-slate-100 outline-none"
              aria-label="Auto refresh interval"
            >
              <option value={0}>Off</option>
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={15}>15s</option>
              <option value={30}>30s</option>
              <option value={60}>60s</option>
            </select>
          </div>
          <button
            onClick={onRefresh}
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200"
            type="button"
          >
            <RefreshCcw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </>
      }
    >
      {error ? (
        <div className="text-sm text-energy-alert">{error}</div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">w_t</div>
              <div className="text-white font-mono mt-1">{fmt(data.reliability_w_t, 3)}</div>
            </div>
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">Inflation</div>
              <div className="text-white font-mono mt-1">{fmt(data.inflation, 3)}</div>
            </div>
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">Drift</div>
              <div className="text-white font-mono mt-1">{String(data.drift_flag ?? 'N/A')}</div>
            </div>
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">Mean PI Width</div>
              <div className="text-white font-mono mt-1">{fmt(data.mean_interval_width_mw, 2)} MW</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">Proposed Action</div>
              <div className="text-white font-mono mt-1">{actionFmt(data.proposed_action)}</div>
            </div>
            <div className="rounded-lg bg-white/3 p-3">
              <div className="text-slate-500">Safe Action</div>
              <div className="text-white font-mono mt-1">{actionFmt(data.safe_action)}</div>
            </div>
          </div>

          <div className="rounded-lg bg-white/3 p-3 text-xs">
            <div className="flex items-center justify-between gap-2 text-slate-400">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Certificate</span>
              </div>
              {auditHref ? (
                <a
                  href={auditHref}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-[11px] text-energy-info hover:underline"
                >
                  Open
                  <ExternalLink className="h-3 w-3" />
                </a>
              ) : null}
            </div>
            <div className="mt-1 font-mono text-slate-200 break-all">
              {data.command_id ?? 'N/A'}
            </div>
            <div className="mt-1 font-mono text-slate-400 break-all">
              {data.certificate_hash ?? 'N/A'}
            </div>
          </div>

          <div className="text-[11px] text-slate-500">
            Auto refresh: {autoRefreshSeconds > 0 ? `${autoRefreshSeconds}s` : 'off'}
          </div>

          <div className="rounded-lg bg-white/3 p-3 text-xs">
            <div className="flex items-center gap-2 text-slate-400">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Command Trace</span>
            </div>
            <div className="mt-1 text-slate-300">
              Matching audit record opens with the same <span className="font-mono">command_id</span>.
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-white/10">
                  <th className="text-left py-2 pr-3">h</th>
                  <th className="text-right py-2 pr-3">Lower</th>
                  <th className="text-right py-2 pr-3">Upper</th>
                  <th className="text-right py-2">Width</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td className="py-3 text-slate-500" colSpan={4}>
                      No uncertainty preview available.
                    </td>
                  </tr>
                ) : (
                  rows.map((row) => (
                    <tr key={row.h} className="border-b border-white/5">
                      <td className="py-2 pr-3 text-slate-400">{row.h}</td>
                      <td className="py-2 pr-3 text-right font-mono text-slate-200">{fmt(row.lower, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono text-slate-200">{fmt(row.upper, 2)}</td>
                      <td className="py-2 text-right font-mono text-slate-300">{fmt(row.width, 2)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Panel>
  );
}
