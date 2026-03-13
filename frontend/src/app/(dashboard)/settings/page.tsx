'use client';

import { Panel } from '@/components/ui/Panel';

const runtimeNotes = [
  'The dashboard is backed by extracted dashboard JSON, report artifacts, and the live DC3S runtime route.',
  'The DC3S live shield card calls the FastAPI backend in real time through /api/dc3s/live.',
  'Other panels prefer repo artifacts and report exports over live streaming telemetry.',
  'Hardware validation and field commissioning remain future work and are intentionally out of scope here.',
];

export default function SettingsPage() {
  return (
    <div className="p-6 space-y-6">
      <div className="space-y-1">
        <h1 className="text-xl font-bold text-white">Settings & Runtime Notes</h1>
        <p className="text-sm text-slate-400">
          Dashboard configuration, data-source expectations, and scope boundaries.
        </p>
      </div>

      <Panel title="Runtime Surface" subtitle="What this dashboard is actually wired to" badge="Research UI" badgeColor="info">
        <div className="space-y-3 text-sm text-slate-300">
          {runtimeNotes.map((note) => (
            <div key={note} className="rounded-lg bg-white/3 px-3 py-2.5">
              {note}
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Scope Boundaries" subtitle="Current dashboard contract">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="rounded-lg bg-white/3 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Included</div>
            <div className="space-y-1 text-slate-300">
              <div>Forecasting and report artifacts</div>
              <div>Optimization and impact summaries</div>
              <div>Monitoring/drift views from extracted artifacts</div>
              <div>Live DC3S step and audit trace</div>
            </div>
          </div>
          <div className="rounded-lg bg-white/3 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Not Included Yet</div>
            <div className="space-y-1 text-slate-300">
              <div>Full operator-grade live telemetry console</div>
              <div>Per-balancing-authority UI decomposition</div>
              <div>Hardware-in-the-loop runtime validation</div>
              <div>Field-device commissioning controls</div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
