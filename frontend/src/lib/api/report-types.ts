import type { ForecastMetrics } from './schema';

export type ReportFile = {
  name: string;
  title: string;
  description: string;
  type: string;
  date: string;
  path: string;
  size_bytes: number;
};

export type ImpactSummary = {
  cost_savings_pct: number | null;
  carbon_reduction_pct: number | null;
  peak_shaving_pct: number | null;
  cost_savings_usd: number | null;
  carbon_reduction_kg: number | null;
  peak_shaving_mw: number | null;
};

export type ReportsApiResponse = {
  reports: ReportFile[];
  metrics: ForecastMetrics[];
  impact: ImpactSummary | null;
  meta: {
    source: 'reports' | 'missing';
    last_updated?: string;
    warnings?: string[];
  };
};
