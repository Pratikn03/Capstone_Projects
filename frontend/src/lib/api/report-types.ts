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

export type RobustnessSummary = {
  perturbation_pct: number;
  infeasible_rate: number | null;
  mean_regret: number | null;
  p95_regret: number | null;
};

export type ReportsApiResponse = {
  reports: ReportFile[];
  metrics: ForecastMetrics[];
  metrics_backtest?: ForecastMetrics[];
  impact: ImpactSummary | null;
  robustness: RobustnessSummary | null;
  regions?: Record<string, RegionReports>;
  meta: {
    source: 'reports' | 'missing';
    last_updated?: string;
    metrics_source?: 'week2_metrics' | 'forecast_point_metrics' | 'missing';
    warnings?: string[];
  };
};

export type TrainingStatus = {
  features_path: string | null;
  features_exists: boolean;
  targets_expected: string[];
  targets_trained: string[];
  targets_missing: string[];
  models_dir: string | null;
  missing_models: Array<{ target: string; missing: string[] }>;
};

export type RegionReports = {
  id: string;
  label: string;
  reports: ReportFile[];
  metrics: ForecastMetrics[];
  metrics_backtest?: ForecastMetrics[];
  impact: ImpactSummary | null;
  robustness: RobustnessSummary | null;
  training_status: TrainingStatus | null;
  meta: {
    source: 'reports' | 'missing';
    last_updated?: string;
    metrics_source?: 'week2_metrics' | 'forecast_point_metrics' | 'missing';
    warnings?: string[];
  };
};
