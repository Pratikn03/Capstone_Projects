export type DispatchSeriesPoint = {
  timestamp: string;
  load_mw: number;
  generation_solar: number;
  generation_wind: number;
  generation_gas: number;
  battery_dispatch: number;
  price_eur_mwh?: number;
  carbon_kg_mwh?: number;
};

export type DispatchCompareResponse = {
  optimized: DispatchSeriesPoint[];
  baseline?: DispatchSeriesPoint[];
  meta: {
    source: 'fastapi' | 'missing';
    horizon_hours?: number;
    generated_at?: string;
    warnings?: string[];
  };
};
