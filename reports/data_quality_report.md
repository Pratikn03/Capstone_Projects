# Data Quality Report

Input file: `data/raw/time_series_60min_singleindex.csv`

## Coverage
- Start: **2014-12-31 23:00:00+00:00**
- End: **2020-09-30 23:00:00+00:00**
- Rows: **50401**
- Duplicate timestamps: **0**
- Missing hourly timestamps (vs full hourly index): **0**

## Missingness (fraction)
```text
DE_solar_generation_actual            0.002063
DE_wind_generation_actual             0.001488
DE_load_actual_entsoe_transparency    0.000020
utc_timestamp                         0.000000
```

## Negative values check (count)
```text
DE_load_actual_entsoe_transparency: 0
DE_solar_generation_actual: 0
DE_wind_generation_actual: 0
```
