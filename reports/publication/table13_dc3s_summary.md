# Table 13: DC3S Controller Summary

Derived from `reports/publication/dc3s_main_table.csv` by averaging metrics across seeds for each `(scenario, controller)` pair.

| scenario | controller | n_seeds | picp_90 | picp_95 | mean_interval_width | expected_cost_usd | intervention_rate | violation_rate | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | deterministic_lp | 10 | 1.000 | 1.000 | 666.98 | 57,538,269.07 | 0.000 | 0.154 | 0.00 | 0.00 |
| nominal | robust_fixed_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| nominal | cvar_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| nominal | dc3s_wrapped | 10 | 1.000 | 1.000 | 696.90 | 57,598,325.71 | 0.000 | 0.000 | 0.00 | 0.00 |
| dropout | deterministic_lp | 10 | 0.962 | 0.962 | 666.98 | 57,511,413.43 | 0.000 | 0.150 | 45.33 | 222.09 |
| dropout | robust_fixed_interval | 10 | 0.962 | 0.962 | 666.98 | 57,498,365.26 | 0.000 | 0.125 | 45.33 | 222.09 |
| dropout | cvar_interval | 10 | 0.962 | 0.962 | 666.98 | 57,498,365.26 | 0.000 | 0.125 | 45.33 | 222.09 |
| dropout | dc3s_wrapped | 10 | 0.962 | 0.962 | 698.39 | 57,610,164.75 | 0.000 | 0.000 | 45.33 | 222.09 |
| delay_jitter | deterministic_lp | 10 | 1.000 | 1.000 | 666.98 | 57,538,269.07 | 0.000 | 0.154 | 0.00 | 0.00 |
| delay_jitter | robust_fixed_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| delay_jitter | cvar_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| delay_jitter | dc3s_wrapped | 10 | 1.000 | 1.000 | 701.40 | 57,598,325.71 | 0.000 | 0.000 | 0.00 | 0.00 |
| out_of_order | deterministic_lp | 10 | 1.000 | 1.000 | 666.98 | 57,538,269.07 | 0.000 | 0.154 | 0.00 | 0.00 |
| out_of_order | robust_fixed_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| out_of_order | cvar_interval | 10 | 1.000 | 1.000 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 0.00 | 0.00 |
| out_of_order | dc3s_wrapped | 10 | 1.000 | 1.000 | 2068.96 | 57,598,325.71 | 0.000 | 0.000 | 0.00 | 0.00 |
| spikes | deterministic_lp | 10 | 0.958 | 0.958 | 666.98 | 57,537,965.43 | 0.000 | 0.154 | 561.57 | 2751.13 |
| spikes | robust_fixed_interval | 10 | 0.958 | 0.958 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 561.57 | 2751.13 |
| spikes | cvar_interval | 10 | 0.958 | 0.958 | 666.98 | 57,529,913.85 | 0.000 | 0.113 | 561.57 | 2751.13 |
| spikes | dc3s_wrapped | 10 | 0.958 | 0.958 | 810.10 | 57,598,325.71 | 0.000 | 0.000 | 561.57 | 2751.13 |
| drift_combo | deterministic_lp | 10 | 0.338 | 0.342 | 666.98 | 61,201,023.12 | 0.000 | 0.150 | 3444.59 | 5848.83 |
| drift_combo | robust_fixed_interval | 10 | 0.338 | 0.342 | 666.98 | 61,149,672.47 | 0.000 | 0.117 | 3444.59 | 5848.83 |
| drift_combo | cvar_interval | 10 | 0.338 | 0.342 | 666.98 | 61,149,672.47 | 0.000 | 0.117 | 3444.59 | 5848.83 |
| drift_combo | dc3s_wrapped | 10 | 0.383 | 0.408 | 2249.64 | 61,310,572.02 | 0.000 | 0.000 | 3444.59 | 5848.83 |
