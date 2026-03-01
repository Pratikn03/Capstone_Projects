# DC3S Ablation Table

Mean CPSBench-IoT `drift_combo` metrics across seeds for four DC3S ablation variants.

| scenario | policy | n_seeds | picp_90 | picp_95 | mean_interval_width | intervention_rate | violation_rate | true_soc_violation_rate | expected_cost_usd | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift_combo | dc3s_no_wt | 5 | 0.898 | 0.958 | 14747.41 | 0.015 | 0.000 | 0.000 | 233,275,836.00 | 3260.22 | 5718.69 |
| drift_combo | dc3s_no_drift | 5 | 0.923 | 0.971 | 19590.73 | 0.015 | 0.000 | 0.000 | 233,275,836.00 | 3260.22 | 5718.69 |
| drift_combo | dc3s_linear | 5 | 0.923 | 0.971 | 19811.59 | 0.015 | 0.000 | 0.000 | 233,275,836.00 | 3260.22 | 5718.69 |
| drift_combo | dc3s_kappa | 5 | 0.977 | 0.981 | 25537.71 | 0.015 | 0.000 | 0.000 | 233,275,836.00 | 3260.22 | 5718.69 |
