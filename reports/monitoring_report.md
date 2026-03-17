# Monitoring Report

```json
{
  "data_drift": {
    "columns": {
      "price_eur_mwh": {
        "ks_stat": 0.10437489967622338,
        "p_value": 0.0499158528040099,
        "drift": false
      },
      "hour": {
        "ks_stat": 0.00030831209405579596,
        "p_value": 1.0,
        "drift": false
      },
      "dayofweek": {
        "ks_stat": 0.0030655031066114113,
        "p_value": 1.0,
        "drift": false
      },
      "month": {
        "ks_stat": 0.5914659212365371,
        "p_value": 7.070267096893439e-56,
        "drift": true
      },
      "is_weekend": {
        "ks_stat": 0.001491643274098231,
        "p_value": 1.0,
        "drift": false
      },
      "season": {
        "ks_stat": 0.7138041601578557,
        "p_value": 5.541037633574362e-86,
        "drift": true
      },
      "load_mw_delta_1h": {
        "ks_stat": 0.0464469722773595,
        "p_value": 0.849736579290959,
        "drift": false
      },
      "load_mw_delta_24h": {
        "ks_stat": 0.061893897573828516,
        "p_value": 0.5279278904502898,
        "drift": false
      },
      "wind_mw_delta_1h": {
        "ks_stat": 0.12040125595580664,
        "p_value": 0.014855479924123316,
        "drift": false
      },
      "wind_mw_delta_24h": {
        "ks_stat": 0.1802886779968914,
        "p_value": 3.417090509497833e-05,
        "drift": true
      },
      "solar_mw_delta_1h": {
        "ks_stat": 0.034048911022108386,
        "p_value": 0.9871738990289245,
        "drift": false
      },
      "solar_mw_delta_24h": {
        "ks_stat": 0.07300536756674225,
        "p_value": 0.32336330319465534,
        "drift": false
      },
      "is_morning_peak": {
        "ks_stat": 6.851379867911511e-05,
        "p_value": 1.0,
        "drift": false
      },
      "is_evening_peak": {
        "ks_stat": 0.0002432239853106255,
        "p_value": 1.0,
        "drift": false
      },
      "is_daylight": {
        "ks_stat": 0.05599632766039081,
        "p_value": 0.6542370395069192,
        "drift": false
      },
      "is_holiday": {
        "ks_stat": 0.0,
        "p_value": 1.0,
        "drift": false
      },
      "is_pre_holiday": {
        "ks_stat": 0.0,
        "p_value": 1.0,
        "drift": false
      },
      "is_post_holiday": {
        "ks_stat": 0.0,
        "p_value": 1.0,
        "drift": false
      },
      "carbon_kg_per_mwh": {
        "ks_stat": 0.19496286552111597,
        "p_value": 5.29189294457347e-06,
        "drift": true
      },
      "load_mw_lag_1": {
        "ks_stat": 0.1842311577265947,
        "p_value": 2.100939735436497e-05,
        "drift": true
      },
      "load_mw_lag_24": {
        "ks_stat": 0.1833267755840311,
        "p_value": 2.3511786997580872e-05,
        "drift": true
      },
      "load_mw_lag_168": {
        "ks_stat": 0.2328867212428012,
        "p_value": 2.1189605464136875e-08,
        "drift": true
      },
      "load_mw_roll_mean_24": {
        "ks_stat": 0.49477924854065614,
        "p_value": 4.779405863847674e-38,
        "drift": true
      },
      "load_mw_roll_std_24": {
        "ks_stat": 0.21534669939668705,
        "p_value": 3.0891501192635615e-07,
        "drift": true
      },
      "load_mw_roll_mean_168": {
        "ks_stat": 0.7091178163282086,
        "p_value": 1.2442812561129528e-84,
        "drift": true
      },
      "load_mw_roll_std_168": {
        "ks_stat": 0.4081229959713887,
        "p_value": 1.6500209486756387e-25,
        "drift": true
      },
      "wind_mw_lag_1": {
        "ks_stat": 0.300789768345059,
        "p_value": 8.1097520090749e-14,
        "drift": true
      },
      "wind_mw_lag_24": {
        "ks_stat": 0.22275108349678768,
        "p_value": 1.0236867579787184e-07,
        "drift": true
      },
      "wind_mw_lag_168": {
        "ks_stat": 0.32848842899817166,
        "p_value": 1.8634767172849624e-16,
        "drift": true
      },
      "wind_mw_roll_mean_24": {
        "ks_stat": 0.296088253602847,
        "p_value": 2.1466589795963708e-13,
        "drift": true
      },
      "wind_mw_roll_std_24": {
        "ks_stat": 0.2717854304428341,
        "p_value": 2.5277829847055966e-11,
        "drift": true
      },
      "wind_mw_roll_mean_168": {
        "ks_stat": 0.7205459179478747,
        "p_value": 5.836879509540325e-88,
        "drift": true
      },
      "wind_mw_roll_std_168": {
        "ks_stat": 0.43915289539313224,
        "p_value": 1.16717618927893e-29,
        "drift": true
      },
      "solar_mw_lag_1": {
        "ks_stat": 0.05609811958985689,
        "p_value": 0.6520220537961758,
        "drift": false
      },
      "solar_mw_lag_24": {
        "ks_stat": 0.06025837532250422,
        "p_value": 0.5622664131178207,
        "drift": false
      },
      "solar_mw_lag_168": {
        "ks_stat": 0.18649504938866124,
        "p_value": 1.5812208863688308e-05,
        "drift": true
      },
      "solar_mw_roll_mean_24": {
        "ks_stat": 0.31028529145769956,
        "p_value": 1.0791158067344108e-14,
        "drift": true
      },
      "solar_mw_roll_std_24": {
        "ks_stat": 0.2704102606264902,
        "p_value": 3.267746342425031e-11,
        "drift": true
      },
      "solar_mw_roll_mean_168": {
        "ks_stat": 0.561045794623037,
        "p_value": 8.801799658539856e-50,
        "drift": true
      },
      "solar_mw_roll_std_168": {
        "ks_stat": 0.5436981007975006,
        "p_value": 1.6336127958760003e-46,
        "drift": true
      },
      "price_eur_mwh_lag_1": {
        "ks_stat": 0.10429268311780848,
        "p_value": 0.05020474356491067,
        "drift": false
      },
      "price_eur_mwh_lag_24": {
        "ks_stat": 0.08160140237958213,
        "p_value": 0.2073306283734493,
        "drift": false
      },
      "price_eur_mwh_lag_168": {
        "ks_stat": 0.2254544422389526,
        "p_value": 6.77372368188472e-08,
        "drift": true
      },
      "price_eur_mwh_roll_mean_24": {
        "ks_stat": 0.1628259788664294,
        "p_value": 0.000258927578564318,
        "drift": true
      },
      "price_eur_mwh_roll_std_24": {
        "ks_stat": 0.42157568034202086,
        "p_value": 2.9030574275674324e-27,
        "drift": true
      },
      "price_eur_mwh_roll_mean_168": {
        "ks_stat": 0.6225437803173559,
        "p_value": 1.284218845195638e-62,
        "drift": true
      },
      "price_eur_mwh_roll_std_168": {
        "ks_stat": 0.5594014634547397,
        "p_value": 1.822523428268748e-49,
        "drift": true
      },
      "carbon_kg_per_mwh_lag_1": {
        "ks_stat": 0.19496286552111597,
        "p_value": 5.29189294457347e-06,
        "drift": true
      },
      "carbon_kg_per_mwh_lag_24": {
        "ks_stat": 0.14098818430603352,
        "p_value": 0.0024265498462473674,
        "drift": true
      },
      "carbon_kg_per_mwh_lag_168": {
        "ks_stat": 0.13488751991794,
        "p_value": 0.004277670941799961,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_mean_24": {
        "ks_stat": 0.24269789721364166,
        "p_value": 4.301240458228108e-09,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_24": {
        "ks_stat": 0.20870086092481882,
        "p_value": 8.054400601901655e-07,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_mean_168": {
        "ks_stat": 0.5954945325988654,
        "p_value": 1.0136829416012473e-56,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_168": {
        "ks_stat": 0.5092493628216723,
        "p_value": 1.9328824966776505e-40,
        "drift": true
      }
    },
    "drift": true
  },
  "model_drift": {
    "current": {
      "rmse": 305.1209624375309,
      "mape": 0.003911572495132256
    },
    "baseline_mape": 0.003911572495132256,
    "decision": {
      "drift": false,
      "ratio": 0.0
    }
  },
  "retraining": {
    "retrain": true,
    "reasons": [
      "data_drift"
    ],
    "last_trained_days_ago": 26
  },
  "dc3s_health": {
    "window_hours": 24,
    "commands_total": 12,
    "intervention_rate": 1.0,
    "low_reliability_rate": 0.0,
    "drift_flag_rate": 0.0,
    "inflation_p95": 1.0,
    "triggered_flags": [],
    "triggered": false,
    "insufficient_data": true,
    "sustained_windows": 3,
    "sustained_breach_counts": {
      "intervention_rate": 0,
      "low_reliability_rate": 0,
      "drift_flag_rate": 0,
      "inflation_p95": 0
    },
    "top_intervention_reasons": [
      "projection_clip"
    ]
  }
}
```
