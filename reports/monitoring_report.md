# Monitoring Report

```json
{
  "data_drift": {
    "columns": {
      "hour": {
        "ks_stat": 0.0003224005851329981,
        "p_value": 1.0,
        "drift": false
      },
      "dayofweek": {
        "ks_stat": 0.000635177272202303,
        "p_value": 1.0,
        "drift": false
      },
      "month": {
        "ks_stat": 0.6725949878738885,
        "p_value": 3.773785328915229e-75,
        "drift": true
      },
      "is_weekend": {
        "ks_stat": 0.000635177272202303,
        "p_value": 1.0,
        "drift": false
      },
      "season": {
        "ks_stat": 0.7531470146668207,
        "p_value": 1.0267275799226061e-98,
        "drift": true
      },
      "load_mw_delta_1h": {
        "ks_stat": 0.05415367440428076,
        "p_value": 0.690779073795915,
        "drift": false
      },
      "load_mw_delta_24h": {
        "ks_stat": 0.07309831774261846,
        "p_value": 0.3185216425839419,
        "drift": false
      },
      "wind_mw_delta_1h": {
        "ks_stat": 0.07667359587327252,
        "p_value": 0.2661159543489,
        "drift": false
      },
      "wind_mw_delta_24h": {
        "ks_stat": 0.14764503214381952,
        "p_value": 0.0012154342036640087,
        "drift": true
      },
      "solar_mw_delta_1h": {
        "ks_stat": 0.029290333756784845,
        "p_value": 0.9980400716273029,
        "drift": false
      },
      "solar_mw_delta_24h": {
        "ks_stat": 0.0642683912691997,
        "p_value": 0.47584723707222965,
        "drift": false
      },
      "is_morning_peak": {
        "ks_stat": 1.924779612727523e-05,
        "p_value": 1.0,
        "drift": false
      },
      "is_evening_peak": {
        "ks_stat": 0.00013954652192327277,
        "p_value": 1.0,
        "drift": false
      },
      "is_daylight": {
        "ks_stat": 0.018545251568695353,
        "p_value": 0.9999999687467211,
        "drift": false
      },
      "price_eur_mwh": {
        "ks_stat": 0.11171902067213302,
        "p_value": 0.028543779536247982,
        "drift": false
      },
      "carbon_kg_per_mwh": {
        "ks_stat": 0.09950148208030185,
        "p_value": 0.0683542219587363,
        "drift": false
      },
      "load_mw_lag_1": {
        "ks_stat": 0.15317877353043075,
        "p_value": 0.0006910076291437835,
        "drift": true
      },
      "load_mw_lag_24": {
        "ks_stat": 0.1565327020056203,
        "p_value": 0.0004857174255740144,
        "drift": true
      },
      "load_mw_lag_168": {
        "ks_stat": 0.20903587789198141,
        "p_value": 7.031453150963161e-07,
        "drift": true
      },
      "load_mw_roll_mean_24": {
        "ks_stat": 0.4184952072987643,
        "p_value": 5.151640468591873e-27,
        "drift": true
      },
      "load_mw_roll_std_24": {
        "ks_stat": 0.1395320860761442,
        "p_value": 0.0026777557260954945,
        "drift": true
      },
      "load_mw_roll_mean_168": {
        "ks_stat": 0.6078646494976325,
        "p_value": 1.0135204685836058e-59,
        "drift": true
      },
      "load_mw_roll_std_168": {
        "ks_stat": 0.6517496246679755,
        "p_value": 6.901628757484839e-70,
        "drift": true
      },
      "wind_mw_lag_1": {
        "ks_stat": 0.14305443276744814,
        "p_value": 0.0019109290516651028,
        "drift": true
      },
      "wind_mw_lag_24": {
        "ks_stat": 0.0885639219309389,
        "p_value": 0.13708650629574115,
        "drift": false
      },
      "wind_mw_lag_168": {
        "ks_stat": 0.16914000846903032,
        "p_value": 0.00012042222673735577,
        "drift": true
      },
      "wind_mw_roll_mean_24": {
        "ks_stat": 0.1390075836316742,
        "p_value": 0.0028136749101987757,
        "drift": true
      },
      "wind_mw_roll_std_24": {
        "ks_stat": 0.2741800438849752,
        "p_value": 1.382584557912556e-11,
        "drift": true
      },
      "wind_mw_roll_mean_168": {
        "ks_stat": 0.3886130038110637,
        "p_value": 3.217351156224932e-23,
        "drift": true
      },
      "wind_mw_roll_std_168": {
        "ks_stat": 0.6591888978711937,
        "p_value": 9.886825656594784e-72,
        "drift": true
      },
      "solar_mw_lag_1": {
        "ks_stat": 0.04935616121954034,
        "p_value": 0.7917624703914998,
        "drift": false
      },
      "solar_mw_lag_24": {
        "ks_stat": 0.04351926704392339,
        "p_value": 0.8959896825931918,
        "drift": false
      },
      "solar_mw_lag_168": {
        "ks_stat": 0.17812391731146782,
        "p_value": 4.167728635646047e-05,
        "drift": true
      },
      "solar_mw_roll_mean_24": {
        "ks_stat": 0.3498671902067213,
        "p_value": 8.873878968840282e-19,
        "drift": true
      },
      "solar_mw_roll_std_24": {
        "ks_stat": 0.24621780806097704,
        "p_value": 2.1106883965970706e-09,
        "drift": true
      },
      "solar_mw_roll_mean_168": {
        "ks_stat": 0.44745351657235244,
        "p_value": 5.149414414059813e-31,
        "drift": true
      },
      "solar_mw_roll_std_168": {
        "ks_stat": 0.4650363783346807,
        "p_value": 1.3250117573084112e-33,
        "drift": true
      },
      "price_eur_mwh_lag_1": {
        "ks_stat": 0.11171902067213302,
        "p_value": 0.028543779536247982,
        "drift": false
      },
      "price_eur_mwh_lag_24": {
        "ks_stat": 0.1120654810024252,
        "p_value": 0.02780454717052916,
        "drift": false
      },
      "price_eur_mwh_lag_168": {
        "ks_stat": 0.11379778265388618,
        "p_value": 0.024356046223997452,
        "drift": false
      },
      "price_eur_mwh_roll_mean_24": {
        "ks_stat": 0.21477653308696154,
        "p_value": 3.059060595679498e-07,
        "drift": true
      },
      "price_eur_mwh_roll_std_24": {
        "ks_stat": 0.43975439812141515,
        "p_value": 6.41207940264132e-30,
        "drift": true
      },
      "price_eur_mwh_roll_mean_168": {
        "ks_stat": 0.2689109596951149,
        "p_value": 3.726263462441287e-11,
        "drift": true
      },
      "price_eur_mwh_roll_std_168": {
        "ks_stat": 0.6297493936944221,
        "p_value": 1.2089566203355804e-64,
        "drift": true
      },
      "carbon_kg_per_mwh_lag_1": {
        "ks_stat": 0.09941486699772883,
        "p_value": 0.068753879107887,
        "drift": false
      },
      "carbon_kg_per_mwh_lag_24": {
        "ks_stat": 0.06524521692266239,
        "p_value": 0.4566283250585944,
        "drift": false
      },
      "carbon_kg_per_mwh_lag_168": {
        "ks_stat": 0.10079589636986563,
        "p_value": 0.06261307962067097,
        "drift": false
      },
      "carbon_kg_per_mwh_roll_mean_24": {
        "ks_stat": 0.1584574816183546,
        "p_value": 0.00039536692369950704,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_24": {
        "ks_stat": 0.20553759094583673,
        "p_value": 1.154350250383791e-06,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_mean_168": {
        "ks_stat": 0.5196038803556993,
        "p_value": 1.846183712218144e-42,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_168": {
        "ks_stat": 0.7230049659314008,
        "p_value": 3.1895243703104825e-89,
        "drift": true
      }
    },
    "drift": true
  },
  "model_drift": {
    "current": {
      "rmse": 211.62248914194316,
      "mape": 0.0025661292774295593
    },
    "baseline_mape": 0.0025661292774295593,
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
    "last_trained_days_ago": 0
  }
}
```
