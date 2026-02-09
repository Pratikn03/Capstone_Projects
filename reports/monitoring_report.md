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
      "carbon_kg_per_mwh": {
        "ks_stat": 0.14553309607983622,
        "p_value": 0.0015646005353441577,
        "drift": true
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
        "ks_stat": 0.023678368823481066,
        "p_value": 0.999963816607781,
        "drift": false
      },
      "is_pre_holiday": {
        "ks_stat": 0.023678368823481066,
        "p_value": 0.999963816607781,
        "drift": false
      },
      "is_post_holiday": {
        "ks_stat": 0.023678368823481066,
        "p_value": 0.999963816607781,
        "drift": false
      },
      "wx_temperature_2m": {
        "ks_stat": 0.453080282511755,
        "p_value": 1.2134193826958065e-31,
        "drift": true
      },
      "wx_relative_humidity_2m": {
        "ks_stat": 0.12265584931662382,
        "p_value": 0.012353636139657717,
        "drift": false
      },
      "wx_precipitation": {
        "ks_stat": 0.0762945192876131,
        "p_value": 0.2745149548530621,
        "drift": false
      },
      "wx_cloud_cover": {
        "ks_stat": 0.16703810933236235,
        "p_value": 0.00016195422254072306,
        "drift": true
      },
      "wx_wind_speed_10m": {
        "ks_stat": 0.4226963703346997,
        "p_value": 2.0588324675282378e-27,
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
        "ks_stat": 0.14545087952142133,
        "p_value": 0.0015772679492010264,
        "drift": true
      },
      "carbon_kg_per_mwh_lag_24": {
        "ks_stat": 0.12250267203814848,
        "p_value": 0.012510755881264537,
        "drift": false
      },
      "carbon_kg_per_mwh_lag_168": {
        "ks_stat": 0.18329545499034938,
        "p_value": 2.3603356032169618e-05,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_mean_24": {
        "ks_stat": 0.2066547452656965,
        "p_value": 1.075099997052564e-06,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_24": {
        "ks_stat": 0.2667447723971608,
        "p_value": 6.434657032947179e-11,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_mean_168": {
        "ks_stat": 0.5390939735262682,
        "p_value": 1.138785315157773e-45,
        "drift": true
      },
      "carbon_kg_per_mwh_roll_std_168": {
        "ks_stat": 0.6076963507593286,
        "p_value": 2.4936024547784393e-59,
        "drift": true
      },
      "wx_temperature_2m_lag_1": {
        "ks_stat": 0.453080282511755,
        "p_value": 1.2134193826958065e-31,
        "drift": true
      },
      "wx_temperature_2m_lag_24": {
        "ks_stat": 0.4692930942006006,
        "p_value": 4.769831521257671e-34,
        "drift": true
      },
      "wx_temperature_2m_lag_168": {
        "ks_stat": 0.44674911812953416,
        "p_value": 9.88399886375292e-31,
        "drift": true
      },
      "wx_temperature_2m_roll_mean_24": {
        "ks_stat": 0.5618679602071858,
        "p_value": 6.109697849317829e-50,
        "drift": true
      },
      "wx_temperature_2m_roll_std_24": {
        "ks_stat": 0.09438460906026469,
        "p_value": 0.09734187849978115,
        "drift": false
      },
      "wx_temperature_2m_roll_mean_168": {
        "ks_stat": 0.6696538682890735,
        "p_value": 6.02206594749473e-74,
        "drift": true
      },
      "wx_temperature_2m_roll_std_168": {
        "ks_stat": 0.6830399963981317,
        "p_value": 1.9410274108383744e-77,
        "drift": true
      },
      "wx_relative_humidity_2m_lag_1": {
        "ks_stat": 0.11670346836424283,
        "p_value": 0.019953288587336093,
        "drift": false
      },
      "wx_relative_humidity_2m_lag_24": {
        "ks_stat": 0.09259346260908374,
        "p_value": 0.10894691154764391,
        "drift": false
      },
      "wx_relative_humidity_2m_lag_168": {
        "ks_stat": 0.355476014297851,
        "p_value": 2.82550288948742e-19,
        "drift": true
      },
      "wx_relative_humidity_2m_roll_mean_24": {
        "ks_stat": 0.18594449207784733,
        "p_value": 1.6949222683246334e-05,
        "drift": true
      },
      "wx_relative_humidity_2m_roll_std_24": {
        "ks_stat": 0.17354936712825386,
        "p_value": 7.654629033717053e-05,
        "drift": true
      },
      "wx_relative_humidity_2m_roll_mean_168": {
        "ks_stat": 0.43796760667598456,
        "p_value": 1.7077171663525914e-29,
        "drift": true
      },
      "wx_relative_humidity_2m_roll_std_168": {
        "ks_stat": 0.5273051956949845,
        "p_value": 1.48119773317206e-43,
        "drift": true
      },
      "wx_precipitation_lag_1": {
        "ks_stat": 0.0762945192876131,
        "p_value": 0.2745149548530621,
        "drift": false
      },
      "wx_precipitation_lag_24": {
        "ks_stat": 0.07645895240444289,
        "p_value": 0.2722199327420044,
        "drift": false
      },
      "wx_precipitation_lag_168": {
        "ks_stat": 0.1429912341488433,
        "p_value": 0.002003306845525364,
        "drift": true
      },
      "wx_precipitation_roll_mean_24": {
        "ks_stat": 0.2615426175403155,
        "p_value": 1.6551348123907988e-10,
        "drift": true
      },
      "wx_precipitation_roll_std_24": {
        "ks_stat": 0.2903839904785396,
        "p_value": 6.839227555209531e-13,
        "drift": true
      },
      "wx_precipitation_roll_mean_168": {
        "ks_stat": 0.5866542950321624,
        "p_value": 7.006787756026468e-55,
        "drift": true
      },
      "wx_precipitation_roll_std_168": {
        "ks_stat": 0.7215520920199042,
        "p_value": 2.934993580887165e-88,
        "drift": true
      },
      "wx_cloud_cover_lag_1": {
        "ks_stat": 0.16703810933236235,
        "p_value": 0.00016195422254072306,
        "drift": true
      },
      "wx_cloud_cover_lag_24": {
        "ks_stat": 0.1675314086828516,
        "p_value": 0.0001531718840885286,
        "drift": true
      },
      "wx_cloud_cover_lag_168": {
        "ks_stat": 0.35772571381590534,
        "p_value": 1.6024901110496628e-19,
        "drift": true
      },
      "wx_cloud_cover_roll_mean_24": {
        "ks_stat": 0.2624949593419543,
        "p_value": 1.3943186808481238e-10,
        "drift": true
      },
      "wx_cloud_cover_roll_std_24": {
        "ks_stat": 0.21714420784345967,
        "p_value": 2.371061028272068e-07,
        "drift": true
      },
      "wx_cloud_cover_roll_mean_168": {
        "ks_stat": 0.25165607639092796,
        "p_value": 9.440558551273604e-10,
        "drift": true
      },
      "wx_cloud_cover_roll_std_168": {
        "ks_stat": 0.3552993465741143,
        "p_value": 2.953679038611925e-19,
        "drift": true
      },
      "wx_wind_speed_10m_lag_1": {
        "ks_stat": 0.4226141537762848,
        "p_value": 2.1114721119597616e-27,
        "drift": true
      },
      "wx_wind_speed_10m_lag_24": {
        "ks_stat": 0.404099278451823,
        "p_value": 5.360814773158499e-25,
        "drift": true
      },
      "wx_wind_speed_10m_lag_168": {
        "ks_stat": 0.36576874439655005,
        "p_value": 2.0409223637646272e-20,
        "drift": true
      },
      "wx_wind_speed_10m_roll_mean_24": {
        "ks_stat": 0.45918535135833505,
        "p_value": 1.550877872858e-32,
        "drift": true
      },
      "wx_wind_speed_10m_roll_std_24": {
        "ks_stat": 0.1049714591090074,
        "p_value": 0.04786234572713588,
        "drift": false
      },
      "wx_wind_speed_10m_roll_mean_168": {
        "ks_stat": 0.8941872893200691,
        "p_value": 2.5828184797968027e-162,
        "drift": true
      },
      "wx_wind_speed_10m_roll_std_168": {
        "ks_stat": 0.5089204965880127,
        "p_value": 2.1958738504926284e-40,
        "drift": true
      }
    },
    "drift": true
  },
  "model_drift": {
    "current": {
      "rmse": 271.1705912291987,
      "mape": 0.0034268962116018524
    },
    "baseline_mape": 0.0034268962116018524,
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
