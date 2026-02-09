#!/usr/bin/env python3
"""Quick evaluation of trained models."""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Load test data
df_test = pd.read_parquet('data/processed/splits/test.parquet')

targets = ['load_mw', 'wind_mw', 'solar_mw']
features = [c for c in df_test.columns if c not in targets + ['timestamp']]
X = df_test[features].values

print('=' * 60)
print('FRESH EVALUATION - NEW MODELS (Feb 9 2026)')
print('=' * 60)

results = []
for target in targets:
    y_true = df_test[target].values
    
    # GBM
    gbm_path = Path(f'artifacts/models/gbm_lightgbm_{target}.pkl')
    if gbm_path.exists():
        loaded = joblib.load(gbm_path)
        # Handle both dict and direct model formats
        if isinstance(loaded, dict):
            gbm = loaded.get('model', loaded)
        else:
            gbm = loaded
        
        # Get expected number of features
        n_features = gbm.n_features_
        
        # Select that many features (excluding targets and timestamp)
        available = [c for c in df_test.columns if c not in targets + ['timestamp']]
        X_model = df_test[available[:n_features]].values
        
        y_pred = gbm.predict(X_model)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        results.append({'target': target, 'model': 'GBM', 'rmse': rmse, 'mae': mae})
        print(f'{target:12} GBM:  RMSE={rmse:>10.2f}  MAE={mae:>10.2f}')

print('=' * 60)
print('\nModel files updated today (Feb 9):')
import os
for f in Path('artifacts/models').glob('*.pkl'):
    stat = f.stat()
    print(f'  {f.name}: {stat.st_size/1e6:.1f} MB')
for f in Path('artifacts/models').glob('*.pt'):
    stat = f.stat()
    print(f'  {f.name}: {stat.st_size/1e6:.1f} MB')
