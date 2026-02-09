#!/usr/bin/env python3
"""Test all module imports."""
import importlib
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# All gridpulse modules to test
modules = [
    # Core forecasting
    'gridpulse.forecasting.ml_gbm',
    'gridpulse.forecasting.dl_lstm',
    'gridpulse.forecasting.dl_tcn',
    'gridpulse.forecasting.predict',
    'gridpulse.forecasting.train',
    'gridpulse.forecasting.evaluate',
    'gridpulse.forecasting.baselines',
    'gridpulse.forecasting.backtest',
    'gridpulse.forecasting.datasets',
    
    # Uncertainty
    'gridpulse.forecasting.uncertainty.conformal',
    
    # Data pipeline
    'gridpulse.data_pipeline.build_features',
    'gridpulse.data_pipeline.build_features_eia930',
    'gridpulse.data_pipeline.download_opsd',
    'gridpulse.data_pipeline.download_weather',
    'gridpulse.data_pipeline.split_time_series',
    'gridpulse.data_pipeline.storage',
    
    # Streaming
    'gridpulse.streaming.consumer',
    'gridpulse.streaming.producer',
    
    # Optimizer
    'gridpulse.optimizer.lp_dispatch',
    'gridpulse.optimizer.risk',
    'gridpulse.optimizer.robust_dispatch',
    
    # Evaluation (new novelty modules)
    'gridpulse.evaluation.stats',
    'gridpulse.evaluation.regret',
    
    # Monitoring
    'gridpulse.monitoring.data_drift',
    'gridpulse.monitoring.model_drift',
    'gridpulse.monitoring.alerts',
    
    # Anomaly
    'gridpulse.anomaly.isolation_forest',
    
    # Utils
    'gridpulse.utils.manifest',
]

passed = 0
failed = 0
errors = []

for mod in modules:
    try:
        importlib.import_module(mod)
        print(f'✓ {mod}')
        passed += 1
    except Exception as e:
        print(f'✗ {mod}: {e}')
        failed += 1
        errors.append((mod, str(e)))

print(f'\n{passed}/{passed+failed} modules imported successfully')
if failed:
    print(f'❌ {failed} modules failed')
    sys.exit(1)
else:
    print('✅ All imports OK')
    sys.exit(0)
