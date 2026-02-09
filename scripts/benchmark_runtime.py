#!/usr/bin/env python3
"""
Runtime benchmark script for paper reproducibility.

This script measures training and inference times for all model types
and optimization components, providing hardware-reproducible benchmarks.
"""

import time
import platform
import json
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import lightgbm as lgb
from scipy.optimize import linprog


def get_hardware_specs():
    """Collect hardware specifications."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "pytorch_device": "MPS" if torch.backends.mps.is_available() else "CPU",
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
    }


def benchmark_gbm_training(n_samples=10000, n_features=50, n_estimators=100):
    """Benchmark GBM training time."""
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    start = time.time()
    model = lgb.LGBMRegressor(n_estimators=n_estimators, verbose=-1)
    model.fit(X, y)
    elapsed = time.time() - start
    
    return {
        "task": "GBM Training",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_estimators": n_estimators,
        "time_seconds": round(elapsed, 3),
    }


def benchmark_gbm_inference(n_iterations=100, horizon=24, n_features=50):
    """Benchmark GBM inference time."""
    X = np.random.randn(1000, n_features)
    y = np.random.randn(1000)
    model = lgb.LGBMRegressor(n_estimators=100, verbose=-1)
    model.fit(X, y)
    
    X_test = np.random.randn(horizon, n_features)
    start = time.time()
    for _ in range(n_iterations):
        model.predict(X_test)
    elapsed = (time.time() - start) / n_iterations * 1000
    
    return {
        "task": "GBM Inference",
        "horizon": horizon,
        "iterations": n_iterations,
        "time_ms": round(elapsed, 3),
    }


def benchmark_lstm_inference(n_iterations=100, seq_len=168, n_features=50):
    """Benchmark LSTM inference time."""
    lstm = nn.LSTM(n_features, 64, num_layers=2, batch_first=True)
    lstm.eval()
    x_tensor = torch.randn(1, seq_len, n_features)
    
    start = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            lstm(x_tensor)
    elapsed = (time.time() - start) / n_iterations * 1000
    
    return {
        "task": "LSTM Inference",
        "sequence_length": seq_len,
        "hidden_size": 64,
        "iterations": n_iterations,
        "time_ms": round(elapsed, 3),
    }


def benchmark_lp_solver(n_iterations=100, horizon=24):
    """Benchmark LP dispatch solver time."""
    n_vars = horizon * 2  # charge + discharge
    c = np.random.randn(n_vars)
    A_ub = np.random.randn(horizon * 4, n_vars)
    b_ub = np.abs(np.random.randn(horizon * 4))
    
    start = time.time()
    for _ in range(n_iterations):
        linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    elapsed = (time.time() - start) / n_iterations * 1000
    
    return {
        "task": "LP Dispatch Solver",
        "horizon": horizon,
        "n_variables": n_vars,
        "iterations": n_iterations,
        "time_ms": round(elapsed, 3),
    }


def main():
    print("=" * 60)
    print("GridPulse Runtime Benchmark")
    print("=" * 60)
    
    # Hardware specs
    print("\n[1/5] Collecting hardware specifications...")
    specs = get_hardware_specs()
    print(f"  Platform: {specs['platform']}")
    print(f"  Processor: {specs['processor']}")
    print(f"  CPU: {specs['cpu_physical_cores']} physical / {specs['cpu_logical_cores']} logical cores")
    print(f"  RAM: {specs['ram_gb']} GB")
    print(f"  PyTorch: {specs['pytorch_device']}")
    
    # Benchmarks
    print("\n[2/5] Benchmarking GBM training...")
    gbm_train = benchmark_gbm_training()
    print(f"  {gbm_train['time_seconds']:.3f}s for {gbm_train['n_samples']} samples")
    
    print("\n[3/5] Benchmarking GBM inference...")
    gbm_infer = benchmark_gbm_inference()
    print(f"  {gbm_infer['time_ms']:.3f}ms per 24h forecast")
    
    print("\n[4/5] Benchmarking LSTM inference...")
    lstm_infer = benchmark_lstm_inference()
    print(f"  {lstm_infer['time_ms']:.3f}ms per 168-step sequence")
    
    print("\n[5/5] Benchmarking LP solver...")
    lp_solve = benchmark_lp_solver()
    print(f"  {lp_solve['time_ms']:.3f}ms per 24h dispatch")
    
    # Save results
    results = {
        "hardware": specs,
        "benchmarks": {
            "gbm_training": gbm_train,
            "gbm_inference": gbm_infer,
            "lstm_inference": lstm_infer,
            "lp_solver": lp_solve,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    out_path = Path("reports/runtime_benchmark.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")
    
    # Print summary for paper
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    print(f"""
Hardware: Apple M-series / {specs['cpu_physical_cores']}-core CPU / {specs['ram_gb']} GB RAM

| Component | Time | Notes |
|-----------|------|-------|
| GBM Training | {gbm_train['time_seconds']:.1f}s | 10k samples, 100 trees |
| GBM Inference | {gbm_infer['time_ms']:.1f}ms | 24h horizon |
| LSTM Inference | {lstm_infer['time_ms']:.1f}ms | 168-step sequence |
| LP Dispatch | {lp_solve['time_ms']:.1f}ms | 24h, 48 variables |
""")


if __name__ == "__main__":
    main()
