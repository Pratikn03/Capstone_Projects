from __future__ import annotations

import argparse
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.datasets import TimeSeriesWindowDataset, SeqConfig

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train LSTM Forecaster")
    parser.add_argument("--config", default="configs/train_dl.yaml")
    parser.add_argument("--train-path", default="data/processed/splits/train.parquet")
    parser.add_argument("--val-path", default="data/processed/splits/val.parquet")
    parser.add_argument("--out-dir", default="artifacts/models")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    target_col = cfg.get("target", "load_mw")
    feature_cols = cfg.get("features", [])
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    lookback = train_cfg.get("lookback", 168)
    horizon = train_cfg.get("horizon", 24)
    batch_size = train_cfg.get("batch_size", 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training LSTM on {device}. Target: {target_col}")

    # 2. Load Data
    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)

    # 3. Leakage-Free Scaling
    # Fit scaler ONLY on training data
    print("Scaling data (fit on train only)...")
    scaler = StandardScaler()
    
    # We scale all input features. 
    # Note: If target is in features, it gets scaled too.
    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # We also need to know which column index is the target to extract y
    try:
        target_idx = feature_cols.index(target_col)
    except ValueError:
        raise ValueError(f"Target {target_col} must be in features list for this implementation.")

    y_train_scaled = X_train_scaled[:, target_idx]
    y_val_scaled = X_val_scaled[:, target_idx]

    # 4. Create Datasets
    seq_cfg = SeqConfig(lookback=lookback, horizon=horizon)
    
    train_ds = TimeSeriesWindowDataset(X_train_scaled, y_train_scaled, seq_cfg)
    val_ds = TimeSeriesWindowDataset(X_val_scaled, y_val_scaled, seq_cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    model = LSTMForecaster(
        n_features=len(feature_cols),
        hidden_size=model_cfg.get("hidden_size", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        horizon=horizon
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 1e-3))
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_val_loss = float("inf")
    patience = train_cfg.get("patience", 5)
    patience_counter = 0
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "lstm_model.pt"

    print("Starting training...")
    for epoch in range(train_cfg.get("epochs", 20)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 7. Save Artifacts
    scaler_path = out_dir / "lstm_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    
    # Save metadata
    meta = {"features": feature_cols, "target": target_col, "target_idx": target_idx, "config": cfg}
    with open(out_dir / "lstm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()