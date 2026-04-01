"""
LSTM RE Forecaster — MAPIRL-DT Project
Hassan Lawal, Cranfield University

Trains an LSTM to predict EMBEDDED_WIND_GENERATION
for the next 1–6 settlement periods (30–180 min ahead).

Output: lstm_forecaster.pt  (saved model weights)
        lstm_scaler.pkl     (MinMaxScaler for inference)
        lstm_training_log.csv

Usage on HPC:
    python lstm_forecaster.py --data uk_battery_dispatch_complete_data.csv
    python lstm_forecaster.py --data uk_battery_dispatch_complete_data.csv --epochs 50 --seq_len 12
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LSTM Wind Generation Forecaster")
parser.add_argument("--data",      type=str,   default="uk_battery_dispatch_complete_data.csv")
parser.add_argument("--seq_len",   type=int,   default=12,   help="Input sequence length (x30 min). Default=12 → 6hrs lookback")
parser.add_argument("--horizon",   type=int,   default=6,    help="Forecast horizon (x30 min). Default=6 → 3hrs ahead")
parser.add_argument("--epochs",    type=int,   default=30)
parser.add_argument("--batch",     type=int,   default=64)
parser.add_argument("--hidden",    type=int,   default=128)
parser.add_argument("--layers",    type=int,   default=2)
parser.add_argument("--lr",        type=float, default=1e-3)
parser.add_argument("--dropout",   type=float, default=0.2)
parser.add_argument("--outdir",    type=str,   default=".")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Config: seq_len={args.seq_len}, horizon={args.horizon}, "
      f"hidden={args.hidden}, layers={args.layers}, epochs={args.epochs}")

# ── Feature columns used as LSTM inputs ───────────────────────────────────────
# Wind + solar (correlated RE sources) + temporal context
FEATURE_COLS = [
    "EMBEDDED_WIND_GENERATION",
    "EMBEDDED_SOLAR_GENERATION",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak",
    "demand_change",
]
TARGET_COL = "EMBEDDED_WIND_GENERATION"

# ── Load & prepare data ────────────────────────────────────────────────────────
print(f"\nLoading {args.data} ...")
df = pd.read_csv(args.data, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"Loaded {len(df):,} rows | {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

# Drop any nulls in feature cols
df = df.dropna(subset=FEATURE_COLS)
print(f"After dropna: {len(df):,} rows")

features = df[FEATURE_COLS].values.astype(np.float32)
target_idx = FEATURE_COLS.index(TARGET_COL)

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler immediately — needed for SAC agent inference
scaler_path = os.path.join(args.outdir, "lstm_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"scaler": scaler, "feature_cols": FEATURE_COLS,
                 "target_col": TARGET_COL, "target_idx": target_idx}, f)
print(f"Scaler saved → {scaler_path}")

# ── Build sequences ────────────────────────────────────────────────────────────
def build_sequences(data, seq_len, horizon, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])                            # (seq_len, n_features)
        y.append(data[i + seq_len : i + seq_len + horizon, target_idx])  # (horizon,)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

print("Building sequences ...")
X, y = build_sequences(features_scaled, args.seq_len, args.horizon, target_idx)
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# ── Train / val / test split (70/15/15) ───────────────────────────────────────
n = len(X)
n_train = int(n * 0.70)
n_val   = int(n * 0.85)

X_train, y_train = X[:n_train],  y[:n_train]
X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
X_test,  y_test  = X[n_val:],    y[n_val:]

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ── Dataset & DataLoader ───────────────────────────────────────────────────────
class REDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(REDataset(X_train, y_train), batch_size=args.batch, shuffle=True,  num_workers=0)
val_loader   = DataLoader(REDataset(X_val,   y_val),   batch_size=args.batch, shuffle=False, num_workers=0)
test_loader  = DataLoader(REDataset(X_test,  y_test),  batch_size=args.batch, shuffle=False, num_workers=0)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class REForecaster(nn.Module):
    """
    Sequence-to-vector LSTM.
    Input:  (batch, seq_len, n_features)
    Output: (batch, horizon)  — normalised wind generation predictions
    """
    def __init__(self, n_features, hidden, layers, horizon, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1]) # last timestep
        return self.fc(out)            # (batch, horizon)

model = REForecaster(
    n_features=len(FEATURE_COLS),
    hidden=args.hidden,
    layers=args.layers,
    horizon=args.horizon,
    dropout=args.dropout,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ── Training ───────────────────────────────────────────────────────────────────
criterion = nn.HuberLoss()          # robust to RE spikes
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

best_val_loss = float("inf")
log_rows = []
model_path = os.path.join(args.outdir, "lstm_forecaster.pt")

print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>10} {'Time(s)':>8}")
print("-" * 45)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    # — train —
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    # — validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(model(xb), yb).item() * len(xb)
    val_loss /= len(X_val)

    scheduler.step(val_loss)
    elapsed = time.time() - t0

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": {
                "n_features": len(FEATURE_COLS),
                "hidden": args.hidden,
                "layers": args.layers,
                "horizon": args.horizon,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
            }
        }, model_path)
        tag = " ✓"
    else:
        tag = ""

    print(f"{epoch:>5} {train_loss:>12.6f} {val_loss:>10.6f} {elapsed:>8.1f}{tag}")
    log_rows.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "time_s": elapsed})

# ── Test evaluation ────────────────────────────────────────────────────────────
print(f"\nBest val loss: {best_val_loss:.6f}")
print("Evaluating on test set ...")

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
        actuals.append(yb.numpy())

preds   = np.concatenate(preds,   axis=0)   # (N, horizon)
actuals = np.concatenate(actuals, axis=0)   # (N, horizon)

# Inverse transform wind column only
def inv_wind(arr_norm):
    """Inverse scale normalised wind predictions back to MW."""
    dummy = np.zeros((len(arr_norm), len(FEATURE_COLS)), dtype=np.float32)
    dummy[:, target_idx] = arr_norm
    return scaler.inverse_transform(dummy)[:, target_idx]

step_labels = [f"t+{i*30}min" for i in range(1, args.horizon + 1)]
print(f"\n{'Step':<12} {'MAE (MW)':>10} {'RMSE (MW)':>12} {'MAPE (%)':>10}")
print("-" * 48)
for h in range(args.horizon):
    p_mw = inv_wind(preds[:, h])
    a_mw = inv_wind(actuals[:, h])
    mae  = mean_absolute_error(a_mw, p_mw)
    rmse = np.sqrt(mean_squared_error(a_mw, p_mw))
    mape = np.mean(np.abs((a_mw - p_mw) / (a_mw + 1e-6))) * 100
    print(f"{step_labels[h]:<12} {mae:>10.1f} {rmse:>12.1f} {mape:>10.2f}")

# ── Save training log ──────────────────────────────────────────────────────────
log_path = os.path.join(args.outdir, "lstm_training_log.csv")
pd.DataFrame(log_rows).to_csv(log_path, index=False)
print(f"\nTraining log saved → {log_path}")
print(f"Model saved        → {model_path}")
print(f"Scaler saved       → {scaler_path}")

# ── Inference helper (copy into your SAC env) ──────────────────────────────────
INFERENCE_SNIPPET = '''
# ── Paste into your SAC environment to get RE_predicted ──────────────────────
import torch, pickle, numpy as np
from lstm_forecaster import REForecaster   # or copy the class inline

def load_lstm(model_path="lstm_forecaster.pt", scaler_path="lstm_scaler.pkl"):
    ckpt   = torch.load(model_path, map_location="cpu")
    cfg    = ckpt["config"]
    model  = REForecaster(cfg["n_features"], cfg["hidden"],
                          cfg["layers"], cfg["horizon"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["scaler"], cfg

def predict_re(model, scaler, recent_window, cfg):
    """
    recent_window: np.array of shape (seq_len, n_features) — raw (unscaled) values
    Returns: np.array of shape (horizon,) — predicted wind in MW
    """
    scaled = scaler.transform(recent_window)
    x = torch.from_numpy(scaled[np.newaxis].astype(np.float32))  # (1, seq_len, n_features)
    with torch.no_grad():
        pred_norm = model(x).numpy()[0]                           # (horizon,)
    # inverse scale wind column only
    dummy = np.zeros((len(pred_norm), len(cfg["feature_cols"])), dtype=np.float32)
    dummy[:, cfg["feature_cols"].index(cfg["target_col"])] = pred_norm
    return scaler.inverse_transform(dummy)[:, cfg["feature_cols"].index(cfg["target_col"])]

# In your SAC env step():
#   re_predicted = predict_re(model, scaler, obs_window, cfg)[0]  # next 30 min
#   state = np.array([v_cell, dT, util, re_now_norm, re_predicted_norm])
'''
print("\n" + "="*60)
print("INFERENCE SNIPPET (copy into SAC env):")
print("="*60)
print(INFERENCE_SNIPPET)
