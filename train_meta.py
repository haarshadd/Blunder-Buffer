"""
Production Training Pipeline - train_meta.py

Connects the PyTorch Wide & Deep architecture to the Experience Replay Buffer 
and triggers the Champion/Challenger Orchestrator.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from core.replay_buffer import build_training_batch
from core.orchestrator import evaluate_and_promote

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "omni_pundit.db"))

def fetch_features_for_batch(batch_events):
    """
    Takes the event_ids from the Replay Buffer and fetches the heavy feature 
    vectors from the SQLite database, including the psychological context.
    """
    event_ids = [row['event_id'] for row in batch_events]
    placeholders = ','.join(['?'] * len(event_ids))
    
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            b.game_id, b.target_result, b.elo_diff, b.has_elo,
            b.w_rest, b.b_rest, b.w_fatigue, b.b_fatigue, b.w_tilt, b.b_tilt, b.black_draw_rate_10,
            base.base_oof_prob_black, base.base_oof_prob_draw, base.base_oof_prob_white,
            eco.eco_oof_prob_black, eco.eco_oof_prob_draw, eco.eco_oof_prob_white,
            mom.momentum_oof_prob_black, mom.momentum_oof_prob_draw, mom.momentum_oof_prob_white
        FROM baseline_features b
        JOIN oof_predictions_baseline base ON b.game_id = base.game_id
        JOIN oof_predictions_eco eco ON b.game_id = eco.game_id
        JOIN oof_predictions_momentum mom ON b.game_id = mom.game_id
        WHERE b.game_id IN ({placeholders})
    """
    
    df_features = pd.read_sql(query, conn, params=event_ids)
    conn.close()
    
    df_features['target_result'] = df_features['target_result'].map({-1: 0, 0: 1, 1: 2})
    
    # Compute the differentials dynamically for the Wide Layer
    df_features['rest_diff'] = df_features['w_rest'] - df_features['b_rest']
    df_features['fatigue_diff'] = df_features['w_fatigue'] - df_features['b_fatigue']
    df_features['tilt_diff'] = df_features['w_tilt'] - df_features['b_tilt']
    
    df_features.dropna(inplace=True)
    return df_features

def run_retrain_loop():
    print("--- Omni-Pundit Self-Learning Loop Initiated ---\n")
    
    # 1. Pull the Training Batch from the Replay Buffer
    print("1. Sampling Experience Replay Buffer...")
    train_batch = build_training_batch(
        db_path=DB_PATH,
        sport='chess',
        model_version_id='baseline_v2', # Bootstrapping from baseline errors
        total_size=15000,
        since_iso='2025-01-01',          
        historical_cutoff_iso='2018-01-01'
    )
    
    df_train = fetch_features_for_batch(train_batch)
    print(f"   Constructed feature matrix for {len(df_train)} games.\n")

    # 2. Pull a Fresh Holdout Batch (Late 2025 until live inference is built)
    print("2. Fetching fresh holdout window for Orchestrator evaluation...")
    conn = sqlite3.connect(DB_PATH)
    holdout_query = """
        SELECT game_id FROM baseline_features 
        WHERE date >= '2025-06-01' AND date < '2026-01-01' 
        ORDER BY RANDOM() LIMIT 5000
    """
    holdout_ids = [{'event_id': row[0]} for row in conn.cursor().execute(holdout_query).fetchall()]
    conn.close()
    
    df_holdout = fetch_features_for_batch(holdout_ids)

    # 3. Define the Arrays with the NEW Psychological Context
    wide_cols = ['elo_diff', 'has_elo', 'rest_diff', 'fatigue_diff', 'tilt_diff', 'black_draw_rate_10']
    deep_cols = [c for c in df_train.columns if 'oof_prob' in c]
    
    X_wide_train = df_train[wide_cols].values
    X_deep_train = df_train[deep_cols].values
    y_train = df_train['target_result'].values

    X_wide_holdout = df_holdout[wide_cols].values
    X_deep_holdout = df_holdout[deep_cols].values
    y_holdout = df_holdout['target_result'].values

    # 4. Scale and Convert to PyTorch Tensors
    scaler = StandardScaler()
    X_wide_train_scaled = scaler.fit_transform(X_wide_train)
    X_wide_holdout_scaled = scaler.transform(X_wide_holdout)

    t_wide_train = torch.FloatTensor(X_wide_train_scaled)
    t_deep_train = torch.FloatTensor(X_deep_train)
    t_y_train = torch.LongTensor(y_train)

    t_wide_holdout = torch.FloatTensor(X_wide_holdout_scaled)
    t_deep_holdout = torch.FloatTensor(X_deep_holdout)

    train_ds = TensorDataset(t_wide_train, t_deep_train, t_y_train)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # 5. Define Neural Network
    class WideAndDeepContext(nn.Module):
        def __init__(self, deep_in, wide_in):
            super().__init__()
            self.deep_path = nn.Sequential(
                nn.Linear(deep_in, 24),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.combine = nn.Sequential(
                nn.Linear(24 + wide_in, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 3)
            )

        def forward(self, wide_x, deep_x):
            deep_out = self.deep_path(deep_x)
            combined = torch.cat((deep_out, wide_x), dim=1)
            return self.combine(combined)

    print("3. Training PyTorch Challenger Model...")
    model = WideAndDeepContext(deep_in=len(deep_cols), wide_in=len(wide_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(150):
        for wide_batch, deep_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(wide_batch, deep_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 6. Evaluate Challenger
    model.eval()
    with torch.no_grad():
        val_outputs = model(t_wide_holdout, t_deep_holdout)
        val_probs = torch.softmax(val_outputs, dim=1).numpy()
        chal_log_loss = log_loss(y_holdout, val_probs, labels=[0, 1, 2])

    print(f"   Challenger Holdout Log Loss: {chal_log_loss:.4f}\n")

    # 7. Orchestrator Hand-off
    print("4. Handing over to Orchestrator...")
    
    evaluate_and_promote(
        db_path=DB_PATH,
        sport='chess',
        layer='meta',
        challenger_id='meta_wide_deep_v4',
        training_window={"start": "buffer_mix", "end": "2026-07-01"},
        challenger_metrics={'fresh_holdout_log_loss': chal_log_loss},
        champion_metrics={'fresh_holdout_log_loss': 0.8500}, # Benchmark to beat
        min_improvement=0.002
    )

if __name__ == "__main__":
    run_retrain_loop()