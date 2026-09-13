"""Pregame-only meta model.

Consumes Baseline + Momentum OOF probabilities and context. ECO is deliberately
absent because the opening is not known at T0.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "omni_pundit.db")
MODEL_PATH = os.path.join(ROOT, "saved_models", "meta_learner_pregame_wide_deep.pkl")
LABELS = ["black", "draw", "white"]


class WideAndDeepContext(nn.Module):
    def __init__(self, deep_in: int, wide_in: int):
        super().__init__()
        self.deep_path = nn.Sequential(nn.Linear(deep_in, 24), nn.ReLU(), nn.Dropout(0.30))
        self.combine = nn.Sequential(
            nn.Linear(24 + wide_in, 16), nn.ReLU(), nn.Dropout(0.20), nn.Linear(16, 3)
        )

    def forward(self, wide_x, deep_x):
        return self.combine(torch.cat((self.deep_path(deep_x), wide_x), dim=1))


def load(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(wide_values: list[float], deep_probs: list[float], path: str = MODEL_PATH):
    artifact = load(path)
    model = WideAndDeepContext(6 if len(artifact["deep_cols"]) == 6 else len(artifact["deep_cols"]), len(artifact["wide_cols"]))
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    scaler = artifact["wide_scaler"]
    wide = scaler.transform([wide_values])
    deep = np.asarray([deep_probs], dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(wide, dtype=torch.float32), torch.tensor(deep, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return dict(zip(LABELS, map(float, probs)))
