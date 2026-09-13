"""Live inference adapter for the existing ECO Expert V3.

ECO V3 is conditional on an observed ECO code. It is NOT a pre-game feature.
This module deliberately keeps the existing trained artifact compatible.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "omni_pundit.db")
MODEL_PATH = os.path.join(ROOT, "saved_models", "eco_expert_lr.pkl")
LABELS = ["black", "draw", "white"]


def _key(name: str) -> str:
    return str(name).replace(" ", "").strip().lower()


def load_eco_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def historical_opening_familiarity(
    white: str,
    black: str,
    eco: str,
    as_of_date: str,
    db_path: str = DB_PATH,
) -> tuple[float, float]:
    """Reproduce ECO V3's player+ECO expanding mean at an as-of date.

    Uses only games strictly before as_of_date. If a player has no history in
    this ECO, V3's original neutral value of 0.5 is returned.
    """
    w = _key(white)
    b = _key(black)
    eco = str(eco or "Unknown").strip() or "Unknown"

    conn = sqlite3.connect(db_path)
    q = """
        SELECT b.date, b.white, b.black, c.opening, b.target_result
        FROM baseline_features b
        JOIN chess_games c ON b.game_id = c.id
        WHERE b.date < ?
          AND c.opening = ?
        ORDER BY b.date ASC
    """
    df = pd.read_sql(q, conn, params=[str(as_of_date), eco])
    conn.close()

    def player_mean(player: str) -> float:
        rows = []
        for r in df.itertuples(index=False):
            wk = _key(r.white)
            bk = _key(r.black)
            if wk == player:
                rows.append(1.0 if r.target_result == 1 else 0.5 if r.target_result == 0 else 0.0)
            elif bk == player:
                rows.append(1.0 if r.target_result == -1 else 0.5 if r.target_result == 0 else 0.0)
        return float(np.mean(rows)) if rows else 0.5

    return player_mean(w), player_mean(b)


def predict_eco(
    white: str,
    black: str,
    eco: str,
    as_of_date: str,
    db_path: str = DB_PATH,
    model_path: str = MODEL_PATH,
) -> Dict[str, float]:
    """Return ECO V3 probabilities conditioned on the observed ECO code."""
    artifact = load_eco_model(model_path)
    model = artifact["model"]
    encoder = artifact["encoder"]
    scaler = artifact["scaler"]

    w_fam, b_fam = historical_opening_familiarity(
        white, black, eco, as_of_date, db_path=db_path
    )
    X_opening = encoder.transform(pd.DataFrame({"opening": [eco]}))
    X_dense = scaler.transform([[w_fam, b_fam, w_fam - b_fam]])

    # scipy sparse + dense; keep exactly the feature ordering used in V3.
    import scipy.sparse as sp
    X = sp.hstack((X_opening, X_dense)).tocsr()
    probs = model.predict_proba(X)[0]
    # sklearn LogisticRegression classes are [-1, 0, 1] => black/draw/white.
    return dict(zip(LABELS, map(float, probs)))
