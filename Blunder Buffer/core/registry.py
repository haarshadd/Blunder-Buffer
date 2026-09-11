"""
Model Registry - core/registry.py

An immutable audit trail for every model trained in the Omni-Pundit ecosystem.
Tracks the model version, the sport it applies to, the architectural layer 
(e.g., 'base', 'meta'), and whether it is the currently promoted 'champion'.

Workflow:
  1. Train a new model (the 'challenger') and call register_model() to save its metrics.
  2. Orchestrator tests the challenger against the current champion.
  3. If it passes the non-regression tests, call promote_model() to flip its 
     is_champion flag to True, demoting the previous champion.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def init_registry(db_path):
    """Creates the model_registry table if it doesn't already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_registry (
            model_version_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL,
            layer TEXT NOT NULL,            -- e.g., 'meta', 'expert_eco', 'expert_momentum'
            training_window TEXT NOT NULL,  -- JSON: {"start": "ISO", "end": "ISO"}
            metrics_json TEXT NOT NULL,     -- JSON: {"walk_forward_acc": 0.61, "log_loss": 0.87}
            registered_at TEXT NOT NULL,
            promoted_at TEXT,               -- NULL until promoted
            is_champion INTEGER DEFAULT 0   -- 1 if live, 0 otherwise
        )
    ''')
    # Indexes for fast querying of champions per sport and layer
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_registry_champion ON model_registry(sport, layer, is_champion)')
    conn.commit()
    conn.close()

def register_model(db_path, model_version_id, sport, layer, training_window, metrics_json):
    """Logs a newly trained challenger model into the registry."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO model_registry
                (model_version_id, sport, layer, training_window, metrics_json, registered_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_version_id, sport, layer, json.dumps(training_window), json.dumps(metrics_json), _now_iso()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Avoids duplicate registrations if the script is re-run
        return False
    finally:
        conn.close()

def get_champion(db_path, sport, layer):
    """Retrieves the model_version_id of the currently active champion."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT model_version_id, metrics_json, promoted_at 
        FROM model_registry 
        WHERE sport = ? AND layer = ? AND is_champion = 1
    ''', (sport, layer))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    return {'model_version_id': row[0], 'metrics': json.loads(row[1]), 'promoted_at': row[2]}

def promote_model(db_path, model_version_id, sport, layer):
    """Promotes a challenger to champion, demoting the previous champion safely."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Safety Check: Verify model actually belongs to this sport/layer lineage
    cursor.execute('SELECT sport, layer FROM model_registry WHERE model_version_id = ?', (model_version_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Model {model_version_id} not found in registry.")
    if row[0] != sport or row[1] != layer:
        conn.close()
        raise ValueError(f"Model {model_version_id} belongs to {row[0]}/{row[1]}, not {sport}/{layer}.")
    

    # 1. Demote the current champion for this sport and layer
    cursor.execute('''
        UPDATE model_registry 
        SET is_champion = 0 
        WHERE sport = ? AND layer = ? AND is_champion = 1
    ''', (sport, layer))
    
    # 2. Promote the new model
    cursor.execute('''
        UPDATE model_registry 
        SET is_champion = 1, promoted_at = ? 
        WHERE model_version_id = ?
    ''', (_now_iso(), model_version_id))
    
    conn.commit()
    conn.close()

def get_model_history(db_path, sport, layer):
    """Returns the full chronological audit trail of all models for a specific layer."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT model_version_id, metrics_json, registered_at, promoted_at, is_champion 
        FROM model_registry 
        WHERE sport = ? AND layer = ? 
        ORDER BY registered_at DESC
    ''', (sport, layer))
    
    cols = ['model_version_id', 'metrics_json', 'registered_at', 'promoted_at', 'is_champion']
    rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
    for row in rows:
        row['metrics_json'] = json.loads(row['metrics_json'])
    conn.close()
    return rows