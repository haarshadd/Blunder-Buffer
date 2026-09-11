"""
backfill_ledger.py

Populates the Prediction Ledger from your EXISTING OOF prediction tables.
Uses BULK INSERTS to process 500,000+ rows in seconds.
"""

import sqlite3
import pandas as pd
from core import ledger
import os
import math
import json

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "omni_pundit.db"))

CLASS_LABELS = ['black', 'draw', 'white']
RESULT_TO_LABEL = {-1: 'black', 0: 'draw', 1: 'white'}

EXPERTS = [
    ('oof_predictions_baseline', 'baseline_v2', 'base_oof_prob'),
    ('oof_predictions_eco', 'eco_v3', 'eco_oof_prob'),
    ('oof_predictions_momentum', 'momentum_v3', 'momentum_oof_prob'),
]

def table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def backfill_expert(conn, table_name, model_version_id, prefix):
    if not table_exists(conn, table_name):
        print(f"  Skipping {model_version_id}: table '{table_name}' not found yet.")
        return 0, 0

    query = f"""
        SELECT o.game_id, o.{prefix}_black, o.{prefix}_draw, o.{prefix}_white,
               b.date, b.target_result
        FROM {table_name} o
        JOIN baseline_features b ON o.game_id = b.game_id
    """
    df = pd.read_sql(query, conn)
    print(f"  {model_version_id}: Formatting {len(df)} rows for bulk insert...")

    insert_data = []
    
    for row in df.itertuples():
        probs = [getattr(row, f'{prefix}_black'), getattr(row, f'{prefix}_draw'), getattr(row, f'{prefix}_white')]
        actual_label = RESULT_TO_LABEL[row.target_result]
        
        idx = CLASS_LABELS.index(actual_label)
        p = max(probs[idx], 1e-9)
        surprise = -math.log(p)
        
        insert_data.append((
            row.game_id, 
            'chess', 
            model_version_id, 
            json.dumps(CLASS_LABELS), 
            json.dumps(probs), 
            actual_label, 
            surprise, 
            str(row.date), 
            str(row.date)  # BUG 1 FIX: Resolved_at matches historical reality
        ))

    print(f"  {model_version_id}: Committing to database...")
    cursor = conn.cursor()
    
    # BUG 2 FIX: Track actual database changes
    initial_changes = conn.total_changes
    
    cursor.executemany('''
        INSERT OR IGNORE INTO predictions_ledger
            (event_id, sport, model_version_id, class_labels, predicted_probs, actual_outcome, surprise_score, predicted_at, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', insert_data)
    conn.commit()
    
    actual_inserts = conn.total_changes - initial_changes
    return actual_inserts, actual_inserts

def backfill_all():
    print("--- Bulk Backfilling Prediction Ledger from existing OOF tables ---")
    ledger.init_ledger(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    total_logged = 0
    total_resolved = 0

    for table_name, model_version_id, prefix in EXPERTS:
        logged, resolved = backfill_expert(conn, table_name, model_version_id, prefix)
        total_logged += logged
        total_resolved += resolved

    conn.close()

    print(f"\nBackfill complete: {total_logged} predictions actually written to disk.")
    summary = ledger.get_calibration_summary(DB_PATH, sport='chess')
    print(f"Ledger now contains {summary['count']} resolved chess predictions "
          f"(mean surprise: {summary['mean_surprise']:.4f})" if summary['count'] else "Ledger is empty.")

if __name__ == "__main__":
    backfill_all()