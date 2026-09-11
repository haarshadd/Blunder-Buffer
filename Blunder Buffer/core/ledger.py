"""
Prediction Ledger - core/ledger.py

The single append-only source of truth for every prediction Omni-Pundit ever
makes, across every sport. Nothing in this file imports anything
sport-specific (no chess, no cricket, no football) - it only ever operates
on (event_id, sport, model_version_id, predicted_probs, class_labels,
actual_outcome, timestamps). That's what makes the self-learning loop
reproducible across sports: everything downstream (replay buffer sampling,
retrain orchestrator, regression tests) reads from this table and this
table alone.

Workflow:
  1. Whenever any model makes a prediction (live or backtest), call
     log_prediction() immediately - before the outcome is known.
  2. Once the real-world result comes in, call resolve_prediction() - this
     computes the surprise_score (how wrong/right the model was, as a
     continuous number) and stamps resolved_at.
  3. Everything else (replay buffer, orchestrator, dashboards) queries this
     table via the read helpers at the bottom - never writes to it directly.
"""

import sqlite3
import json
import math
import os
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_ledger(db_path):
    """Creates the predictions_ledger table if it doesn't already exist.
    Safe to call every time before using the ledger - idempotent."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            model_version_id TEXT NOT NULL,
            class_labels TEXT NOT NULL,     -- JSON array, e.g. ["black","draw","white"]
            predicted_probs TEXT NOT NULL,  -- JSON array, same order as class_labels
            actual_outcome TEXT,            -- one of class_labels, NULL until resolved
            surprise_score REAL,            -- -log(P(actual)), NULL until resolved
            predicted_at TEXT NOT NULL,
            resolved_at TEXT,
            UNIQUE(event_id, model_version_id)
        )
    ''')
    # Indexes for the read patterns the replay buffer / orchestrator will
    # actually use: filtering by sport+resolved state, and sorting by
    # surprise for the prioritized replay slice.
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_sport_resolved ON predictions_ledger(sport, resolved_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ledger_surprise ON predictions_ledger(surprise_score)')
    conn.commit()
    conn.close()


def log_prediction(db_path, event_id, sport, model_version_id, class_labels, predicted_probs, predicted_at=None):
    """Logs a prediction BEFORE the outcome is known. Call this at the
    moment any model makes a call, whether live or during a backtest run.

    class_labels and predicted_probs must be same-length, same-order lists,
    e.g. class_labels=["black","draw","white"], predicted_probs=[0.2,0.15,0.65].

    Returns True if logged, False if this (event_id, model_version_id) pair
    was already logged (duplicate calls are safely ignored, not errors -
    a backtest re-run shouldn't crash on already-logged predictions).
    """
    if len(class_labels) != len(predicted_probs):
        raise ValueError("class_labels and predicted_probs must be the same length")

    prob_sum = sum(predicted_probs)
    if not (0.98 <= prob_sum <= 1.02):
        raise ValueError(f"predicted_probs must sum to ~1.0, got {prob_sum:.4f}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO predictions_ledger
                (event_id, sport, model_version_id, class_labels, predicted_probs, predicted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_id, sport, model_version_id, json.dumps(class_labels), json.dumps(predicted_probs),
              predicted_at or _now_iso()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Already logged for this event_id + model_version_id - not an
        # error, just a no-op, so re-running a pipeline step is always safe.
        return False
    finally:
        conn.close()


def resolve_prediction(db_path, event_id, model_version_id, actual_outcome):
    """Call once the real-world result is known. Computes surprise_score =
    -log(P(actual_outcome)) - the standard log-loss-per-prediction, which is
    what makes 'wrong' a continuous signal instead of a binary flag. A
    near-perfect game that loses to one blunder isn't the same kind of
    'wrong' as a confidently-mispredicted upset, and this number reflects
    that distinction directly.

    Returns the computed surprise_score, or None if no matching logged
    prediction was found (e.g. logged under a different model_version_id).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT class_labels, predicted_probs FROM predictions_ledger
        WHERE event_id = ? AND model_version_id = ?
    ''', (event_id, model_version_id))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    class_labels = json.loads(row[0])
    predicted_probs = json.loads(row[1])

    if actual_outcome not in class_labels:
        conn.close()
        raise ValueError(f"actual_outcome '{actual_outcome}' not in logged class_labels {class_labels}")

    idx = class_labels.index(actual_outcome)
    # Clip away from exactly 0 to avoid -log(0) = inf on a rare case where a
    # model assigned literally zero probability to what actually happened.
    p = max(predicted_probs[idx], 1e-9)
    surprise = -math.log(p)

    cursor.execute('''
        UPDATE predictions_ledger
        SET actual_outcome = ?, surprise_score = ?, resolved_at = ?
        WHERE event_id = ? AND model_version_id = ?
    ''', (actual_outcome, surprise, _now_iso(), event_id, model_version_id))
    conn.commit()
    conn.close()
    return surprise


# ============================================================
# Read helpers - the replay buffer's three slices (prioritized / random /
# historical-anchor) and general-purpose queries are built on these.
# ============================================================

def _row_to_dict(row, columns):
    d = dict(zip(columns, row))
    if d.get('class_labels'):
        d['class_labels'] = json.loads(d['class_labels'])
    if d.get('predicted_probs'):
        d['predicted_probs'] = json.loads(d['predicted_probs'])
    return d


_COLUMNS = ['id', 'event_id', 'sport', 'model_version_id', 'class_labels', 'predicted_probs',
            'actual_outcome', 'surprise_score', 'predicted_at', 'resolved_at']


def get_unresolved(db_path, sport=None):
    """Predictions logged but not yet resolved - i.e. events still pending."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if sport:
        cursor.execute('SELECT * FROM predictions_ledger WHERE resolved_at IS NULL AND sport = ?', (sport,))
    else:
        cursor.execute('SELECT * FROM predictions_ledger WHERE resolved_at IS NULL')
    rows = [_row_to_dict(r, _COLUMNS) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_resolved_since(db_path, since_iso, sport=None, model_version_id=None):
    """All resolved predictions with resolved_at >= since_iso. This is the
    base pool the replay buffer's 'random recent slice' samples from."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = 'SELECT * FROM predictions_ledger WHERE resolved_at IS NOT NULL AND resolved_at >= ?'
    params = [since_iso]
    if sport:
        query += ' AND sport = ?'
        params.append(sport)
    if model_version_id:
        query += ' AND model_version_id = ?'
        params.append(model_version_id)
    cursor.execute(query, params)
    rows = [_row_to_dict(r, _COLUMNS) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_top_surprise(db_path, k, since_iso=None, sport=None, model_version_id=None):
    """Top-K most surprising (highest log-loss) resolved predictions - the
    replay buffer's 'prioritized mistakes' slice. Weighting by continuous
    surprise rather than binary right/wrong means a confidently-wrong call
    ranks above a near-miss coin flip, which is the point."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = 'SELECT * FROM predictions_ledger WHERE resolved_at IS NOT NULL'
    params = []
    if since_iso:
        query += ' AND resolved_at >= ?'
        params.append(since_iso)
    if sport:
        query += ' AND sport = ?'
        params.append(sport)
    if model_version_id:
        query += ' AND model_version_id = ?'
        params.append(model_version_id)
    query += ' ORDER BY surprise_score DESC LIMIT ?'
    params.append(k)
    cursor.execute(query, params)
    rows = [_row_to_dict(r, _COLUMNS) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_random_sample(db_path, n, since_iso=None, sport=None, model_version_id=None):
    """Uniform random sample of resolved predictions - right or wrong. This
    is what keeps the replay buffer honest: without it, retraining only on
    prioritized mistakes over-represents upsets/edge cases relative to their
    true frequency, which quietly wrecks calibration.

    Note: `seed` is accepted for API stability but not currently honored -
    sqlite's RANDOM() has no seed hook. If reproducible sampling is needed
    later, fetch all matching rows and sample with Python's `random.Random(seed)`
    instead of relying on `ORDER BY RANDOM()`.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = 'SELECT * FROM predictions_ledger WHERE resolved_at IS NOT NULL'
    params = []
    if since_iso:
        query += ' AND resolved_at >= ?'
        params.append(since_iso)
    if sport:
        query += ' AND sport = ?'
        params.append(sport)
    if model_version_id:
        query += ' AND model_version_id = ?'
        params.append(model_version_id)
    query += ' ORDER BY RANDOM() LIMIT ?'
    params.append(n)
    cursor.execute(query, params)
    rows = [_row_to_dict(r, _COLUMNS) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_historical_anchor_sample(db_path, n, before_iso, sport=None, model_version_id=None):
    """Random sample of resolved predictions from BEFORE a cutoff - the
    replay buffer's 'historical anchor' slice, so a retrain never fully
    forgets the broad patterns learned early on."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = 'SELECT * FROM predictions_ledger WHERE resolved_at IS NOT NULL AND resolved_at < ?'
    params = [before_iso]
    if sport:
        query += ' AND sport = ?'
        params.append(sport)
    if model_version_id:
        query += ' AND model_version_id = ?'
        params.append(model_version_id)
    query += ' ORDER BY RANDOM() LIMIT ?'
    params.append(n)
    cursor.execute(query, params)
    rows = [_row_to_dict(r, _COLUMNS) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_calibration_summary(db_path, sport=None, since_iso=None, n_bins=10):
    """Quick health check: mean surprise score and resolved/unresolved
    counts. Cheap sanity check to run before/after any retrain."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = 'SELECT surprise_score FROM predictions_ledger WHERE resolved_at IS NOT NULL'
    params = []
    if sport:
        query += ' AND sport = ?'
        params.append(sport)
    if since_iso:
        query += ' AND resolved_at >= ?'
        params.append(since_iso)
    cursor.execute(query, params)
    surprises = [r[0] for r in cursor.fetchall()]
    conn.close()

    if not surprises:
        return {'count': 0, 'mean_surprise': None}
    return {
        'count': len(surprises),
        'mean_surprise': sum(surprises) / len(surprises),
        'min_surprise': min(surprises),
        'max_surprise': max(surprises),
    }
