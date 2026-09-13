"""Two-stage prediction API.

T0: pregame prediction before the game starts.
T1: opening-conditioned update once an ECO code is actually observed.

The two predictions are intentionally separate ledger events/models so a late
ECO observation can never overwrite or contaminate the T0 forecast.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import ledger
from live.eco import predict_eco
from live.pregame import predict as predict_pregame
from live.opening_update import predict_opening_meta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "omni_pundit.db")


def _key(s):
    return str(s).replace(" ", "").strip().lower()


def _latest_player_elo(conn, player, before_date):
    k = _key(player)
    q = """
    SELECT white, black, white_elo, black_elo, date
    FROM baseline_features
    WHERE date < ? AND (lower(replace(white,' ','')) = ? OR lower(replace(black,' ','')) = ?)
    ORDER BY date DESC LIMIT 1
    """
    r = conn.execute(q, [before_date, k, k]).fetchone()
    if not r:
        return np.nan
    white, black, we, be, _ = r
    return float(we if _key(white) == k else be) if (we if _key(white) == k else be) is not None else np.nan


def _historical_h2h(conn, white, black, before_date):
    wk, bk = _key(white), _key(black)
    q = """
    SELECT white, black, target_result FROM baseline_features
    WHERE date < ?
      AND ((lower(replace(white,' ',''))=? AND lower(replace(black,' ',''))=?)
        OR (lower(replace(white,' ',''))=? AND lower(replace(black,' ',''))=?))
    ORDER BY date ASC
    """
    rows = conn.execute(q, [before_date, wk, bk, bk, wk]).fetchall()
    w = b = d = 0
    for rw, rb, result in rows:
        if _key(rw) == wk:
            w += int(result == 1); b += int(result == -1)
        else:
            w += int(result == -1); b += int(result == 1)
        d += int(result == 0)
    return w, b, d


def build_pregame_features(white, black, as_of_date, db_path=DB_PATH):
    """Build a conservative T0 context vector.

    For a live tournament, event-specific rest/fatigue should come from the
    tournament scheduler once available. Historical fallbacks are neutral.
    """
    conn = sqlite3.connect(db_path)
    we = _latest_player_elo(conn, white, as_of_date)
    be = _latest_player_elo(conn, black, as_of_date)
    wh, bh, draws = _historical_h2h(conn, white, black, as_of_date)
    conn.close()

    has_elo = int(np.isfinite(we) and np.isfinite(be))
    if not has_elo:
        we = be = 0.0
    elo_diff = we - be

    # T0 context defaults: no game-time result information is allowed.
    # The production tournament adapter should supply rest/fatigue/tilt once
    # the event schedule and roster history are known.
    return {
        "white_elo": float(we), "black_elo": float(be), "elo_diff": float(elo_diff),
        "has_elo": has_elo,
        "white_historical_wins": wh, "black_historical_wins": bh,
        "historical_draws": draws,
        "rest_diff": 0.0, "fatigue_diff": 0.0, "tilt_diff": 0.0,
        "black_draw_rate_10": 0.0,
    }


def stage0(white, black, as_of_date, model_version_id="pregame_meta_wd_v1", db_path=DB_PATH):
    """Run T0 and return the immutable pregame forecast."""
    from live.features_live import baseline_and_momentum
    f = build_pregame_features(white, black, as_of_date, db_path)
    base, mom, extra = baseline_and_momentum(white, black, as_of_date, db_path)
    wide = [f["elo_diff"], f["rest_diff"], f["fatigue_diff"], f["tilt_diff"], f["black_draw_rate_10"]]
    # The pregame model uses 5 wide context features, matching the original WD architecture.
    deep = [base[k] for k in ("black", "draw", "white")] + [mom[k] for k in ("black", "draw", "white")]
    probs = predict_pregame(wide, deep)
    event_id = f"{as_of_date[:10]}-{_key(white)}-{_key(black)}"
    ledger.init_ledger(db_path)
    ledger.log_prediction(db_path, event_id, "chess", model_version_id, ["black","draw","white"], [probs[k] for k in ("black","draw","white")])
    return {"event_id": event_id, "stage": "T0", "white": white, "black": black, "probabilities": probs, "base": base, "momentum": mom}


def stage1(white, black, eco, as_of_date, event_id, model_version_id="eco_update_v3", db_path=DB_PATH):
    """Run the ECO expert after ECO is observed; never replaces T0."""
    probs = predict_eco(white, black, eco, as_of_date, db_path=db_path)
    update_id = f"{event_id}:eco:{eco}"
    ledger.log_prediction(db_path, update_id, "chess", model_version_id, ["black","draw","white"], [probs[k] for k in ("black","draw","white")])
    return {"event_id": update_id, "stage": "T1-ECO", "eco": eco, "probabilities": probs}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--white", required=True)
    p.add_argument("--black", required=True)
    p.add_argument("--date", required=True, help="ISO timestamp/date used as information cutoff")
    p.add_argument("--eco")
    args = p.parse_args()
    result = stage0(args.white, args.black, args.date)
    print(json.dumps(result, indent=2))
    if args.eco:
        print(json.dumps(stage1(args.white, args.black, args.eco, args.date, result["event_id"]), indent=2))
