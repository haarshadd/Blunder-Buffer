import sqlite3

from core.ledger import init_ledger
from live.events import ChessGameEvent
from workers.realtime_chess import RealtimeChessWorker
from live.config import TournamentConfig


class FakeAdapter:
    def __init__(self):
        self.poll_count = 0

    def fetch_events(self, round_number):
        self.poll_count += 1

        result = None if self.poll_count == 1 else "white"

        return [
            ChessGameEvent(
                tournament_id="recovery-test",
                section="open",
                round_number=round_number,
                board=1,
                white="Carlsen, Magnus",
                black="Caruana, Fabiano",
                result=result,
                source="test",
            )
        ]

    def close(self):
        pass


def test_worker_resolves_finished_event_after_state_restart(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")

    init_ledger(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE baseline_features (
            game_id INTEGER,
            date TEXT,
            white TEXT,
            black TEXT,
            white_elo REAL,
            black_elo REAL,
            elo_diff REAL,
            has_elo INTEGER,
            white_historical_wins INTEGER,
            black_historical_wins INTEGER,
            historical_draws INTEGER,
            target_result INTEGER,
            w_rest REAL,
            b_rest REAL,
            w_fatigue REAL,
            b_fatigue REAL,
            w_tilt REAL,
            b_tilt REAL,
            black_draw_rate_10 REAL
        )
    """)
    conn.executemany(
        """
        INSERT INTO baseline_features (
            game_id, date, white, black,
            white_elo, black_elo, elo_diff, has_elo,
            white_historical_wins, black_historical_wins,
            historical_draws, target_result,
            w_rest, b_rest, w_fatigue, b_fatigue,
            w_tilt, b_tilt, black_draw_rate_10
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "2026-09-10", "Carlsen, Magnus", "Other, Player",
             2840, 2500, 340, 1, 50, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0),
            (2, "2026-09-10", "Other, Player", "Caruana, Fabiano",
             2500, 2790, -290, 1, 10, 20, 5, -1, 0, 0, 0, 0, 0, 0, 0),
        ],
    )
    conn.commit()
    conn.close()

    import workers.realtime_chess as worker_module

    monkeypatch.setattr(worker_module, "DB_PATH", db_path)

    config = TournamentConfig(
        tournament_id="recovery-test",
        section="open",
        poll_seconds=5,
    )

    adapter = FakeAdapter()
    worker = RealtimeChessWorker(config, adapter)

    # First poll: game has not finished, so T0 is generated.
    first = worker.poll_round(1)
    assert len(first["new"]) == 1
    assert first["new"][0].has_result is False

    # Simulate worker restart by creating a new worker with fresh state.
    restarted_worker = RealtimeChessWorker(config, adapter)

    # Second poll: game is now finished and should resolve the existing T0.
    second = restarted_worker.poll_round(1)

    assert len(second["new"]) == 1
    assert second["new"][0].has_result is True

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT actual_outcome, surprise_score
        FROM predictions_ledger
        WHERE event_id = ?
          AND model_version_id = ?
        """,
        (
            second["new"][0].event_id,
            "pregame_meta_wd_v1",
        ),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "white"
    assert row[1] is not None
    assert row[1] >= 0