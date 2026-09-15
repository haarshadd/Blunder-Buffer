from workers.realtime_chess import RealtimeChessWorker
from live.config import TournamentConfig
from live.events import ChessGameEvent


class FakeAdapter:

    def fetch_events(self, round_number):
        return [
            ChessGameEvent(
                tournament_id="test-event",
                section="open",
                round_number=round_number,
                board=1,
                white="Carlsen, Magnus",
                black="Caruana, Fabiano",
                result=None,
                source="test",
            )
        ]

    def close(self):
        pass


def test_worker_generates_t0():
    config = TournamentConfig(
        tournament_id="test-event",
        section="open",
        poll_seconds=5,
    )

    worker = RealtimeChessWorker(
        config,
        FakeAdapter(),
    )

    result = worker.poll_round(1)

    assert len(result["new"]) == 1
    assert result["new"][0].has_result is False