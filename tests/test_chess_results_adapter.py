from workers.adapters.chess_results import ChessResultsAdapter


def test_completed_team_round_has_games():
    adapter = ChessResultsAdapter("1474895")

    try:
        events = adapter.fetch_events(1)

        assert len(events) == 24

        assert events[0].white == "Alexander, Easther"
        assert events[0].black == "Ainul Fikri, Aqilah Husna"

        assert events[0].white_rating == 1562
        assert events[0].black_rating == 1881

        assert events[0].result == "black"

    finally:
        adapter.close()


def test_unpublished_olympiad_round_returns_empty():
    adapter = ChessResultsAdapter("1469895")

    try:
        events = adapter.fetch_events(1)

        assert events == []

    finally:
        adapter.close()