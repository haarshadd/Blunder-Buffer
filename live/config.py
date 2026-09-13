"""Configuration for a real-time chess tournament run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TournamentConfig:
    tournament_id: str

    section: str = "open"

    # Source adapter name.
    source: str = "chess_results"

    # Polling interval.
    poll_seconds: int = 60

    start_round: int | None = None
    end_round: int | None = None

    # Prediction integration will be enabled after the event layer
    # is verified.
    predict_pregame: bool = False

    def rounds(self) -> list[int] | None:
        if self.start_round is None:
            return None

        end = (
            self.end_round
            if self.end_round is not None
            else self.start_round
        )

        if end < self.start_round:
            raise ValueError(
                "end_round must be >= start_round"
            )

        return list(
            range(self.start_round, end + 1)
        )