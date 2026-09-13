"""Interface for live chess data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from live.events import ChessGameEvent


class ChessSourceAdapter(ABC):
    """Convert one provider's data into normalized ChessGameEvent objects."""

    @abstractmethod
    def fetch_events(
        self,
        round_number: int,
    ) -> list[ChessGameEvent]:
        """Return all known games for one round."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook."""
        return None