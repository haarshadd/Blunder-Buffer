"""Normalized chess event objects used by all live tournament sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import re


def normalize_name(name: str) -> str:
    value = " ".join(str(name or "").split()).strip().lower()
    return value


def make_event_id(
    tournament_id: str,
    section: str,
    round_number: int,
    board: Optional[int],
    white: str,
    black: str,
    white_fide_id: Optional[str] = None,
    black_fide_id: Optional[str] = None,
) -> str:
    """Create a stable event ID independent of the source provider."""

    white_key = str(white_fide_id or normalize_name(white))
    black_key = str(black_fide_id or normalize_name(black))
    board_key = str(board) if board is not None else "na"

    section_key = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(section or "open").lower(),
    )

    return (
        f"{tournament_id}:{section_key}:"
        f"r{int(round_number)}:b{board_key}:"
        f"{white_key}:vs:{black_key}"
    )


@dataclass(frozen=True)
class ChessGameEvent:
    tournament_id: str
    section: str
    round_number: int

    white: str
    black: str

    board: Optional[int] = None

    white_rating: Optional[int] = None
    black_rating: Optional[int] = None

    white_fide_id: Optional[str] = None
    black_fide_id: Optional[str] = None

    white_team: Optional[str] = None
    black_team: Optional[str] = None

    scheduled_time: Optional[datetime] = None

    # scheduled | live | finished | cancelled
    status: str = "scheduled"

    # white | draw | black
    result: Optional[str] = None

    eco: Optional[str] = None

    source: str = "unknown"
    source_url: Optional[str] = None

    moves: tuple[str, ...] = field(default_factory=tuple)

    @property
    def event_id(self) -> str:
        return make_event_id(
            self.tournament_id,
            self.section,
            self.round_number,
            self.board,
            self.white,
            self.black,
            self.white_fide_id,
            self.black_fide_id,
        )

    @property
    def has_result(self) -> bool:
        return self.result in {"white", "draw", "black"}

    def with_updates(self, **changes) -> "ChessGameEvent":
        values = self.__dict__.copy()
        values.update(changes)
        return ChessGameEvent(**values)