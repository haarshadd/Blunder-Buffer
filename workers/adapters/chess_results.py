"""
Chess-Results adapter.

Parses Chess-Results Board Pairings pages into normalized
ChessGameEvent objects.

The parser works at the HTML <tr> level because Chess-Results
renders each individual game as its own table row.

Example:

    1.1 | Player A | 1562 | IND | - | Player B | 1881 | USA | 0 - 1

Team tournament notation:

    1.1 = team match 1, individual board 1
    1.2 = team match 1, individual board 2
    2.1 = team match 2, individual board 1
"""

from __future__ import annotations

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from live.events import ChessGameEvent
from workers.adapters.base import ChessSourceAdapter


class ChessResultsAdapter(ChessSourceAdapter):
    """Chess-Results Board Pairings adapter."""

    BASE_URL = "https://chess-results.com"

    def __init__(
        self,
        tournament_id: str,
        section: str = "open",
        timeout: int = 30,
    ) -> None:
        self.tournament_id = str(tournament_id)
        self.section = section
        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/140.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    # ================================================================
    # URL
    # ================================================================

    def _round_url(self, round_number: int) -> str:
        return (
            f"{self.BASE_URL}/"
            f"tnr{self.tournament_id}.aspx"
            f"?lan=1"
            f"&art=3"
            f"&rd={int(round_number)}"
            f"&flag=30"
            f"&turdet=YES"
            f"&zeilen=9999"
        )

    # ================================================================
    # Text helpers
    # ================================================================

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(
            str(value or "").split()
        ).strip()

    @staticmethod
    def _normalize_result(
        value: Optional[str],
    ) -> Optional[str]:

        if not value:
            return None

        value = (
            value
            .replace("½", "1/2")
            .replace("–", "-")
            .replace("—", "-")
        )

        value = re.sub(
            r"\s+",
            "",
            value,
        )

        if value == "1-0":
            return "white"

        if value == "0-1":
            return "black"

        if value == "1/2-1/2":
            return "draw"

        return None

    # ================================================================
    # Titles
    # ================================================================

    @staticmethod
    def _is_title(value: str) -> bool:
        return value.upper() in {
            "GM",
            "IM",
            "FM",
            "CM",
            "WGM",
            "WIM",
            "WFM",
            "WCM",
            "AFM",
            "AIM",
        }

    @classmethod
    def _clean_player_name(
        cls,
        value: str,
    ) -> str:

        value = cls._clean(value)

        parts = [
            part
            for part in value.split()
            if not cls._is_title(part)
        ]

        return cls._clean(
            " ".join(parts)
        )

    # ================================================================
    # Rating / result
    # ================================================================

    @staticmethod
    def _is_rating(value: str) -> bool:
        """
        Chess-Results ratings in the relevant range.

        This deliberately excludes years such as 2026.
        """

        return bool(
            re.fullmatch(
                r"(?:1\d{3}|2\d{3})",
                value,
            )
        )

    @staticmethod
    def _extract_result(
        cells: list[str],
    ) -> Optional[str]:

        for value in reversed(cells):

            normalized = (
                value
                .replace("½", "1/2")
                .replace("–", "-")
                .replace("—", "-")
            )

            if re.fullmatch(
                r"(?:1\s*-\s*0|0\s*-\s*1|1/2\s*-\s*1/2)",
                normalized,
            ):
                return ChessResultsAdapter._normalize_result(
                    normalized
                )

        return None

    # ================================================================
    # Individual row parser
    # ================================================================

    @classmethod
    def _parse_game_row(
        cls,
        row,
    ) -> Optional[dict]:
        """
        Parse one actual Chess-Results individual-game <tr>.

        Real example observed from Chess-Results:

            [
                '1.1',
                '',
                'Alexander, Easther',
                '',
                'Alexander, Easther',
                '1562',
                'NSN',
                '-',
                '',
                'Ainul Fikri, Aqilah Husna',
                '',
                'Ainul Fikri, Aqilah Husna',
                '1881',
                'WP',
                '0 - 1'
            ]

        Important:
        Some rows contain a title such as AFM/WCM between the
        separator and the player name.
        """

        cells = [
            cls._clean(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )
            for cell in row.find_all(
                ["td", "th"]
            )
        ]

        cells = [
            value
            for value in cells
            if value
            or value == "-"
        ]

        if not cells:
            return None

        # ------------------------------------------------------------
        # First cell must be an individual board marker.
        #
        # Accept:
        #   1.1
        #   2.4
        #   10.3
        # ------------------------------------------------------------

        board_match = re.fullmatch(
            r"(\d+)\.(\d+)",
            cells[0],
        )

        if not board_match:
            return None

        team_board = int(
            board_match.group(1)
        )

        player_board = int(
            board_match.group(2)
        )

        if team_board < 1:
            return None

        if not 1 <= player_board <= 20:
            return None

        # ------------------------------------------------------------
        # Find the separator between white and black.
        # ------------------------------------------------------------

        separator_index = None

        for index, value in enumerate(
            cells[1:],
            start=1,
        ):
            if value == "-":
                separator_index = index
                break

        if separator_index is None:
            return None

        white_cells = cells[
            1:separator_index
        ]

        black_cells = cells[
            separator_index + 1:
        ]

        if not white_cells or not black_cells:
            return None

        # ------------------------------------------------------------
        # Find the white rating.
        #
        # It should occur before the federation code.
        # ------------------------------------------------------------

        white_rating_index = None

        for index, value in enumerate(
            white_cells
        ):
            if cls._is_rating(value):
                white_rating_index = index
                break

        if white_rating_index is None:
            return None

        white_rating = int(
            white_cells[
                white_rating_index
            ]
        )

        # ------------------------------------------------------------
        # Find the black rating.
        # ------------------------------------------------------------

        black_rating_index = None

        for index, value in enumerate(
            black_cells
        ):
            if cls._is_rating(value):
                black_rating_index = index
                break

        if black_rating_index is None:
            return None

        black_rating = int(
            black_cells[
                black_rating_index
            ]
        )

        # ------------------------------------------------------------
        # Player names.
        #
        # Chess-Results may duplicate a linked player name:
        #
        #   ''
        #   'Alexander, Easther'
        #   ''
        #   'Alexander, Easther'
        #
        # We choose the longest meaningful text before the rating.
        #
        # Titles such as AFM/WCM are ignored.
        # ------------------------------------------------------------

        white_name_candidates = []

        for value in white_cells[
            :white_rating_index
        ]:
            if not value:
                continue

            if cls._is_title(value):
                continue

            white_name_candidates.append(
                value
            )

        black_name_candidates = []

        for value in black_cells[
            :black_rating_index
        ]:
            if not value:
                continue

            if cls._is_title(value):
                continue

            black_name_candidates.append(
                value
            )

        if not white_name_candidates:
            return None

        if not black_name_candidates:
            return None

        white = max(
            white_name_candidates,
            key=len,
        )

        black = max(
            black_name_candidates,
            key=len,
        )

        white = cls._clean_player_name(
            white
        )

        black = cls._clean_player_name(
            black
        )

        if not white or not black:
            return None

        # ------------------------------------------------------------
        # Result.
        # ------------------------------------------------------------

        result = cls._extract_result(
            cells
        )

        return {
            "team_board": team_board,
            "player_board": player_board,
            "white": white,
            "black": black,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "result": result,
        }

    # ================================================================
    # Fetch round
    # ================================================================

    def fetch_events(
        self,
        round_number: int,
    ) -> list[ChessGameEvent]:

        url = self._round_url(
            round_number
        )

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        page_text = self._clean(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        lower_text = page_text.lower()

        # ------------------------------------------------------------
        # If Board Pairings is absent, this round is not published.
        # ------------------------------------------------------------

        if "board pairings" not in lower_text:
            return []

        # ------------------------------------------------------------
        # Parse ONLY actual table rows.
        #
        # This is the critical fix.
        # ------------------------------------------------------------

        parsed_games = []

        for row in soup.find_all("tr"):

            parsed = self._parse_game_row(
                row
            )

            if parsed is not None:
                parsed_games.append(
                    parsed
                )

        # ------------------------------------------------------------
        # No actual game rows means pairings haven't been published
        # yet, even if the page contains the "Board Pairings" heading.
        # ------------------------------------------------------------

        if not parsed_games:
            return []

        # ------------------------------------------------------------
        # Convert into normalized events.
        # ------------------------------------------------------------

        events: dict[
            str,
            ChessGameEvent,
        ] = {}

        for game in parsed_games:

            event = ChessGameEvent(
                tournament_id=self.tournament_id,
                section=self.section,
                round_number=int(
                    round_number
                ),
                board=game[
                    "player_board"
                ],
                white=game["white"],
                black=game["black"],
                white_rating=game[
                    "white_rating"
                ],
                black_rating=game[
                    "black_rating"
                ],
                result=game["result"],
                source="chess_results",
                source_url=url,
            )

            events[event.event_id] = event

        return list(
            events.values()
        )

    # ================================================================
    # Cleanup
    # ================================================================

    def close(self) -> None:
        self.session.close()