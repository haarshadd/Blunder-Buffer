"""
Chess-Results adapter.

Parses Chess-Results Board Pairings pages into ChessGameEvent objects.

The parser is designed for team tournaments where individual games are
represented in the form:

    1.1
    Player A
    1562
    IND
    -
    Player B
    1881
    USA
    0 - 1

or, after HTML rendering:

    1.1 Player A 1562 IND - Player B 1881 USA 0 - 1

The parser deliberately looks for complete individual-game patterns rather
than treating arbitrary HTML table rows as games.
"""

from __future__ import annotations

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from live.events import ChessGameEvent
from workers.adapters.base import ChessSourceAdapter


class ChessResultsAdapter(ChessSourceAdapter):
    """Adapter for Chess-Results Board Pairings pages."""

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

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: str) -> str:
        return " ".join(
            str(value or "").split()
        ).strip()

    @staticmethod
    def _normalize_result(
        result: Optional[str],
    ) -> Optional[str]:

        if not result:
            return None

        value = (
            result
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

    # ------------------------------------------------------------------
    # Player cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_titles(name: str) -> str:
        titles = {
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

        parts = name.split()

        return " ".join(
            part
            for part in parts
            if part.upper() not in titles
        )

    @classmethod
    def _clean_player_name(
        cls,
        name: str,
    ) -> str:

        name = cls._clean_text(name)

        name = cls._remove_titles(name)

        return cls._clean_text(name)

    # ------------------------------------------------------------------
    # Game pattern
    # ------------------------------------------------------------------

    @classmethod
    def _game_pattern(cls) -> re.Pattern:
        """
        Match one complete individual game.

        Example:

            1.1 Alexander, Easther 1562 NSN -
            Ainul Fikri, Aqilah Husna 1881 WP 0 - 1

        The player names are captured lazily until a rating + federation
        pair is encountered.

        This is intentionally based on the rendered text rather than
        specific HTML tags.
        """

        return re.compile(
            r"""
            (?<![\w.])
            (?P<team_board>\d+)
            \.
            (?P<player_board>\d+)

            \s+

            (?P<white>.*?)
            \s+
            (?P<white_rating>\d{3,4})
            \s+
            (?P<white_fed>[A-Z]{3})

            \s*-\s*

            (?P<black>.*?)
            \s+
            (?P<black_rating>\d{3,4})
            \s+
            (?P<black_fed>[A-Z]{3})

            (?:
                \s+
                (?P<result>
                    1\s*-\s*0
                    |
                    0\s*-\s*1
                    |
                    ½\s*-\s*½
                    |
                    1/2\s*-\s*1/2
                )
            )?

            (?=\s|$)
            """,
            re.VERBOSE,
        )

    # ------------------------------------------------------------------
    # Parse games
    # ------------------------------------------------------------------

    @classmethod
    def _parse_games(
        cls,
        page_text: str,
    ) -> list[dict]:

        pattern = cls._game_pattern()

        games: list[dict] = []

        for match in pattern.finditer(page_text):

            white = cls._clean_player_name(
                match.group("white")
            )

            black = cls._clean_player_name(
                match.group("black")
            )

            if not white or not black:
                continue

            team_board = int(
                match.group("team_board")
            )

            player_board = int(
                match.group("player_board")
            )

            # ----------------------------------------------------------
            # Safety checks.
            #
            # These prevent navigation/page metadata from being
            # interpreted as a game.
            # ----------------------------------------------------------

            if player_board < 1 or player_board > 20:
                continue

            if team_board < 1:
                continue

            # A genuine player name should not contain page-navigation
            # phrases.
            bad_phrases = (
                "board pairings",
                "search for player",
                "search board",
                "ranking list",
                "statistics",
                "creator/last upload",
                "last update",
                "round on",
            )

            combined = (
                f"{white} {black}"
            ).lower()

            if any(
                phrase in combined
                for phrase in bad_phrases
            ):
                continue

            result = cls._normalize_result(
                match.group("result")
            )

            games.append(
                {
                    "team_board": team_board,
                    "player_board": player_board,
                    "white": white,
                    "black": black,
                    "white_rating": int(
                        match.group("white_rating")
                    ),
                    "black_rating": int(
                        match.group("black_rating")
                    ),
                    "result": result,
                }
            )

        return games

    # ------------------------------------------------------------------
    # Fetch round
    # ------------------------------------------------------------------

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

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Use spaces here, not newlines.
        #
        # Chess-Results can split one player's information over several
        # HTML elements. get_text(" ") reconstructs it into a stable
        # searchable representation.
        # --------------------------------------------------------------

        page_text = self._clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        lower_text = page_text.lower()

        # --------------------------------------------------------------
        # If the actual Board Pairings section isn't present, the
        # requested round has not been published.
        # --------------------------------------------------------------

        if "board pairings" not in lower_text:
            return []

        # --------------------------------------------------------------
        # Parse complete individual games.
        # --------------------------------------------------------------

        games = self._parse_games(
            page_text
        )

        # --------------------------------------------------------------
        # Future round / unpublished pairings.
        #
        # Chess-Results may still display the Board Pairings navigation
        # or heading even when there are zero actual game records.
        #
        # Therefore zero parsed games is a valid "not published yet"
        # state, not automatically a parser failure.
        # --------------------------------------------------------------

        if not games:
            return []

        # --------------------------------------------------------------
        # Deduplicate.
        #
        # event_id currently contains:
        # tournament + section + round + board + players
        #
        # This protects against duplicated HTML content.
        # --------------------------------------------------------------

        events: dict[str, ChessGameEvent] = {}

        for game in games:

            event = ChessGameEvent(
                tournament_id=self.tournament_id,
                section=self.section,
                round_number=int(
                    round_number
                ),

                # Store the individual board.
                #
                # 1.1 -> board 1
                # 1.2 -> board 2
                # etc.
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

        return list(events.values())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.session.close()