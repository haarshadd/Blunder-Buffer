"""Generic real-time chess tournament worker.

Responsibilities:

    source adapter
        ↓
    normalized ChessGameEvent
        ↓
    event state tracking
        ↓
    future prediction / result handlers

Model inference is intentionally not hard-wired yet.
"""

from __future__ import annotations

import argparse
import time

from dataclasses import dataclass
from datetime import datetime, timezone

from live.config import TournamentConfig
from live.events import ChessGameEvent
from live.staged_predict import stage0, DB_PATH
from core.ledger import resolve_prediction

from workers.adapters.base import (
    ChessSourceAdapter,
)

from workers.adapters.chess_results import (
    ChessResultsAdapter,
)


@dataclass
class WorkerState:

    events: dict[str, ChessGameEvent]

    def __init__(self):
        self.events = {}

    def apply(
        self,
        incoming: list[ChessGameEvent],
    ):
        """
        Compare incoming events against previous state.

        Returns:

            new_events
            changed_events
        """

        new_events = []
        changed_events = []

        for event in incoming:

            previous = self.events.get(
                event.event_id
            )

            if previous is None:

                self.events[
                    event.event_id
                ] = event

                new_events.append(event)

            elif previous != event:

                self.events[
                    event.event_id
                ] = event

                changed_events.append(event)

        return (
            new_events,
            changed_events,
        )


class RealtimeChessWorker:

    def __init__(
        self,
        config: TournamentConfig,
        adapter: ChessSourceAdapter,
    ):

        self.config = config
        self.adapter = adapter

        self.state = WorkerState()

        self._stop = False

    def stop(self):
        self._stop = True

    def poll_round(
        self,
        round_number: int,
    ):

        events = (
            self.adapter.fetch_events(
                round_number
            )
        )

        (
            new_events,
            changed_events,
        ) = self.state.apply(events)

        finished = [
            event
            for event in (
                changed_events
                + new_events
            )
            if event.has_result
        ]

        now = datetime.now(
            timezone.utc
        ).isoformat()

        print(
            f"[{now}] "
            f"round={round_number} "
            f"fetched={len(events)} "
            f"new={len(new_events)} "
            f"changed={len(changed_events)} "
            f"finished={len(finished)}"
        )

        for event in new_events:

            print(
                f"  NEW  "
                f"{event.event_id} | "
                f"{event.white} vs "
                f"{event.black} | "
                f"{event.status}"
            )

            if not event.has_result:

                prediction = stage0(
                    event.white,
                    event.black,
                    datetime.now(timezone.utc).date().isoformat(),
                    event_id=event.event_id,
                    db_path=DB_PATH
                )

                probs = prediction["probabilities"]

                print(
                    f"       T0 "
                    f"black={probs['black']:.3f} "
                    f"draw={probs['draw']:.3f} "
                    f"white={probs['white']:.3f}"
                )

            for event in changed_events:

                print(
                f"  UPD  "
                f"{event.event_id} | "
                f"status={event.status} "
                f"result={event.result}"
            )

        for event in finished:

            surprise = resolve_prediction(
                DB_PATH,
                event.event_id,
                "pregame_meta_wd_v1",
                event.result,
            )

            if surprise is not None:
                print(
                    f"       T0 RESOLVED "
                    f"actual={event.result} "
                    f"surprise={surprise:.4f}"
                )
        return {
            "events": events,
            "new": new_events,
            "changed": changed_events,
            "finished": finished,
        }

    def run(
        self,
        rounds: list[int],
    ):

        try:

            while not self._stop:

                for round_number in rounds:

                    self.poll_round(
                        round_number
                    )

                if self._stop:
                    break

                time.sleep(
                    max(
                        5,
                        self.config.poll_seconds,
                    )
                )

        finally:

            self.adapter.close()


def build_adapter(
    config: TournamentConfig,
) -> ChessSourceAdapter:

    if config.source == "chess_results":

        return ChessResultsAdapter(
            tournament_id=config.tournament_id,
            section=config.section,
        )

    raise ValueError(
        f"Unsupported chess source: "
        f"{config.source}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Blunder Buffer "
            "real-time chess worker"
        )
    )

    parser.add_argument(
        "--tournament-id",
        required=True,
    )

    parser.add_argument(
        "--section",
        default="open",
    )

    parser.add_argument(
        "--round",
        type=int,
        action="append",
        dest="rounds",
    )

    parser.add_argument(
        "--poll",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    rounds = (
        args.rounds
        or [1]
    )

    config = TournamentConfig(
        tournament_id=args.tournament_id,
        section=args.section,
        poll_seconds=max(
            5,
            args.poll,
        ),
    )

    worker = RealtimeChessWorker(
        config,
        build_adapter(config),
    )

    worker.run(rounds)


if __name__ == "__main__":
    main()