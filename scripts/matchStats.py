"""Playing a match and scoring it, in one place.

Five scripts here play two players against each other over a seed range and
count wins, and until 2026-08-23 each carried its own copy: `Result` was
byte-identical in `baseline.py` and `compareCheckpoints.py`, the Wald interval
was written out seven times, and `evaluateSearch.py` had started importing
`Result` from `compareCheckpoints` for lack of anywhere better. The convention
that seat 1 moves first and that the first-mover advantage is real -- about five
points between two checkpoints -- is the kind of thing that has to be stated
once rather than re-derived per script.

**Players are built by a factory, not passed in.** The callers need three
different things: a fresh `Computer` per game (`baseline`), one long-lived
`NeuralComputer` reused across every game so its checkpoint is loaded once
(`compareCheckpoints`), and a player whose strategy is patched after
construction (`ablateTerms`, `fitWeights`). A factory covers all three, and it
takes the game index as well as the seat so a pool can be cycled by seed --
which is what keeps `fitWeights`' comparisons paired.

**`attachTo` is called by duck-typing, deliberately.** A `NeuralComputer` has
to be pointed at each new game object; a `Computer` has no such method. Testing
for the method rather than the type is what keeps this module free of any
import from `env/`, and therefore of torch -- `baseline.py` and `ablateTerms.py`
are pure-engine scripts and must stay that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from game.gameManager import ComputedGame
from game.player import Computer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from game.player import HalmaPlayer

    # (seat, game index) -> the player to seat there for that game.
    MakePlayer = Callable[[int, int], HalmaPlayer]

SEAT_ONE = 1
SEAT_TWO = 2


def marginOfError(rate: float, games: int) -> float:
    """Rough 95% interval half-width for a win rate.

    The usual Wald interval. It is honest for sampled play and **far too tight
    for argmax play against a deterministic opponent**: the opening is fixed and
    both sides answer identically, so a seed varies only the play order and 30
    games are about 5 distinct game lines rather than 30 independent draws. Use
    a forced-opening census (`scripts/openingSweep.py`) when that matters.
    """
    if not games:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / games)


@dataclass
class Result:
    """Wins, losses and draws of a match, from one side's point of view."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def winRate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def marginOfError(self) -> float:
        return marginOfError(self.winRate, self.games)

    def flipped(self) -> Result:
        """The same match read from the other side."""
        return Result(wins=self.losses, losses=self.wins, draws=self.draws, moves=self.moves)

    def __add__(self, other: Result) -> Result:
        return Result(
            wins=self.wins + other.wins,
            losses=self.losses + other.losses,
            draws=self.draws + other.draws,
            moves=self.moves + other.moves,
        )


@dataclass(frozen=True)
class Bot:
    """A factory seating a fresh heuristic bot, which is the common case.

    A class rather than a closure because it has to survive pickling:
    `baseline.py` and `fitWeights.py` hand factories to a `multiprocessing.Pool`,
    and a lambda cannot be pickled. Instances of a module-level frozen
    dataclass can.
    """

    name: str

    def __call__(self, seat: int, _index: int) -> HalmaPlayer:
        return Computer(seat, self.name)


@dataclass(frozen=True)
class BotPool:
    """A factory cycling a pool of bots by game index.

    Every candidate in a fit therefore meets the same opponent on the same
    seed, which is what makes those comparisons paired: a seed that happens to
    favour one challenger favours all of them.
    """

    names: tuple[str, ...]

    def __call__(self, seat: int, index: int) -> HalmaPlayer:
        return Computer(seat, self.names[index % len(self.names)])


@dataclass(frozen=True)
class Fixed:
    """A factory seating one long-lived player in every game.

    What a `NeuralComputer` needs: constructing one loads a checkpoint from
    disk, so rebuilding it per game would pay that cost hundreds of times.
    Not picklable in practice -- the player it holds is a loaded network -- and
    it does not need to be, since no run seats a checkpoint across processes.

    **Only for a player that stays on one seat.** A `HalmaPlayer` carries its
    seat in `identifier` from construction, so handing the same instance to
    the swapped leg of :func:`playBothSeats` would seat it on the wrong home
    corner. Use :class:`Seated` there.
    """

    player: HalmaPlayer

    def __call__(self, _seat: int, _index: int) -> HalmaPlayer:
        return self.player


@dataclass
class Seated:
    """A factory picking the instance built for whichever seat is asked for.

    A `HalmaPlayer`'s seat is fixed at construction and `Initializer` hands out
    home corners by list position, so swapping seats means swapping *instances*,
    not reusing one. Callers that load a checkpoint once per seat -- which is
    every checkpoint comparison here -- pass that mapping in and
    :func:`playBothSeats` then works without the caller having to think about
    it.

    A `Mapping` rather than a `dict`: callers hand in a dict of whatever
    concrete player they built, and `dict` is invariant in its value type, so
    `dict[int, NeuralComputer]` would not be accepted.
    """

    bySeat: Mapping[int, HalmaPlayer]

    def __call__(self, seat: int, _index: int) -> HalmaPlayer:
        return self.bySeat[seat]


def playMatch(
    makeSeatOne: MakePlayer,
    makeSeatTwo: MakePlayer,
    games: int,
    seed: int = 0,
    scoredSeat: int = SEAT_ONE,
) -> Result:
    """Play ``games`` games and count them from ``scoredSeat``'s side.

    Each game is seeded ``seed + i``, which fixes both the play order and the
    bots' tie-breaking, so a match is reproducible from its seed alone. Note
    that play order is *drawn* per game rather than alternating, so over enough
    games the first-mover advantage averages out within a single direction --
    but not reliably at small ``games``, which is why the seat-swapping
    :func:`playBothSeats` exists.
    """
    result = Result()
    for index in range(games):
        game = ComputedGame()
        game.seed(seed + index)
        players = [makeSeatOne(SEAT_ONE, index), makeSeatTwo(SEAT_TWO, index)]
        game.initGame(players)
        for player in players:
            attach = getattr(player, "attachTo", None)
            if attach is not None:
                attach(game)
        winner = game.play()
        result.moves += game.gameLength()
        if winner is None:
            result.draws += 1
        elif winner == scoredSeat:
            result.wins += 1
        else:
            result.losses += 1
    return result


def playBothSeats(
    makeA: MakePlayer, makeB: MakePlayer, games: int, seed: int = 0
) -> tuple[Result, Result, Result]:
    """Play A against B in both seat directions, all scored from A's side.

    Returns ``(combined, aOnSeatOne, aOnSeatTwo)``. One direction alone bakes
    in the first-mover advantage, which is worth roughly five points here, and
    folds it into the result invisibly. The reverse leg is flipped here rather
    than at the call site, where the inversion is easy to miss.
    """
    forward = playMatch(makeA, makeB, games, seed)
    reverse = playMatch(makeB, makeA, games, seed).flipped()
    return forward + reverse, forward, reverse
