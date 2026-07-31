"""Measure how the existing bots do against each other.

Run before training anything:

    python -m scripts.baseline --games 200

These numbers are the yardstick. A trained agent is only worth keeping if it
beats them, and without them "the bot seems better" is an opinion. Play order
is randomised per game, so first-move advantage averages out.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.strategy import STRATEGY_NAMES


@dataclass
class Result:
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
        """Rough 95% interval half-width for the win rate."""
        if not self.games:
            return 0.0
        p = self.winRate
        return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / self.games)


def playMatch(strategyA: str, strategyB: str, games: int, seed: int = 0) -> Result:
    """Play ``games`` games of A (seat 1) against B (seat 2)."""
    result = Result()
    for i in range(games):
        game = ComputedGame()
        game.seed(seed + i)
        game.initGame(
            [Computer(1, strategyA), Computer(2, strategyB)],
        )
        winner = game.play()
        result.moves += game.gameLength()
        if winner is None:
            result.draws += 1
        elif winner == 1:
            result.wins += 1
        else:
            result.losses += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per matchup")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Every pairing, so the ranking is complete rather than inferred. Seats do
    # not confer an advantage -- play order is shuffled per game -- so each
    # unordered pair is played once. random against itself is kept as a sanity
    # check: two aimless players should never finish.
    matchups = [("random", "random"), *combinations(STRATEGY_NAMES, 2)]

    print(f"{args.games} games per matchup, seed {args.seed}\n")
    header = f"{'seat 1':<18} {'seat 2':<18} {'win%':>14}  {'draws':>6}  {'avg moves':>9}"
    print(header)
    print("-" * len(header))
    for a, b in matchups:
        r = playMatch(a, b, args.games, args.seed)
        winRate = f"{r.winRate * 100:5.1f} +/- {r.marginOfError * 100:4.1f}"
        print(f"{a:<18} {b:<18} {winRate:>14}  {r.draws:>6}  {r.moves / r.games:>9.0f}")


if __name__ == "__main__":
    main()
