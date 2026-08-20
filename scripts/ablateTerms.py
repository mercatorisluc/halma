"""Drop each of `shaped`'s shape terms and play the result against `shaped`.

    python -m scripts.ablateTerms --games 200

Correlation says whether two terms measure the same thing; it cannot say
whether a term is worth anything. This does, the only way that counts: the
challenger is `shaped` minus one term, the reference is `shaped`, and a win
rate near 50% means the term was not paying for itself.

Each ablation is played twice. Removing a term also shrinks the total magnitude
of the shape sum, and `SHAPE_WEIGHT` is known to respond sharply to that
magnitude -- so `--rescale` compensates by the missing term's share, separating
"this term carries information" from "a smaller shape weight happens to help".
Both numbers are reported because they answer different questions.
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.strategy import Strategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from game.board import HalmaBoard
    from game.player import HalmaPlayer

# The shape terms of `shaped`, by the name this script refers to them by.
TERMS: dict[str, str] = {
    "clustering": "clusteringScore",
    "stragglerLag": "stragglerLagScore",
    "jumpPotential": "jumpPotentialScore",
}


class AblatedShaped(Strategy):
    """`shaped` with a subset of its shape terms, otherwise identical."""

    def __init__(self, keep: list[str], rescale: bool) -> None:
        super().__init__("shaped")
        self.keep = keep
        self.factor = len(TERMS) / len(keep) if rescale else 1.0

    def shaped(self, board: HalmaBoard, player: HalmaPlayer) -> float:
        home = board.unfilledTargetScore(player)
        shape = sum(float(getattr(board, TERMS[name])(player)) for name in self.keep)
        return (
            board.openTargetDistanceScore(player)
            + home
            + self.SHAPE_WEIGHT * self.factor * home * shape
        )


def playMatch(
    makeChallenger: Callable[[], Strategy], games: int, seed: int
) -> tuple[int, int, int]:
    """Wins, draws and losses for the challenger against full `shaped`."""
    wins = draws = losses = 0
    for i in range(games):
        game = ComputedGame()
        game.seed(seed + i)
        challenger, reference = Computer(1, "shaped"), Computer(2, "shaped")
        challenger.strategy = makeChallenger()
        game.initGame([challenger, reference])
        winner = game.play()
        if winner is None:
            draws += 1
        elif winner == 1:
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


def report(label: str, wins: int, draws: int, losses: int) -> None:
    games = wins + draws + losses
    rate = wins / games
    margin = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / games)
    print(f"{label:<42}{rate * 100:5.1f} +/- {margin * 100:4.1f}   draws {draws}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200, help="games per ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--term", help="ablate only this term")
    parser.add_argument(
        "--rescale",
        choices=["off", "on", "both"],
        default="both",
        help="compensate for the missing term's magnitude",
    )
    args = parser.parse_args()

    dropped = [args.term] if args.term else list(TERMS)
    rescales = {"off": [False], "on": [True], "both": [False, True]}[args.rescale]
    print(f"challenger vs. full `shaped`, {args.games} games each\n")
    for term in dropped:
        keep = [name for name in TERMS if name != term]
        for rescale in rescales:
            suffix = ", rescaled" if rescale else ""
            wins, draws, losses = playMatch(
                lambda keep=keep, rescale=rescale: AblatedShaped(keep, rescale),
                args.games,
                args.seed,
            )
            report(f"without {term}{suffix}", wins, draws, losses)


if __name__ == "__main__":
    main()
