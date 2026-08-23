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
from typing import TYPE_CHECKING

from game.player import Computer
from heuristics.strategy import Strategy
from scripts.matchStats import Bot, Result, playMatch

if TYPE_CHECKING:
    from game.board import HalmaBoard
    from game.player import HalmaPlayer
    from scripts.matchStats import MakePlayer

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


def challenger(keep: list[str], rescale: bool) -> MakePlayer:
    """Seat `shaped`, then swap its scorer for the ablated one.

    The bot is built the normal way and patched afterwards rather than
    registered as a strategy of its own: an ablation is a throwaway variant and
    has no business in `Strategy.SCORERS`, which is the single source of truth
    for which bots exist.
    """

    def make(seat: int, _index: int) -> HalmaPlayer:
        player = Computer(seat, "shaped")
        player.strategy = AblatedShaped(keep, rescale)
        return player

    return make


def report(label: str, result: Result) -> None:
    print(
        f"{label:<42}{result.winRate * 100:5.1f} +/- {result.marginOfError * 100:4.1f}"
        f"   draws {result.draws}"
    )


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
            result = playMatch(challenger(keep, rescale), Bot("shaped"), args.games, args.seed)
            report(f"without {term}{suffix}", result)


if __name__ == "__main__":
    main()
