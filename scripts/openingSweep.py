"""Compare two checkpoints over every combination of the two opening moves.

    python -m scripts.openingSweep models/Talos1.0 models/Talos1.1
    python -m scripts.openingSweep --openings 5 models/a models/b

``scripts/compareCheckpoints.py`` has to choose between two bad options: argmax
play is deterministic against a fixed opening, so a pairing has only two
possible games however many are asked for, while ``--sampled`` buys variety by
adding noise to the very thing being measured. Forcing the opening gets variety
without either. Each side has exactly 20 legal opening moves, and that count
does not depend on what the other side played, so fixing both enumerates 400
games per play order -- 800 in all, every one played out deterministically.

That makes the result a census rather than a sample: these are *all* the
two-ply openings, so no confidence interval applies to the total. What is
genuinely uncertain is whether forced openings represent free play, which no
amount of extra games would settle.

Both play orders are swept, and the split is printed, because the first-mover
advantage is real (~5 points between these two checkpoints) and a single
direction would fold it into the result invisibly.
"""

from __future__ import annotations

import argparse
from collections import Counter

from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from game.boardTypes import MovePath
from game.gameManager import ComputedGame
from game.player import HalmaPlayer


def buildGame(
    seatOne: NeuralComputer, seatTwo: NeuralComputer, firstSeat: int
) -> tuple[ComputedGame, list[HalmaPlayer]]:
    """A fresh game with the play order forced rather than drawn.

    ``seed`` still fixes the bots' tie-breaking, but the play order is set
    directly: leaving it to the draw would sample the two orders unevenly (11
    to 9 over seeds 0-19) and skew a sweep meant to weigh them equally.
    """
    game = ComputedGame()
    game.seed(0)
    game.initGame([seatOne, seatTwo])
    seatOne.attachTo(game)
    seatTwo.attachTo(game)
    order = [p for p in game.players if p.identifier == firstSeat]
    order += [p for p in game.players if p.identifier != firstSeat]
    game.playOrder = order
    return game, order


def openingPairs(
    seatOne: NeuralComputer, seatTwo: NeuralComputer, firstSeat: int, limit: int | None
) -> list[tuple[MovePath, MovePath]]:
    """Every (first move, reply) pair, or the first ``limit`` of each."""
    game, order = buildGame(seatOne, seatTwo, firstSeat)
    firstMoves = game.board.allValidMovesWithWay(order[0])[: limit or None]
    pairs = []
    for opener in firstMoves:
        probe, probeOrder = buildGame(seatOne, seatTwo, firstSeat)
        probe.playMove(probeOrder[0], opener)
        replies = probe.board.allValidMovesWithWay(probeOrder[1])[: limit or None]
        pairs += [(opener, reply) for reply in replies]
    return pairs


def playForced(
    seatOne: NeuralComputer,
    seatTwo: NeuralComputer,
    firstSeat: int,
    opening: tuple[MovePath, MovePath],
) -> tuple[int | None, int]:
    """Play one game with both opening moves forced, the rest argmax."""
    game, order = buildGame(seatOne, seatTwo, firstSeat)
    game.playMove(order[0], opening[0])
    game.playMove(order[1], opening[1])
    for _ in range(game.MAX_MOVES - 2):
        if game.winner() is not None:
            break
        game.playNextMove(game.currentPlayer())
    return game.winner(), game.gameLength()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs=2, help="the two model paths to compare")
    parser.add_argument(
        "--openings",
        type=int,
        default=None,
        help="use only the first N moves per side (default: all, giving 800 games)",
    )
    args = parser.parse_args()

    pathOne, pathTwo = args.checkpoints
    seatOne = NeuralComputer(HalmaEnv.AGENT_SEAT, pathOne, deterministic=True)
    seatTwo = NeuralComputer(HalmaEnv.OPPONENT_SEAT, pathTwo, deterministic=True)

    total: Counter[str] = Counter()
    drawnOpenings: list[str] = []
    for firstSeat in (HalmaEnv.AGENT_SEAT, HalmaEnv.OPPONENT_SEAT):
        pairs = openingPairs(seatOne, seatTwo, firstSeat, args.openings)
        tally: Counter[str] = Counter()
        moves = 0
        for opening in pairs:
            winner, length = playForced(seatOne, seatTwo, firstSeat, opening)
            key = "draw" if winner is None else (pathOne if winner == 1 else pathTwo)
            tally[key] += 1
            moves += length
            if winner is None:
                drawnOpenings.append(
                    f"{opening[0][0]}->{opening[0][-1]} / {opening[1][0]}->{opening[1][-1]}"
                    f"  (seat {firstSeat} first)"
                )
        starter = pathOne if firstSeat == HalmaEnv.AGENT_SEAT else pathTwo
        print(f"\n{starter} moves first  ({len(pairs)} openings)")
        for key in (pathOne, pathTwo, "draw"):
            print(f"    {key:<28} {tally[key]:>4}  ({tally[key] / len(pairs) * 100:5.1f} %)")
        print(f"    {'average length':<28} {moves / len(pairs):>4.0f}")
        total += tally

    games = sum(total.values())
    print(f"\n=== {games} games")
    for key in (pathOne, pathTwo, "draw"):
        print(f"    {key:<28} {total[key]:>4}  ({total[key] / games * 100:5.1f} %)")
    if drawnOpenings:
        # Every draw here is the move cap running out, which for two
        # deterministic policies means a cycle -- replay one to see it.
        print("\ndrawn openings (replay with scripts/replayGame.py --opening):")
        for entry in drawnOpenings:
            print(f"    {entry}")


if __name__ == "__main__":
    main()
