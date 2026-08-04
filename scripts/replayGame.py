"""Play two checkpoints against each other and step through it in the window.

    python -m scripts.replayGame models/Talos1.0 models/Talos1.1
    python -m scripts.replayGame --opening 6,26 66,67 --start 70 models/a models/b

A win rate says who won, not what happened. This plays the game out first, then
hands the finished game to ``GameVisualization``, so the whole thing can be
stepped through with the arrow keys rather than watched once at ten frames a
second. It is how the drawn games ``scripts/openingSweep.py`` reports were
found to be deadlocks: one piece per side shuffling between two fields.

``--opening`` forces both first moves, which is what makes a specific game from
that sweep reproducible here -- without it the two argmax policies play the one
game their play order determines.
"""

from __future__ import annotations

import argparse

from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from game.boardTypes import MovePath
from game.gameManager import ComputedGame
from game.player import HalmaPlayer
from visual.gameVisualization import GameVisualization


def parseEndpoints(text: str) -> tuple[int, int]:
    start, end = text.split(",")
    return int(start), int(end)


def buildGame(
    seatOne: NeuralComputer, seatTwo: NeuralComputer, firstSeat: int
) -> tuple[ComputedGame, list[HalmaPlayer]]:
    game = ComputedGame()
    game.seed(0)
    game.initGame([seatOne, seatTwo])
    seatOne.attachTo(game)
    seatTwo.attachTo(game)
    order = [p for p in game.players if p.identifier == firstSeat]
    order += [p for p in game.players if p.identifier != firstSeat]
    game.playOrder = order
    return game, order


def forcedMove(game: ComputedGame, player: HalmaPlayer, endpoints: tuple[int, int]) -> MovePath:
    """The full jump path whose endpoints are ``endpoints``.

    The board deals in paths, the command line in endpoints, and applying the
    wrong one of the two is the classic mistake here -- so the lookup happens
    once, loudly, rather than being passed around as a pair.
    """
    for move in game.board.allValidMovesWithWay(player):
        if (move[0], move[-1]) == endpoints:
            return move
    raise SystemExit(f"{endpoints[0]}->{endpoints[1]} is not legal for seat {player.identifier}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs=2, help="seat 1 and seat 2 model paths")
    parser.add_argument(
        "--opening",
        nargs=2,
        default=None,
        metavar=("START,END", "START,END"),
        help="force both opening moves, as endpoint pairs",
    )
    parser.add_argument(
        "--firstSeat", type=int, default=HalmaEnv.AGENT_SEAT, choices=(1, 2), help="who moves first"
    )
    parser.add_argument("--start", type=int, default=0, help="half-move the window opens at")
    parser.add_argument(
        "--sampled", action="store_true", help="play both policies' distributions instead of argmax"
    )
    args = parser.parse_args()

    pathOne, pathTwo = args.checkpoints
    deterministic = not args.sampled
    seatOne = NeuralComputer(HalmaEnv.AGENT_SEAT, pathOne, deterministic=deterministic)
    seatTwo = NeuralComputer(HalmaEnv.OPPONENT_SEAT, pathTwo, deterministic=deterministic)

    game, order = buildGame(seatOne, seatTwo, args.firstSeat)
    if args.opening:
        for player, text in zip(order, args.opening, strict=True):
            game.playMove(player, forcedMove(game, player, parseEndpoints(text)))
    for _ in range(game.MAX_MOVES - len(game.moves)):
        if game.winner() is not None:
            break
        game.playNextMove(game.currentPlayer())

    home = {
        p.identifier: sum(
            1 for field in p.endPositions if game.board.fields[field].playerID == p.identifier
        )
        for p in game.players
    }
    print(f"seat 1 {pathOne}, seat 2 {pathTwo}")
    print(f"{game.gameLength()} half-moves, winner {game.winner()}, pieces home {home}")
    print("window: right/left step the history, up/down rotate the board")

    visualization = GameVisualization(game)
    # GamePlaybackController starts its cursor at 0 and assumes the board
    # matches. Here the game has already been played out, so the board is at
    # the end -- without this the rewind below does nothing and the first
    # forward step replays move 0 onto a board where that piece has long since
    # moved, tripping an assert in removePiece.
    visualization.playback.moveTraveler = len(game.moves)
    visualization.prepareGameVisualization()
    visualization.playback.gameStateAt(min(args.start, len(game.moves)))
    visualization.runGame(game)


if __name__ == "__main__":
    main()
