"""Score a checkpoint against the bots with the one-ply search in front of it.

    python -m scripts.evaluateSearch models/Talos1.2 --bots lookahead2 --games 20
    python -m scripts.evaluateSearch models/Talos1.2 --bots lookahead2 --noSearch

``--noSearch`` seats a plain ``NeuralComputer`` instead and is the control: the
same games, the same seeds, the same loop, the policy's bare argmax. Any
difference between the two runs is the search and nothing else, which is why
this script exists rather than comparing against the numbers
``scripts/evaluateAgainstBots.py`` produces -- that one drives ``HalmaEnv``
directly and a searching player cannot be seated in it.

Both seat directions are played, because the first-mover advantage here is
worth about five points (ARCHITECTURE.md's opening sweep) and one direction
would fold it into the result invisibly.

Draws are reported rather than swallowed. They are what a policy without a
repetition signal does against a deterministic opponent, and the search's
``seen`` set is meant to remove them, so the column is part of the result.
"""

from __future__ import annotations

import argparse
import time

from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from env.searchPlayer import SearchingComputer
from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.strategy import STRATEGY_NAMES
from scripts.compareCheckpoints import Result


def buildAgent(seat: int, checkpoint: str, candidates: int, replies: int, search: bool):
    if not search:
        return NeuralComputer(seat, checkpoint)
    return SearchingComputer(seat, checkpoint, candidates=candidates, replies=replies)


def playMatch(agent: NeuralComputer, botName: str, games: int, seed: int) -> Result:
    """``games`` games of ``agent`` against ``botName``, scored from the agent."""
    result = Result()
    botSeat = (
        HalmaEnv.OPPONENT_SEAT if agent.identifier == HalmaEnv.AGENT_SEAT else HalmaEnv.AGENT_SEAT
    )
    for i in range(games):
        game = ComputedGame()
        game.seed(seed + i)
        bot = Computer(botSeat, botName)
        players = [agent, bot] if agent.identifier < botSeat else [bot, agent]
        game.initGame(players)
        agent.attachTo(game)
        winner = game.play()
        result.moves += game.gameLength()
        if winner is None:
            result.draws += 1
        elif winner == agent.identifier:
            result.wins += 1
        else:
            result.losses += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="model path")
    parser.add_argument("--bots", nargs="+", default=["lookahead2"], help=f"of {STRATEGY_NAMES}")
    parser.add_argument("--games", type=int, default=20, help="games per seat direction")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--candidates", type=int, default=6, help="own moves expanded")
    parser.add_argument("--replies", type=int, default=3, help="opponent answers per candidate")
    parser.add_argument(
        "--noSearch", action="store_true", help="control: the bare policy through the same loop"
    )
    args = parser.parse_args()

    search = not args.noSearch
    mode = (
        f"search, {args.candidates} candidates x {args.replies} replies" if search else "no search"
    )
    print(f"{args.checkpoint} -- {mode}")
    print(f"{args.games} games per seat direction, seed {args.seed}\n")

    agents = [
        buildAgent(seat, args.checkpoint, args.candidates, args.replies, search)
        for seat in (HalmaEnv.AGENT_SEAT, HalmaEnv.OPPONENT_SEAT)
    ]

    header = (
        f"{'bot':<18} {'win%':>14}  {'W':>3} {'L':>3} {'D':>3}  {'avg moves':>9}  {'s/game':>7}"
    )
    print(header)
    print("-" * len(header))
    for botName in args.bots:
        started = time.perf_counter()
        combined = Result()
        for agent in agents:
            half = playMatch(agent, botName, args.games, args.seed)
            combined = Result(
                wins=combined.wins + half.wins,
                losses=combined.losses + half.losses,
                draws=combined.draws + half.draws,
                moves=combined.moves + half.moves,
            )
        elapsed = time.perf_counter() - started
        rate = f"{combined.winRate * 100:5.1f} +/- {combined.marginOfError * 100:4.1f}"
        print(
            f"{botName:<18} {rate:>14}  {combined.wins:>3} {combined.losses:>3} "
            f"{combined.draws:>3}  {combined.moves / combined.games:>9.0f}  "
            f"{elapsed / combined.games:>7.1f}"
        )


if __name__ == "__main__":
    main()
