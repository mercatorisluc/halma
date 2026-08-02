"""Play trained checkpoints against each other, not just against bots.

    python -m scripts.compareCheckpoints models/tunedEnt000 models/cloned
    python -m scripts.compareCheckpoints --games 30 --sampled models/a models/b models/c

Analogous to scripts/baseline.py, but for policy checkpoints instead of
heuristic names. Two checkpoints can be seated directly against each other
now that env/halmaEnv.py's ``selfSeat`` lets a NeuralComputer build its
observation from either seat -- previously each was only ever measurable
against a common panel of bots.

Each checkpoint is loaded twice, once per seat, and reused across every
matchup it appears in -- reloading per game would mean a network load for
every one of possibly hundreds of games.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations

from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from game.gameManager import ComputedGame


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


def buildAgents(
    checkpoints: list[str], deterministic: bool
) -> dict[str, dict[int, NeuralComputer]]:
    return {
        path: {
            seat: NeuralComputer(seat, path, deterministic=deterministic)
            for seat in (HalmaEnv.AGENT_SEAT, HalmaEnv.OPPONENT_SEAT)
        }
        for path in checkpoints
    }


def playMatch(agentA: NeuralComputer, agentB: NeuralComputer, games: int, seed: int = 0) -> Result:
    """Play ``games`` games of A (seat 1) against B (seat 2)."""
    result = Result()
    for i in range(games):
        game = ComputedGame()
        game.seed(seed + i)
        game.initGame([agentA, agentB])
        agentA.attachTo(game)
        agentB.attachTo(game)
        winner = game.play()
        result.moves += game.gameLength()
        if winner is None:
            result.draws += 1
        elif winner == HalmaEnv.AGENT_SEAT:
            result.wins += 1
        else:
            result.losses += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="model paths, every pairing played once")
    parser.add_argument("--games", type=int, default=30, help="games per matchup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampled", action="store_true", help="play each policy's distribution instead of argmax"
    )
    args = parser.parse_args()

    agents = buildAgents(args.checkpoints, deterministic=not args.sampled)
    matchups = list(combinations(args.checkpoints, 2))
    width = max(len(c) for c in args.checkpoints)

    print(
        f"{args.games} games per matchup, seed {args.seed}, "
        f"{'sampled' if args.sampled else 'argmax'}\n"
    )
    header = f"{'seat 1':<{width}} {'seat 2':<{width}} {'win%':>14}  {'draws':>6}  {'avg moves':>9}"
    print(header)
    print("-" * len(header))
    for a, b in matchups:
        agentA, agentB = agents[a][HalmaEnv.AGENT_SEAT], agents[b][HalmaEnv.OPPONENT_SEAT]
        r = playMatch(agentA, agentB, args.games, args.seed)
        winRate = f"{r.winRate * 100:5.1f} +/- {r.marginOfError * 100:4.1f}"
        print(f"{a:<{width}} {b:<{width}} {winRate:>14}  {r.draws:>6}  {r.moves / r.games:>9.0f}")


if __name__ == "__main__":
    main()
