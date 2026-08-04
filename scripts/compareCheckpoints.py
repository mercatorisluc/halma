"""Play trained checkpoints against each other, not just against bots.

    python -m scripts.compareCheckpoints models/Talos1.0 models/Talos1.1
    python -m scripts.compareCheckpoints --games 30 --sampled models/a models/b models/c

Analogous to scripts/baseline.py, but for policy checkpoints instead of
heuristic names. Two checkpoints can be seated directly against each other
now that env/halmaEnv.py's ``selfSeat`` lets a NeuralComputer build its
observation from either seat -- previously each was only ever measurable
against a common panel of bots.

Each checkpoint is loaded twice, once per seat, and reused across every
matchup it appears in -- reloading per game would mean a network load for
every one of possibly hundreds of games.

Two things about this measurement are easy to get wrong, so the script handles
both rather than leaving them to the caller:

*Argmax play makes --games a lie.* The opening is fixed and a deterministic
policy answers a given position the same way every time, so the only thing a
seed still varies is the play order -- see HalmaGame.seed. With two players
that is two possible games, whatever --games says. Measured: 20 different seeds
produced exactly 2 distinct games, one repeated 11 times and the other 9, and
the printed "55.0% +/- 21.8" was just the 11/20 of seeds that let seat 1 start.
So argmax mode reports the outcome per play order rather than a percentage
pretending to be a sample. Use --sampled for a real win rate: the same 20 seeds
then give 17 distinct games.

*One seat direction bakes in the first-mover advantage*, which is large here.
Every pairing is therefore played twice, swapping seats, and the win rate is
over both halves; the per-seat split is printed alongside so a lopsided pair is
visible rather than averaged away.
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


def playPairing(
    agents: dict[str, dict[int, NeuralComputer]], a: str, b: str, games: int, seed: int
) -> tuple[Result, Result, Result]:
    """Play ``a`` against ``b`` in both seat directions, scored from a's view.

    Returns ``(combined, aOnSeat1, aOnSeat2)``. The reverse leg is played with
    ``b`` on seat 1, so its wins are a's losses and vice versa -- flipped here
    rather than at the call site, where the inversion is easy to miss.
    """
    forward = playMatch(
        agents[a][HalmaEnv.AGENT_SEAT], agents[b][HalmaEnv.OPPONENT_SEAT], games, seed
    )
    reverse = playMatch(
        agents[b][HalmaEnv.AGENT_SEAT], agents[a][HalmaEnv.OPPONENT_SEAT], games, seed
    )
    flipped = Result(
        wins=reverse.losses, losses=reverse.wins, draws=reverse.draws, moves=reverse.moves
    )
    combined = Result(
        wins=forward.wins + flipped.wins,
        losses=forward.losses + flipped.losses,
        draws=forward.draws + flipped.draws,
        moves=forward.moves + flipped.moves,
    )
    return combined, forward, flipped


def playOrderOutcomes(
    agents: dict[str, dict[int, NeuralComputer]], a: str, b: str, seed: int
) -> tuple[str, str]:
    """Play the two games that exist under argmax: one per opening move order.

    Seeds are scanned until one of each play order has been seen, because the
    seed picks the order and nothing else once both policies are deterministic.
    """
    outcomes: dict[int, str] = {}
    probe = seed
    while len(outcomes) < 2 and probe < seed + 100:
        game = ComputedGame()
        game.seed(probe)
        agentA, agentB = agents[a][HalmaEnv.AGENT_SEAT], agents[b][HalmaEnv.OPPONENT_SEAT]
        game.initGame([agentA, agentB])
        agentA.attachTo(game)
        agentB.attachTo(game)
        winner = game.play()
        first = game.playOrder[0].identifier
        if first not in outcomes:
            label = "draw" if winner is None else ("a" if winner == HalmaEnv.AGENT_SEAT else "b")
            outcomes[first] = f"{label} ({game.gameLength()})"
        probe += 1
    return outcomes.get(HalmaEnv.AGENT_SEAT, "?"), outcomes.get(HalmaEnv.OPPONENT_SEAT, "?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="model paths, every pairing played once")
    parser.add_argument("--games", type=int, default=30, help="games per seat direction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampled", action="store_true", help="play each policy's distribution instead of argmax"
    )
    args = parser.parse_args()

    agents = buildAgents(args.checkpoints, deterministic=not args.sampled)
    matchups = list(combinations(args.checkpoints, 2))
    width = max(len(c) for c in args.checkpoints)

    if not args.sampled:
        # Rather than print a win rate over --games near-identical replays, say
        # what argmax can actually tell us: who wins under each play order.
        print(
            f"argmax, seed {args.seed} -- deterministic, so only two games exist per pairing.\n"
            f"Reporting the winner under each opening move order; "
            f"rerun with --sampled for a win rate.\n"
        )
        header = f"{'a':<{width}} {'b':<{width}} {'a starts':>14}  {'b starts':>14}"
        print(header)
        print("-" * len(header))
        for a, b in matchups:
            aFirst, bFirst = playOrderOutcomes(agents, a, b, args.seed)
            print(f"{a:<{width}} {b:<{width}} {aFirst:>14}  {bFirst:>14}")
        return

    print(
        f"{args.games} games per seat direction ({args.games * 2} per pairing), "
        f"seed {args.seed}, sampled\n"
    )
    header = (
        f"{'a':<{width}} {'b':<{width}} {'a win%':>14}  "
        f"{'a on s1':>7}  {'a on s2':>7}  {'draws':>6}  {'avg moves':>9}"
    )
    print(header)
    print("-" * len(header))
    for a, b in matchups:
        combined, onSeat1, onSeat2 = playPairing(agents, a, b, args.games, args.seed)
        winRate = f"{combined.winRate * 100:5.1f} +/- {combined.marginOfError * 100:4.1f}"
        print(
            f"{a:<{width}} {b:<{width}} {winRate:>14}  "
            f"{onSeat1.winRate * 100:6.1f}%  {onSeat2.winRate * 100:6.1f}%  "
            f"{combined.draws:>6}  {combined.moves / combined.games:>9.0f}"
        )


if __name__ == "__main__":
    main()
