"""Fit a calibrated bot's weights instead of sweeping one knob by hand.

    python -m scripts.fitWeights --variant calibrated --candidates 192

Successive halving on the real objective: many weight vectors get a few games
each, the survivors get more, and only a handful ever play a full match. Noise
is spent where candidates are still close together instead of uniformly, which
is what makes a four-dimensional search affordable -- giving every vector enough
games to rank it reliably costs an order of magnitude more.

Every candidate in a round plays the same seeds against the same opponents, so
the comparisons are paired: a seed that happens to favour the challenger favours
all of them, and what is left is the difference the weights make.

**Which variant is fitted decides which weights are searched.** A member of the
`CALIBRATED_VARIANTS` family is nothing but a weight vector, and a term it
switches off is a weight of 0; the fit searches exactly the terms that variant
uses and leaves the rest at 0. So the same command fits the full five-term bot
and each of the deliberately partial ones.

Two dead ends are recorded here because both look obviously right and both cost
a run to rule out:

- **Do not fit on agreement with `lookahead2`.** It is the tempting fitness --
  deterministic, thousands of vectors a second, no game noise. But `lookahead2`
  searches on `straggler`, and over sampled positions `straggler` already agrees
  with it ~98% of the time: the search rarely departs from its own leaf. So
  "agree with the search" means "be `straggler`", and the vectors maximising it
  lost every game they played. Kept as a diagnostic, never as a target.
- **Do not fit against one opponent.** Fitting on `shaped` alone produced a
  vector that beat `shaped` over fresh seeds and was *weaker* than `shaped`
  against everyone else. These bots are not transitive enough for that: what it
  found was a counter to one opponent, not a better bot. Hence a pool, cycled
  by seed. Figures in ARCHITECTURE.md.

Weights are scale-free -- multiplying all of them by a positive constant
reorders nothing -- so `distance` is pinned at 1.0 and never searched.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.strategy import CALIBRATED_VARIANTS, LookaheadStrategy, Strategy
from scripts.matchStats import BotPool, marginOfError, playMatch

if TYPE_CHECKING:
    from game.player import HalmaPlayer

SAMPLING_BOTS = ["distance", "tipDistance", "shaped", "straggler"]
ANCHORS = ["shaped", "straggler", "distance"]

# Plies between sampled positions in the diagnostic. Consecutive positions are
# near-duplicates, and the reference move costs a ~35 ms search apiece.
STRIDE = 7

# Bounds for the searched weights, log-uniform. Wide on purpose: the point of
# calibrating was that nobody knows what these should be.
WEIGHT_RANGE = (0.01, 10.0)

# Share of candidates kept per round, and how the games per candidate grow to
# match. Four rounds take a 192-vector search from 24 games each to 300 for the
# last three.
SURVIVAL = 0.25
GAME_SCHEDULE = [24, 48, 120, 300]


@dataclass
class Position:
    """One position, reduced to what the agreement diagnostic needs."""

    features: np.ndarray  # (candidates, weights), from Strategy.calibratedFeatures
    anchors: dict[str, np.ndarray]  # bot name -> its score per candidate
    reference: int  # index of the move lookahead2 played


def collect(positions: int, seed: int) -> list[Position]:
    """Sample positions from panel games and reduce each one."""
    scorer = Strategy("calibrated")
    anchorStrategies = {name: Strategy(name) for name in ANCHORS}
    search = LookaheadStrategy()
    collected: list[Position] = []
    for a, b in combinations(SAMPLING_BOTS, 2):
        for i in range(1000):
            game = ComputedGame()
            game.seed(seed + i)
            game.initGame([Computer(1, a), Computer(2, b)])
            for ply in range(game.MAX_MOVES):
                player, board = game.currentPlayer(), game.board
                if ply % STRIDE == 0:
                    moves = board.allValidMovesWithWay(player)
                    if len(moves) > 1:
                        features = []
                        anchors: dict[str, list[float]] = {name: [] for name in ANCHORS}
                        for move in moves:
                            with board.moveApplied(move, player):
                                features.append(scorer.calibratedFeatures(board, player))
                                for name, strategy in anchorStrategies.items():
                                    anchors[name].append(strategy.scoringFunction(board, player))
                        collected.append(
                            Position(
                                np.array(features),
                                {name: np.array(values) for name, values in anchors.items()},
                                moves.index(search.bestMove(moves, board, player)),
                            )
                        )
                        if len(collected) >= positions:
                            return collected
                game.playNextMove(player)
                if game.winner() is not None:
                    break
    return collected


def agreement(sample: list[Position], scores: list[np.ndarray]) -> float:
    """How often the reference move is among the lowest-scoring candidates.

    Among rather than equal to, because `bestMove` breaks ties at random: a
    vector that leaves the search's move tied for best has not disagreed with
    it, and counting that as a miss would punish coarse terms for being coarse.
    """
    hits = sum(
        float(row[position.reference] <= row.min() + 1e-12)
        for position, row in zip(sample, scores, strict=True)
    )
    return hits / len(sample)


def activeTerms(variant: str) -> list[str]:
    """The weights this variant actually uses, `distance` excluded as pinned."""
    weights = CALIBRATED_VARIANTS[variant]
    return [name for name in Strategy.WEIGHT_ORDER[1:] if weights[name] != 0.0]


def randomWeights(count: int, active: list[str], rng: np.random.Generator) -> np.ndarray:
    """Log-uniform vectors over the active terms; everything else stays 0."""
    low, high = (math.log(bound) for bound in WEIGHT_RANGE)
    vectors = np.zeros((count, len(Strategy.WEIGHT_ORDER)))
    vectors[:, 0] = 1.0
    for name in active:
        vectors[:, Strategy.WEIGHT_ORDER.index(name)] = np.exp(rng.uniform(low, high, size=count))
    return vectors


@dataclass(frozen=True)
class Weighted:
    """Seat `variant`, then overwrite its weight vector with the candidate's.

    A frozen dataclass rather than a closure because `rank` hands this to a
    `multiprocessing.Pool` and a lambda cannot be pickled.
    """

    variant: str
    weights: tuple[float, ...]

    def __call__(self, seat: int, _index: int) -> HalmaPlayer:
        player = Computer(seat, self.variant)
        player.strategy.weights = dict(zip(Strategy.WEIGHT_ORDER, self.weights, strict=True))
        return player


def playCandidate(
    weights: list[float], games: int, seed: int, opponents: list[str], variant: str
) -> int:
    """Wins for this weight vector against a pool, over a fixed seed set.

    The pool is cycled by game index rather than played as separate matches, so
    every candidate meets the same opponent on the same seed and the pairing
    holds.
    """
    result = playMatch(Weighted(variant, tuple(weights)), BotPool(tuple(opponents)), games, seed)
    return result.wins


def rank(
    candidates: np.ndarray, games: int, seed: int, opponents: list[str], variant: str, jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Win rate of every candidate, best first; returns (order, rates)."""
    play = partial(playCandidate, games=games, seed=seed, opponents=opponents, variant=variant)
    rows = [row.tolist() for row in candidates]
    if jobs > 1:
        with Pool(jobs) as pool:
            wins = pool.map(play, rows)
    else:
        wins = [play(row) for row in rows]
    rates = np.array(wins) / games
    return rates.argsort()[::-1], rates


def describe(weights: np.ndarray) -> str:
    return "  ".join(
        f"{name}={weight:.3f}" for name, weight in zip(Strategy.WEIGHT_ORDER, weights, strict=True)
    )


def diagnose(positions: int, seed: int) -> None:
    sample = collect(positions, seed)
    print(f"agreement with `lookahead2` over {len(sample)} positions")
    for name in ANCHORS:
        print(f"  {name:<26}{agreement(sample, [p.anchors[name] for p in sample]) * 100:5.1f}%")
    defaults = np.array([CALIBRATED_VARIANTS["calibrated"][n] for n in Strategy.WEIGHT_ORDER])
    scores = [position.features @ defaults for position in sample]
    print(f"  {'calibrated (defaults)':<26}{agreement(sample, scores) * 100:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="calibrated", choices=sorted(CALIBRATED_VARIANTS))
    parser.add_argument("--candidates", type=int, default=192, help="random weight vectors")
    parser.add_argument(
        "--opponentPool",
        nargs="+",
        default=["shaped", "straggler"],
        help="bots the fitness is measured against, cycled by seed",
    )
    parser.add_argument("--positions", type=int, default=200, help="for the agreement diagnostic")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    if args.diagnostic:
        diagnose(args.positions, args.seed)
        print()

    active = activeTerms(args.variant)
    rng = np.random.default_rng(args.seed)
    # The variant's current weights ride along as the control, so anything the
    # search reports has to beat where it started rather than just beat the field.
    control = np.array([CALIBRATED_VARIANTS[args.variant][n] for n in Strategy.WEIGHT_ORDER])
    candidates = np.vstack([control[None, :], randomWeights(args.candidates, active, rng)])

    print(f"fitting `{args.variant}` on {active} against {args.opponentPool}, {args.jobs} jobs")
    for round_, games in enumerate(GAME_SCHEDULE):
        order, rates = rank(
            candidates, games, args.seed, args.opponentPool, args.variant, args.jobs
        )
        last = round_ == len(GAME_SCHEDULE) - 1
        keep = 3 if last else max(2, int(len(candidates) * SURVIVAL))
        print(f"\nround {round_ + 1}: {len(candidates)} vectors x {games} games -> {keep}")
        for index in order[:keep]:
            note = "  (control)" if np.array_equal(candidates[index], control) else ""
            rate = float(rates[index])
            print(
                f"  {rate * 100:5.1f} +/- {marginOfError(rate, games) * 100:4.1f}"
                f"   {describe(candidates[index])}{note}"
            )
        candidates = candidates[order[:keep]]

    best = dict(zip(Strategy.WEIGHT_ORDER, (round(w, 3) for w in candidates[0]), strict=True))
    print(f'\n    "{args.variant}": {best},')
    print("Re-measure on fresh seeds before adopting; the winner's rate is selection-biased.")


if __name__ == "__main__":
    main()
