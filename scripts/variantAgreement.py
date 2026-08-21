"""Measure whether the calibrated family actually plays differently.

    python -m scripts.variantAgreement

The family exists to be an *opponent pool*, not a ladder. A pool whose members
pick the same move nine times in ten teaches an agent to beat one opponent
wearing five hats, however well each hat scores on its own. So the number that
decides pool membership is not a win rate -- `scripts/baseline.py` already
reports those -- but pairwise move agreement over a shared position set.

The measurement is cheap for the reason `scripts/fitWeights.py` documents: a
candidate's features do not depend on the weights, so one position reduces to a
small matrix and every bot after that costs a matrix-vector multiply. Only the
non-calibrated anchors have to be scored the slow way, one call per candidate.

**Read the phase split, not just the headline.** `calibratedFeatures` multiplies
all three shape terms by `unfilledTargetScore`, which falls to 0 as pieces
arrive, so late in a game every member of the family *is* the same bot by
construction -- the terms that distinguish them have faded out. Averaging over
whole games therefore reports agreement that is high for a reason which has
nothing to do with the weights.

**The midgame is the discriminating bucket**, which is not where the fade
argument alone would put it. Measured over 1500 positions on 2026-08-21, the
family agrees *more* early than in the middle -- `calibratedCluster` against
`calibrated` runs 65.1% early, 37.0% in the middle, 58.9% late. Early the
shape terms have little to disagree about, because every candidate is some way
of advancing a packed home triangle; late the fade has switched them off. Only
in between are the pieces spread out enough for clustering, lag and jump
potential to point in different directions. Judge pool membership there.

Agreement counts a shared *best set* rather than an identical move, matching
`fitWeights.agreement`: `bestMove` breaks ties at random, so two bots that leave
the same move tied for best have not disagreed, and counting that as a
difference would credit coarse terms with diversity they do not have.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.strategy import CALIBRATED_VARIANTS, Strategy

# Bots whose games supply the positions. The family plays itself: these are the
# positions a pool made of them would actually put an agent in, which is not
# the same distribution the old panel produces.
SAMPLING_BOTS = [*CALIBRATED_VARIANTS]

# Non-calibrated bots to compare the family against. A variant that merely
# re-derives `straggler` adds nothing to a pool that could hold `straggler`
# itself, and that is invisible in a family-only comparison.
ANCHORS = ["distance", "shaped", "straggler"]

# Plies between sampled positions. Consecutive positions are near-duplicates,
# and agreement measured over them would mostly count the board not moving.
STRIDE = 7

# Ties are compared against the best score with this slack. The scores are sums
# of calibrated terms in [0, 1] with weights near 1, so absolute slack is safe.
TIE = 1e-12


@dataclass
class Position:
    """One position, reduced to what the agreement measurement needs."""

    features: np.ndarray  # (candidates, weights), from Strategy.calibratedFeatures
    anchors: dict[str, np.ndarray]  # bot name -> its score per candidate
    ply: int


def collect(positions: int, seed: int, samplingBots: list[str]) -> list[Position]:
    """Sample positions from games between `samplingBots` and reduce each one."""
    scorer = Strategy("calibrated")
    anchorStrategies = {name: Strategy(name) for name in ANCHORS}
    collected: list[Position] = []
    for a, b in combinations(samplingBots, 2):
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
                                ply,
                            )
                        )
                        if len(collected) >= positions:
                            return collected
                game.playNextMove(player)
                if game.winner() is not None:
                    break
    return collected


def bestSets(sample: list[Position], scores: list[np.ndarray]) -> list[np.ndarray]:
    """Per position, the boolean mask of candidates tied for best. Lower is better."""
    return [row <= row.min() + TIE for row in scores]


def scoreAll(sample: list[Position], bots: list[str]) -> dict[str, list[np.ndarray]]:
    """Every bot's score per candidate, calibrated ones by one multiply each."""
    scores: dict[str, list[np.ndarray]] = {}
    for name in bots:
        if name in CALIBRATED_VARIANTS:
            weights = np.array([CALIBRATED_VARIANTS[name][n] for n in Strategy.WEIGHT_ORDER])
            scores[name] = [position.features @ weights for position in sample]
        else:
            scores[name] = [position.anchors[name] for position in sample]
    return scores


def agreementMatrix(masks: dict[str, list[np.ndarray]], keep: list[int]) -> np.ndarray:
    """Pairwise share of the kept positions where two bots' best sets intersect."""
    names = list(masks)
    matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            hits = sum(bool((masks[a][k] & masks[b][k]).any()) for k in keep)
            matrix[i, j] = hits / len(keep) if keep else float("nan")
    return matrix


def tieSizes(masks: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """Mean number of candidates tied for best, per bot.

    The guard on the whole measurement. Agreement is scored as an intersection
    of best sets, so a bot that leaves ten moves tied intersects with everyone
    and would look like a near-duplicate of bots it has nothing in common with.
    A coarse, ties-heavy scorer next to a smooth one is exactly the case where
    the headline number lies, and `distance` sums two terms that can come out
    integral. Read any high agreement against these.
    """
    return {name: float(np.mean([mask.sum() for mask in rows])) for name, rows in masks.items()}


def report(title: str, names: list[str], matrix: np.ndarray) -> None:
    width = max(len(name) for name in names) + 2
    print(f"\n{title}")
    print(" " * width + "".join(f"{name[:7]:>8}" for name in names))
    for i, name in enumerate(names):
        cells = "".join(f"{matrix[i, j] * 100:7.1f}%" for j in range(len(names)))
        print(f"{name:<{width}}{cells}")


def phases(sample: list[Position]) -> list[tuple[str, list[int]]]:
    """Split the positions into three by ply, at the terciles of what was
    collected. Absolute ply thresholds would not survive a change of bots."""
    plies = np.array([position.ply for position in sample])
    low, high = np.quantile(plies, [1 / 3, 2 / 3])
    buckets = [
        (f"early (ply < {low:.0f})", [i for i, p in enumerate(plies) if p < low]),
        (f"middle (ply {low:.0f}-{high:.0f})", [i for i, p in enumerate(plies) if low <= p < high]),
        (f"late (ply >= {high:.0f})", [i for i, p in enumerate(plies) if p >= high]),
    ]
    return [(label, keep) for label, keep in buckets if keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--samplingBots",
        nargs="+",
        default=SAMPLING_BOTS,
        help="bots whose games supply the positions",
    )
    parser.add_argument(
        "--bots",
        nargs="+",
        default=[*CALIBRATED_VARIANTS, *ANCHORS],
        help="bots to compare; calibrated ones cost a multiply, others a full scoring pass",
    )
    args = parser.parse_args()

    print(f"sampling {args.positions} positions from games between {args.samplingBots}")
    sample = collect(args.positions, args.seed, args.samplingBots)
    candidateCount = float(np.mean([len(position.features) for position in sample]))
    print(f"collected {len(sample)}, {candidateCount:.0f} candidates each")

    masks = {name: bestSets(sample, rows) for name, rows in scoreAll(sample, args.bots).items()}
    names = list(masks)
    print("\nmean candidates tied for best (a large one inflates every row below)")
    for name, size in tieSizes(masks).items():
        print(f"  {name:<20}{size:5.2f}")
    report("agreement over all positions", names, agreementMatrix(masks, list(range(len(sample)))))
    for label, keep in phases(sample):
        report(f"{label}, {len(keep)} positions", names, agreementMatrix(masks, keep))

    print(
        "\nJudge pool membership on the middle bucket. Early the candidates are all"
        "\nways of advancing a packed triangle and late the fade has switched the"
        "\nshape terms off, so both ends agree for reasons unrelated to the weights."
    )


if __name__ == "__main__":
    main()
