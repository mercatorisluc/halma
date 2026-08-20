"""Ask which scoring primitives are saying the same thing.

    python -m scripts.scoreCorrelation --games 8

Two matrices, because there are two different questions and only the second one
is about the bots:

- **Over the game.** How the primitives move as a game progresses. Expect this
  to be high everywhere and read it with suspicion: every primitive drifts with
  the game's progress, so a pair can correlate at 0.9 here while disagreeing
  about every move on the board. It measures the drift, not the redundancy.
- **Within a position.** How the primitives rank the *candidate moves* of one
  position, averaged over positions. This is the one that decides whether a term
  can be dropped, because ranking candidates is the only thing a scorer does.

The same run reports how much of a vote each primitive has: its spread across
the candidates of a position, on the calibrated scale so the numbers compare,
and how often it is flat across every candidate and so cannot affect the choice
at all.

Correlations are Spearman -- rank based, so a primitive being on a different
scale or a nonlinear one does not register as disagreement. Ties share the
average rank, which matters here: the coarse primitives are mostly ties.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.calibration import PRIMITIVES, calibrator

DEFAULT_BOTS = ["distance", "tipDistance", "shaped", "straggler"]

# Sample the candidate set every so many plies rather than at each one. Scoring
# every legal move under every primitive is ~60x the cost of scoring the
# position itself, and consecutive plies are near-duplicates anyway.
STRIDE = 5


def ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which the coarse primitives depend on."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        result[order[start : stop + 1]] = (start + stop) / 2
        start = stop + 1
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, or NaN if either side is constant and has no ranking."""
    rankedA, rankedB = ranks(a), ranks(b)
    if rankedA.std() == 0 or rankedB.std() == 0:
        return float("nan")
    return float(np.corrcoef(rankedA, rankedB)[0, 1])


def collect(bots: list[str], games: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Primitives at every ply, and per sampled position over its candidates.

    Returns `(positions, candidateSets)`: an array of shape (plies, primitives)
    and a list of (candidates, primitives) arrays.
    """
    positions: list[list[float]] = []
    candidateSets: list[np.ndarray] = []
    for a, b in combinations(bots, 2):
        for i in range(games):
            game = ComputedGame()
            game.seed(seed + i)
            game.initGame([Computer(1, a), Computer(2, b)])
            for ply in range(game.MAX_MOVES):
                player = game.currentPlayer()
                positions.append([float(getattr(game.board, name)(player)) for name in PRIMITIVES])
                if ply % STRIDE == 0:
                    moves = game.board.allValidMovesWithWay(player)
                    scored = []
                    for move in moves:
                        with game.board.moveApplied(move, player):
                            scored.append(
                                [float(getattr(game.board, name)(player)) for name in PRIMITIVES]
                            )
                    if len(scored) > 1:
                        candidateSets.append(np.array(scored))
                game.playNextMove(player)
                if game.winner() is not None:
                    break
    return np.array(positions), candidateSets


def correlationMatrix(values: np.ndarray) -> np.ndarray:
    """Spearman between every pair of primitives, over one sample."""
    size = len(PRIMITIVES)
    matrix = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            matrix[i, j] = spearman(values[:, i], values[:, j])
    return matrix


def meanCorrelationMatrix(candidateSets: list[np.ndarray]) -> np.ndarray:
    """Spearman within each position's candidates, averaged over positions.

    Positions where a primitive is flat contribute nothing to its pairs rather
    than counting as disagreement -- there is no ranking to compare against.
    """
    size = len(PRIMITIVES)
    total = np.zeros((size, size))
    counted = np.zeros((size, size))
    for candidates in candidateSets:
        matrix = correlationMatrix(candidates)
        usable = ~np.isnan(matrix)
        total[usable] += matrix[usable]
        counted[usable] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counted > 0, total / np.maximum(counted, 1), np.nan)


def votes(candidateSets: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """Per primitive: calibrated spread across candidates, and how often flat."""
    calibrators = [calibrator(name) for name in PRIMITIVES]
    spreads: list[list[float]] = [[] for _ in PRIMITIVES]
    flat = [0] * len(PRIMITIVES)
    for candidates in candidateSets:
        for index, calibrate in enumerate(calibrators):
            column = np.array([calibrate(value) for value in candidates[:, index]])
            spreads[index].append(float(column.std()))
            if column.std() == 0:
                flat[index] += 1
    return (
        [float(np.mean(values)) for values in spreads],
        [count / len(candidateSets) for count in flat],
    )


def printMatrix(title: str, matrix: np.ndarray) -> None:
    print(f"\n{title}")
    labels = [name.replace("Score", "")[:11] for name in PRIMITIVES]
    print(f"{'':<26}" + "".join(f"{label:>12}" for label in labels))
    for i, name in enumerate(PRIMITIVES):
        cells = "".join(
            "           -" if j >= i else f"{matrix[i, j]:>12.2f}" for j in range(len(PRIMITIVES))
        )
        print(f"{name:<26}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8, help="games per pairing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bots", nargs="+", default=DEFAULT_BOTS)
    args = parser.parse_args()

    positions, candidateSets = collect(args.bots, args.games, args.seed)
    print(f"{len(positions)} positions, {len(candidateSets)} candidate sets")

    spreads, flatness = votes(candidateSets)
    print(f"\n{'primitive':<26}{'spread':>10}{'flat':>10}")
    print("-" * 46)
    for name, spread, flatShare in zip(PRIMITIVES, spreads, flatness, strict=True):
        print(f"{name:<26}{spread:>10.4f}{flatShare * 100:>9.1f}%")

    printMatrix("Spearman over the game (drift, not redundancy):", correlationMatrix(positions))
    printMatrix(
        "Spearman within a position (what ranks moves):", meanCorrelationMatrix(candidateSets)
    )


if __name__ == "__main__":
    main()
