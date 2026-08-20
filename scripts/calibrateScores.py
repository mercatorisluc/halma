"""Measure what each scoring primitive's distribution actually looks like.

    python -m scripts.calibrateScores --games 40

Plays every pairing of the scoring bots, records each primitive in
`heuristics/calibration.PRIMITIVES` for both players at every ply, fits the two
calibrations that module offers, and writes the one that measures better per
primitive to `heuristics/calibrationData.py`. That file is checked in, so
nothing at runtime depends on this script having been run; regenerate it when a
primitive's definition changes, and re-run `scripts.baseline` afterwards,
because a recalibrated primitive is a different summand rather than a rescaled
one.

The sample is the positions the bots *reach*, not the ones they *evaluate* --
the candidates scored during move selection are those same positions one ply
on, which is the same distribution to within a move. Both players are recorded
at every ply, so there is no side-to-move bias.

`random` is left out: its games never finish, and would fill the sample with
positions no bot ever reaches. `lookahead2` is left out by default for speed
alone, and can be added with --bots.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from game.gameManager import ComputedGame
from game.player import Computer
from heuristics.calibration import (
    DISCRETE_MAX_LEVELS,
    PRIMITIVES,
    logisticCalibrator,
    tableCalibrator,
)

DEFAULT_BOTS = ["distance", "tipDistance", "shaped", "straggler"]
OUTPUT = Path(__file__).resolve().parent.parent / "heuristics" / "calibrationData.py"

# Knots in the quantile table for a continuous primitive. The table is uniform
# to within a bin, so this is the accuracy dial; 33 puts the worst-case error
# near 1/64 while keeping the generated module small enough to read.
KNOTS = 33

# Above this the logistic fit is judged too poor to use and the primitive is
# tabulated instead. A Kolmogorov-Smirnov statistic is the largest gap between
# the calibrated values and a straight uniform, so 0.05 means "no quantile is
# off by more than 5 percentage points".
MAX_LOGISTIC_KS = 0.05


def sample(bots: list[str], games: int, seed: int) -> dict[str, np.ndarray]:
    """Every primitive at every ply of every pairing, keyed by primitive."""
    collected: dict[str, list[float]] = {name: [] for name in PRIMITIVES}
    for a, b in combinations(bots, 2):
        for i in range(games):
            game = ComputedGame()
            game.seed(seed + i)
            game.initGame([Computer(1, a), Computer(2, b)])
            for _ in range(game.MAX_MOVES):
                for player in game.players:
                    for name in PRIMITIVES:
                        collected[name].append(float(getattr(game.board, name)(player)))
                game.playNextMove(game.currentPlayer())
                if game.winner() is not None:
                    break
    return {name: np.array(values) for name, values in collected.items()}


def kolmogorovSmirnov(uniforms: np.ndarray) -> float:
    """How far the calibrated values are from uniform, at their worst point."""
    ordered = np.sort(uniforms)
    n = len(ordered)
    above = np.arange(1, n + 1) / n - ordered
    below = ordered - np.arange(n) / n
    return float(max(above.max(), below.max()))


def bestPossibleKs(values: np.ndarray) -> float:
    """The KS no calibration of this primitive can beat.

    Every position sharing a value has to land on the same point of [0, 1], so
    the heaviest value alone forces a gap of half its share. Negligible for the
    fine-grained primitives, and the whole story for the two coarse ones -- it
    is the difference between a fit that failed and a fit that hit the ceiling.
    """
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max()) / (2 * len(values))


def discreteKnots(values: np.ndarray) -> tuple[list[float], list[float]]:
    """Each distinct value, and the middle of the interval it occupies.

    Mid-rank rather than the cumulative frequency itself, because a discrete
    variable cannot be made uniform: sending a level to the *middle* of its
    share splits the unavoidable error evenly instead of piling it on one side.
    """
    levels, counts = np.unique(values, return_counts=True)
    uniforms = (np.cumsum(counts) - counts / 2) / len(values)
    return [round(float(v), 6) for v in levels], [round(float(u), 6) for u in uniforms]


def continuousKnots(values: np.ndarray) -> tuple[list[float], list[float]]:
    """Evenly spaced quantiles, deduplicated.

    A skewed sample can put several quantiles on the same value; keeping both
    would leave a zero-width bin, so repeated knots collapse to one and take
    the middle of the range they covered -- the same mid-rank argument as for
    the discrete case.
    """
    probabilities = np.linspace(0.0, 1.0, KNOTS)
    quantiles = np.quantile(values, probabilities)
    knots: list[float] = []
    uniforms: list[float] = []
    for knot in np.unique(quantiles):
        matching = probabilities[quantiles == knot]
        knots.append(round(float(knot), 6))
        uniforms.append(round(float(matching.mean()), 6))
    return knots, uniforms


def fits(values: np.ndarray) -> tuple[str, float, float]:
    """Choose between the two calibrations, and report what both scored."""
    knots, uniforms = knotsFor(values)
    calibrateTable = tableCalibrator(knots, uniforms)
    calibrateLogistic = logisticCalibrator(float(values.mean()), float(values.std()))
    tableKs = kolmogorovSmirnov(np.array([calibrateTable(v) for v in values]))
    logisticKs = kolmogorovSmirnov(np.array([calibrateLogistic(v) for v in values]))
    choice = "logistic" if logisticKs <= MAX_LOGISTIC_KS and logisticKs <= tableKs else "table"
    return choice, tableKs, logisticKs


def knotsFor(values: np.ndarray) -> tuple[list[float], list[float]]:
    if len(np.unique(values)) <= DISCRETE_MAX_LEVELS:
        return discreteKnots(values)
    return continuousKnots(values)


def render(
    results: dict[str, tuple[str, np.ndarray]], bots: list[str], games: int, seed: int
) -> str:
    """The generated module, as source."""
    logistic = {
        name: (round(float(values.mean()), 6), round(float(values.std()), 6))
        for name, (choice, values) in results.items()
        if choice == "logistic"
    }
    tables = {
        name: knotsFor(values) for name, (choice, values) in results.items() if choice == "table"
    }
    sampleSize = len(next(iter(results.values()))[1])
    return f'''"""Generated by `python -m scripts.calibrateScores` -- do not edit by hand.

Measured {datetime.now(UTC).date()} over {games} games of every pairing of
{", ".join(bots)} at seed {seed}: {sampleSize} positions. See
`heuristics/calibration.py` for what the two fits mean and why a primitive gets
one rather than the other.
"""

from __future__ import annotations

# primitive -> (mean, standard deviation) of the raw values.
LOGISTIC_FIT: dict[str, tuple[float, float]] = {logistic!r}

# primitive -> (knots, the point in [0, 1] each knot maps to).
QUANTILE_KNOTS: dict[str, tuple[list[float], list[float]]] = {tables!r}
'''


def report(
    results: dict[str, tuple[str, np.ndarray]], diagnostics: dict[str, tuple[float, float]]
) -> None:
    header = (
        f"{'primitive':<26}{'min':>8}{'max':>8}{'mean':>8}{'std':>8}"
        f"{'values':>8}{'KS tbl':>8}{'KS log':>8}{'KS min':>8}  fit"
    )
    print(header)
    print("-" * len(header))
    for name, (choice, values) in results.items():
        tableKs, logisticKs = diagnostics[name]
        print(
            f"{name:<26}{values.min():>8.3f}{values.max():>8.3f}{values.mean():>8.3f}"
            f"{values.std():>8.3f}{len(np.unique(values)):>8}"
            f"{tableKs:>8.3f}{logisticKs:>8.3f}{bestPossibleKs(values):>8.3f}  {choice}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40, help="games per pairing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bots", nargs="+", default=DEFAULT_BOTS, help="bots to sample from")
    parser.add_argument("--dryRun", action="store_true", help="report without writing the module")
    args = parser.parse_args()

    samples = sample(args.bots, args.games, args.seed)
    results: dict[str, tuple[str, np.ndarray]] = {}
    diagnostics: dict[str, tuple[float, float]] = {}
    for name, values in samples.items():
        choice, tableKs, logisticKs = fits(values)
        results[name] = (choice, values)
        diagnostics[name] = (tableKs, logisticKs)

    report(results, diagnostics)
    if args.dryRun:
        return
    OUTPUT.write_text(render(results, args.bots, args.games, args.seed))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
