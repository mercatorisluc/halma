"""Map a raw scoring primitive onto [0, 1], as close to uniform as it gets.

The primitives in `game/board.py` are measurements, not scores on a common
scale: `tipDistanceScore` runs to 12.5 while `jumpPotentialScore` has a
standard deviation of 0.06. Summed with nominally equal weights in
`Strategy.shaped`, the scale therefore *is* the weight, and it was never
chosen -- it fell out of the ad-hoc divisors the primitives carry.

This module removes that accident. Push a random variable through its own
cumulative distribution function and the result is uniform on [0, 1] by
construction, so a calibrated primitive contributes on the same footing as
every other one and the weights in `heuristics/strategy.py` mean what they say.

**One fit: a quantile table.** Knots taken from the measured distribution,
linearly interpolated between. Uniform to within the knot spacing whatever the
shape of the distribution, which is what the discrete primitives need too:
`stragglerTravelScore` takes 13 values and `unfilledTargetScore` 15, so their
knots are those values and each maps to its mid-rank -- the best a discrete
variable admits, since no map can spread 13 values evenly over an interval.

A logistic fit was the obvious cheaper alternative -- the logistic CDF *is* the
sigmoid -- and it was carried here until 2026-08-23 without ever being
selected. Over 57k positions its worst-case deviation from uniform ran 0.044 to
0.194 where the table managed 0.015 to 0.021, so every primitive failed it.
These distributions are simply not bell-shaped: `jumpPotentialScore` is sharply
peaked and `tipDistanceScore` close to flat over its range. Recorded here so
nobody reaches for it again without measuring; git at `d194c5d` has the code.

A calibrated term costs 0.13 us, measured over 2M calls. That is affordable but
not nothing: `shaped` calibrates five terms on top of 14.7 us a candidate
(+5%), where `straggler` calibrates three on top of 1.7 us (+23%), and
`straggler` is the leaf `lookahead2` searches on.

Calibration is not free of consequence. The slope of a quantile map is
1/density, so it spreads differences where positions are dense and compresses
the tails -- and ranking candidate moves depends on exactly those differences.
Recalibrating a primitive gives a different bot, not the same bot rescaled, and
`scripts/baseline.py` is the only thing that can say whether it is a better one.

The constants live in `calibrationData.py`, measured by
`scripts/calibrateScores.py`. The map is a fixed function of the value alone,
never of the position: `LookaheadStrategy` compares nodes at different depths
and would break under anything else.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

from heuristics.calibrationData import QUANTILE_KNOTS

if TYPE_CHECKING:
    from collections.abc import Callable

# The primitives that get calibrated, and the single source of truth for that
# set: scripts/calibrateScores.py samples exactly these. `distance` and
# `targetDistances` on the board are geometry rather than scores, and
# `calculateOpenTargetDistance` is the raw sum behind openTargetDistanceScore.
PRIMITIVES = [
    "tipDistanceScore",
    "openTargetDistanceScore",
    "clusteringScore",
    "stragglerLagScore",
    "stragglerTravelScore",
    "jumpPotentialScore",
    "unfilledTargetScore",
]

# Below this many distinct values a primitive is tabulated at the values
# themselves rather than at quantiles. Nothing subtle about the threshold --
# the measured counts are 12 and 15 for the two discrete primitives and 156 or
# more for every other one, so anything in between separates them.
DISCRETE_MAX_LEVELS = 32


def tableCalibrator(knots: list[float], uniforms: list[float]) -> Callable[[float], float]:
    """Piecewise-linear map through the measured quantiles.

    Values outside the observed range flatten onto the end knots rather than
    extrapolating, so the result stays inside [0, 1] and stays monotone.
    """
    top = len(knots) - 1

    def calibrate(value: float) -> float:
        index = bisect_left(knots, value)
        if index <= 0:
            return uniforms[0]
        if index > top:
            return uniforms[top]
        lower, upper = knots[index - 1], knots[index]
        span = upper - lower
        if span <= 0:
            return uniforms[index]
        return uniforms[index - 1] + (value - lower) / span * (
            uniforms[index] - uniforms[index - 1]
        )

    return calibrate


def calibrator(name: str) -> Callable[[float], float]:
    """The calibration for one primitive, as a closure to call per candidate.

    Built once and held, not looked up per call: scoring a candidate move costs
    1.7 us in `straggler`, so a dict lookup per term is not free.
    """
    if name in QUANTILE_KNOTS:
        knots, uniforms = QUANTILE_KNOTS[name]
        return tableCalibrator(knots, uniforms)
    raise ValueError(f"no calibration measured for {name!r}; known: {sorted(PRIMITIVES)}")
