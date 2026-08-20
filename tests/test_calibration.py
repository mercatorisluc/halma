"""The calibration layer that puts every scoring primitive on [0, 1].

The measured constants themselves are not asserted on -- they are regenerated
by `scripts/calibrateScores.py` whenever a primitive changes. What is locked
here is what the layer promises regardless of the numbers: bounded, monotone,
and defined for every primitive.
"""

import pytest

from heuristics.calibration import (
    PRIMITIVES,
    calibrator,
    logisticCalibrator,
    tableCalibrator,
)


@pytest.fixture(params=PRIMITIVES)
def primitive(request):
    return request.param


def test_every_primitive_has_a_calibration(primitive):
    assert callable(calibrator(primitive))


def test_unknown_primitive_is_rejected():
    with pytest.raises(ValueError, match="no calibration measured"):
        calibrator("noSuchScore")


def test_calibrated_values_stay_in_the_unit_interval(primitive):
    calibrate = calibrator(primitive)
    # Deliberately far outside any range the board can produce, since the
    # calibration is also what the RL side would rely on for bounded inputs.
    for value in (-100.0, -1.0, 0.0, 0.5, 1.0, 5.0, 13.0, 100.0):
        assert 0.0 <= calibrate(value) <= 1.0


def test_calibration_is_monotone(primitive):
    calibrate = calibrator(primitive)
    values = [i / 20 for i in range(-20, 300)]
    calibrated = [calibrate(value) for value in values]
    assert calibrated == sorted(calibrated)


def test_table_reproduces_its_knots_exactly():
    knots, uniforms = [1.0, 2.0, 4.0], [0.1, 0.5, 0.9]
    calibrate = tableCalibrator(knots, uniforms)
    assert [calibrate(knot) for knot in knots] == uniforms


def test_table_interpolates_between_knots():
    calibrate = tableCalibrator([0.0, 1.0], [0.0, 1.0])
    assert calibrate(0.25) == pytest.approx(0.25)


def test_table_flattens_outside_the_observed_range():
    calibrate = tableCalibrator([1.0, 2.0], [0.2, 0.8])
    assert calibrate(-5.0) == pytest.approx(0.2)
    assert calibrate(99.0) == pytest.approx(0.8)


def test_table_survives_a_repeated_knot():
    # A mass point can put two quantiles on one value; the bin has zero width
    # and must not divide by it.
    calibrate = tableCalibrator([1.0, 1.0, 2.0], [0.2, 0.5, 0.8])
    assert 0.0 <= calibrate(1.0) <= 1.0


def test_logistic_is_centred_on_its_mean():
    assert logisticCalibrator(3.0, 1.0)(3.0) == pytest.approx(0.5)


def test_board_primitives_calibrate_at_the_starting_position(board, game):
    player = game.players[0]
    for name in PRIMITIVES:
        value = float(getattr(board, name)(player))
        assert 0.0 <= calibrator(name)(value) <= 1.0
