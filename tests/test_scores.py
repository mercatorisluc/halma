"""Characterization tests for the heuristic scoring functions.

These lock the exact scores of the fixed starting position. They are not a
statement of "correct" values, only a tripwire: if a refactor changes the
numbers the bots rely on, these tests catch it.
"""

import pytest


@pytest.fixture
def player(game):
    return game.players[0]


def test_simple_distance_score(board, player):
    assert board.simpleDistanceScore(player) == pytest.approx(12.5)


def test_advanced_distance_score(board, player):
    assert board.advancedDistanceScore(player) == pytest.approx(6.694444444444445)


def test_sparsity_score(board, player):
    assert board.sparsityScore(player) == pytest.approx(0.30666666666666664)


def test_home_bonus_score_is_one_before_any_piece_arrives(board, player):
    assert board.homeBonusScore(player) == pytest.approx(1.0)


def test_potential_jump_score(board, player):
    assert board.potentialJumpScore(player) == pytest.approx(0.8400000000000001)
