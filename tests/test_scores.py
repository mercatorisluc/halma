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


# ``player.distanceScore`` is maintained incrementally by
# updatePlayerDistanceScore instead of being recomputed. Two invariants make
# that safe, and both are relied on elsewhere: Strategy.bestMove scores a move
# by applying and then reversing it, and the playback controller steps the
# history backwards the same way.


def test_distance_score_is_restored_by_reversing_a_move(board, player):
    before = player.distanceScore
    for move in sorted(board.allValidMoves(player)):
        board.applyMoveForPlayer(move, player)
        board.applyMoveForPlayer((move[-1], move[0]), player)
        assert player.distanceScore == pytest.approx(before)


def test_incremental_distance_score_matches_a_full_recomputation(game):
    # Play a whole game so pieces actually reach their target base, which is
    # what exercises the correction branches in updatePlayerDistanceScore.
    game.play()
    for player in game.players:
        recomputed = game.board.calculatePlayerDistanceScore(player)
        assert player.distanceScore == pytest.approx(recomputed)
