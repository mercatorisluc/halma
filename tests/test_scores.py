"""Characterization tests for the heuristic scoring functions.

These lock the exact scores of the fixed starting position. They are not a
statement of "correct" values, only a tripwire: if a refactor changes the
numbers the bots rely on, these tests catch it.
"""

import pytest

from game.gameManager import ComputedGame
from game.player import Computer


@pytest.fixture
def player(game):
    return game.players[0]


def test_simple_distance_score(board, player):
    assert board.tipDistanceScore(player) == pytest.approx(12.5)


def test_player_distance_score(board, player):
    assert board.openTargetDistanceScore(player) == pytest.approx(0.8888888888888888)


def test_sparsity_score(board, player):
    assert board.clusteringScore(player) == pytest.approx(0.30666666666666664)


def test_home_bonus_score_is_one_before_any_piece_arrives(board, player):
    assert board.unfilledTargetScore(player) == pytest.approx(1.0)


def test_bottleneck_score(board, player):
    # At the start every piece is 12 steps from the nearest free target field.
    assert board.stragglerTravelScore(player) == pytest.approx(12.0)


def test_bottleneck_score_is_zero_once_everything_has_arrived(board, game):
    player = game.players[0]
    for field in board.fields:
        if field.playerID == player.identifier:
            field.removePlayer()
    for target in player.endPositions:
        board.fields[target].playerID = player.identifier
    player.positions = set(player.endPositions)
    player.nonArrived = set()
    player.openEndPositions = set()
    assert board.stragglerTravelScore(player) == 0.0


def test_bottleneck_score_is_the_worst_piece_not_the_average(board, game):
    # Independently recomputed: the largest, over pieces still on their way, of
    # the distance to the nearest free target. The distinction from a mean is
    # the whole point -- one straggler decides when the game ends.
    player = game.players[0]
    game.playNextMove(player)
    perPiece = [
        min(board.distanceMatrix[piece][target] for target in player.openEndPositions)
        for piece in player.nonArrived
    ]
    assert board.stragglerTravelScore(player) == pytest.approx(max(perPiece))
    assert max(perPiece) > sum(perPiece) / len(perPiece), "expected a spread to distinguish them"


def test_bottleneck_score_matches_the_literal_formula_all_game_long():
    # stragglerTravelScore seeds a bound and stops the inner scan early -- both
    # shortcuts are exact, but neither is obvious, and a wrong bound would show
    # up as a slightly-too-small value in midgame positions only. So compare it
    # against the literal max-over-mins for every position of a real game,
    # rather than trusting the three fixed positions above.
    game = ComputedGame()
    game.seed(7)
    game.initGame([Computer(1, "straggler"), Computer(2, "distance")])
    board = game.board
    checked = 0
    for _ in range(game.MAX_MOVES):
        if game.winner() is not None:
            break
        game.playNextMove(game.currentPlayer())
        for player in game.players:
            if not player.nonArrived or not player.openEndPositions:
                continue
            literal = max(
                min(board.distanceMatrix[piece][target] for target in player.openEndPositions)
                for piece in player.nonArrived
            )
            assert board.stragglerTravelScore(player) == pytest.approx(literal)
            checked += 1
    assert checked > 100, f"expected a full game's worth of positions, got {checked}"


def test_potential_jump_score(board, player):
    assert board.jumpPotentialScore(player) == pytest.approx(0.8400000000000001)


# ``player.distanceScore`` is maintained incrementally by
# updateOpenTargetDistance instead of being recomputed. Two invariants make
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
    # what exercises the correction branches in updateOpenTargetDistance.
    game.play()
    for player in game.players:
        recomputed = game.board.calculateOpenTargetDistance(player)
        assert player.distanceScore == pytest.approx(recomputed)
