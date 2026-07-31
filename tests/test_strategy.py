"""Characterization tests for heuristics/strategy.py's Strategy class."""

import pytest

from heuristics.strategy import STRATEGY_NAMES, LookaheadStrategy, Strategy, makeStrategy


@pytest.fixture
def player(game):
    return game.players[0]


def test_advanced_dist_score_dispatches_to_the_advanced_formula(board, player):
    strategy = Strategy("advancedDistScore")
    expected = (board.advancedDistanceScore(player) + board.homeBonusScore(player)) / 2
    assert strategy.scoringFunction(board, player) == pytest.approx(expected)


def test_simple_dist_score_dispatches_to_the_simple_formula(board, player):
    strategy = Strategy("simpleDistScore")
    expected = (board.simpleDistanceScore(player) + board.homeBonusScore(player)) / 2
    assert strategy.scoringFunction(board, player) == pytest.approx(expected)


def test_sparsity_score_dispatches_to_the_combined_formula(board, player):
    strategy = Strategy("sparsityScore")
    expected = (
        board.advancedDistanceScore(player)
        + board.sparsityScore(player)
        + board.playerSparsityScore(player)
        + board.potentialJumpScore(player)
        + board.homeBonusScore(player)
    )
    assert strategy.scoringFunction(board, player) == pytest.approx(expected)


def test_bottleneck_dispatches_to_advanced_dist_plus_the_straggler(board, player):
    strategy = Strategy("bottleneck")
    expected = (
        board.advancedDistanceScore(player) + board.homeBonusScore(player)
    ) / 2 + Strategy.BOTTLENECK_WEIGHT * board.bottleneckScore(player)
    assert strategy.scoringFunction(board, player) == pytest.approx(expected)


def test_every_registered_strategy_can_be_built_and_scored(board, player):
    # SCORERS is the single source of truth; STRATEGY_NAMES adds the search on
    # top. Anything listed must actually work, or baseline.py breaks mid-run.
    for name in STRATEGY_NAMES:
        strategy = makeStrategy(name)
        assert strategy.strategyName == name
        if name in Strategy.SCORERS:
            assert isinstance(strategy.scoringFunction(board, player), float)


def test_unknown_strategy_is_rejected_at_construction():
    # Previously this surfaced as a bare KeyError on the first scoring call,
    # deep inside a running game.
    with pytest.raises(ValueError, match="unknown strategy"):
        makeStrategy("noSuchStrategy")


def test_lookahead_searches_and_leaves_the_board_untouched(board, game):
    # The search nests moveApplied for TWO different players -- ours, then the
    # opponent's reply inside it. Each undo has to be an exact inverse and the
    # blocks have to unwind in order, for both players' derived state.
    player = game.players[0]
    strategy = makeStrategy(LookaheadStrategy.NAME)
    assert isinstance(strategy, LookaheadStrategy)

    def snapshot():
        return {
            p.identifier: (
                set(p.positions),
                set(p.nonArrived),
                set(p.openEndPositions),
                p.distanceScore,
            )
            for p in game.players
        }

    beforeBoard = board.boardState().copy()
    before = snapshot()

    chosen = strategy.bestMove(board.allValidMovesWithWay(player), board, player)

    assert chosen in board.allValidMovesWithWay(player)
    assert (board.boardState() == beforeBoard).all()
    assert snapshot() == before


def test_seating_gives_every_player_its_opponents(game):
    # The searching strategy needs to score the opposition, and the board holds
    # no player list.
    for player in game.players:
        assert player.opponents == [p for p in game.players if p is not player]


def test_random_strategy_scores_every_position_equally(board, player):
    assert Strategy("random").scoringFunction(board, player) == 1


def test_best_move_leaves_board_and_player_state_unchanged(board, game):
    # bestMove must try each candidate move and undo it, not just the winner.
    player = game.players[0]
    moves = board.allValidMoves(player)
    beforeBoard = board.boardState().copy()
    beforePositions = set(player.positions)

    Strategy("advancedDistScore").bestMove(moves, board, player)

    assert (board.boardState() == beforeBoard).all()
    assert player.positions == beforePositions


def test_board_is_restored_when_scoring_raises(board, game):
    # bestMove scores candidates by mutating the real board and undoing the
    # move afterwards. If a scoring function raises, the undo must still run --
    # otherwise the half-evaluated move stays applied to the live game.
    player = game.players[0]
    before = board.boardState().copy()
    beforePositions = set(player.positions)
    beforeScore = player.distanceScore

    strategy = _RaisingStrategy("advancedDistScore")

    with pytest.raises(ValueError):
        strategy.bestMove(sorted(board.allValidMoves(player)), board, player)

    assert (board.boardState() == before).all()
    assert player.positions == beforePositions
    assert player.distanceScore == pytest.approx(beforeScore)


class _RaisingStrategy(Strategy):
    """Stands in for a scoring function that blows up mid-evaluation."""

    def scoringFunction(self, board, player):
        raise ValueError("scoring blew up")


def test_random_strategy_returns_one_of_the_offered_moves(board, game):
    player = game.players[0]
    moves = board.allValidMoves(player)
    chosen = Strategy("random").bestMove(moves, board, player)
    assert chosen in set(moves)


def test_best_move_selects_a_minimum_scoring_move(board, game):
    player = game.players[0]
    moves = sorted(board.allValidMoves(player))
    strategy = Strategy("simpleDistScore")

    scores = {}
    for move in moves:
        board.applyMoveForPlayer(move, player)
        scores[move] = strategy.scoringFunction(board, player)
        board.applyMoveForPlayer((move[-1], move[0]), player)

    chosen = strategy.bestMove(moves, board, player)
    assert scores[chosen] == pytest.approx(min(scores.values()))
