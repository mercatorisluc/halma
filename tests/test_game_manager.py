"""Characterization tests for HalmaGame/ComputedGame/InteractiveGame in game/gameManager.py."""

from game.gameManager import ComputedGame, InteractiveGame
from game.player import Computer, HumanPlayer


def test_computed_game_seats_two_bots_with_their_strategies():
    game = ComputedGame()
    game.initStandardGame()
    assert [type(p) for p in game.players] == [Computer, Computer]
    assert game.players[0].strategy.strategyName == "advancedDistScore"
    assert game.players[1].strategy.strategyName == "sparsityScore"
    assert all(not p.isHuman() for p in game.players)


def test_computed_game_is_not_a_human_game():
    game = ComputedGame()
    game.initStandardGame()
    assert game.isHumanGame() is False


def test_interactive_game_seats_a_human_and_a_bot():
    game = InteractiveGame()
    game.initStandardGame()
    assert isinstance(game.players[0], HumanPlayer)
    assert isinstance(game.players[1], Computer)
    assert game.players[0].isHuman() is True
    assert game.players[1].isHuman() is False


def test_interactive_game_is_a_human_game():
    game = InteractiveGame()
    game.initStandardGame()
    assert game.isHumanGame() is True


def _assert_three_players_are_seated_without_overlap(game):
    assert len(game.players) == 3
    homeBases = [p.homeBase for p in game.players]
    assert len(set(homeBases)) == 3
    allPositions = set()
    for player in game.players:
        assert len(player.positions) == 15
        allPositions |= player.positions
    # No two players start on the same field.
    assert len(allPositions) == 45


def test_computed_three_player_game_seats_three_bots_without_overlap():
    game = ComputedGame()
    game.init3PlayerGame()
    _assert_three_players_are_seated_without_overlap(game)
    assert all(isinstance(p, Computer) for p in game.players)


def test_interactive_three_player_game_seats_a_human_among_two_bots():
    game = InteractiveGame()
    game.init3PlayerGame()
    _assert_three_players_are_seated_without_overlap(game)
    assert sum(1 for p in game.players if isinstance(p, HumanPlayer)) == 1


def test_play_runs_a_full_bot_game_to_a_winner(game):
    # Integration test: exercises getNextMove -> Computer.chooseMove ->
    # Strategy.bestMove end to end, not just the pre-recorded moves the other
    # tests apply directly.
    result = game.play()
    assert result == game.winner()
    assert result in (1, 2)
    assert 0 < game.gameLength() <= game.MAX_MOVES


def test_reset_can_be_called_repeatedly(game):
    # The Initializer instance is reused across resets, so anything it
    # accumulates per call (its field list) must be cleared each time. The RL
    # env resets once per episode, so a leak here corrupts every field id.
    game.play()
    for _ in range(3):
        game.reset()
        assert len(game.initializer.fields) == 121
        assert len(game.board.fields) == 121
        # The board and its mapper are reused rather than rebuilt, so they must
        # be overwritten in place, never appended to.
        assert len(game.board.fieldPositionsMapper.coordById) == 121
        assert len(game.board.fieldPositionsMapper.idByCoord) == 121
        assert game.gameLength() == 0
        assert game.winner() is None
        assert all(len(p.positions) == 15 for p in game.players)


def test_reset_returns_a_playable_game(game):
    game.reset()
    player = game.currentPlayer()
    assert len(game.board.allValidMoves(player)) > 0


def test_current_player_rotates_through_full_play_order():
    # 3 players so the rotation period is distinguishable from a coin flip.
    game = ComputedGame()
    game.init3PlayerGame()
    seenIdentifiers = []
    for _ in range(len(game.playOrder)):
        player = game.currentPlayer()
        seenIdentifiers.append(player.identifier)
        move = min(game.board.allValidMoves(player))
        game.playMove(player, move)
    assert seenIdentifiers == [p.identifier for p in game.playOrder]
