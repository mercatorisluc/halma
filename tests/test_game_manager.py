"""Characterization tests for HalmaGame/ComputedGame/InteractiveGame in game/gameManager.py."""

import pytest

from game.gameManager import ComputedGame, InteractiveGame
from game.player import Computer, HumanPlayer


@pytest.mark.parametrize("gameClass", [ComputedGame, InteractiveGame])
@pytest.mark.parametrize("setUp", ["initStandardGame", "init3PlayerGame"])
def test_player_identifiers_are_seat_numbers(gameClass, setUp):
    # Identifiers must be ints numbered from 1, never names, and never 0 --
    # 0 is what an empty field holds. boardState() gathers them into a numpy
    # array that the RL code does arithmetic on, so a string identifier would
    # make it a string array where every empty field compares unequal to 0 and
    # reads as an opponent piece.
    game = gameClass()
    getattr(game, setUp)()
    identifiers = [p.identifier for p in game.players]
    assert identifiers == list(range(1, len(game.players) + 1))

    boardState = game.board.boardState()
    assert boardState.dtype.kind == "i"
    assert int((boardState != 0).sum()) == 15 * len(game.players)
    for player in game.players:
        assert int((boardState == player.identifier).sum()) == 15


def test_computed_game_seats_two_bots_with_their_strategies():
    game = ComputedGame()
    game.initStandardGame()
    bots = game.players
    assert [type(p) for p in bots] == [Computer, Computer]
    assert all(isinstance(p, Computer) for p in bots)
    strategyNames = [p.strategy.strategyName for p in bots if isinstance(p, Computer)]
    assert strategyNames == ["advancedDistScore", "sparsityScore"]
    assert all(not p.isHuman() for p in bots)


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
        assert len(game.board.fields) == 121
        # The board is reused rather than rebuilt, so its fields and coordinate
        # index must be replaced wholesale, never appended to.
        assert [f.id for f in game.board.fields] == list(range(121))
        assert len(game.board.idByCoord) == 121
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
