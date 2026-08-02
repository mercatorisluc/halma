"""Characterization tests for game and player initialisation."""

from game.gameManager import ComputedGame
from game.player import Computer


def test_reseating_a_player_object_does_not_leak_its_previous_positions():
    """Every existing caller constructs a fresh player per game, so this bug
    was invisible until something (scripts/compareCheckpoints.py, to avoid
    reloading a checkpoint from disk for every game) reused player objects
    across games -- prepareForGameStart used to *update* positions rather
    than reset it, so a player kept whatever it hadn't yet moved home from a
    previous game, and move generation (board.py reads player.positions
    directly) would then read phantom pieces the fresh board never placed."""
    agent = Computer(1, "advancedDistScore")
    opponent = Computer(2, "advancedDistScore")

    first = ComputedGame()
    first.seed(0)
    first.initGame([agent, opponent])
    first.play()
    assert agent.positions != agent.startPositions  # actually moved somewhere

    second = ComputedGame()
    second.seed(0)
    second.initGame([agent, opponent])
    assert agent.positions == agent.startPositions
    assert opponent.positions == opponent.startPositions


def test_standard_game_has_two_players(game):
    assert [p.identifier for p in game.players] == [1, 2]


def test_each_player_starts_with_fifteen_pieces(game):
    for player in game.players:
        assert len(player.startPositions) == 15
        assert len(player.endPositions) == 15
        assert len(player.positions) == 15
        # At the start no piece has reached its target yet.
        assert player.nonArrived == player.startPositions
        assert player.openEndPositions == player.endPositions


def test_pieces_are_placed_on_the_board(game):
    # 2 players * 15 pieces = 30 occupied fields at the start.
    occupied = sum(1 for f in game.board.fields if not f.isEmpty())
    assert occupied == 30


def test_fresh_game_has_no_winner_and_no_moves(game):
    assert game.winner() is None
    assert game.gameLength() == 0


def test_play_order_contains_all_players(game):
    assert sorted(p.identifier for p in game.playOrder) == [1, 2]
