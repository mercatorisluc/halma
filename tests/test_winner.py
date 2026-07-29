"""Characterization tests for win detection."""


def _move_player_into_target(game, player):
    """Place all of ``player``'s pieces exactly onto their target fields."""
    for field in game.board.fields:
        if field.playerID == player.identifier:
            field.removePlayer()
    for target_id in player.endPositions:
        game.board.fields[target_id].playerID = player.identifier
    player.positions = set(player.endPositions)


def test_player_is_winning_when_pieces_fill_target(game):
    player = game.players[0]
    assert player.isWinning() is False
    _move_player_into_target(game, player)
    assert player.isWinning() is True


def test_game_reports_the_winning_player_identifier(game):
    player = game.players[0]
    assert game.winner() is None
    _move_player_into_target(game, player)
    assert game.winner() == player.identifier


def test_winning_by_fully_blocked_target(game):
    # A player also wins if all 15 target fields are occupied (by any piece),
    # per playerIsWinningByBlockedFields.
    player = game.players[0]
    for target_id in player.endPositions:
        if game.board.fields[target_id].isEmpty():
            game.board.fields[target_id].playerID = 2
    assert game.playerIsWinningByBlockedFields(player) is True
