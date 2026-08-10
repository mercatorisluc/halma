"""Tests for the human interaction state, which killed a live game.

``visual/`` is otherwise untested -- it is due to be replaced by the browser
front-end -- but ``HumanInputHandler`` imports no pygame and owns the one piece
of state that the playback cursor can invalidate underneath it. The crash was
not in playing a move but in *drawing* one: the renderer reconstructs each
legal move's full jump path to outline it, and a path the board no longer
supports fails an assertion inside the engine.
"""

from game.gameManager import InteractiveGame
from game.player import Computer, HumanPlayer
from visual.gamePlaybackController import GamePlaybackController
from visual.humanInputHandler import HumanInputHandler


def gameAtHumanTurn(plies=6):
    """A game with history behind it, stopped on the human's turn."""
    game = InteractiveGame()
    game.seed(0)
    game.initGame([Computer(1, "advancedDistScore"), HumanPlayer(2)])
    playback = GamePlaybackController(game)
    for _ in range(plies):
        player = game.currentPlayer()
        if player.isHuman():
            game.playMove(player, game.board.allValidMovesWithWay(player)[0])
        else:
            game.playNextMove(player)
        playback.moveTraveler += 1
    assert game.currentPlayer().isHuman()
    return game, playback, HumanInputHandler(playback)


def drawableWithoutCrashing(game, player, moves):
    """What ``BoardRenderer.drawValidMoves`` does to every move it outlines."""
    for way in moves:
        game.createMoveForPlayer(way, player).reconstructFullMove(game.board)


def test_the_drawn_moves_survive_a_rewind():
    """The crash: two presses of LEFT, then a click.

    The move list used to be computed once per turn and kept, while the arrow
    keys mutate the board underneath it. Rewinding past the human's own last
    move left cached jump paths whose intermediate landings no longer existed.
    """
    game, playback, handler = gameAtHumanTurn()
    handler.adaptToHumanInteraction(game)
    assert handler.validHumanMoves

    playback.backwardGame()
    playback.backwardGame()
    handler.adaptToHumanInteraction(game)

    drawableWithoutCrashing(game, game.players[1], handler.validHumanMoves or [])


def test_a_rewound_board_is_not_the_humans_turn():
    """Nothing may be played into a position the game has moved on from -- the
    move would be appended to a history it does not follow from."""
    game, playback, handler = gameAtHumanTurn()
    handler.adaptToHumanInteraction(game)
    assert handler.waitingForHumanMove

    playback.backwardGame()
    handler.adaptToHumanInteraction(game)

    assert not handler.waitingForHumanMove
    assert handler.validHumanMoves is None


def test_returning_to_the_front_restores_the_turn():
    """Reviewing history and stepping back to the present is not a one-way
    door: the human must be able to move again afterwards."""
    game, playback, handler = gameAtHumanTurn()
    playback.backwardGame()
    handler.adaptToHumanInteraction(game)
    assert not handler.waitingForHumanMove

    playback.forwardGame()
    handler.adaptToHumanInteraction(game)

    assert handler.waitingForHumanMove
    drawableWithoutCrashing(game, game.players[1], handler.validHumanMoves)


def test_the_move_list_tracks_the_board_it_is_drawn_against():
    """Recomputed per frame, not cached for the turn."""
    game, playback, handler = gameAtHumanTurn()
    handler.adaptToHumanInteraction(game)
    before = handler.validHumanMoves

    playback.backwardGame()
    playback.forwardGame()
    handler.adaptToHumanInteraction(game)

    assert handler.validHumanMoves == before
    assert handler.validHumanMoves is not before
