"""Tests for the (pygame-free) playback controller."""

from visual.gamePlaybackController import GamePlaybackController


def test_game_state_at_replays_to_each_position(game):
    # Record a few bot moves, snapshotting the board after each.
    states = [game.board.boardState().copy()]
    for _ in range(4):
        game.playNextMove(game.currentPlayer())
        states.append(game.board.boardState().copy())

    controller = GamePlaybackController(game)
    controller.moveTraveler = len(game.moves)  # board currently at the final state

    # Jumping to position k must reproduce the board as it was after k moves.
    for k in range(len(game.moves) + 1):
        controller.gameStateAt(k)
        assert (game.board.boardState() == states[k]).all()
