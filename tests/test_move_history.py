"""Characterization tests for move recording and jump-move reconstruction."""

import pytest

from game.move import Move


def test_play_move_records_history(game):
    player = game.currentPlayer()
    move = min(game.board.allValidMoves(player))
    assert game.gameLength() == 0
    game.playMove(player, move)
    assert game.gameLength() == 1
    recorded = game.moves[-1]
    assert (recorded.start, recorded.end) == (move[0], move[-1])
    assert recorded.player is player


def test_current_player_alternates_with_move_count(game):
    first = game.currentPlayer()
    move = min(game.board.allValidMoves(first))
    game.playMove(first, move)
    second = game.currentPlayer()
    assert first is not second


def test_single_step_move_needs_no_reconstruction(board, game):
    player = game.players[0]
    # Pick a genuine single-neighbour step (not a jump): from the packed
    # start, several legal moves are jumps into the empty centre, so we must
    # select one whose destination is a direct neighbour.
    start, end = next((s, e) for s, e in board.allValidMoves(player)
                      if e in board.fields[s].neighbours)
    move = Move([start, end], player)
    move.reconstructFullMove(board)
    assert move.jumpedOvers == []
    assert move.fullStepsList() == [start, end]


def test_full_steps_list_refuses_to_run_before_reconstruction(game):
    # jumpedOvers is None until reconstructFullMove fills it in, so the flat
    # path cannot be built yet. Both renderer call sites reconstruct first;
    # this pins the contract so a future caller gets a clear error instead of
    # "object of type 'NoneType' has no len()".
    move = Move([0, 1], game.players[0])
    assert move.needsReconstruction() is True
    with pytest.raises(RuntimeError, match="reconstructFullMove"):
        move.fullStepsList()


def test_double_jump_reconstruction(board, game):
    # Craft a two-hop jump: 0 -over 2-> 5 -over 9-> 18.
    start, over1, mid, over2, end = (0, 2, 5, 9, 18)
    for f in board.fields:
        f.playerID = 0
    board.fields[start].playerID = 1
    board.fields[over1].playerID = 2
    board.fields[over2].playerID = 2

    move = Move([start, end], game.players[0])
    assert move.needsReconstruction() is True
    move.reconstructFullMove(board)

    assert move.steps == [start, mid, end]
    assert move.jumpedOvers == [over1, over2]
    assert move.fullStepsList() == [start, over1, mid, over2, end]
    assert move.partMoves() == [(start, mid), (mid, end)]
