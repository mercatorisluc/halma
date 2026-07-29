"""Characterization tests for move generation and application."""

import pytest


def test_fresh_start_offers_only_single_step_moves(board, game):
    player = game.players[0]
    simple = board.allValidMoves(player)
    with_way = board.allValidMovesWithWay(player)
    # From the packed starting corner every legal move is a single step to an
    # adjacent empty field; there are 20 of them for player 1.
    assert len(simple) == 20
    assert len(with_way) == 20
    assert all(len(way) == 2 for way in with_way)


def test_valid_move_targets_are_empty_and_allowed(board, game):
    player = game.players[0]
    for start, end in board.allValidMoves(player):
        assert board.fields[start].playerID == player.identifier
        assert board.fields[end].isEmpty()
        assert board.fields[end].allows(player)


def test_apply_move_moves_the_piece(board, game):
    player = game.players[0]
    start, end = sorted(board.allValidMoves(player))[0]
    board.applyMoveForPlayer((start, end), player)
    assert board.fields[start].isEmpty()
    assert board.fields[end].playerID == player.identifier
    assert end in player.positions and start not in player.positions


def test_apply_then_reverse_restores_the_board(board, game):
    player = game.players[0]
    before = board.boardState().copy()
    start, end = sorted(board.allValidMoves(player))[0]
    board.applyMoveForPlayer((start, end), player)
    board.applyMoveForPlayer((end, start), player)  # reverse == undo
    assert (board.boardState() == before).all()
    assert board.fields[start].playerID == player.identifier
    assert board.fields[end].isEmpty()


def test_is_jump_move(board, game):
    player = game.players[0]
    start = next(f.id for f in board.fields
                 if f.playerID == player.identifier and f.neighbours)
    neighbour = board.fields[start].neighbours[0]
    assert board.isJumpMove(start, neighbour) is False


def test_jump_over_occupied_field_lands_on_empty(board):
    # Clear the board, place a jumper and a single blocker so exactly one
    # jump target is reachable.
    field = next(f for f in board.fields if f.jumpNeighbours)
    jumped_over, landing = next(iter(field.jumpNeighbours.items()))
    for f in board.fields:
        f.playerID = 0
    board.fields[field.id].playerID = 1
    board.fields[jumped_over].playerID = 2
    assert board.getValidJumpFields(field.id) == [landing]


def test_jump_blocked_when_landing_is_occupied(board):
    field = next(f for f in board.fields if f.jumpNeighbours)
    jumped_over, landing = next(iter(field.jumpNeighbours.items()))
    for f in board.fields:
        f.playerID = 0
    board.fields[field.id].playerID = 1
    board.fields[jumped_over].playerID = 2
    board.fields[landing].playerID = 2  # landing blocked
    assert board.getValidJumpFields(field.id) == []
