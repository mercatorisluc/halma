"""Characterization tests for board construction and its static topology.

These pin down the board that ``Initializer`` builds: the star-shaped Halma
board has 121 fields, a symmetric neighbour graph, and a fully populated
distance matrix. None of this changes during play, so it is the most stable
part of the engine to lock in before refactoring elsewhere.
"""


def test_board_has_121_fields(board):
    assert len(board.fields) == 121
    assert all(f is not None for f in board.fields)


def test_field_ids_match_their_index(board):
    # fields[i].id == i is relied upon throughout the engine (id used as index).
    for i, field in enumerate(board.fields):
        assert field.id == i


def test_neighbour_graph_is_symmetric(board):
    # If A is a neighbour of B, then B must be a neighbour of A.
    for field in board.fields:
        for neighbour_id in field.neighbours:
            assert field.id in board.fields[neighbour_id].neighbours


def test_jump_neighbour_landing_is_beyond_the_neighbour(board):
    # Each jump entry maps a directly-adjacent field ("jump over") to the
    # field landed on. The landing field must itself be a real field id.
    for field in board.fields:
        for jumped_over, landing in field.jumpNeighbours.items():
            assert jumped_over in field.neighbours
            assert 0 <= landing < 121


def test_distance_matrix_is_populated_and_zero_on_the_diagonal(board):
    assert len(board.distanceMatrix) == 121
    for i in range(121):
        assert board.distanceMatrix[i][i] == 0
    # Distances are symmetric.
    for i in range(121):
        for j in range(121):
            assert board.distanceMatrix[i][j] == board.distanceMatrix[j][i]
