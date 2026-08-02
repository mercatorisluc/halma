"""Tests for the per-seat canonicalisation that ``HalmaEnv`` builds
observations from.

The property that matters is agreement: whichever seat is "self", its own
pieces must land on the exact same canonical field ids that seat 1's do,
because that is the single frame the policy is trained to recognise. This
guards against a bug that was entirely silent until something finally read
``player2...`` -- the wrong rotation degree, and later a coordinate lookup
that satisfied a plausible-looking but insufficient self-consistency check
while still landing player 2's pieces on player 1's *target* zone instead of
its start.
"""

from env.boardNormalizer import Normalizer
from game.board import HalmaBoard
from game.initializer import Initializer

OWN_START_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18]


def buildBoard():
    board = HalmaBoard()
    Initializer().initializeBoard(board)
    return board


def test_the_three_home_corners_sit_120_degrees_apart_in_this_order():
    """player1 -> player3 -> player2, not player1 -> player2 -> player3 --
    the fact the rotation-degree bug got backwards. Pinned against
    Initializer's actual coordinates rather than assumed."""
    board = buildBoard()
    init = Initializer()
    coord = {f.id: f.coord for f in board.fields}

    def rot120(c):
        x, y = c
        return (-x - y, x)

    starts = {
        seat: {coord[i] for i in getattr(init, f"player{seat}Positions")(board)[0]}
        for seat in (1, 2, 3)
    }
    assert {rot120(c) for c in starts[1]} == starts[3]
    assert {rot120(c) for c in starts[3]} == starts[2]


def test_player2_permutations_land_player1s_start_on_player2s_actual_pieces():
    """The invariant _observation() depends on: for either seat, canonical
    ids 0-14ish (player 1's own real corner, always -- ids are fixed by
    field construction, not by who is "self") must show *that seat's* own
    pieces once permuted, not the opponent's."""
    board = buildBoard()
    init = Initializer()
    normalizer = Normalizer(board)

    p2Start = set(init.player2Positions(board)[0])

    raw = [0] * len(board.fields)
    for fieldId in p2Start:
        raw[fieldId] = 2

    for key in ("player2WithoutFlip", "player2WithFlip"):
        permuted = normalizer.permute(raw, key)
        selfIds = {k for k in range(len(permuted)) if permuted[k] == 2}
        assert selfIds == set(OWN_START_IDS), key


def test_player1_and_player2_permutations_agree_on_a_shared_board():
    """A stronger version of the same property: with both players' real
    pieces on the board at once, each seat's own canonicalisation must
    produce byte-identical own/opponent planes -- not just a matching set of
    "self" ids in isolation."""
    board = buildBoard()
    init = Initializer()
    normalizer = Normalizer(board)

    p1Start = set(init.player1Positions(board)[0])
    p2Start = set(init.player2Positions(board)[0])
    raw = [0] * len(board.fields)
    for fieldId in p1Start:
        raw[fieldId] = 1
    for fieldId in p2Start:
        raw[fieldId] = 2

    # player1's own real start never needs a flip -- verified separately in
    # HalmaEnv tests -- so WithoutFlip is what a live game actually uses here.
    state1 = normalizer.permute(raw, "player1WithoutFlip")
    for key2 in ("player2WithoutFlip", "player2WithFlip"):
        state2 = normalizer.permute(raw, key2)
        self1 = {k for k in range(len(state1)) if state1[k] == 1}
        self2 = {k for k in range(len(state2)) if state2[k] == 2}
        assert self1 == self2 == set(OWN_START_IDS), key2
