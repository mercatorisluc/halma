"""Tests for the cloning script's compact storage.

``collect`` does not store observations; it stores the two piece planes and a
bit-packed action mask, and ``fit`` reassembles both per batch. That saves
14.2 GB at 500k samples and introduces the one failure mode worth guarding
against here: a reconstruction that disagrees with the environment would train
the policy on a board it will never be shown, and nothing would raise. So what
is pinned is equality with what ``HalmaEnv`` actually produced.
"""

import numpy as np

from env.halmaEnv import HalmaEnv
from scripts.pretrain import DYNAMIC_PLANES, collect, rebuildBoards, staticBlock


def test_the_stored_planes_rebuild_the_exact_observation():
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    board = observation["board"]
    static = staticBlock(env)

    stored = board[:DYNAMIC_PLANES].astype(np.uint8)
    rebuilt = rebuildBoards(stored[None], static)[0]

    np.testing.assert_array_equal(rebuilt, board)
    assert rebuilt.dtype == board.dtype


def test_the_static_block_is_everything_the_dynamic_half_leaves_out():
    """The split has to be exhaustive: two planes stored plus the block must be
    the whole board, or rebuildBoards would silently pad with uninitialised
    memory -- it allocates with np.empty."""
    env = HalmaEnv()
    assert DYNAMIC_PLANES + len(staticBlock(env)) == env.boardPlanes


def test_the_piece_planes_really_are_the_only_ones_that_move():
    """The premise of not storing the rest. If a later change makes any plane
    from index 2 on position-dependent, storing it once per run would feed the
    fit stale data -- and the loss would still go down."""
    env = HalmaEnv()
    rng = np.random.default_rng(0)
    observation, _ = env.reset(seed=0)
    opening = observation["board"][DYNAMIC_PLANES:].copy()
    for _ in range(30):
        legal = np.flatnonzero(env.action_masks())
        observation, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        np.testing.assert_array_equal(observation["board"][DYNAMIC_PLANES:], opening)
        if terminated or truncated:
            break


def test_collect_round_trips_through_the_packed_mask():
    """Unpacking has to give back the legal moves exactly -- packbits pads to a
    byte boundary, so the count argument is load-bearing and an off-by-one
    would hand the policy a mask shifted against the action space."""
    env = HalmaEnv()
    dynamic, scalars, masks, actions = collect("bottleneck", "advancedDistScore", 40, seed=0)

    assert len(dynamic) == len(scalars) == len(masks) == len(actions) == 40
    assert dynamic.dtype == np.uint8
    unpacked = np.unpackbits(masks, axis=1, count=env.actionCount).astype(bool)
    assert unpacked.shape == (40, env.actionCount)
    # The expert's own move has to be legal in the position it was recorded in.
    assert all(unpacked[i, actions[i]] for i in range(40))
    # And a real position leaves far more illegal than legal -- a mask that
    # unpacked to all-true would pass the check above and be useless.
    assert unpacked.sum(axis=1).max() < 200
