"""Tests for the Gymnasium environment, focused on the reward.

Shaping is the part where a mistake is invisible: training would still run, the
agent would just learn the wrong thing. So these pin the two properties it is
supposed to have -- a signal on every step, and no change to which policy is
optimal.
"""

import numpy as np
import pytest

from env.halmaEnv import HalmaEnv

GAMMA = 0.99


def playRandomEpisode(env, seed):
    """Play through with uniformly random legal moves, returning the rewards."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    startingPotential = env._potential()
    rewards, terminated = [], False
    while True:
        legal = np.flatnonzero(env.action_masks())
        _, reward, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        rewards.append(reward)
        if terminated or truncated:
            return rewards, startingPotential, terminated


def discountedReturn(rewards):
    return sum(GAMMA**t * reward for t, reward in enumerate(rewards))


def test_env_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(HalmaEnv(), skip_render_check=True)


def test_unshaped_reward_is_almost_always_zero():
    # The problem shaping exists to solve: one signal per episode, and a random
    # agent never wins at all, so it sees nothing but the final -1.
    rewards, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=0.0), seed=0)
    nonZero = [r for r in rewards if r != 0]
    assert len(nonZero) <= 1
    assert len(rewards) > 30


def test_shaping_puts_a_signal_on_every_step():
    rewards, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=1.0, gamma=GAMMA), seed=0)
    assert all(reward != 0 for reward in rewards)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
def test_shaping_does_not_change_which_policy_is_optimal(seed):
    # Potential-based shaping shifts the discounted return of *every* policy by
    # the same -phi(s0), so no policy gains on another. Same seed and same RNG
    # means both runs play identical moves, which is what makes them
    # comparable.
    shaped, startingPotential, terminated = playRandomEpisode(
        HalmaEnv(shapingWeight=1.0, gamma=GAMMA), seed
    )
    plain, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=0.0, gamma=GAMMA), seed)
    if not terminated:
        pytest.skip("relation is deliberately not upheld across a time-limit truncation")
    assert discountedReturn(shaped) - discountedReturn(plain) == pytest.approx(
        -startingPotential, abs=1e-9
    )


def test_shaping_stays_smaller_than_winning():
    # If progress outweighed the result, the agent would learn to advance
    # rather than to win.
    shaped, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=1.0, gamma=GAMMA), seed=0)
    plain, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=0.0, gamma=GAMMA), seed=0)
    assert abs(discountedReturn(shaped) - discountedReturn(plain)) < 1.0


def test_potential_favours_the_leader():
    # Higher is better for the agent, so handing the opponent the whole game
    # has to lower it.
    env = HalmaEnv()
    env.reset(seed=0)
    even = env._potential()
    assert even == pytest.approx(0.0, abs=0.5), "an even opening should sit near zero"

    opponent = env._player(env.OPPONENT_SEAT)
    for field in env.board.fields:
        if field.playerID == opponent.identifier:
            field.removePlayer()
    for target in opponent.endPositions:
        env.board.fields[target].playerID = opponent.identifier
    opponent.positions = set(opponent.endPositions)
    opponent.nonArrived = set()
    opponent.openEndPositions = set()
    # distanceScore is maintained incrementally, so moving pieces by hand
    # leaves it stale -- and a stale one here reads as a huge opponent score,
    # flipping the sign of the potential. Recompute it, the way
    # prepareForGameStart does.
    opponent.distanceScore = env.board.calculatePlayerDistanceScore(opponent)

    assert env._potential() < even


def test_info_reports_the_unshaped_outcome():
    # Evaluation has to score wins, not shaped reward.
    env = HalmaEnv(shapingWeight=1.0)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    while True:
        legal = np.flatnonzero(env.action_masks())
        _, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        if terminated:
            assert info["outcome"] in (1.0, -1.0)
            assert reward != info["outcome"]  # shaping was applied on top
            return
        if truncated:
            assert info["outcome"] == 0.0
            return
        assert info["outcome"] == 0.0
