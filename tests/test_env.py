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


def test_shaping_stays_comparable_to_winning():
    # The potential is normalised so a whole episode's shaping sums to about 1,
    # the same order as the +/-1 for the result. Progress has to be worth
    # something -- an agent that never wins learns from nothing else -- but not
    # so much that the result stops mattering. Unnormalised it would be ~7.7.
    shaped, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=1.0, gamma=GAMMA), seed=0)
    plain, _, _ = playRandomEpisode(HalmaEnv(shapingWeight=0.0, gamma=GAMMA), seed=0)
    assert abs(discountedReturn(shaped) - discountedReturn(plain)) < 1.5


def test_potential_measures_own_progress_only():
    # It has to answer "how far along am I", not "how far ahead am I".
    # Rewarding a lead lets the agent hold it by obstructing, which is what an
    # earlier version learned to do instead of playing.
    env = HalmaEnv()
    env.reset(seed=0)
    opening = env._potential()
    assert opening == pytest.approx(0.0), "ground covered, so the opening sits at 0"

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
    # leaves it stale. Recompute it, the way prepareForGameStart does.
    opponent.distanceScore = env.board.calculatePlayerDistanceScore(opponent)

    assert env._potential() == pytest.approx(opening), "the opponent must not move it"


def test_standing_still_is_never_rewarded():
    """The sign bug that made three training runs learn to stall.

    With a discount below 1 the shaping for an unchanged position is
    ``(gamma - 1) * phi``. Measuring ground *remaining* puts the potential at
    -1 and turns that into +0.01 a step for doing nothing -- +1.25 over a game,
    against -1 for losing, so stalling outpaid winning. Measuring ground
    covered keeps the potential at or above 0, where the same term cannot be
    positive.
    """
    env = HalmaEnv()
    env.reset(seed=0)
    assert env._potential() >= 0.0

    for phi in (0.0, 0.25, 0.5, 1.0):
        env.previousPotential = phi
        idle = env.shapingWeight * (env.gamma * phi - phi)
        assert idle <= 0.0, f"an unchanged position must not pay, phi={phi}"


def test_potential_rises_as_the_agent_advances():
    env = HalmaEnv()
    env.reset(seed=0)
    before = env._potential()
    agent = env._player(env.AGENT_SEAT)
    for field in env.board.fields:
        if field.playerID == agent.identifier:
            field.removePlayer()
    for target in agent.endPositions:
        env.board.fields[target].playerID = agent.identifier
    agent.positions = set(agent.endPositions)
    agent.nonArrived = set()
    agent.openEndPositions = set()
    agent.distanceScore = env.board.calculatePlayerDistanceScore(agent)

    assert env._potential() > before


def test_running_out_of_moves_costs_as_much_as_losing():
    # Stalling used to score 0 against -1 for losing, so it was strictly the
    # better play and an agent found that out.
    env = HalmaEnv()
    env.reset(seed=0)
    env.game.MAX_MOVES = 8  # far too few to finish, so the cap is what stops it
    rng = np.random.default_rng(0)
    while True:
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            break
    assert env.game.winner() is None, "the cap should have stopped it, not a win"
    assert info["outcome"] == -1.0
    # The move cap is a rule of the game, so it ends the episode outright
    # rather than cutting it short with something left to bootstrap from.
    assert terminated
    assert not truncated


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
