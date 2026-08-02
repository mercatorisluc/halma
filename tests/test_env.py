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
    # Exactly 1, not merely more: the potential measures travel still to be
    # done, and a won position has none left. The measure it replaced bottomed
    # out at 0.84 instead, because two thirds of it was the distance to the tip
    # field of the target triangle rather than to the triangle -- so 16% of the
    # shaping budget was unreachable, and pieces already home were still being
    # paid to shuffle towards the tip.
    assert env._potential() == pytest.approx(1.0)


def test_progress_is_zero_only_once_every_piece_is_home():
    # 15 pieces, 15 target fields: no travel left to do means each piece stands
    # on one of them, which is the win condition. That makes the potential's
    # upper end coincide with winning rather than approximating it.
    env = HalmaEnv()
    env.reset(seed=0)
    agent = env._player(env.AGENT_SEAT)
    assert env._progress(agent) > 0

    board = env.board
    for field in board.fields:
        if field.playerID == agent.identifier:
            field.removePlayer()
    targets = sorted(agent.endPositions)
    # One piece short of home, parked a step outside the zone.
    for target in targets[1:]:
        board.fields[target].playerID = agent.identifier
    outside = next(n for n in board.fields[targets[0]].neighbours if board.fields[n].isEmpty())
    board.fields[outside].playerID = agent.identifier
    agent.positions = {outside, *targets[1:]}
    assert env._progress(agent) == pytest.approx(1.0), "one piece, one step out"

    board.fields[outside].removePlayer()
    board.fields[targets[0]].playerID = agent.identifier
    agent.positions = set(targets)
    assert env._progress(agent) == pytest.approx(0.0)


def test_progress_counts_every_straggler_rather_than_averaging_them():
    """Summed, not averaged -- the flaw in the measure this replaced.

    That one divided by the number of pieces still out, so a piece arriving
    shrank the numerator and the divisor together and the average could sit
    still on real progress. A sum falls by the distance the piece had left,
    every time.
    """
    env = HalmaEnv()
    env.reset(seed=0)
    agent = env._player(env.AGENT_SEAT)
    board = env.board
    for field in board.fields:
        if field.playerID == agent.identifier:
            field.removePlayer()

    targets = sorted(agent.endPositions)
    far = max(range(len(board.fields)), key=lambda f: env.distanceToTarget[f])
    # Everything home but one straggler at the far end of the board.
    agent.positions = {far, *targets[1:]}
    withStraggler = env._progress(agent)
    # The same straggler one step closer.
    closer = min(board.fields[far].neighbours, key=lambda n: env.distanceToTarget[n])
    agent.positions = {closer, *targets[1:]}
    assert env._progress(agent) == pytest.approx(withStraggler - 1.0)


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


def test_the_legal_move_cache_never_goes_stale_during_play():
    """The mask is memoised per position; a stale one would be silent.

    An action mask that lags a move behind would not raise anything -- the
    agent would simply be offered moves it cannot make and denied ones it can,
    and step() would forfeit the episode as if the policy had misbehaved. So
    this replays a whole game checking the memoised answer against a freshly
    generated one at every step, on both sides of the move.
    """
    env = HalmaEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(3)

    def freshlyGenerated():
        player = env.game.currentPlayer()
        moves = env.board.allValidMoves(player)
        key = env._permutationKey(player)
        return sorted(env.encodeAction(m) for m in env.normalizer.permuteMoves(moves, key))

    steps = 0
    while True:
        assert sorted(env._legalActions()) == freshlyGenerated()
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
        assert not info["illegalAction"]
        steps += 1
        if terminated or truncated:
            break
        assert sorted(env._legalActions()) == freshlyGenerated()
    assert steps > 20


def test_reset_clears_the_legal_move_cache():
    """A new game restarts the move count, which is the cache key.

    Deliberately a white-box assertion. The collision it guards against cannot
    currently be caught through behaviour: reset always hands back a position
    with the agent on turn, and the two openings that share a key are either
    the pristine board (identical, so a stale entry is accidentally right) or
    the board after one opponent move, which is played in the far corner and
    never changes what the agent may do. That makes the entry unobservable
    today and wrong the moment either holds -- if the opponent's opening
    reached across the board, or if reset left someone else on turn. Cheaper to
    clear it and pin that than to rely on the coincidence.
    """
    env = HalmaEnv()
    env.reset(seed=1)
    # A poisoned entry under the key the next game will open on. Without the
    # clear, reset hands it straight back; the two real openings happen to
    # agree, so only a planted answer shows the difference.
    env._legalCache = (0, [12345])
    env.reset(seed=2)
    assert env.game.gameLength() == 0, "seed chosen so the new game shares the key"
    assert env._legalActions() != [12345]
    player = env.game.currentPlayer()
    assert sorted(env._legalActions()) == sorted(
        env.encodeAction(m)
        for m in env.normalizer.permuteMoves(
            env.board.allValidMoves(player), env._permutationKey(player)
        )
    )
