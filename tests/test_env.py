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


def test_a_single_opponent_strategy_never_changes_across_resets():
    env = HalmaEnv(opponentStrategy="bottleneck")
    for seed in range(10):
        env.reset(seed=seed)
        assert env.opponentStrategy == "bottleneck"


def test_an_opponent_pool_is_drawn_from_on_every_reset():
    pool = ["advancedDistScore", "sparsityScore", "bottleneck"]
    env = HalmaEnv(opponentStrategy=pool)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        assert env.opponentStrategy in pool
        seen.add(env.opponentStrategy)
    # Not a strict guarantee for any RNG, but 20 draws from 3 options landing
    # on a single one every time would mean the draw is not happening at all.
    assert len(seen) > 1


def test_the_opponent_draw_is_reproducible_from_a_seed():
    pool = ["advancedDistScore", "sparsityScore", "bottleneck"]
    first = HalmaEnv(opponentStrategy=pool)
    second = HalmaEnv(opponentStrategy=pool)
    for seed in range(10):
        first.reset(seed=seed)
        second.reset(seed=seed)
        assert first.opponentStrategy == second.opponentStrategy


def test_opponent_sampling_needs_a_checkpoint_to_sample():
    # A heuristic has no distribution, so this combination would quietly do
    # nothing -- and a run configured that way would look varied and not be.
    with pytest.raises(ValueError, match="opponentSampling needs"):
        HalmaEnv(opponentStrategy="bottleneck", opponentSampling=0.5)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_opponent_sampling_outside_zero_to_one_is_refused(fraction, checkpoint):
    with pytest.raises(ValueError, match="probability"):
        HalmaEnv(opponentModel=checkpoint, opponentSampling=fraction)


def opponentStyles(env, episodes=20):
    """Whether the neural opponent played argmax, per episode."""
    styles = []
    for seed in range(episodes):
        env.reset(seed=seed)
        assert env._opponentNeural is not None
        styles.append(env._opponentNeural.deterministic)
    return styles


def test_a_checkpoint_opponent_plays_argmax_unless_sampling_is_asked_for(checkpoint):
    # The default has to stay exactly what every recorded result was measured
    # against: the opponent's actual best move, every episode.
    assert all(opponentStyles(HalmaEnv(opponentModel=checkpoint)))


def test_full_opponent_sampling_makes_it_play_its_distribution_every_episode(checkpoint):
    assert not any(opponentStyles(HalmaEnv(opponentModel=checkpoint, opponentSampling=1.0)))


def test_partial_opponent_sampling_mixes_both_kinds_of_episode(checkpoint):
    # The point of the fraction: the argmax opponent stays in the data
    # alongside the sampled one, rather than being replaced by it.
    styles = opponentStyles(HalmaEnv(opponentModel=checkpoint, opponentSampling=0.5))
    assert any(styles) and not all(styles)


def test_the_opponent_style_draw_is_reproducible_from_a_seed(checkpoint):
    first = HalmaEnv(opponentModel=checkpoint, opponentSampling=0.5)
    second = HalmaEnv(opponentModel=checkpoint, opponentSampling=0.5)
    assert opponentStyles(first) == opponentStyles(second)


def openingPosition(env, seed):
    """The two piece planes after a reset, as a hashable snapshot."""
    obs, _ = env.reset(seed=seed)
    return obs["board"][:2].tobytes()


def ownOpeningPieces(env, seed):
    """Just the agent's own plane after a reset."""
    obs, _ = env.reset(seed=seed)
    return obs["board"][0].tobytes()


def test_the_opening_is_fixed_unless_random_plies_are_asked_for():
    # By default the agent's first decision is always made from its home
    # corner untouched -- which is what every earlier checkpoint trained and
    # was measured on. (The *opponent's* pieces do vary a little even here,
    # because the play order is drawn and it may have replied already.)
    env = HalmaEnv()
    assert len({ownOpeningPieces(env, seed) for seed in range(10)}) == 1


def test_random_opening_plies_vary_the_starting_position():
    env = HalmaEnv(randomOpeningPlies=6)
    assert len({openingPosition(env, seed) for seed in range(10)}) == 10
    # Both sides open at random, not just the opponent: an even count splits
    # the plies evenly, so the agent has played three of these six itself.
    assert len({ownOpeningPieces(env, seed) for seed in range(10)}) > 1


def test_random_opening_plies_leave_the_agent_on_turn():
    # reset() must hand back a position the agent can actually step from,
    # whether the random opening ended on its turn or the opponent's.
    env = HalmaEnv(randomOpeningPlies=6)
    for seed in range(10):
        env.reset(seed=seed)
        assert env._isAgentsTurn()


def test_the_random_opening_is_reproducible_from_a_seed():
    # It is drawn from the env's own generator, so an episode seed reproduces
    # the opening along with everything else -- not a second, loose source of
    # randomness beside it.
    first, second = HalmaEnv(randomOpeningPlies=6), HalmaEnv(randomOpeningPlies=6)
    for seed in range(10):
        assert openingPosition(first, seed) == openingPosition(second, seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
def test_shaping_still_telescopes_from_a_random_opening(seed):
    # The random plies move the board before phi(s0) is taken, so the property
    # that shaping cannot redirect the agent has to hold from wherever they
    # left it -- otherwise a varied opening would quietly reintroduce the bias
    # potential-based shaping exists to avoid.
    shaped, startingPotential, terminated = playRandomEpisode(
        HalmaEnv(shapingWeight=1.0, gamma=GAMMA, randomOpeningPlies=6), seed
    )
    plain, _, _ = playRandomEpisode(
        HalmaEnv(shapingWeight=0.0, gamma=GAMMA, randomOpeningPlies=6), seed
    )
    if not terminated:
        pytest.skip("relation is deliberately not upheld across a time-limit truncation")
    assert discountedReturn(shaped) - discountedReturn(plain) == pytest.approx(
        -startingPotential, abs=1e-9
    )


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


def test_needs_flip_is_fixed_to_the_seats_home_corner_not_current_pieces():
    """Seat 1's own pieces never cross the sign threshold in real play, so a
    version keyed on *current* positions always happened to agree with one
    keyed on the fixed home corner -- for seat 1 only. Seat 2's home corner
    sits on the opposite side, and its pieces migrate across the threshold
    over a real game; keying on current positions made the canonical frame
    flip discontinuously almost every ply once its pieces straddled the
    threshold, which training (seat 1 only) never produced. Measured: a
    checkpoint at 99% from seat 1 lost 20/20 games from seat 2 with the
    current-position version and 29/30 with this one.
    """
    env = HalmaEnv(selfSeat=HalmaEnv.OPPONENT_SEAT)
    env.reset(seed=0)
    player = env._player(HalmaEnv.OPPONENT_SEAT)
    before = env._needsFlip(HalmaEnv.OPPONENT_SEAT)

    # Move every piece to the far (positive-x) side without touching
    # startPositions -- current occupancy now disagrees in sign with the
    # fixed home corner, the exact situation that used to flap the frame.
    player.positions = set(player.endPositions)
    after = env._needsFlip(HalmaEnv.OPPONENT_SEAT)

    assert before == after


def test_the_geometry_planes_look_the_same_from_either_seat():
    """The point of the canonical frame, stated on the new planes.

    A policy only ever learns one orientation, so the four zone planes and the
    two distance maps must be *identical* whichever seat builds them -- even
    though the two seats' zones are four different corners of the star and the
    permutation taking each to canonical is a different one. If this drifts,
    a checkpoint seated on OPPONENT_SEAT sees a board whose geometry contradicts
    its pieces, which is precisely the failure invariant 7 was about.
    """
    fromSeat1 = HalmaEnv(selfSeat=HalmaEnv.AGENT_SEAT).geometryPlanes
    fromSeat2 = HalmaEnv(selfSeat=HalmaEnv.OPPONENT_SEAT).geometryPlanes
    np.testing.assert_array_equal(fromSeat1, fromSeat2)


def test_the_geometry_planes_are_aligned_with_the_piece_planes():
    """Built by a different route than the piece planes, so alignment is a
    property to pin rather than assume: at the opening every own piece stands
    on its own start zone and no piece is anywhere else, so plane 0 and the own
    start plane have to agree cell for cell."""
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    board = observation["board"]
    if env.game.gameLength() == 0:
        np.testing.assert_array_equal(board[0], board[4])
    # Whatever the play order did, the opponent's pieces start on the
    # opponent's start zone, and no opponent move can have emptied it entirely.
    assert float((board[1] * board[6]).sum()) > 0


def test_closeness_peaks_exactly_on_the_target_zone():
    """The distance maps are 1 on the zone and below 1 everywhere else, which
    is what makes them a gradient towards it rather than a second zone plane."""
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    board = observation["board"]
    ownTarget, ownCloseness = board[3], board[7]
    assert float(ownCloseness[ownTarget > 0].min()) == 1.0
    offZone = env.boardMask.astype(bool) & (ownTarget == 0)
    assert float(ownCloseness[offZone].max()) < 1.0


def test_the_geometry_planes_do_not_move_during_a_game():
    """Constant is the whole premise: they are painted once in __init__ and
    copied into every observation, so a game that changed them would mean the
    observation and the cached planes had come apart."""
    env = HalmaEnv()
    rng = np.random.default_rng(0)
    observation, _ = env.reset(seed=0)
    opening = observation["board"][3:].copy()
    for _ in range(20):
        legal = np.flatnonzero(env.action_masks())
        observation, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        np.testing.assert_array_equal(observation["board"][3:], opening)
        if terminated or truncated:
            break


def test_no_jump_ever_changes_a_pieces_parity_class():
    """The claim the whole parity feature rests on, checked against the real
    move generator rather than against the coordinate arithmetic it was derived
    from. Every jump delta is even in both coordinates, so a jump -- and so any
    chain of them, however long -- lands in the class it started in. A single
    step always leaves it.
    """
    env = HalmaEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    jumps = steps = 0
    for _ in range(40):
        player = env.game.currentPlayer()
        for way in env.board.allValidMovesWithWay(player):
            sameClass = env.parityClass[way[0]] == env.parityClass[way[-1]]
            if env.board.isJumpMove(way[0], way[-1]):
                jumps += 1
                assert sameClass, f"jump {way} left its class"
            else:
                steps += 1
                assert not sameClass, f"single step {way} stayed in its class"
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            break
    # A run that generated no jumps would pass vacuously.
    assert jumps > 100 and steps > 100


def test_the_parity_mismatch_is_zero_at_both_ends_of_the_game():
    """Start and target zone have the same class distribution (6/3/3/3), so the
    opening is already matched, and a won position has neither stragglers nor
    open targets. That is what keeps the potential running exactly 0 to 1 with
    the penalty switched on -- the penalty is a detour, not a shift of scale."""
    env = HalmaEnv()
    env.reset(seed=0)
    agent = env._player(HalmaEnv.AGENT_SEAT)
    assert env._parityMismatch(agent) == 0
    assert env._potential() == pytest.approx(0.0)

    for field in env.board.fields:
        if field.playerID == agent.identifier:
            field.removePlayer()
    for target in agent.endPositions:
        env.board.fields[target].playerID = agent.identifier
    agent.positions = set(agent.endPositions)
    agent.nonArrived = set()
    agent.openEndPositions = set()
    agent.distanceScore = env.board.calculatePlayerDistanceScore(agent)

    assert env._parityMismatch(agent) == 0
    assert env._potential() == pytest.approx(1.0)


def test_the_parity_penalty_keeps_the_potential_in_range_and_never_flat():
    """The two properties the subtractive form broke, pinned together because
    the obvious fix for either one breaks the other: subtracting sent early
    plies negative, and clamping that at zero made consecutive clamped plies
    produce exactly zero shaping. Multiplying keeps the potential in [0, 1]
    without ever flattening it while the agent is moving."""
    env = HalmaEnv(parityWeight=1.0, gamma=GAMMA)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(200):
        assert 0.0 <= env._potential() <= 1.0
        legal = np.flatnonzero(env.action_masks())
        _, reward, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        assert reward != 0.0, "shaping must still put a signal on every step"
        if terminated or truncated:
            break


def test_turning_the_parity_weight_off_restores_the_travel_only_potential():
    """The control every parity result has to be measured against: at weight 0
    the potential must be bit-for-bit what it was before the penalty existed."""
    penalised = HalmaEnv(parityWeight=0.5)
    control = HalmaEnv(parityWeight=0.0)
    rng = np.random.default_rng(0)
    penalised.reset(seed=0)
    control.reset(seed=0)
    differed = False
    for _ in range(60):
        agent = control._player(HalmaEnv.AGENT_SEAT)
        travel = 1.0 - control._progress(agent) / control.openingProgress
        assert control._potential() == pytest.approx(travel)
        if control._potential() != pytest.approx(penalised._potential()):
            differed = True
        legal = np.flatnonzero(control.action_masks())
        action = int(rng.choice(legal))
        _, _, terminated, truncated, _ = control.step(action)
        penalised.step(action)
        if terminated or truncated:
            break
    # Otherwise the control would be trivially satisfied by a penalty that
    # never fires at all.
    assert differed


def test_the_class_planes_are_one_hot_over_every_raster_cell():
    """One-hot rather than a number because the four classes are mutually one
    single step apart -- no ordering to encode -- and painted over all 289
    cells rather than the 121 real fields, because parity belongs to the raster
    and stopping it at the star's edge would fake a discontinuity right where a
    3x3 kernel straddles the boundary."""
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    classPlanes = observation["board"][9:13]
    assert set(np.unique(classPlanes)) == {0.0, 1.0}
    np.testing.assert_array_equal(classPlanes.sum(axis=0), np.ones((17, 17)))
    # The void included: 289, not the 121 the mask covers.
    assert float(classPlanes.sum()) == 289.0


def test_every_class_is_one_single_step_from_every_other():
    """Why the encoding carries no metric. All three step deltas are odd in at
    least one coordinate, so from any class a single step reaches all three
    others -- the classes form a complete graph, and any encoding implying near
    and far classes (an ordinal 1..4, or a two-bit Hamming code) would misstate
    the board."""
    env = HalmaEnv()
    reachable = {cls: set() for cls in range(4)}
    for field in env.board.fields:
        for neighbour in field.neighbours:
            reachable[int(env.parityClass[field.id])].add(int(env.parityClass[neighbour]))
    for cls in range(4):
        assert reachable[cls] == {0, 1, 2, 3} - {cls}


def test_the_class_planes_partition_the_board_the_way_the_raw_classes_do():
    """The labels may differ and the partition may not.

    ``parityClass`` is read off raw coordinates and the planes off canonical
    ones, and a rotation permutes the four labels -- which is exactly why
    ``_parityMismatch`` may sum over all four and ignore the frame. What must
    survive is the grouping: two fields share a class in one frame iff they
    share one in the other. Nothing cross-references the labels themselves, and
    this is the test that says that is safe.
    """
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    classPlanes = observation["board"][9:13]
    rows, cols = env.rasterIndex[:, 0], env.rasterIndex[:, 1]
    canonical = classPlanes.argmax(axis=0)[rows, cols]
    # The raw labels pushed through the same permutation the planes went
    # through, so both are indexed by canonical field id.
    raw = env.normalizer.permute(env.parityClass, env._selfKey)

    mapping = {}
    for rawLabel, canonicalLabel in zip(raw, canonical, strict=True):
        mapping.setdefault(int(rawLabel), int(canonicalLabel))
        assert mapping[int(rawLabel)] == int(canonicalLabel), "the partition differs"
    assert len(set(mapping.values())) == len(mapping) == 4, "the relabelling must be a bijection"


def test_the_same_class_distance_map_is_not_a_rescaling_of_the_other():
    """Plane 8 restricts the target set to the field's own class, so it is
    always at least the unrestricted distance and strictly more wherever the
    nearest target sits across a class boundary -- 56 of 121 fields."""
    env = HalmaEnv()
    agent = env._player(HalmaEnv.AGENT_SEAT)
    sameClass = env._sameClassDistances(agent)
    anyClass = env.distanceToTarget
    assert np.all(sameClass >= anyClass)
    assert int((sameClass > anyClass).sum()) == 56
