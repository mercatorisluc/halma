"""Tests for the one-ply search seated as a player.

Everything here fails silently in production. A search that ranks by the wrong
quantity still returns a legal move, and the only symptom is a policy that
plays slightly worse than the one it wraps -- which is what was in fact
measured (ARCHITECTURE.md), so "it plays badly" cannot be used as evidence that
anything is broken. Hence these pin the properties rather than the strength:
the shaping correction, the repetition rule, and that ranking hands back the
moves it was given.

The checkpoint is untrained, as in ``test_neural_player.py``: what is under
test is the machinery around the network.
"""

import numpy as np
import pytest
import torch

from env.halmaEnv import HalmaEnv
from env.searchPlayer import SearchingComputer
from game.gameManager import ComputedGame
from game.player import Computer


def searchGame(checkpoint, **kwargs):
    """A started game with the search on AGENT_SEAT and a bot opposite."""
    agent = SearchingComputer(HalmaEnv.AGENT_SEAT, checkpoint, **kwargs)
    game = ComputedGame()
    game.seed(0)
    game.initGame([agent, Computer(2, "distance")])
    agent.attachTo(game)
    return agent, game


def test_the_critic_estimate_has_the_shaping_potential_added_back(checkpoint):
    """The correction the module docstring calls easy to miss.

    PPO trained the critic on the shaped return, so ``V'(s) = V(s) - w*phi(s)``
    and the offset is position-dependent -- it does not cancel between sibling
    leaves. Drop the ``+ w*phi`` from ``_value`` and this is what notices.
    """
    agent, game = searchGame(checkpoint)
    # Measured mid-game, not at the opening: the potential is ground *covered*,
    # so it is exactly 0 before anyone moves and the check would pass against a
    # missing correction.
    for _ in range(10):
        game.playNextMove(game.currentPlayer())

    agent.encoder._legalCache = None
    tensor, _ = agent.model.policy.obs_to_tensor(agent.encoder._observation())
    with torch.no_grad():
        raw = float(agent.model.policy.predict_values(tensor)[0].item())

    corrected = agent._value()
    potential = agent.encoder.shapingWeight * agent.encoder._potential()
    assert potential > 0.0
    assert corrected == pytest.approx(raw + potential)


def test_ranking_returns_exactly_the_moves_it_was_given(checkpoint):
    """A permutation, not a re-derivation.

    ``_rank`` encodes endpoints to look moves up in the policy's distribution,
    and the root ranks *full jump paths*. If it ever returned what it encoded
    rather than what it was handed, a jump would come back as its endpoints and
    the engine would be asked to play a path it never generated.

    Played on past the opening deliberately: the opening offers no multi-field
    jump path at all, so there a path and its endpoints are the same thing and
    this could not tell the two apart.
    """
    agent, game = searchGame(checkpoint)
    for _ in range(10):
        game.playNextMove(game.currentPlayer())
    moves = game.board.allValidMovesWithWay(agent)
    assert any(len(move) > 2 for move in moves), "the position must offer a jump to be a test"

    ranked = agent._rank(agent.encoder, agent, moves)

    assert len(ranked) == len(moves)
    assert {tuple(move) for move in ranked} == {tuple(move) for move in moves}


def test_ranking_is_ordered_by_the_policy_probability(checkpoint):
    """Move ordering is the whole reason a 6-candidate search is worth anything."""
    agent, game = searchGame(checkpoint)
    moves = game.board.allValidMovesWithWay(agent)

    ranked = agent._rank(agent.encoder, agent, moves)

    agent.encoder._legalCache = None
    actions = agent._encode(agent.encoder, agent, ranked)
    mask = np.zeros(agent.encoder.actionCount, dtype=bool)
    mask[actions] = True
    tensor, _ = agent.model.policy.obs_to_tensor(agent.encoder._observation())
    with torch.no_grad():
        distribution = agent.model.policy.get_distribution(tensor, action_masks=mask)
        probabilities = distribution.distribution.probs[0].numpy()  # pyright: ignore[reportAttributeAccessIssue]

    ordered = [float(probabilities[action]) for action in actions]
    assert ordered == sorted(ordered, reverse=True)


def test_the_search_plays_a_whole_game_legally(checkpoint):
    """The integration guard: 30 plies with the engine validating every move.

    The searched position is a hypothetical built by mutating the real board,
    so a leaked ``moveApplied`` or a stale ``_legalCache`` would surface here as
    an illegal move rather than as a slightly worse game.
    """
    _, game = searchGame(checkpoint, candidates=3, replies=2)

    for _ in range(30):
        if game.winner() is not None:
            break
        game.playNextMove(game.currentPlayer())

    assert game.gameLength() > 0


def test_the_search_leaves_the_board_as_it_found_it(checkpoint):
    """Candidates are scored by mutating the real board and undoing it."""
    agent, game = searchGame(checkpoint)
    before = game.board.boardState().copy()
    beforePositions = set(agent.positions)

    agent.chooseMove(game.board.allValidMovesWithWay(agent), game.board)

    np.testing.assert_array_equal(game.board.boardState(), before)
    assert agent.positions == beforePositions


def test_zero_replies_stops_before_the_opponent_answers(checkpoint):
    """``replies=0`` is the diagnostic arm, and it must still play.

    It separates the two ways the search can fail -- the critic, and "the
    opponent plays as I would". With no reply ply only the critic is left.
    """
    agent, game = searchGame(checkpoint, replies=0)

    moves = game.board.allValidMovesWithWay(agent)
    chosen = agent.chooseMove(moves, game.board)

    assert chosen in moves


def test_the_position_played_from_is_remembered(checkpoint):
    """``seen`` is what the repetition rule is built on."""
    agent, game = searchGame(checkpoint)
    assert agent.seen == set()

    agent.chooseMove(game.board.allValidMovesWithWay(agent), game.board)

    assert game.board.boardState().tobytes() in agent.seen


def statesAfter(game, agent, moves):
    """The board state each move leads to, keyed by the move."""
    states = {}
    for move in moves:
        with game.board.moveApplied(move, agent):
            states[tuple(move)] = game.board.boardState().tobytes()
    return states


def favouring(states, wanted):
    """A stub ``_valueAfterReply`` that likes exactly one resulting position.

    The critic is untrained, so left to itself it scores every sibling almost
    identically and a test cannot tell "declined the repeat" from "happened to
    prefer another move". This makes the repeating move the *tempting* one, so
    only the repetition rule can steer away from it.
    """

    def value(board, opponent):
        return 100.0 if board.boardState().tobytes() == states[tuple(wanted)] else 0.0

    return value


def test_a_repeating_move_is_declined_even_when_it_evaluates_best(checkpoint):
    """The rule that removed the deadlocks (draws 8 -> 0/2)."""
    agent, game = searchGame(checkpoint, candidates=4, replies=1)
    moves = game.board.allValidMovesWithWay(agent)
    ranked = agent._rank(agent.encoder, agent, moves)[:4]
    states = statesAfter(game, agent, ranked)

    # The best-valued candidate is the one that repeats.
    agent.seen.add(states[tuple(ranked[0])])
    agent._valueAfterReply = favouring(states, ranked[0])  # pyright: ignore[reportAttributeAccessIssue]

    chosen = agent.chooseMove(moves, game.board)

    assert chosen != ranked[0]
    # The first non-repeating candidate, all remaining ones being level.
    assert chosen == ranked[1]


def test_every_candidate_repeating_falls_back_to_the_best_of_them(checkpoint):
    """Declining to move is not an option, so the rule cannot be absolute --
    and the fallback still has to pick the best repeat rather than any move."""
    agent, game = searchGame(checkpoint, candidates=3, replies=1)
    moves = game.board.allValidMovesWithWay(agent)
    ranked = agent._rank(agent.encoder, agent, moves)[:3]
    states = statesAfter(game, agent, ranked)

    for move in ranked:
        agent.seen.add(states[tuple(move)])
    agent._valueAfterReply = favouring(states, ranked[-1])  # pyright: ignore[reportAttributeAccessIssue]

    assert agent.chooseMove(moves, game.board) == ranked[-1]


def test_attaching_to_a_game_forgets_the_previous_one(checkpoint):
    """These players are reused across games by every script that runs a match,
    so history from the last game would decline moves in this one."""
    agent, game = searchGame(checkpoint)
    agent.chooseMove(game.board.allValidMovesWithWay(agent), game.board)
    assert agent.seen

    nextGame = ComputedGame()
    nextGame.seed(1)
    nextGame.initGame([agent, Computer(2, "distance")])
    agent.attachTo(nextGame)

    assert agent.seen == set()


def test_a_won_position_short_circuits_the_reply_search(checkpoint):
    """A move that wins is worth ``WIN_VALUE``, whatever the opponent answers.

    The branch is unreachable from a fresh game inside a test's budget, so the
    win itself is faked; what is pinned is that the check happens before the
    reply ply and returns a value no critic estimate can compete with.
    """
    agent, game = searchGame(checkpoint)
    opponent = agent.opponents[0]

    ordinary = agent._valueAfterReply(game.board, opponent)
    agent.isWinning = lambda: True  # pyright: ignore[reportAttributeAccessIssue]
    won = agent._valueAfterReply(game.board, opponent)

    assert won == SearchingComputer.WIN_VALUE
    assert won > abs(ordinary) * 1000


def test_the_opponent_encoder_is_seated_opposite(checkpoint):
    """The reply is chosen from the opponent's viewpoint by the same network.
    Seat them both the same way and the search would pick the reply *it* would
    like, not the one that hurts it."""
    agent = SearchingComputer(HalmaEnv.AGENT_SEAT, checkpoint)
    assert agent.opponentEncoder.selfSeat == HalmaEnv.OPPONENT_SEAT

    fromOtherSeat = SearchingComputer(HalmaEnv.OPPONENT_SEAT, checkpoint)
    assert fromOtherSeat.opponentEncoder.selfSeat == HalmaEnv.AGENT_SEAT
