"""Tests for seating a trained policy as an ordinary player.

The failure this guards against is silent. A policy handed a subtly different
observation than it trained on still returns a legal move, so a human would see
a working opponent that simply plays badly, and nothing would point at the
encoding. So the property pinned here is agreement: the seated player and the
environment must choose the same move from the same position.
"""

import pytest
from sb3_contrib import MaskablePPO

from env.features import HalmaFeatures
from env.halmaEnv import HalmaEnv
from env.neuralPlayer import NeuralComputer
from env.policy import FactoredMaskablePolicy
from game.gameManager import ComputedGame, InteractiveGame
from game.player import Computer, HumanPlayer


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An untrained policy on disk. Untrained is enough -- what is under test
    is the encoding around the network, not the network."""
    env = HalmaEnv()
    model = MaskablePPO(
        FactoredMaskablePolicy,
        env,
        policy_kwargs={"features_extractor_class": HalmaFeatures},
        seed=0,
    )
    path = tmp_path_factory.mktemp("models") / "untrained"
    model.save(path)
    return str(path)


def openingEnvironment():
    """A HalmaEnv sitting at the untouched opening, agent to move.

    Play order is randomised, so some seeds have the opponent move first and
    leave a position no fresh game shares.
    """
    env = HalmaEnv()
    for seed in range(20):
        env.reset(seed=seed)
        if env.game.gameLength() == 0:
            return env
    raise AssertionError("no seed left the agent on move at the opening")


def test_the_seated_policy_plays_what_the_environment_would_play(checkpoint):
    env = openingEnvironment()
    model = MaskablePPO.load(checkpoint)
    action, _ = model.predict(
        env._observation(), action_masks=env.action_masks(), deterministic=True
    )
    expected = env.normalizer.inverseMove(
        env.decodeAction(int(action)), env._permutationKey(env._player(HalmaEnv.AGENT_SEAT))
    )

    agent = NeuralComputer(HalmaEnv.AGENT_SEAT, checkpoint)
    game = InteractiveGame()
    game.initGame([agent, HumanPlayer(2)])
    agent.attachTo(game)

    chosen = agent.chooseMove(game.board.allValidMovesWithWay(agent), game.board)
    assert (chosen[0], chosen[-1]) == expected


def test_a_policy_refuses_a_seat_whose_view_it_was_not_built_for(checkpoint):
    with pytest.raises(ValueError, match="seat"):
        NeuralComputer(2, checkpoint)


def test_the_seated_policy_plays_a_whole_game_legally(checkpoint):
    """Every move legal for 40 plies, which is what the raise in chooseMove
    would catch if the encoder and the live game drifted apart mid-game."""
    agent = NeuralComputer(HalmaEnv.AGENT_SEAT, checkpoint)
    game = ComputedGame()
    game.seed(0)
    game.initGame([agent, Computer(2, "advancedDistScore")])
    agent.attachTo(game)

    for _ in range(40):
        if game.winner() is not None:
            break
        game.playNextMove(game.currentPlayer())
    assert game.gameLength() > 0
