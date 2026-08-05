"""Fixtures shared by more than one test module.

The untrained checkpoint moved here once a second module needed a policy on
disk: ``test_neural_player.py`` seats one as a player, ``test_env.py`` has the
environment draw one as its opponent. Two copies of the policy and
feature-extractor wiring would be two things to keep in step with ``env/``.
"""

import pytest
from sb3_contrib import MaskablePPO

from env.features import HalmaFeatures
from env.halmaEnv import HalmaEnv
from env.policy import FactoredMaskablePolicy


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    """An untrained policy on disk. Untrained is enough -- what is under test
    is the machinery around the network, not the network."""
    model = MaskablePPO(
        FactoredMaskablePolicy,
        HalmaEnv(),
        policy_kwargs={"features_extractor_class": HalmaFeatures},
        seed=0,
    )
    path = tmp_path_factory.mktemp("models") / "untrained"
    model.save(path)
    return str(path)
