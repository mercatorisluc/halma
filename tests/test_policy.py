"""Tests for the network wiring: the factored action head and the split trunk.

Both are silent when wrong. A misaligned action head still returns a legal
move, and a critic accidentally reading the policy's features still produces a
number PPO will happily train on -- neither shows up as anything but a weaker
agent, which is indistinguishable from a bad run.
"""

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.features import HalmaFeatures
from env.halmaEnv import HalmaEnv
from env.policy import (
    DEFAULT_NET_ARCH,
    FactoredActionHead,
    FactoredMaskablePolicy,
    SplitMlpExtractor,
)


def buildModel(**policyKwargs):
    return MaskablePPO(
        FactoredMaskablePolicy,
        HalmaEnv(),
        policy_kwargs={"features_extractor_class": HalmaFeatures, **policyKwargs},
        seed=0,
    )


def test_the_extractor_output_splits_into_a_policy_and_a_value_half():
    model = buildModel()
    extractor = model.policy.features_extractor
    assert isinstance(extractor, HalmaFeatures)
    assert extractor.features_dim == extractor.policyDim + extractor.valueDim


def test_the_two_nets_read_disjoint_halves_of_the_features():
    """The whole point of the split: perturbing the value half must not move
    the actor's latent, and vice versa. A shared MlpExtractor -- which is what
    this replaces -- would fail both halves of this."""
    model = buildModel()
    extractor = model.policy.features_extractor
    assert isinstance(extractor, HalmaFeatures)
    mlp = model.policy.mlp_extractor
    assert isinstance(mlp, SplitMlpExtractor)

    torch.manual_seed(0)
    features = torch.randn(4, extractor.features_dim)
    actor, critic = mlp(features)

    disturbed = features.clone()
    disturbed[:, extractor.policyDim :] += 1.0
    torch.testing.assert_close(mlp.forward_actor(disturbed), actor)

    disturbed = features.clone()
    disturbed[:, : extractor.policyDim] += 1.0
    torch.testing.assert_close(mlp.forward_critic(disturbed), critic)


def test_the_value_net_is_wider_than_the_policy_net_by_default():
    """The critic's capacity is the thing DEFAULT_NET_ARCH exists to raise, so
    a default quietly reverting to sb3's symmetric [64, 64] should fail here."""
    model = buildModel()
    assert model.policy.mlp_extractor.latent_dim_pi == DEFAULT_NET_ARCH["pi"][-1]
    assert model.policy.mlp_extractor.latent_dim_vf == DEFAULT_NET_ARCH["vf"][-1]
    assert model.policy.mlp_extractor.latent_dim_vf > model.policy.mlp_extractor.latent_dim_pi


def test_an_explicit_net_arch_still_wins():
    model = buildModel(net_arch={"pi": [32], "vf": [48]})
    assert model.policy.mlp_extractor.latent_dim_pi == 32
    assert model.policy.mlp_extractor.latent_dim_vf == 48


def test_the_action_head_scores_a_pair_as_the_sum_of_its_halves():
    """``logit[start, end] = startScore[start] + endScore[end]`` is the
    property that makes one update inform hundreds of pairs, and it also fixes
    the flattening order the environment's encodeAction assumes."""
    model = buildModel()
    head = model.policy.action_net
    assert isinstance(head, FactoredActionHead)
    torch.manual_seed(0)
    latent = torch.randn(1, model.policy.mlp_extractor.latent_dim_pi)
    with torch.no_grad():
        logits = head(latent)[0]
        start = head.startScore(latent)[0]
        end = head.endScore(latent)[0]
    fields = head.fieldCount
    assert logits.shape == (fields * fields,)
    for a, b in ((0, 0), (5, 7), (120, 120), (17, 103)):
        torch.testing.assert_close(logits[a * fields + b], start[a] + end[b])


def test_the_policy_consumes_the_environments_observation():
    """End to end: the extractor's declared channel count and the environment's
    plane count are set independently, so a mismatch is a live possibility."""
    env = HalmaEnv()
    observation, _ = env.reset(seed=0)
    model = buildModel()
    action, _ = model.predict(observation, action_masks=env.action_masks(), deterministic=True)
    assert int(action) in env._legalActions()
    with torch.no_grad():
        value = model.policy.predict_values(model.policy.obs_to_tensor(observation)[0])
    assert np.isfinite(float(value))
