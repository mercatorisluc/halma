"""A policy whose action head knows that an action is a pair of fields.

Actions are encoded ``start * fieldCount + end``. A plain head is one
``Linear(latent, 14641)``: 951,665 parameters, one row per pair, each learned
alone. Nothing ties "piece 5 to field 7" to "piece 5 to field 8", and with only
about 65 of 14641 actions legal at a time each row is almost never trained.

This scores the two halves separately and adds them::

    logit[start, end] = startScore[start] + endScore[end]

Two ``Linear(latent, 121)`` layers, some 15,000 parameters, and every move from
the same piece now shares its start term while every move onto the same field
shares its end term -- so one update informs hundreds of pairs instead of one.

Additive rather than a product on purpose: it cannot express "this piece to
here is good but that piece to here is not", but it can express "fields near
the target are worth moving to", which is the greedy distance policy the
heuristic bots already win with.

The policy also carries the second half of the split trunk in
``env/features.py``: :class:`SplitMlpExtractor` cuts the extractor's output
back into a policy half and a value half, so the two MLPs behind it read
different features despite Stable-Baselines running one extractor. The value
MLP is wider than the policy's by default (``vf=[256, 256]`` against
``pi=[64, 64]``) for the reason recorded in ``env/features.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn

from env.features import HalmaFeatures

# Wider than the policy's, and than Stable-Baselines' [64, 64] default for
# both. Overridable through policy_kwargs like any other net_arch.
DEFAULT_NET_ARCH = {"pi": [64, 64], "vf": [256, 256]}


class FactoredActionHead(nn.Module):
    """Scores start and end fields separately, then adds them into pair logits."""

    def __init__(self, latentDim: int, fieldCount: int) -> None:
        super().__init__()
        self.fieldCount = fieldCount
        self.startScore = nn.Linear(latentDim, fieldCount)
        self.endScore = nn.Linear(latentDim, fieldCount)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        start = self.startScore(latent)
        end = self.endScore(latent)
        # Outer sum, flattened in the same start*fieldCount + end order the
        # environment encodes actions with.
        return (start.unsqueeze(2) + end.unsqueeze(1)).flatten(1)


class SplitMlpExtractor(nn.Module):
    """Stable-Baselines' ``MlpExtractor``, but each net reads its own slice.

    Stock behaviour is that the features extractor produces one vector and both
    the policy and the value MLP read all of it. ``HalmaFeatures`` instead
    returns its policy branch and its value branch concatenated, and this cuts
    them apart again at ``policyDim``. Why the branches differ at all is in
    ``env/features.py``; what matters here is only where to cut.

    The three methods and the two ``latent_dim_*`` attributes are the whole
    interface ``ActorCriticPolicy`` uses: ``forward`` on the shared path,
    ``forward_actor`` from ``get_distribution`` and ``forward_critic`` from
    ``predict_values``.
    """

    def __init__(
        self,
        policyDim: int,
        valueDim: int,
        netArch: dict[str, list[int]],
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.policyDim = policyDim
        policyArch = netArch["pi"]
        valueArch = netArch["vf"]
        # create_mlp with output_dim=-1 builds hidden layers only, ending on an
        # activation -- which is what a latent extractor is, since the action
        # and value heads are built separately by the policy.
        self.policy_net = nn.Sequential(*create_mlp(policyDim, -1, policyArch, activation))
        self.value_net = nn.Sequential(*create_mlp(valueDim, -1, valueArch, activation))
        self.latent_dim_pi = policyArch[-1] if policyArch else policyDim
        self.latent_dim_vf = valueArch[-1] if valueArch else valueDim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[:, : self.policyDim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features[:, self.policyDim :])


class FactoredMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO's policy with the flat action head swapped for a factored one."""

    def __init__(self, *args: Any, fieldCount: int = 121, **kwargs: Any) -> None:
        self.fieldCount = fieldCount
        # Explicitly rather than setdefault: sb3 may pass net_arch=None through
        # rather than omit it, and None is what has to be replaced.
        if kwargs.get("net_arch") is None:
            kwargs["net_arch"] = DEFAULT_NET_ARCH
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Slice the split trunk apart, when there is one to slice.

        Any other extractor -- an sb3 stock one, or a checkpoint predating the
        split -- has a single undivided output, so the base class's shared
        ``MlpExtractor`` is the right thing and is left in place.
        """
        extractor = self.features_extractor
        if not isinstance(extractor, HalmaFeatures):
            super()._build_mlp_extractor()
            return
        netArch = self.net_arch if isinstance(self.net_arch, dict) else DEFAULT_NET_ARCH
        self.mlp_extractor = SplitMlpExtractor(
            extractor.policyDim,
            extractor.valueDim,
            netArch,
            self.activation_fn,
        )

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        self.action_net = FactoredActionHead(self.mlp_extractor.latent_dim_pi, self.fieldCount)
        # The optimizer was built over the old head's parameters, so it has to
        # be rebuilt to see the new ones at all.
        settings = {"lr": lr_schedule(1), **self.optimizer_kwargs}
        self.optimizer = self.optimizer_class(self.parameters(), **settings)
