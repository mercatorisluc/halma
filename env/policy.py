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
"""

from __future__ import annotations

from typing import Any

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn


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


class FactoredMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO's policy with the flat action head swapped for a factored one."""

    def __init__(self, *args: Any, fieldCount: int = 121, **kwargs: Any) -> None:
        self.fieldCount = fieldCount
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        self.action_net = FactoredActionHead(self.mlp_extractor.latent_dim_pi, self.fieldCount)
        # The optimizer was built over the old head's parameters, so it has to
        # be rebuilt to see the new ones at all.
        settings = {"lr": lr_schedule(1), **self.optimizer_kwargs}
        self.optimizer = self.optimizer_class(self.parameters(), **settings)
