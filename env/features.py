"""Feature extractor for the Halma observation.

A flat vector of 121 fields hides the board's geometry: nothing in it says that
field 5 borders field 6, so a plain MLP has to learn adjacency from data. The
observation is laid out on the 17x17 raster instead (see
``HalmaEnv._rasterCell``), and this runs a small convolutional stack over it so
neighbourhood comes for free.

Small on purpose. The board is 17x17 with 121 useful cells, far below the sizes
Stable-Baselines' stock ``NatureCNN`` is built for, and its stride-4 first layer
would throw most of the board away immediately.

Small also because training is bound by the network, not by the game: measured
on one machine, the environment alone steps at 1648/s and the environment with
PPO in the loop at 87/s, so the engine is 5% of a training step and everything
else is here.

**The trunk is shared, the branches are not.** The convolutions run once and
both branches read their output, but each squeezes and flattens it its own way,
because the two consumers want different things from the same board:

* the policy wants "which piece, where", so its branch keeps the full 17x17
  resolution and squeezes to few channels -- position is the signal and the
  1x1 is only there to keep the flatten affordable;
* the value wants "is this structure won", which is a property of the whole
  position rather than of one cell. Its branch gets a convolution of its own,
  a wider squeeze, and -- concatenated -- a global average over the trunk's
  channels, which is the one summary a per-cell flatten does not hand it.

That split is the answer to a measured problem. ``env/searchPlayer.py`` ranked
sibling positions with the critic and played *worse* than the policy it wrapped
(RESULTS.md), and of the two candidate causes -- the PPO objective, which
only ever asks the critic to be a baseline, and capacity -- this addresses the
second: the value head used to be 20,673 parameters behind a straggler whose
shape the policy had chosen. It does not address the first, and the sibling-
ranking probe recorded in RESULTS.md is still the measurement that says
which of the two was actually binding.

The output is the two branches concatenated, which
:class:`~env.policy.SplitMlpExtractor` slices apart again -- Stable-Baselines
runs one extractor and hands its whole output to both networks, so carrying
both halves in one vector is how a shared trunk with unshared branches fits
that interface without paying for a second trunk.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HalmaFeatures(BaseFeaturesExtractor):
    """Convolutions over the board raster, with separate policy and value branches."""

    def __init__(
        self,
        observationSpace: gym.spaces.Dict,
        features: int = 256,
        valueFeatures: int = 256,
        valueChannels: int = 16,
    ) -> None:
        # The two branches are concatenated, so the declared width is their
        # sum; SplitMlpExtractor cuts at self.policyDim.
        super().__init__(observationSpace, features_dim=features + valueFeatures)
        self.policyDim = features
        self.valueDim = valueFeatures

        boardSpace = observationSpace["board"]
        assert isinstance(boardSpace, gym.spaces.Box)
        channels = int(boardSpace.shape[0])
        scalarSpace = observationSpace["scalars"]
        assert isinstance(scalarSpace, gym.spaces.Box)
        scalarCount = int(scalarSpace.shape[0])

        # Stride 1 and padding 1 throughout: at 17x17 there is nothing to
        # downsample, and every field matters.
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # The 1x1 is a channel squeeze, and it is what the flatten costs.
        # Ending on 32 channels hands Linear 32*17*17 = 9248 numbers, which was
        # 2.37M of the extractor's 2.43M parameters -- the convolutions
        # themselves are only 57k. Squeezing to 8 channels first cuts that to
        # 0.59M without touching the spatial resolution, which has to stay: the
        # policy needs to know *where* a piece is, so pooling the board away is
        # not an option.
        self.policyBranch = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # One more convolution than the policy gets, and twice the channels
        # through the squeeze. The extra layer is there because "is this
        # position won" depends on longer-range structure than "is this piece
        # worth moving" -- one more 3x3 widens the receptive field by a ring.
        self.valueBranch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, valueChannels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Averaged over the board, so it survives where a piece happens to be:
        # 64 numbers saying how much of each learned pattern is present at all.
        # Cheap, and the one thing a per-cell flatten cannot express.
        self.valuePool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        with torch.no_grad():
            sample = torch.zeros(1, channels, *boardSpace.shape[1:])
            trunk = self.trunk(sample)
            policyOut = int(self.policyBranch(trunk).shape[1])
            valueOut = int(self.valueBranch(trunk).shape[1]) + int(self.valuePool(trunk).shape[1])

        # Both branches see the scalars: they are five numbers, and the value
        # in particular is largely a question about the two "pieces home" counts.
        self.policyHead = nn.Sequential(nn.Linear(policyOut + scalarCount, features), nn.ReLU())
        self.valueHead = nn.Sequential(nn.Linear(valueOut + scalarCount, valueFeatures), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        trunk = self.trunk(observations["board"])
        scalars = observations["scalars"]
        policy = self.policyHead(torch.cat([self.policyBranch(trunk), scalars], dim=1))
        value = self.valueHead(
            torch.cat([self.valueBranch(trunk), self.valuePool(trunk), scalars], dim=1)
        )
        return torch.cat([policy, value], dim=1)
