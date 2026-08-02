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
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HalmaFeatures(BaseFeaturesExtractor):
    """Convolutions over the board raster, concatenated with the scalars."""

    def __init__(self, observationSpace: gym.spaces.Dict, features: int = 256) -> None:
        super().__init__(observationSpace, features_dim=features)

        boardSpace = observationSpace["board"]
        assert isinstance(boardSpace, gym.spaces.Box)
        channels = int(boardSpace.shape[0])
        scalarSpace = observationSpace["scalars"]
        assert isinstance(scalarSpace, gym.spaces.Box)
        scalarCount = int(scalarSpace.shape[0])

        # Stride 1 and padding 1 throughout: at 17x17 there is nothing to
        # downsample, and every field matters.
        #
        # The 1x1 at the end is a channel squeeze, and it is what the flatten
        # costs. Ending on 32 channels hands Linear 32*17*17 = 9248 numbers,
        # which was 2.37M of the extractor's 2.43M parameters -- the
        # convolutions themselves are only 57k. Squeezing to 8 channels first
        # cuts that to 0.59M without touching the spatial resolution, which has
        # to stay: the policy needs to know *where* a piece is, so pooling the
        # board away is not an option.
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, channels, *boardSpace.shape[1:])
            convOut = int(self.conv(sample).shape[1])

        self.head = nn.Sequential(nn.Linear(convOut + scalarCount, features), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        board = self.conv(observations["board"])
        return self.head(torch.cat([board, observations["scalars"]], dim=1))
