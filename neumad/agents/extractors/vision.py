from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
import torch


class OverviewObsEncoder(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, feature_dims: int, in_channels: int = 3):
        super().__init__(observation_space, feature_dims)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=feature_dims, kernel_size=5, padding=2),
            nn.BatchNorm2d(feature_dims),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return x


class PartialViewObsEncoder(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, feature_dims, in_channels=3):
        super().__init__(observation_space, feature_dims)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=feature_dims, kernel_size=5, padding=2),
            nn.BatchNorm2d(feature_dims),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return x
