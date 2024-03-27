from typing import Dict

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from stable_baselines3.common.utils import get_device


class LanguageExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.device = get_device("auto")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # the policy model calls preprocess_obs which converts Box features to float()
        # we have to undo this operation
        observations = observations.long()
        return self._on_forward(observations)

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class TextEncoder(LanguageExtractor):
    """ Treat each word as a word embedding and train a language model on it"""

    def __init__(self, observation_space: gym.spaces.Box, hparams: Dict, shared_word_embeddings: nn.Embedding = None):
        super().__init__(observation_space, hparams["agent.features"])
        self.vocab_size = observation_space.high[0]
        if shared_word_embeddings is not None:
            self.word_embedding = shared_word_embeddings
        else:
            self.word_embedding = nn.Embedding(self.vocab_size, hparams["word_embedding_dims"], padding_idx=0)
        self.text_rnn = nn.GRU(hparams["word_embedding_dims"], hparams["agent.features"], batch_first=True)

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        word_embeddings = self.word_embedding(observations)
        _, hidden = self.text_rnn(word_embeddings)
        return hidden[-1]
