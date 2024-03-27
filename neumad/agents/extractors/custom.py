from typing import Dict

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from neumad.agents.extractors.fusion import FiLM
from neumad.agents.extractors.language import TextEncoder
from neumad.agents.extractors.vision import OverviewObsEncoder, PartialViewObsEncoder
from neumad.agents.utils import count_parameters


class VLFeaturesExtractor(BaseFeaturesExtractor):
    """ Vision and Language features extractor taking two obs space names"""

    def __init__(self, observation_space: gym.spaces.Dict, hparams: Dict):
        super().__init__(observation_space, features_dim=1)  # dummy value
        self.logger = None  # could be set later (from outside)
        self.hparams = hparams

        extractors = {}
        for name, subspace in observation_space.spaces.items():
            if name == hparams["agent.obs.vision"]:
                vision_encoder = self._create_vision_encoder(subspace, hparams)
                print("VLFeaturesExtractor: vision_encoder ->",
                      count_parameters(vision_encoder, only_trainable=True), "trainable parameters")
                extractors[name] = vision_encoder
            elif name == hparams["agent.obs.pos"]:
                pos_encoder = self._create_pos_encoder(subspace, hparams)
                print("VLFeaturesExtractor: pos_encoder ->",
                      count_parameters(pos_encoder, only_trainable=True), "trainable parameters")
                extractors[name] = pos_encoder
            elif name == hparams["agent.obs.text"]:
                text_encoder = self._create_text_encoder(subspace, hparams)
                print("VLFeaturesExtractor: text_encoder ->",
                      count_parameters(text_encoder, only_trainable=True), "trainable parameters")
                extractors[name] = text_encoder
            else:
                print("VLFeaturesExtractor: ignores observation subspace ->", name)
        self.extractors = nn.ModuleDict(extractors)

        feature_dims = hparams["agent.features"]
        self.film_vision = FiLM(in_features=feature_dims, out_features=feature_dims,
                                in_channels=feature_dims, imm_channels=feature_dims)
        self.film_vision_norm = nn.LayerNorm(feature_dims)
        self.film_pos = FiLM(in_features=feature_dims, out_features=feature_dims,
                             in_channels=feature_dims, imm_channels=feature_dims)
        self.film_pos_norm = nn.LayerNorm(feature_dims)
        self.film_pool = nn.AdaptiveMaxPool2d((1, 1))
        print("Trainable parameters (film_controller):",
              count_parameters(self.film_vision, only_trainable=True) * 2)

        # Note: feature_dims is used to init the policy mlp-extractor; stays the same for average
        if hparams["agent.fusion"] == "concat":
            self._features_dim = hparams["agent.features"] * 2
        else:
            self._features_dim = hparams["agent.features"]

    def _create_pos_encoder(self, subspace, hparams):
        print("VLFeaturesExtractor: pos_encoder -> OverviewObsEncoder")
        return OverviewObsEncoder(subspace, hparams["agent.features"], in_channels=4)

    def _create_vision_encoder(self, subspace, hparams):
        print("VLFeaturesExtractor: vision_encoder -> PartialViewObsEncoder")
        return PartialViewObsEncoder(subspace, hparams["agent.features"])

    def _create_text_encoder(self, subspace: gym.spaces.Box, hparams):
        print(f"VLFeaturesExtractor: text_encoder -> "
              f"GRU({hparams['agent.features']}, {hparams['word_embedding_dims']})")
        self.vocab_size = subspace.high[0]
        self.word_embeddings = nn.Embedding(self.vocab_size, hparams["word_embedding_dims"], padding_idx=0)
        return nn.Sequential(TextEncoder(subspace, hparams=hparams, shared_word_embeddings=self.word_embeddings),
                             nn.LayerNorm(hparams["agent.features"]))

    def forward(self, observations: TensorDict) -> torch.Tensor:

        key = self.hparams["agent.obs.vision"]
        vision_embeddings = self.extractors[key](observations[key])

        key = self.hparams["agent.obs.pos"]
        pos_embeddings = self.extractors[key](observations[key])

        key = self.hparams["agent.obs.text"]
        language_embeddings = self.extractors[key](observations[key])

        filmed_vision = self.film_pool(self.film_vision(vision_embeddings, language_embeddings))
        filmed_vision = filmed_vision.reshape(filmed_vision.shape[0], -1)
        filmed_vision = self.film_vision_norm(filmed_vision)

        filmed_pos = self.film_pool(self.film_pos(pos_embeddings, language_embeddings))
        filmed_pos = filmed_pos.reshape(filmed_pos.shape[0], -1)
        filmed_pos = self.film_pos_norm(filmed_pos)

        if self.hparams["agent.fusion"] == "concat":
            x = torch.concatenate([filmed_vision, filmed_pos], dim=1)
        else:  # additive
            x = filmed_pos + filmed_vision
        return x

    @classmethod
    def get_name(cls):
        return cls.__name__
