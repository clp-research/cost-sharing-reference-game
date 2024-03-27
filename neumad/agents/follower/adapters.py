from typing import Dict, Union, Tuple

import numpy as np

from neumad.agents.ppo import PPOAgent
from neumad.envs import SpeakerAction, FollowerAction


class NeuralFollowerAgent(PPOAgent):

    def __init__(self, agent_path, agent_name, hparams: Dict = None):
        super().__init__(agent_path, agent_name, hparams)
        self.deterministic = True
        self.episode_start = np.ones((1,), dtype=bool)
        self.states = None

    def on_reset(self, current_task, current_board, current_gripper_coords, info):
        self.states = None
        self.episode_start = np.ones((1,), dtype=bool)

    def on_step(self, obs: Union[Dict, np.array], action: SpeakerAction = None) -> Tuple[np.array, FollowerAction]:
        """
        :param obs: the observations for the speaker
        :param action: optional to give only for verbalization of a speaker action
        :return: the utterance and the speaker action (possibly chosen by the obs only)
        """
        assert action is None, f"Argument 'action' should be None, but is {type(action)}"
        action, states = self.algorithm.predict(obs, state=self.states,
                                                episode_start=self.episode_start,
                                                deterministic=self.deterministic)
        self.episode_start = False
        self.states = states
        return action
