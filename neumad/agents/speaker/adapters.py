from typing import Dict, Union

import numpy as np

from neumad.agents import HeuristicAgent
from neumad.agents.ppo import PPOAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.agents.speaker.intent_level_2 import IntentSpeakerL2
from neumad.envs import SpeakerAction
from neumad.envs.take_env import OBS_SYMBOLIC_POS
from typing import Tuple


class HeuristicSpeakerAgent(HeuristicAgent):

    def __init__(self, speaker: HeuristicSpeaker):
        self.speaker = speaker
        self.current_utterance = None
        td = speaker.last_feedback_distance_threshold
        tt = speaker.last_feedback_time_threshold
        super().__init__(agent_name=f"speaker_td={td}_tt={tt}")

    def on_reset(self, current_task, current_board, current_gripper_coords, info):
        self.speaker.on_reset(current_task, current_board, current_gripper_coords, info)

    def on_step(self, obs, action=None) -> Union[np.array, int, str]:
        # speaker in env is called two times:
        # once to determine the action and once to verbalize the action
        if action is None:
            # 1st call: return action, store utterance
            gripper_pos = obs[OBS_SYMBOLIC_POS]
            utterance, action = self.speaker.generate_utterance(gripper_pos)
            self.current_utterance = utterance
            return action
        # 2nd call: return utterance
        return self.current_utterance


class NeuralSpeakerAgent(PPOAgent):
    """ Decorates an agent with speaker functionality (the verbalizer)"""

    def __init__(self, agent_path, agent_name, hparams: Dict = None):
        super().__init__(agent_path, agent_name, hparams)
        self.verbalizer = IntentSpeakerL2()
        self.deterministic = True
        self.episode_start = np.ones((1,), dtype=bool)
        self.states = None

    def on_reset(self, current_task, current_board, current_gripper_coords, info):
        self.verbalizer.on_reset(current_task, current_board, current_gripper_coords, info)
        self.states = None
        self.episode_start = np.ones((1,), dtype=bool)

    def on_step(self, obs: Union[Dict, np.array], action: SpeakerAction = None) -> Tuple[np.array, SpeakerAction]:
        """
        :param obs: the observations for the speaker
        :param action: optional to give only for verbalization of a speaker action
        :return: the utterance and the speaker action (possibly chosen by the obs only)
        """
        if action is None:
            action, states = self.algorithm.predict(obs, state=self.states,
                                                episode_start=self.episode_start,
                                                deterministic=self.deterministic)
            self.episode_start = False
            self.states = states
            return SpeakerAction.from_object(action)
        assert isinstance(action, SpeakerAction), \
            f"Argument 'action' should be a SpeakerAction, but is {type(action)}"
        gripper_pos = obs[OBS_SYMBOLIC_POS]
        speaker_action = SpeakerAction(action)
        utterance = self.verbalizer.verbalize(speaker_action.to_list(), gripper_pos)
        return utterance
