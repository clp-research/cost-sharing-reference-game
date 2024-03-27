import abc
import random
from typing import Tuple, Union, Dict, List

import numpy as np

from cogrip.pentomino.state import Board
from cogrip.tasks import Task
from neumad.agents.speaker import Speaker
from neumad.agents.speaker.reference import IAMissionSpeaker
from neumad.envs import SpeakerIntent


class IntentSpeaker(abc.ABC, Speaker):
    """
    A speaker that allows a finer-grained control over language production:

    - when to speak (silence or not)
    - what to say (confirm, correct, initiate)
    - how to say it (preference order, directions)

    The intent realisation utilizes language templates.
    """
    PREFERENCE_ORDERS = ["CSP", "CPS", "PSC", "PCS", "SPC", "SCP"]

    def __init__(self):
        super().__init__()
        self.mission_speakers = dict([(po, IAMissionSpeaker(po))
                                      for po in IntentSpeaker.PREFERENCE_ORDERS])
        self.current_action: Union[np.array, List] = None

    def on_reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        super().on_reset(task, board, gripper_pos, info)
        for mission_speaker in self.mission_speakers.values():
            mission_speaker.on_reset(task, board, gripper_pos, info)

    def generate_reference(self, prefer_mode=None) -> Tuple[str, str]:
        self.log_debug(f"generate_reference")
        if prefer_mode == "position":
            preference_order = random.choice(["PSC", "PCS"])
        elif prefer_mode == "color":
            preference_order = random.choice(["CSP", "SCP"])
        elif prefer_mode == "shape":
            preference_order = random.choice(["SCP", "SPC"])
        else:  # Default: choose a preference order randomly
            preference_order = random.choice(IntentSpeaker.PREFERENCE_ORDERS)
        speaker = self.mission_speakers[preference_order]
        utterance = speaker.generate_reference()
        return utterance, preference_order

    def verbalize(self, action: Union[np.array, List], gripper_pos: Tuple[int, int]) -> str:
        """ Always assumes a tree-like structure """
        self.current_gripper_pos = gripper_pos
        self.current_action = action
        self.log_debug(self.current_action)
        intent = self.current_action[0]
        utterance = self._on_intent(intent)
        if utterance is None:
            raise ValueError(f"Speaker action cannot be translated: {self.current_action}")
        return utterance

    @abc.abstractmethod
    def get_level(self) -> int:
        pass

    @abc.abstractmethod
    def _on_intent(self, intent: SpeakerIntent) -> str:
        pass
