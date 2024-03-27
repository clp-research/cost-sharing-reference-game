import math
from typing import Tuple

from neumad.agents.speaker.intent_level_0 import IntentSpeakerL0
from neumad.envs import SpeakerAction
from neumad.envs.take_env import SILENCE


class HeuristicSpeaker(IntentSpeakerL0):

    def __init__(self, distance_threshold: int, time_threshold: int):
        super().__init__(distance_threshold, time_threshold)

    def generate_utterance(self, gripper_pos: Tuple[int, int]) -> Tuple[str, int]:
        """        This method is invoked at each time step        """
        self.current_gripper_pos = gripper_pos
        if self.is_first:  # speaker has not mentioned the main objective yet (has just been reset)
            self.is_first = False
            utterance, preference_order = self.generate_reference()
            action = SpeakerAction.get_reference(preference_order)
        else:
            # this code is similar to say_something in IntentSpeakerL0
            self.current_distance_to_target = math.dist(self.current_gripper_pos, self.target_pos)
            if self._has_gripper_moved():
                if self._is_gripper_over_piece():
                    utterance = self._generate_take_feedback()
                else:
                    utterance = self._generate_move_feedback()

                if "yes" in utterance:
                    action = SpeakerAction.confirmation
                elif "not" in utterance:
                    action = SpeakerAction.correction
                else:
                    action = SpeakerAction.get_directive(utterance)

                self._mark_reference_point()
            else:  # follower is not yet over the distance threshold
                utterance, po = self._generate_wait_feedback()
                if "yes" in utterance:
                    action = SpeakerAction.confirmation
                elif "not" in utterance:
                    action = SpeakerAction.correction
                elif "go" in utterance:
                    action = SpeakerAction.get_directive(utterance)
                else:
                    if po:
                        action = SpeakerAction.get_reference(po)
                    elif utterance == SILENCE:
                        action = SpeakerAction.silence
                    else:
                        action = SpeakerAction.get_directive(utterance)
        if utterance == SILENCE:
            action = SpeakerAction.silence
        return utterance, action
