import math
import random
from typing import Tuple, Dict, Optional

from cogrip.pentomino.state import Board
from cogrip.tasks import Task
from neumad.agents.speaker.intent import IntentSpeaker
from neumad.envs import SpeakerIntent


class IntentSpeakerL0(IntentSpeaker):
    """
    A speaker that automatically performs language production:

    - when to speak (silence or not)
    -> automatically resolves what to say
    -> automatically resolves how to say it

    The intent realisation utilizes language templates.
    """

    def __init__(self, distance_threshold: int, time_threshold: int):
        """
        :param distance_threshold: the threshold for "essential" movement
        :param time_threshold: the number of steps where no "essential" movement happened
        """
        # (a) a distance-determined (cross a distance of 3 compared to the last gripper position)
        super().__init__()
        self.last_feedback_distance_threshold = distance_threshold
        # (b) a time-determined (e.g. waiting for 5 seconds)
        self.last_feedback_time_threshold = time_threshold  # actions
        self.wait_counter = 0
        self.current_distance_to_target: float = -1
        self.previous_state = None

    def get_level(self) -> int:
        return 0

    def on_reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        super().on_reset(task, board, gripper_pos, info)
        self.current_distance_to_target = math.dist(gripper_pos, self.target_pos)
        self.previous_state = None

    def _on_intent(self, intent: SpeakerIntent) -> str:
        if intent == SpeakerIntent.silence:
            if self._has_gripper_moved():
                self.previous_state = None
            utterance = self.say_nothing()
        else:
            utterance = self.say_something()
        self._mark_reference_point()  # todo: is it a bug to reset wait_counter here?
        return utterance

    def say_something(self) -> str:
        self.log_debug(f"generate_feedback")
        self.current_distance_to_target = math.dist(self.current_gripper_pos, self.target_pos)
        if self._has_gripper_moved():
            if self._is_gripper_over_piece():
                utterance = self._generate_take_feedback()
            else:
                utterance = self._generate_move_feedback()
        else:
            utterance = self._generate_wait_feedback()
        return utterance

    def _mark_reference_point(self):
        self.last_gripper_pos = self.current_gripper_pos
        self.wait_counter = 0  # reset wait counter

    def _generate_wait_feedback(self) -> Tuple[str, Optional[str]]:
        self.log_debug(f"_generate_wait_feedback")
        self.wait_counter += 1
        if self.__is_time_threshold_reached():
            # print("_time_threshold_reached")
            # mark reference point, so that the wait action not immediately again
            self._mark_reference_point()
            feedback, po = self._on_gripper_waits()
        else:
            po = None
            feedback = self.say_nothing()
        return feedback, po

    def _generate_move_feedback(self) -> str:
        self.log_debug(f"_generate_move_feedback")
        last_distance_to_target = math.dist(self.last_gripper_pos, self.target_pos)
        if self.current_distance_to_target < last_distance_to_target:
            return self.say_confirm_direction()
        if self.previous_state == "decline_move":  # do not repeat
            self.previous_state = None
            return self.say_initiate_direction()
        self.previous_state = "decline_move"
        return self.say_decline_direction()

    def _generate_take_feedback(self, initiate_take=False) -> str:
        self.log_debug(f"_generate_piece_feedback")
        if self._is_gripper_over_target():
            if initiate_take:
                return self.say_initiate_take()
            else:
                return self.say_confirm_take()
        if self.previous_state == "decline_take":  # do not repeat
            self.previous_state = None
            return self.say_initiate_direction()
        self.previous_state = "decline_take"
        return self.say_decline_take()

    def __is_time_threshold_reached(self):
        self.log_debug(f"__is_time_threshold_reached: {self.wait_counter}")
        return self.wait_counter >= self.last_feedback_time_threshold

    def _has_gripper_moved(self) -> bool:
        distance_crossed_since_last_feedback = math.dist(self.last_gripper_pos, self.current_gripper_pos)
        self.log_debug(f"__has_gripper_moved: {distance_crossed_since_last_feedback}")
        return distance_crossed_since_last_feedback >= self.last_feedback_distance_threshold

    def _on_gripper_waits(self) -> Tuple[str, Optional[str]]:
        self.log_debug(f"_on_gripper_waits")
        if self._is_gripper_over_piece():  # gripper waits over a piece
            return self._generate_take_feedback(initiate_take=True), None

        if self.previous_state == "reference":  # do not repeat the reference
            self.log_debug(f"Previous_state: {self.previous_state}")
            self.previous_state = None
            return self.say_initiate_direction(), None

        if self._is_target_visible():  # produce reference if target in view
            self.log_debug(f"Target is visible")
            reference, po = self.generate_reference(prefer_mode="color")
            self.previous_state = "reference"
            return reference, po

        utterance_type = random.choice([0, 1])  # choose reference or direction on wait
        self.log_debug(f"Choose utterance_type: {utterance_type}")
        if utterance_type == 0:
            reference, po = self.generate_reference(prefer_mode="position")
            self.previous_state = "reference"
            return reference, po

        self.previous_state = None
        return self.say_initiate_direction(), None
