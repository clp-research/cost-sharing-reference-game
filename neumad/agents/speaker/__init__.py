import math
from typing import Tuple, Dict, Union

import numpy as np

from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board
from cogrip.tasks import Task
from neumad.agents import NoopAgent
from neumad.envs import SpeakerAction
from neumad.envs.take_env import SILENCE


class Speaker:  # and Monitor (world model) # todo separate Speaker and Monitor logic
    """ Speaker works solely in string space """

    def __init__(self, debug=False):
        """ Attributes reset at each episode"""
        self.mission: str = None
        self.last_gripper_pos: Tuple[int, int] = None
        self.target_pos: Tuple[int, int] = None
        self.task: Task = None
        self.board: Board = None
        self.is_first = True
        self.info = None
        self.debug = debug

        self.current_gripper_pos: Tuple[int, int] = (0, 0)
        self.proximity_threshold = 1  # Euclidean distance for "nearness"
        self.proximity_threshold_upper = 6

    def log_info_step(self, measure, value=1):
        if self.info is not None:
            self.info[f"step/speaker/{measure}"] = value

    def log_debug(self, message):
        if self.debug:
            print(f"{self.__class__.__name__}: {message}")

    def on_reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        self.mission = None
        self.last_gripper_pos = gripper_pos
        self.current_gripper_pos = gripper_pos
        self.target_pos = task.target_piece.get_centroid()
        self.task = task
        self.board = board
        self.info = info
        self.is_first = True

    def get_target_piece(self) -> Piece:
        return self.task.target_piece

    def say_nothing(self):
        self.log_info_step("silence")
        return SILENCE

    def say_confirm_direction(self):
        self.log_info_step("confirm")
        utterance = "yes"
        if self._is_target_adjacent():
            utterance += " a bit more"
        utterance += " this way"
        return utterance

    def say_confirm_take(self):
        self.log_info_step("confirm")
        if self._is_gripper_over_piece():
            return "yes take this " + self.__piece()
        return "yes take"  # not "take the" to distinguish for reference

    def say_decline_direction(self):
        self.log_info_step("decline")
        return "not this way"

    def say_decline_take(self):
        self.log_info_step("decline")
        if self._is_gripper_over_piece():
            return "not the " + self.__piece()
        return "not take"  # not "take the" to distinguish for reference

    def say_initiate_take(self):
        self.log_info_step("directive")
        self.log_info_step("take")
        if self._is_gripper_over_piece():
            return "take this " + self.__piece()
        return "take a piece"  # not "take the" to distinguish for reference

    def say_initiate_direction(self):
        self.log_info_step("directive")
        self.log_info_step("move")
        return "go " + self.__directions()

    def __directions(self) -> str:
        utterance = ""
        if self._is_target_adjacent():
            utterance += "a bit more "
        utterance = self._on_directions()
        return utterance

    def _is_target_visible(self):
        current_distance_to_target = math.dist(self.current_gripper_pos, self.target_pos)
        self.log_debug(f"_is_target_visible: {current_distance_to_target} < {self.proximity_threshold_upper}")
        return current_distance_to_target < self.proximity_threshold_upper

    def _is_target_adjacent(self):
        current_distance_to_target = math.dist(self.current_gripper_pos, self.target_pos)
        return current_distance_to_target < self.proximity_threshold

    def _on_directions(self) -> str:
        """ Default: return ground-truth directions """
        utterance = ""
        h, v = self._gt_target_direction()
        if h and not v:
            utterance += h
        if v and not h:
            utterance += v
        if v and h:
            utterance += f"{h} {v}"
        return utterance

    def _gt_target_direction(self) -> (str, str):
        # todo this might not be accurate enough b.c. pieces span a range of tiles
        g_x, g_y = self.current_gripper_pos
        t_x, t_y = self.target_pos
        horizontal = ""
        if g_x < t_x:
            horizontal = "right"
        if g_x > t_x:
            horizontal = "left"
        vertical = ""
        if g_y < t_y:
            vertical = "down"
        if g_y > t_y:
            vertical = "up"
        return horizontal, vertical

    def __piece(self) -> str:
        self.log_debug(f"{self.__class__.__name__}: piece")
        piece = self._get_underlying_piece()
        if piece is None:
            raise ValueError("No underlying piece! Guard this with _is_gripper_over_piece")
        color = piece.piece_config.color.value_name.lower()
        shape = piece.piece_config.shape.name.lower()
        return f"{color} {shape}"

    def _get_underlying_piece(self) -> Piece:
        gripper_tile = self.board.get_tile(*self.current_gripper_pos)
        piece = None
        if gripper_tile.objects:
            piece = gripper_tile.objects[-1]
        return piece

    def _is_gripper_over_piece(self) -> bool:
        gripper_tile = self.board.get_tile(*self.current_gripper_pos)
        return len(gripper_tile.objects) > 0

    def _is_gripper_over_target(self) -> bool:
        underlying_piece = self._get_underlying_piece()
        if underlying_piece is None:
            return False
        return self.task.target_piece.id_n == underlying_piece.id_n


class NoopSpeaker(NoopAgent):

    def __init__(self):
        super().__init__(agent_name="noop_speaker")

    def on_step(self, obs: Union[Dict, np.array], action: SpeakerAction = None) -> Union[np.array, SpeakerAction]:
        if action is None:
            speaker_action = SpeakerAction(SpeakerAction.silence)
            return speaker_action
        return SILENCE
