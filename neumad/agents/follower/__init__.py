import abc
from typing import Tuple, Dict, Union

import numpy as np

from cogrip.pentomino.state import Board
from cogrip.tasks import Task
from neumad.agents import NoopAgent, HeuristicAgent
from neumad.envs import FollowerAction, SpeakerAction


class HeuristicFollower(abc.ABC):

    def __init__(self, debug=False):
        self.task = None
        self.board = None
        self.current_pos = None
        self.current_pos_area = None
        self.info = None
        self.debug = debug

    def log_debug(self, message):
        if self.debug:
            print(f"{self.__class__.__name__}: {message}")

    def on_reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        self.task = task
        self.board = board
        self.current_pos = gripper_pos
        self.current_pos_area = None
        self.info = info

    def _translate_view_coord_to_global(self, coord, fov_size):
        """
        Relative coords are in (0,0) to (V,V) in current view port.
        The gripper is always in (V/2,V/2) (middle). We map to the top left coord and then translate.
        """
        if coord is None:
            return None
        fx, fy = fov_size // 2, fov_size // 2
        dx, dy = coord
        gx, gy = self.current_pos
        vx, vy = gx - fx, gy - fy
        ax, ay = vx + dx, vy + dy
        return ax, ay

    def log_info(self, key, value):
        if self.info:
            self.info[key] = value

    @abc.abstractmethod
    def on_step(self, obs) -> FollowerAction:
        pass


class NoopFollower(NoopAgent):

    def __init__(self):
        super().__init__(agent_name="noop_follower")

    def on_step(self, obs: Union[Dict, np.array], action: SpeakerAction = None) -> Union[np.array, SpeakerAction]:
        raise RuntimeError("on_step should not be called for NoopFollower")


class HeuristicFollowerAgent(HeuristicAgent):
    """ Wrapper for evaluating a heuristic follower (not vectorized) """

    def __init__(self, heuristic_follower: HeuristicFollower):
        self.heuristic_follower: HeuristicFollower = heuristic_follower
        strategy = heuristic_follower.strategy
        confidence = int(heuristic_follower.confidence[0] * 100)
        super().__init__(agent_name=f"follower_{strategy}_c={confidence}")

    def on_reset(self, current_task, current_board, current_gripper_coords, info) -> None:
        self.heuristic_follower.on_reset(current_task, current_board, current_gripper_coords, info)

    def on_step(self, observation, action=None) -> Union[np.array, int]:
        action = self.heuristic_follower.on_step(observation)
        return action

    def get_env(self):  # for compatibility with callback
        return self.env
