import abc
import random
from typing import Tuple, Optional, Any, Dict

import gymnasium as gym
from tqdm import tqdm

from cogrip import render
from cogrip.core.engine import Engine
from cogrip.pentomino.config import PentoConfig
from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board

from cogrip.tasks import TaskLoader, Task

EPISODE_STEP_COUNT = "episode/step/count"
EPISODE_OUTCOME_FAILURE = "episode/outcome/failure"
EPISODE_OUTCOME_SUCCESS = "episode/outcome/success"
EPISODE_OUTCOME_ABORT = "episode/outcome/abort"

GRIPPER_ID = 0


class CoGripEnv(gym.Env, abc.ABC):
    """
        CoGriEnv defines an environment with two agents: a follower and a speaker
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

    def __init__(self, task_loader: TaskLoader, speaker, follower, hparams: Dict, debug: bool = False):
        self.task_loader = task_loader

        self.engine = Engine(PentoConfig())  # reset the state on each env reset (e.g. size of the maps)
        self.debug = debug
        self.center_spawn = hparams["env.center_spawn"]

        """ Agents """
        self.speaker = speaker
        self.follower = follower
        print("CoGripEnv speaker :", speaker.get_path())
        print("CoGripEnv follower:", follower.get_path())

        """ State """
        self.current_task: Task = None
        self.current_board: Board = None
        self.obs = None  # for autoplay: obs are returned but not fed back on step

        self.step_count = 0
        self.info = {}

        self.progress_bar = None
        if hparams["env.show_progress"]:
            progress_length = len(task_loader)
            self.progress_bar = tqdm(total=progress_length)

    def log_debug(self, message):
        if self.debug:
            print(message)

    def seed(self, seed=None):
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        self.log_debug("\n NEW EPISODE")
        self.info = {}
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        """ We go through the tasks; and restart if reached the end """
        self.current_task = self.task_loader.next_task()
        # reset board
        self.current_board = self.current_task.create_board()
        self.engine.set_state(self.current_board)  # this new board has no gripper yet
        if self.center_spawn:
            self.engine.add_gr(GRIPPER_ID)  # we expect exactly one gripper (that can be controlled by an agent)
        else:
            grx = random.randint(0, self.engine.get_width() - 1)
            gry = random.randint(0, self.engine.get_height() - 1)
            self.engine.add_gr(GRIPPER_ID, grx, gry)
        # reset agents (potentially possible with less information)
        self.follower.on_reset(self.current_task, self.current_board, self.current_gripper_coords, self.info)
        self.speaker.on_reset(self.current_task, self.current_board, self.current_gripper_coords, self.info)
        self.step_count = 0
        self.obs = self.on_reset()
        self.info.clear()  # remove infos (if accidentally added during reset)
        if self.current_task.idx is not None:
            self.info["task/idx"] = self.current_task.idx
        return self.obs, self.info

    @abc.abstractmethod
    def on_reset(self) -> Any:
        pass

    def render(self, mode="rgb_array", channel_first=False):
        if mode in ["rgb_array", "human"]:
            return render.to_rgb_array(self.current_board,
                                       self.current_gripper_coords,
                                       channel_first=channel_first)  # return RGB frame suitable for video
        else:
            super().render(mode=mode)  # just raise an exception

    def step(self, action: object) -> Tuple[Any, float, bool, bool, dict]:
        self.step_count += 1

        self.obs, piece_gripped = self.on_step(action)

        if self.is_abort_condition():
            self.info[EPISODE_OUTCOME_ABORT] = 1
            self.info[EPISODE_OUTCOME_SUCCESS] = 0
            self.info[EPISODE_OUTCOME_FAILURE] = 0
            self.info[EPISODE_STEP_COUNT] = self.step_count
            done = True
            reward = self.on_failure_reward(None)
        elif self.is_terminal_condition(piece_gripped):
            self.info[EPISODE_OUTCOME_ABORT] = 0
            self.info[EPISODE_STEP_COUNT] = self.step_count
            done = True
            if self.is_success_condition(piece_gripped):
                self.info[EPISODE_OUTCOME_SUCCESS] = 1
                self.info[EPISODE_OUTCOME_FAILURE] = 0
                reward = self.on_success_reward(piece_gripped)
            else:
                self.info[EPISODE_OUTCOME_SUCCESS] = 0
                self.info[EPISODE_OUTCOME_FAILURE] = 1
                reward = self.on_failure_reward(piece_gripped)
        else:
            done = False
            reward = self.on_step_reward()

        # future package gymnasium will have a distinction between terminated and truncated
        self.info["done"] = done
        truncated = False
        return self.obs, reward, done, truncated, self.info

    @abc.abstractmethod
    def on_step(self, action: object) -> Tuple[Dict, Piece]:
        pass

    @abc.abstractmethod
    def on_success_reward(self, piece_gripped: Piece) -> float:
        pass

    @abc.abstractmethod
    def on_failure_reward(self, piece_gripped: Piece) -> float:
        pass

    @abc.abstractmethod
    def on_step_reward(self) -> float:
        pass

    @property
    def current_gripper_coords(self) -> Tuple[int, int]:
        """ Returns (x,y) """
        x, y = self.engine.get_gripper_coords(GRIPPER_ID)
        return int(x), int(y)

    @property
    def map_size(self) -> int:
        return self.current_board.grid_config.width

    @property
    def max_steps(self) -> int:
        if self.engine.get_width() == 7:
            return 15
        if self.engine.get_width() == 12: # fix naive-12
            return 30
        return self.current_task.max_steps

    @abc.abstractmethod
    def is_terminal_condition(self, piece_gripped: Piece):
        pass

    @abc.abstractmethod
    def is_success_condition(self, piece_gripped: Piece):
        pass

    @abc.abstractmethod
    def is_abort_condition(self):
        pass

    def _grip(self) -> Piece:
        self.engine.grip(GRIPPER_ID)
        piece = self.engine.get_gripped_obj(GRIPPER_ID)
        self.engine.state.ungrip(GRIPPER_ID)  # immediately ungrip
        return piece

    def _move(self, dx: int, dy: int):
        return self.engine.mover.apply_movement(self.engine, "move", GRIPPER_ID, x_steps=dx, y_steps=dy)

    def close(self):
        pass
