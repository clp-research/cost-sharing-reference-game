import abc
from typing import Tuple, Any, Dict

import numpy as np

from cogrip import render, language
from cogrip.base_env import CoGripEnv
from cogrip.constants import POS_NAME_TO_IDX, COLOR_NAME_TO_IDX, SHAPE_NAME_TO_IDX
from cogrip.language import encode_sent, decode_sent
from cogrip.pentomino.objects import Piece
from gymnasium import spaces

from cogrip.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from cogrip.pentomino.symbolic.types import RelPositions, PropertyNames, Colors, Shapes
from cogrip.tasks import TaskLoader
from neumad.agents import Agent, NoopAgent
from neumad.envs import FollowerAction, SpeakerIntent, SpeakerAction

OBS_RGB_PARTIAL = "partial_view_with_pixels"  # using FOV_SIZE
OBS_SYMBOLIC_PARTIAL = "partial_view_with_symbols"  # needed for heuristic follower

OBS_POS_FULL_CURRENT = "full_view_with_position_current"  # padded towards MAX_SIZE, for the follower
OBS_POS_FULL_TARGET = "full_view_with_position_target"  # padded towards MAX_SIZE, for the speaker
OBS_SYMBOLIC_POS = "gr_coords_pos"  # needed for heuristic speaker
OBS_SYMBOLIC_POS_AREA = "gr_coords_pos_area"  # needed for heuristic follower

OBS_LANGUAGE = "language"
OBS_TARGET_DESCRIPTION = "target_description"
OBS_TARGET_DESCRIPTION_SYMBOLS = "target_description_symbols"

FOV_SIZE = 7  # partial view size
FOV_SIZE_5 = 7 * 5  # partial view size
MAX_SIZE = 30  # maximal map size

VOCAB_SIZE = 54  # number of tokens
MAX_SENT = 16  # maximal sentence length

MAX_POSITION = len(RelPositions)
RGB = 3  # red green blue
CSI = 3  # color shape id
BPG = 3  # board position gripper
BPTG = 4  # board position target gripper

NUM_COLORS = len(list(Colors))
NUM_SHAPES = len(list(Shapes))
NUM_POS = len(list(RelPositions))

SILENCE = ""


def tokenize(text: str):
    if text == SILENCE:
        return np.zeros(shape=(MAX_SENT,), dtype=np.uint8)  # only padding
    tokens = language.encode_sent(text, pad_length=MAX_SENT)
    return np.array(tokens, dtype=np.uint8)


def trans_obs(obs, utt_obs_key):
    utterance = None
    if OBS_LANGUAGE in utt_obs_key:
        utterance = decode_sent(obs[OBS_LANGUAGE])
        if utterance == "":
            utterance = "<silence>"
    if OBS_TARGET_DESCRIPTION in utt_obs_key:
        target_desc = decode_sent(obs[OBS_TARGET_DESCRIPTION])
        utterance = utterance + f"\ntgt: {target_desc}" if utterance else target_desc
    partial = obs[OBS_RGB_PARTIAL]
    if utt_obs_key == OBS_TARGET_DESCRIPTION:
        overview = obs[OBS_POS_FULL_TARGET]
    else:
        overview = obs[OBS_POS_FULL_CURRENT]
    return utterance, partial, overview


def overview_to_greyscale(overview_obs):
    grey_scale = np.zeros(shape=(3, 2 * MAX_SIZE, 2 * MAX_SIZE), dtype=np.uint8)
    assert overview_obs.shape in [(4, 2 * MAX_SIZE, 2 * MAX_SIZE), (3, MAX_SIZE, MAX_SIZE)]  # speaker has also target
    channels, height, width = overview_obs.shape
    channel2_color = 200 if channels == 4 else 100
    for idx in range(channels):
        for y in range(height):
            for x in range(width):
                if overview_obs[idx, y, x] == 255:
                    if idx == 0:
                        color = 255
                    if idx == 1:
                        color = 235
                    if idx == 2:
                        color = channel2_color
                    if idx == 3:
                        color = 100
                    grey_scale[:, y, x] = color
    grey_scale = np.moveaxis(grey_scale, 0, -1)
    return grey_scale


class TakePieceEnv(CoGripEnv, abc.ABC):
    """
        TakePieceEnv defines the observations and the dynamics of the environment (steps and terminal condition)
    """

    def __init__(self, task_loader: TaskLoader, speaker: Agent, follower: Agent, hparams: Dict, debug: bool = False):
        super().__init__(task_loader, speaker, follower, hparams, debug)

        """ Reward computation """
        self.speaker_effort = 0.
        self.follower_effort = 0.

        """ Action space """
        self.speaker_action: SpeakerAction = None
        self.follower_action: FollowerAction = None
        # subclasses must define the action space

        """ Observation space """
        dict_spaces = dict()
        # partial vision
        dict_spaces[OBS_RGB_PARTIAL] = spaces.Box(low=0, high=255, shape=(RGB, FOV_SIZE_5, FOV_SIZE_5), dtype=np.uint8)
        dict_spaces[OBS_SYMBOLIC_PARTIAL] = spaces.Box(low=0, high=255, shape=(CSI, FOV_SIZE, FOV_SIZE), dtype=np.uint8)

        # position overview
        dict_spaces[OBS_POS_FULL_TARGET] = spaces.Box(low=0, high=255,
                                                      shape=(BPTG, 2 * MAX_SIZE, 2 * MAX_SIZE),
                                                      dtype=np.uint8)
        dict_spaces[OBS_POS_FULL_CURRENT] = spaces.Box(low=0, high=255,
                                                       shape=(BPTG, 2 * MAX_SIZE, 2 * MAX_SIZE),
                                                       dtype=np.uint8)
        dict_spaces[OBS_SYMBOLIC_POS] = spaces.Box(low=0, high=MAX_SIZE, shape=(2,), dtype=np.uint8)
        dict_spaces[OBS_SYMBOLIC_POS_AREA] = spaces.Box(low=0, high=len(POS_NAME_TO_IDX), shape=(1,), dtype=np.uint8)

        # in theory the language observation space should be given by the speaker, but we simplify here
        dict_spaces[OBS_LANGUAGE] = spaces.Box(low=0, high=VOCAB_SIZE, shape=(MAX_SENT,), dtype=np.uint8)
        dict_spaces[OBS_TARGET_DESCRIPTION] = spaces.Box(low=0, high=VOCAB_SIZE, shape=(MAX_SENT,), dtype=np.uint8)
        dict_spaces[OBS_TARGET_DESCRIPTION_SYMBOLS] = spaces.Box(low=0, high=20, shape=(3,), dtype=np.uint8)

        self.observation_space = spaces.Dict(dict_spaces)

        # value holders for re-use and easier computation
        self.target_description = np.zeros(shape=dict_spaces[OBS_TARGET_DESCRIPTION].shape)
        self.target_description_symbol = np.zeros(shape=dict_spaces[OBS_TARGET_DESCRIPTION_SYMBOLS].shape)

        self.only_center_pos = hparams["env.only_center_pos"]

        self.minimal_descriptor = None
        if hparams["env.trg_desc"] == "minimal":
            self.minimal_descriptor = PentoIncrementalAlgorithm(
                [PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION])

    def get_target_descriptors(self):
        target_symbol = self.current_task.target_piece.piece_config
        properties = {
            PropertyNames.COLOR: target_symbol.color,
            PropertyNames.SHAPE: target_symbol.shape,
            PropertyNames.REL_POSITION: target_symbol.rel_position
        }
        if self.minimal_descriptor is not None:
            properties, _ = self.minimal_descriptor.generate(self.current_task.piece_symbols, target_symbol,
                                                             is_selection_in_pieces=True, return_expression=False)
        descriptor = ""
        color_symbol, shape_symbol, pos_symbol = 0, 0, 0
        if PropertyNames.COLOR in properties:
            color_value = properties[PropertyNames.COLOR].value_name
            descriptor += color_value
            descriptor += " "
            color_symbol = COLOR_NAME_TO_IDX[color_value]
        if PropertyNames.SHAPE in properties:
            shape_value = properties[PropertyNames.SHAPE].value
            descriptor += shape_value
            descriptor += " "
            shape_symbol = SHAPE_NAME_TO_IDX[shape_value]
        if PropertyNames.REL_POSITION in properties:
            pos_value = properties[PropertyNames.REL_POSITION].value
            descriptor += pos_value
            pos_value_key = "_".join(pos_value.split(" ")).upper()
            pos_symbol = POS_NAME_TO_IDX[pos_value_key]
        text = descriptor.strip().lower()
        symbols = (color_symbol, shape_symbol, pos_symbol)
        return text, symbols

    def compute_full_target_pos(self):
        width, height = self.current_board.grid_config.width, self.current_board.grid_config.height
        pos_full_target_board = np.full(shape=(1, height, width), fill_value=255, dtype=np.uint8)
        target_position = self.current_task.target_piece.piece_config.rel_position
        if self.only_center_pos:
            pos_full_target_pos = pos_full_target_board.copy()
        else:
            pos_full_target_pos = render.compute_pos_mask(target_position, width, height)
        pos_full_target_piece = render.compute_target_mask(self.current_board, self.current_task.target_piece)
        pos_full_target_gr = np.zeros(shape=(1, height, width), dtype=np.uint8)
        grx, gry = self.current_gripper_coords
        pos_full_target_gr[0, gry, grx] = 255
        full_target_pos = np.concatenate(
            [pos_full_target_board, pos_full_target_pos,
             pos_full_target_piece, pos_full_target_gr], axis=0)
        full_target_pos = render.pad_with_zeros_to_center(self.current_gripper_coords, full_target_pos, MAX_SIZE)
        return full_target_pos

    def compute_full_current_pos(self):
        width, height = self.current_board.grid_config.width, self.current_board.grid_config.height
        pos_full_current_board = np.full(shape=(1, height, width), fill_value=255, dtype=np.uint8)
        width, height = self.current_board.grid_config.width, self.current_board.grid_config.height
        pos_full_current_gr = np.zeros(shape=(1, height, width), dtype=np.uint8)
        grx, gry = self.current_gripper_coords
        pos_full_current_gr[0, gry, grx] = 255
        if self.only_center_pos:
            pos_current_mask = pos_full_current_board.copy()
        else:
            pos_current = RelPositions.from_coords(grx, gry, width, height)
            pos_current_mask = render.compute_pos_mask(pos_current, width, height)
        pos_current_pieces = render.compute_pieces_mask(self.current_board)
        full_current_pos = np.concatenate([pos_full_current_board, pos_current_mask,
                                           pos_current_pieces, pos_full_current_gr], axis=0)
        full_current_pos = render.pad_with_zeros_to_center(self.current_gripper_coords, full_current_pos, MAX_SIZE)
        return full_current_pos

    def on_reset(self) -> Any:
        self.follower_action = FollowerAction.wait
        self.speaker_action = SpeakerAction.silence
        self.speaker_effort = 0.
        self.follower_effort = 0.
        text, symbols = self.get_target_descriptors()
        self.target_description = np.array(encode_sent(text, pad_length=MAX_SENT))
        self.target_description_symbol = np.array(symbols)
        obs = self._gen_obs()
        return obs

    def _gen_obs(self):
        full_view_rgb = render.to_rgb_array(self.current_board, self.current_gripper_coords)
        partial_view_rgb = render.compute_fov(full_view_rgb, self.current_gripper_coords, fov_size=FOV_SIZE)
        full_view_symbols = render.to_symbolic_array(self.current_board)
        partial_view_symbols = render.compute_fov(full_view_symbols, self.current_gripper_coords,
                                                  fov_size=FOV_SIZE, margin=0)
        pos_full_target = self.compute_full_target_pos()
        pos_full_current = self.compute_full_current_pos()
        pos_symbolic = self.current_gripper_coords
        size = self.engine.get_width()
        pos_area = RelPositions.from_coords(pos_symbolic[0], pos_symbolic[1], size, size)
        pos_area_symbolic = POS_NAME_TO_IDX[pos_area.name]
        pos_area_symbolic = np.array([pos_area_symbolic])
        obs = {
            # partial vision
            OBS_RGB_PARTIAL: partial_view_rgb,  # for the neural follower
            OBS_SYMBOLIC_PARTIAL: partial_view_symbols,  # for heuristic follower
            # position overview
            OBS_POS_FULL_TARGET: pos_full_target,
            OBS_POS_FULL_CURRENT: pos_full_current,
            OBS_SYMBOLIC_POS: pos_symbolic,  # for heuristic agents
            OBS_SYMBOLIC_POS_AREA: pos_area_symbolic,  # for heuristic agents
            # language
            OBS_TARGET_DESCRIPTION_SYMBOLS: self.target_description_symbol,
            OBS_TARGET_DESCRIPTION: self.target_description,  # for the neural speaker
            OBS_LANGUAGE: np.zeros(shape=(MAX_SENT,))  # by default silence
        }

        return obs

    def is_terminal_condition(self, piece_gripped: Piece):
        return piece_gripped is not None

    def is_success_condition(self, piece_gripped: Piece):
        if piece_gripped is None:
            return False
        return self.current_task.target_piece.id_n == piece_gripped.id_n

    def is_abort_condition(self):
        return self.step_count >= self.max_steps

    @abc.abstractmethod
    def on_step(self, action: object) -> Tuple[Dict, Piece]:
        pass

    def step_transition(self, follower_action: FollowerAction) -> Piece:
        piece_gripped = None
        if follower_action == FollowerAction.wait:
            pass
        elif follower_action == FollowerAction.left:
            self._move(-1, 0)
        elif follower_action == FollowerAction.right:
            self._move(1, 0)
        elif follower_action == FollowerAction.up:
            self._move(0, -1)
        elif follower_action == FollowerAction.down:
            self._move(0, 1)
        elif follower_action == FollowerAction.take:
            piece_gripped = self._grip()
        else:
            raise ValueError(f"Unknown action: {follower_action}")
        return piece_gripped

    def step_effort(self):
        if self.speaker_action == SpeakerAction.silence:
            self.speaker_effort += 0.0
        elif self.speaker_action in [SpeakerAction.confirmation, SpeakerAction.correction]:
            self.speaker_effort += 1.0
        elif self.speaker_action.is_directive():
            self.speaker_effort += 2.0
        elif self.speaker_action.is_reference():
            self.speaker_effort += 3.0
        else:
            raise ValueError("Unknown action: " + str(self.speaker_action))

        if self.follower_action == FollowerAction.wait:
            self.follower_effort += 0.0
            self.info["step/follower/wait"] = 1
        elif self.follower_action == FollowerAction.take:
            self.follower_effort += 3.0
            self.info["step/follower/take"] = 1
        else:  # movements
            self.follower_effort += 2.0
            self.info["step/follower/move"] = 1

    def on_step_reward(self):
        self.step_effort()
        return 0

    def on_failure_reward(self, piece_gripped: Piece) -> float:
        if piece_gripped is None:
            return self.final_reward(goal_reward=-1)  # no selection penalty
        gripped_shape = piece_gripped.piece_config.shape
        gripped_color = piece_gripped.piece_config.color
        gripped_pos = piece_gripped.piece_config.rel_position
        if gripped_pos == self.current_task.target_piece.piece_config.rel_position:
            if gripped_color == self.current_task.target_piece.piece_config.color:
                return self.final_reward(goal_reward=0)
            if gripped_shape == self.current_task.target_piece.piece_config.shape:
                return self.final_reward(goal_reward=0)
        return self.final_reward(goal_reward=-1)  # wrong selection penalty

    def on_success_reward(self, piece_gripped: Piece) -> float:
        return self.final_reward(goal_reward=1)  # bonus

    def final_reward(self, goal_reward: int):
        time_reward = self.reward(self.step_count)  # "metabolic cost"
        self.info["episode/reward/time"] = time_reward

        self.info["episode/effort/speaker_abs"] = self.speaker_effort
        self.info["episode/effort/speaker_rel"] = self.speaker_effort / self.step_count
        speaker_reward = self.reward(self.speaker_effort)
        self.info["episode/reward/speaker"] = speaker_reward

        self.info["episode/effort/follower_abs"] = self.follower_effort
        self.info["episode/effort/follower_rel"] = self.follower_effort / self.step_count
        follower_reward = self.reward(self.follower_effort)
        self.info["episode/reward/follower"] = follower_reward

        joint_effort = (self.follower_effort + self.speaker_effort) / 2
        self.info["episode/effort/mean_joint_abs"] = joint_effort
        self.info["episode/effort/mean_joint_rel"] = joint_effort / self.step_count

        joint_reward = (speaker_reward + follower_reward) / 2  # coordination cost
        self.info["episode/reward/joint"] = joint_reward

        final_reward = (time_reward + joint_reward) / 2  # balanced final reward
        final_reward = final_reward + goal_reward

        self.info["episode/reward/goal"] = goal_reward
        self.info["episode/reward/final"] = final_reward
        return final_reward

    def reward(self, value: float) -> float:
        reward = 1 - (0.9 * (value / self.max_steps))
        return reward


class FollowerTakePieceEnv(TakePieceEnv):
    """ Follower is serving the actions on step """

    def __init__(self, task_loader: TaskLoader, speaker: Agent, follower: Agent, hparams: Dict, debug: bool = False):
        super().__init__(task_loader, speaker, follower, hparams, debug)
        self.action_space: spaces.Space = spaces.Discrete(len(FollowerAction))

    def on_reset(self) -> Any:
        # for follower env we initially also provide an utterance
        # thus the follower agent has actually one more step
        # the speaker can only act the first time on_step
        # in the speaker env after reset
        obs = super().on_reset()
        # let the speaker produce something initially
        speaker_action = self.speaker.on_step(obs)
        self.speaker_action = speaker_action
        utterance = self.speaker.on_step(obs, action=speaker_action)
        obs[OBS_LANGUAGE] = tokenize(utterance)
        # we have to compute the speakers initial effort here already
        self.step_effort()
        return obs

    def on_step(self, action: object) -> Tuple[Dict, Piece]:
        if action == "auto":
            action = self.follower.on_step(self.obs)
        # can be a follower action, np.array or int
        self.follower_action = FollowerAction.from_object(action)
        # transition the environment with respect to the follower action
        piece_gripped = self.step_transition(self.follower_action)
        obs = self._gen_obs()
        # let the speaker produce something in response to the follower's action
        # if terminal condition is reached, then this is simply ignored
        speaker_action = self.speaker.on_step(obs)
        self.speaker_action = speaker_action
        utterance = self.speaker.on_step(obs, action=speaker_action)
        obs[OBS_LANGUAGE] = tokenize(utterance)
        return obs, piece_gripped


class SpeakerTakePieceEnv(TakePieceEnv):
    """ Speaker is serving the actions on step """

    def __init__(self, task_loader: TaskLoader, speaker: Agent, follower: Agent, hparams: Dict,
                 debug: bool = False):
        super().__init__(task_loader, speaker, follower, hparams, debug)
        # silence(1), confirmation(1), correction(1), directive(5), reference(6) => 14
        self.action_space: spaces.Space = spaces.Discrete(len(SpeakerAction))

    def on_step(self, action: object) -> Tuple[Dict, Piece]:
        # can be a speaker action, np.array or int
        self.speaker_action = SpeakerAction.from_object(action)
        utterance = self.speaker.on_step(self.obs, action=self.speaker_action)
        obs = self._gen_obs()
        obs[OBS_LANGUAGE] = tokenize(utterance)
        # let the follower react to the speaker's utterance and the other obs
        self.follower_action = self.follower.on_step(obs)
        # transition the environment with respect to the follower action
        piece_gripped = self.step_transition(self.follower_action)
        # if terminal condition is reached, then this is simply ignored
        obs = self._gen_obs()
        # after that the language obs is empty again (but last utt. is provided for visualizer)
        # this is input for the speaker (who the language obs)
        obs[OBS_LANGUAGE] = tokenize(utterance)
        return obs, piece_gripped
