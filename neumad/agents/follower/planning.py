import random
from typing import Tuple, List, Dict

import numpy as np

from cogrip import language
from cogrip.constants import SHAPE_NAME_TO_IDX, COLOR_NAME_TO_IDX, POSITIONS_8, COLORS_6, SHAPES_9, IDX_TO_POS, SHAPES_7
from cogrip.pentomino.state import Board
from cogrip.pentomino.symbolic.types import SymbolicPiece, RelPositions
from cogrip.tasks import Task
from neumad.agents.follower import HeuristicFollower
from neumad.envs import FollowerAction
from neumad.envs.take_env import OBS_SYMBOLIC_POS, OBS_LANGUAGE, OBS_SYMBOLIC_PARTIAL, FOV_SIZE, OBS_SYMBOLIC_POS_AREA

OFFSETS_XY = {
    FollowerAction.wait: (0, 0),
    FollowerAction.left: (-1, 0),
    FollowerAction.right: (1, 0),
    FollowerAction.up: (0, -1),
    FollowerAction.down: (0, 1),
    FollowerAction.take: (0, 0),
}


def confidence_fn(g, x):
    return g ** x


def compute_confidences(plan_actions: List[FollowerAction], confidence: Tuple[float, float]):
    # the confidence here is defined by: a decay factor, a lower bound (threshold)
    discount, lower = confidence
    return [max(confidence_fn(discount, t), lower) for t in range(len(plan_actions))]


class LimitedHorizontPlanner(HeuristicFollower):

    def __init__(self, confidence: Tuple[float, float], debug=False):
        super().__init__(debug)

        # confidence of the follower is a variable that defines the probability
        # of following the actions in the plan versus performing (intermediate) wait actions
        # the confidence naturally decreases with the length of the plan
        # so that each action in the plan has a confidence scores attached to it e.g. [L1 L.8, U.6, U.5, U.5]
        self.confidence = confidence

        # the plan is limited to the horizont, so it eventually runs out of actions
        # diagonals = 10 with agent_view=11 (agent in the middle)
        # diagonals = 6 with agent_view=7 (agent in the middle)
        self.view_size = FOV_SIZE
        self.plan_horizon: int = self.view_size - 1

        # world model
        self.current_plan: List[FollowerAction] = []
        self.current_confidences: List[float] = []

        self.pos_area_centers: Dict[RelPositions, Tuple[int, int]] = dict()
        self.map_size: int = None

        self.current_target_descriptor: SymbolicPiece = SymbolicPiece()
        self.previous_target_descriptor: SymbolicPiece = SymbolicPiece()

        self.current_view_target_candidates: Dict = dict()
        self.previous_view_target_candidates: List[Tuple] = []

        self.current_obs = None
        self.current_utterance = None

    def on_reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        super().on_reset(task, board, gripper_pos, info)
        self.current_plan = []
        self.current_confidences = []
        self.current_target_descriptor: SymbolicPiece = SymbolicPiece()
        self.current_obs = None
        self.current_utterance = None

        self.map_size = board.grid_config.width
        for rel_pos in list(RelPositions):
            self.pos_area_centers[rel_pos] = rel_pos.get_area_center(self.map_size)

    def on_step(self, obs) -> FollowerAction:
        self.current_obs = obs
        current_pos = obs[OBS_SYMBOLIC_POS]  # x,y
        self.current_pos = current_pos[0], current_pos[1]
        pos_area_idx = obs[OBS_SYMBOLIC_POS_AREA]
        pos_area_idx = pos_area_idx.item()
        self.current_pos_area = IDX_TO_POS[pos_area_idx]

        self.current_utterance = language.decode_sent(obs[OBS_LANGUAGE])
        self.log_debug("on_step\n ")
        self.log_debug(f"Received: {self.current_utterance if self.current_utterance else '<silence>'}")

        if self.is_silence():
            self._on_silence()
        elif self.is_reference():
            self._on_reference()
        elif self.is_correction():
            self._on_correction()
        elif self.is_confirmation():
            self._on_confirmation()
        else:  # should be a directive
            self._on_directive()
        self.log_debug(f"Current Plan: {self.current_plan} ({self.current_confidences})")

        if self.current_plan:
            action = self._fetch_next_action()
        else:  # if plan is empty
            action = FollowerAction.wait

        self.log_debug(f"Perform action: {action}")
        self.log_debug(f"Remaining Plan: {self.current_plan} ({self.current_confidences})")
        return action

    def _on_reference(self):
        raise NotImplementedError()

    def _on_silence(self):
        pass

    def _fetch_next_action(self):
        candidate_action = self.current_plan[0]
        confidence = self.current_confidences[0]
        self.log_debug(f"Consider: {str(candidate_action)} ({confidence})")
        chosen_action = np.random.choice([FollowerAction.wait, candidate_action], p=[1 - confidence, confidence])
        chosen_action = list(FollowerAction)[chosen_action]
        if chosen_action != FollowerAction.wait:  # remove only, when actually taken
            self.current_plan.pop(0)
            self.current_confidences.pop(0)
        # check if applicable (otherwise the plan leads "nowhere")
        if self.is_impossible(chosen_action):
            self.log_debug("Impossible Action: Erase plan")
            self._install_new_plan([])
            return FollowerAction.wait
        return chosen_action

    def _update_target_descriptor(self):
        self.previous_target_descriptor = self.current_target_descriptor
        for color in COLORS_6:
            if str(color) in self.current_utterance:  # ignore two word colors for now
                self.current_target_descriptor.color = color
        for shape in SHAPES_7:
            # split words b.c. we look for letters in the sentence
            candidate_shapes = self.current_utterance.split(" ")
            if str(shape).lower() in candidate_shapes:
                self.current_target_descriptor.shape = shape
        is_pos_in_utterance = False
        for pos in POSITIONS_8:
            if str(pos) in self.current_utterance:
                self.current_target_descriptor.rel_position = pos
                is_pos_in_utterance = True
        # additionally check center, only when nothing has been found yet (otherwise also matches on top center)
        if not is_pos_in_utterance:
            if str(RelPositions.CENTER) in self.current_utterance:
                self.current_target_descriptor.rel_position = RelPositions.CENTER
        self.log_debug(f"New target descriptor: {self.current_target_descriptor}")
        return self.current_target_descriptor

    def _update_target_candidates(self):
        target_color = self.current_target_descriptor.color
        target_shape = self.current_target_descriptor.shape

        symbols = self.current_obs[OBS_SYMBOLIC_PARTIAL]
        max_size = symbols.shape[1]
        color_idx = None
        if target_color is not None:
            color_idx = COLOR_NAME_TO_IDX[target_color.value_name]
        shape_idx = None
        if target_shape is not None:
            shape_idx = SHAPE_NAME_TO_IDX[target_shape.value]
        # find particular coordinates for target description
        target_coords = []
        color_coords = []
        shape_coords = []
        other_coords = []
        for y in range(max_size):
            for x in range(max_size):
                global_coord = self._translate_view_coord_to_global((x, y), self.view_size)
                is_target_color = symbols[0, y, x] == color_idx
                is_target_shape = symbols[1, y, x] == shape_idx
                is_piece = symbols[0, y, x] not in [0, 1]  # 0=oow, 1=empty

                if self.current_target_descriptor.rel_position is None:
                    if is_target_color and is_target_shape:
                        target_coords.append(global_coord)
                    elif is_target_color:
                        color_coords.append(global_coord)
                    elif is_target_shape:
                        shape_coords.append(global_coord)
                    elif is_piece:
                        other_coords.append(global_coord)
                else:  # consider only tile that fall into position area
                    width = self.board.grid_config.width
                    tile_area = RelPositions.from_coords(global_coord[0], global_coord[1], width, width)
                    if tile_area == self.current_target_descriptor.rel_position:
                        if is_target_color and is_target_shape:
                            target_coords.append(global_coord)
                        elif is_target_color:
                            color_coords.append(global_coord)
                        elif is_target_shape:
                            shape_coords.append(global_coord)
                        elif is_piece:
                            other_coords.append(global_coord)

        for candidates in self.current_view_target_candidates.values():
            self.previous_view_target_candidates = [coord for coord in candidates]
        self.current_view_target_candidates["both"] = target_coords
        self.current_view_target_candidates["color"] = color_coords
        self.current_view_target_candidates["shape"] = shape_coords
        self.current_view_target_candidates["other"] = other_coords

    def _select_target_candidate_for_color_shape(self) -> Tuple[int, int]:
        # look for target attributes in vision obs
        target_color = self.current_target_descriptor.color
        target_shape = self.current_target_descriptor.shape

        is_shape_unknown = target_shape is None
        is_color_unknown = target_color is None

        preferred_selector = None
        if self.current_view_target_candidates["both"]:  # is color and shape visible? approach piece
            preferred_selector = "both"
        elif self.current_view_target_candidates["color"] and is_shape_unknown:  # is color visible? approach color
            preferred_selector = "color"  # favor color over shape
        elif self.current_view_target_candidates["shape"] and is_color_unknown:  # is shape visible? approach shape
            preferred_selector = "shape"

        if preferred_selector:
            coords = self._select_target_candidate(preferred_selector)
            self.log_debug(f"Approach {coords} determined by selector: {preferred_selector}")
            return coords
        else:
            self.log_debug(f"Found nothing in current view for target shape or color: {self.current_target_descriptor}")
            return None

    def _select_target_candidate(self, selector):
        if self.current_view_target_candidates[selector]:
            coords = random.choice(self.current_view_target_candidates[selector])
            return coords
        return None

    def _on_directive(self):
        uttered_actions = FollowerAction.from_utterance(self.current_utterance)
        if FollowerAction.take in uttered_actions:
            plan_actions = [FollowerAction.take]
        else:
            if "a bit" in self.current_utterance:
                plan_actions = [uttered_actions[0]]
            elif len(uttered_actions) == 1:
                plan_actions = [uttered_actions[0] for _ in range(self.plan_horizon)]
            elif len(uttered_actions) == 2:
                # handle "left up" by using list of actions
                actions = uttered_actions
                plan_actions = [actions[idx % 2] for idx in range(self.plan_horizon)]
            else:
                raise ValueError(f"uttered_actions: {uttered_actions}")

        self.log_debug(f"Install new plan (directives:{uttered_actions}): {plan_actions}")
        self._install_new_plan(plan_actions)

    def _install_new_plan(self, plan_actions):
        self.current_plan = plan_actions
        self.current_confidences = compute_confidences(self.current_plan, self.confidence)

    def is_impossible(self, chosen_action: FollowerAction) -> bool:
        if chosen_action in [FollowerAction.take, FollowerAction.wait]:
            return False
        symbols = self.current_obs[OBS_SYMBOLIC_PARTIAL]
        dx, dy = OFFSETS_XY[chosen_action]
        center_x, center_y = int(self.view_size / 2), int(self.view_size / 2)
        planned_x, planned_y = center_x + dx, center_y + dy
        if symbols.shape[0] in [3, 4, 5]:
            next_symbol = symbols[0, planned_y, planned_x]
        else:
            next_symbol = symbols[planned_y, planned_x, 0]
        if next_symbol == 0:  # oow
            return True
        return False

    def is_silence(self):
        return self.current_utterance == ""

    def is_reference(self):
        return self.current_utterance.startswith("take the")

    def is_confirmation(self):
        return self.current_utterance.startswith("yes") and not self.current_utterance.startswith("yes take")

    def is_correction(self):
        return self.current_utterance.startswith("not")

    def _on_correction(self):
        # corrections should lead to dropping the plan and wait to follow new directives
        # for example "no this way" [L L U] -> []
        # as a consequence corrections always include a wait (until next utterance (might vary based on autonomy))
        self._install_new_plan([])

    def _on_confirmation(self):
        # Option 1: confirmations should fill up short-term actions
        # in the plan with the current action until the time-horizon is reached
        # for example "yes left" [L U U] -> [L L L U U] (keeping later already planned actions)
        # confirmation do not need wait action (re-planning time-step)
        # the problem here is that it changes the initial plan which might be already correct

        # Option 2: we could also say confirmation does nothing to the current plan, but
        # it simply increases the certainty of following the action in the current plan:
        # probability of taking the action in the plan vs. the wait action

        # We use Option 2
        """
        We should not take on "yes take" but need initiate take
        if "yes take" in self.current_utterance:
            self.current_plan = [FollowerActions.take]
            self.current_confidences = [1.0]
        """
        self.current_confidences = [1. for _ in range(len(self.current_plan))]

    def is_within_area(self, rel_position):
        is_reached = rel_position == self.current_pos_area
        self.log_debug(
            f"Reference behavior: is_target_area={is_reached} "
            f"(from:{self.current_pos_area}<=>to:{rel_position}).")
        return is_reached

    def is_target_position_center(self, rel_position) -> bool:
        """ Did we reach the center of the target position yet?"""
        if rel_position is None:
            return False
        target_position = self.pos_area_centers[rel_position]
        is_reached = (self.current_pos[0] == target_position[0]) and (self.current_pos[1] == target_position[1])
        self.log_debug(
            f"Reference behavior: is_target_position_center={is_reached} "
            f"(from:{self.current_pos_area}<=>to:{rel_position}).")
        return is_reached

    def _get_coord_for_target_position(self, target_area: RelPositions):
        self.log_debug(f"Get coord for target area: {target_area}")
        target_position = self.pos_area_centers[target_area]
        return target_position
