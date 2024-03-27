from typing import Tuple

from neumad.agents.follower import path_utils
from neumad.agents.follower.planning import LimitedHorizontPlanner
from neumad.envs import FollowerAction


class CautiousPlanner(LimitedHorizontPlanner):

    def __init__(self, confidence: Tuple[float, float], debug=False):
        super().__init__(confidence, debug)
        self.current_target_coord: Tuple[int, int] = None
        self.strategy = "cautious"
        self.do_take: bool = False

    def _on_silence(self):
        if self.current_target_descriptor.rel_position is None:
            return  # not sure where to go
        if self.current_plan:
            return  # we already have plan
        self._on_reference()

    def _reference_behavior(self):
        """ The 'main behavior' """
        if self.current_target_descriptor.rel_position is None:
            self.log_debug(f"Reference behavior: Check view for any piece (no target position)")
            # simply look for any piece in the current view; we do not know where to go anyway
            self.do_take = True
            return self._select_target_candidate_for_color_shape()

        if self.is_within_area(self.current_target_descriptor.rel_position):
            self.log_debug(f"Reference behavior: Check view for target color and shape within target area.")
            target_coord = self._select_target_candidate_for_color_shape()
            if target_coord is not None:
                self.do_take = True
                return target_coord

            self.log_debug(f"Reference behavior: No piece with color or shape. Look for other pieces in target area.")

            # we are already in the correct position, but see no color or shape
            # we have to wait for a new reference or a direction hint
            # if possible we select a piece that is within the view
            # e.g. when reference only contains the position description (the only piece in that position)
            target_coord = self._select_target_candidate("other")
            if target_coord is not None:
                self.do_take = True
                return target_coord
            self.log_debug(f"Reference behavior: No piece at all")

        if not self.is_target_position_center(self.current_target_descriptor.rel_position):
            self.log_debug(f"Go further towards: {self.current_target_descriptor.rel_position}.")
            # default: we are somewhere but not yet in the target position
            # we try to approach the target position if possible (priority!)
            self.do_take = False
            return self._get_coord_for_target_position(self.current_target_descriptor.rel_position)

        # we have done all we can do: correct position, looking for a piece; but cannot find s.t.
        # act as we have heard nothing (wait for a direction or "eagerly" explore)
        return None

    def _on_reference(self):
        self._update_target_descriptor()
        self._update_target_candidates()

        target_coord = self._reference_behavior()
        if target_coord is None:
            self.log_debug(f"Reference behavior: No target coord.")
            # self._on_silence()  # wait for a direction or "eagerly" explore
            return

        # do not re-plan, when target coords do already match the target descriptor and were visible before
        # and we still have actions ahead to perform
        if self.current_plan:
            if self.current_target_descriptor == self.previous_target_descriptor:
                if self.current_target_coord in self.previous_view_target_candidates:  # still in view
                    # cool, this works automatically with directions, because the previous candidates move forward
                    return  # avoid jitter of target coords
        self._install_reference_plan(target_coord, do_take_if_possible=self._do_take_on_reference())

    def _do_take_on_reference(self):
        return self.do_take

    def _install_reference_plan(self, target_coord, do_take_if_possible):
        self.current_target_coord = target_coord
        if target_coord is None:
            if FollowerAction.take in self.current_plan:
                self.log_debug(f"Erase current approaching plan b.c. new reference could not be found in view")
                self._install_new_plan([])
            return
        # we compute shorted path now globally (not just within view) and cut-off at the horizon
        path = path_utils.compute_shortest_path(self.current_pos, target_coord, self.map_size)
        plan_actions = path_utils.translate_path_to_actions(path)
        plan_actions = plan_actions[:self.plan_horizon]
        # the reference starts with "take the", so we add the take action, if the horizon allows it
        if do_take_if_possible and len(plan_actions) < self.plan_horizon:
            plan_actions.append(FollowerAction.take)
        self.log_debug(f"Install new plan (reference target ({target_coord})): {plan_actions}")
        self._install_new_plan(plan_actions)
