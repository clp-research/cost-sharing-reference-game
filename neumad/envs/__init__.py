from enum import IntEnum
from typing import Union, List, Tuple

import numpy as np


class FollowerAction(IntEnum):
    wait = 0
    left = 1
    right = 2
    up = 3
    down = 4
    take = 5

    @staticmethod
    def from_utterance(utterance: str) -> List:
        actions = []
        for action in list(FollowerAction):
            if str(action) in utterance:
                actions.append(action)
        if not actions:
            raise ValueError("Cannot find any action in utterance: " + utterance)
        return actions

    @staticmethod
    def from_object(action: object):
        if isinstance(action, FollowerAction):
            return action
        if isinstance(action, np.ndarray):  # unwrap 1d-array
            if len(action.shape) == 0:
                action = action.item()
            elif len(action.shape) == 1:
                if action.shape[0] == 1:
                    action = action[0]
        return FollowerAction(action)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name)


class FollowerAutonomy(IntEnum):
    # autonomy of the follower is a variable that defines how often re-planning occurs
    # none: follower does not formulate a plan at all (but requires the input of the speaker at each step)
    # +cautious: uses speakers input the re-plan (but does not explore alone)
    # (+on_step: additionally re-plans at each step based on the mission and new vision)
    # +eager: if the plan runs out the follower chooses a random movement plan (for each step until horizon)
    # => hypothesis: with increasing autonomy the speaker has to produce fewer utterances
    none = 0
    cautious = 1
    eager = 2
    pretrained = 3  # no heuristic planner

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name)


class SpeakerAction(IntEnum):
    # basics
    silence = 0
    confirmation = 1
    correction = 2

    # directives
    left = 3
    right = 4
    up = 5
    down = 6
    take = 7

    # references
    pcs = 8
    psc = 9
    cps = 10
    csp = 11
    spc = 12
    scp = 13

    def is_reference(self):
        return self in [SpeakerAction.pcs, SpeakerAction.psc,
                        SpeakerAction.cps, SpeakerAction.csp,
                        SpeakerAction.spc, SpeakerAction.scp]

    def is_directive(self):
        return self in [SpeakerAction.left, SpeakerAction.right,
                        SpeakerAction.up, SpeakerAction.down,
                        SpeakerAction.take]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name)

    def to_list(self) -> List:
        if self == 0:
            return [SpeakerIntent.silence]
        if self == 1:
            return [SpeakerIntent.confirmation]
        if self == 2:
            return [SpeakerIntent.correction]
        if 3 <= self <= 7:
            return [SpeakerIntent.directive, SpeakerDirective(self - 3), None]
        if 8 <= self:
            return [SpeakerIntent.reference, None, SpeakerReference(self - 8)]
        raise ValueError()

    @staticmethod
    def from_list(action: Union[np.array, List, Tuple[int, int, int]]) -> int:
        assert action is not None
        speaker_action = action[0]
        if speaker_action == SpeakerIntent.silence:
            return SpeakerAction(0)
        if speaker_action == SpeakerIntent.confirmation:
            return SpeakerAction(1)
        if speaker_action == SpeakerIntent.correction:
            return SpeakerAction(2)
        if speaker_action == SpeakerIntent.directive:
            return SpeakerAction(action[1] + 3)
        if speaker_action == SpeakerIntent.reference:
            return SpeakerAction(action[2] + 8)  # 3 + 5
        raise ValueError()

    @staticmethod
    def from_object(action: object):
        if isinstance(action, SpeakerAction):
            return action
        if isinstance(action, np.ndarray):  # unwrap 1d-array
            if len(action.shape) == 0:
                action = action.item()
            elif len(action.shape) == 1:
                if action.shape[0] == 1:
                    action = action[0]
        return SpeakerAction(action)

    @staticmethod
    def get_reference(preference_order: str):
        preference_order = preference_order.lower()
        for value in SpeakerAction:
            if str(value.name) == preference_order:
                return value
        raise ValueError(preference_order)

    @staticmethod
    def get_directive(utterance: str):
        utterance = utterance.lower()
        for value in SpeakerAction:
            if str(value.name) in utterance:
                return value
        raise ValueError(utterance)


class SpeakerIntent(IntEnum):
    # save effort
    # hypothesis:
    # there should be less effort needed for more autonomous followers
    silence = 0
    # positive feedback "yes (go left)" "yes (take this piece)"
    # hypothesis:
    # this effects the confidence of the follower: a more confident follower should need less confirmation acts
    confirmation = 1
    # negative feedback "not (this way but go left)" "not (this piece but go left)"
    # this assumes that the follower will dismiss the current (planned) actions
    # can the speaker align with the language that is understood by the follower?
    # hypothesis:
    # in early episodes a lot of corrections should happen to test for mutual understanding
    # later the corrections should become less soften (only as interventions)
    correction = 2
    # intent to formulate the goal "take the X" (reference is "global")
    reference = 3
    # intent to ask for a specific action "go left" "take the piece"
    directive = 4

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name)


class SpeakerDirective(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    take = 4

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name)


class SpeakerReference(IntEnum):  # the goal
    pcs = 0
    psc = 1
    cps = 2
    spc = 3
    csp = 4
    scp = 5

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.name).upper()
