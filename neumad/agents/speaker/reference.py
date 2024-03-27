from typing import Union

import numpy as np

from cogrip.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from cogrip.pentomino.symbolic.types import PropertyNames
from neumad.agents.speaker import Speaker


class IAMissionSpeaker(Speaker):
    """ A speaker that mentions the shape and color of a piece (but provides no feedback) and ignores positions """

    _pref_order_by_name = dict(
        CSP=[PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION],
        CPS=[PropertyNames.COLOR, PropertyNames.REL_POSITION, PropertyNames.SHAPE],
        PSC=[PropertyNames.REL_POSITION, PropertyNames.SHAPE, PropertyNames.COLOR],
        PCS=[PropertyNames.REL_POSITION, PropertyNames.COLOR, PropertyNames.SHAPE],
        SPC=[PropertyNames.SHAPE, PropertyNames.REL_POSITION, PropertyNames.COLOR],
        SCP=[PropertyNames.SHAPE, PropertyNames.COLOR, PropertyNames.REL_POSITION]
    )

    def __init__(self, preference_order):
        super().__init__()
        preference_order = preference_order.upper()
        assert preference_order in ["CSP", "CPS", "PSC", "PCS", "SPC", "SCP"]
        self.preference_order = preference_order
        if self.preference_order == "ALL":
            self.reg = [PentoIncrementalAlgorithm(po, start_tokens=["take"])
                        for po in IAMissionSpeaker._pref_order_by_name.values()]
        else:
            po = IAMissionSpeaker._pref_order_by_name[preference_order]
            self.reg = PentoIncrementalAlgorithm(po, start_tokens=["take"])  # , "select", "get"])

    def generate_reference(self) -> str:
        self.log_debug("IAMissionSpeaker: generate_mission")
        if self.mission is None:
            reg = self.reg
            re = reg.generate(self.task.piece_symbols, self.task.target_piece_symbol, is_selection_in_pieces=True)
            self.mission = re[0].lower()
            self.mission = self.mission.replace("one of", "")  # keep ambiguity
        self.log_info_step("reference")
        # todo: only for eval !
        self.log_info_step(f"reference/{self.preference_order}")
        return self.mission
