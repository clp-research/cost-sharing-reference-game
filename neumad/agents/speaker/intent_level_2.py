from neumad.agents.speaker.intent_level_1 import IntentSpeakerL1
from neumad.envs import SpeakerDirective, SpeakerReference


class IntentSpeakerL2(IntentSpeakerL1):
    """
    A speaker that allows a finer-grained control over language production:

    - when to speak (silence or not)
    - what to say (confirm, correct, directive, reference)
    - how to say it (preference order, directions)

    The intent realisation utilizes language templates.
    """

    def __init__(self):
        super().__init__()

    def get_level(self) -> int:
        return 2

    def generate_directive(self) -> str:
        self.log_debug(f"generate_directive")
        direction_intent = self.current_action[1]
        if direction_intent == SpeakerDirective.take:
            utterance = self.say_initiate_take()
        else:
            utterance = self.say_initiate_direction()
        return utterance

    def _on_directions(self) -> str:
        direction_intent = self.current_action[1]
        direction_intent = list(SpeakerDirective)[direction_intent]
        direction = str(direction_intent)
        return direction

    def generate_reference(self, prefer_mode=None) -> str:
        preference_order = self.current_action[2]
        preference_order = list(SpeakerReference)[preference_order]
        preference_order = str(preference_order)
        speaker = self.mission_speakers[preference_order]
        utterance = speaker.generate_reference()
        return utterance
