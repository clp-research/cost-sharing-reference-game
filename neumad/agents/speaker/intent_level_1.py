from neumad.agents.speaker.intent import IntentSpeaker
from neumad.envs import SpeakerIntent


class IntentSpeakerL1(IntentSpeaker):
    """
    A speaker that semi-automatically performs language production:

    - when to speak (silence or not)
    - what to say (confirm, correct, directive, reference)
    -> automatically resolves how to say it

    The intent realisation utilizes language templates.
    """

    def __init__(self):
        super().__init__()

    def get_level(self) -> int:
        return 1

    def _on_intent(self, intent: SpeakerIntent) -> str:
        utterance = None
        if intent == SpeakerIntent.silence:
            utterance = self.say_nothing()
        if intent == SpeakerIntent.confirmation:
            utterance = self.generate_confirmation()
        if intent == SpeakerIntent.correction:
            utterance = self.generate_correction()
        if intent == SpeakerIntent.directive:
            utterance = self.generate_directive()
        if intent == SpeakerIntent.reference:
            utterance = self.generate_reference()
        return utterance

    def generate_confirmation(self) -> str:
        self.log_debug(f"generate_confirmation")
        if self._is_gripper_over_piece():
            utterance = self.say_confirm_take()
        else:
            utterance = self.say_confirm_direction()
        return utterance

    def generate_correction(self) -> str:
        self.log_debug(f"generate_correction")
        if self._is_gripper_over_piece():
            utterance = self.say_decline_take()
        else:
            utterance = self.say_decline_direction()
        return utterance

    def generate_directive(self) -> str:
        self.log_debug(f"generate_directive")
        if self._is_gripper_over_piece():
            utterance = self.say_initiate_take()
        else:
            utterance = self.say_initiate_direction()
        return utterance
