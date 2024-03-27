from neumad import CautiousPlanner
from neumad.agents import Role
from neumad.agents.follower import NoopFollower, HeuristicFollowerAgent
from neumad.agents.speaker import NoopSpeaker
from neumad.agents.speaker.adapters import HeuristicSpeakerAgent, NeuralSpeakerAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.envs.manual_play import Plotter
from neumad.envs.take_env import OBS_TARGET_DESCRIPTION, OBS_LANGUAGE
from neumad.hparams import take_env

agent_role = Role.follower
# speaker = HeuristicSpeakerAgent(HeuristicSpeaker(distance_threshold=3, time_threshold=3))
# speaker = NeuralSpeakerAgent("", "")
speaker = NoopSpeaker()
follower = NoopFollower()  # HeuristicFollowerAgent(CautiousPlanner(confidence=(1., 1.)))
# follower = HeuristicFollowerAgent(CautiousPlanner(confidence=(.9, .5)))

plotter = Plotter(f"tasks-didact-12.json", "train", utt_obs_key=OBS_TARGET_DESCRIPTION)
plotter.prepare_environment(agent_role, speaker, follower,
                            hparams=take_env("didact",
                                             "full",
                                             False,
                                             False,
                                             False),
                            shuffle=True, task_index=None)
plotter.run_interactive()
