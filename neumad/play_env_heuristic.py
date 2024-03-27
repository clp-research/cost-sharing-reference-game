from neumad import CautiousPlanner
from neumad.agents import Role
from neumad.agents.follower import HeuristicFollowerAgent
from neumad.agents.speaker.adapters import HeuristicSpeakerAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.envs.manual_play import Plotter
from neumad.envs.take_env import OBS_LANGUAGE, OBS_TARGET_DESCRIPTION
from neumad.hparams import take_env

agent_role = Role.follower
tt = 5
speaker = HeuristicSpeakerAgent(HeuristicSpeaker(distance_threshold=tt, time_threshold=tt))
follower = HeuristicFollowerAgent(CautiousPlanner(confidence=(.99, .5), debug=True))  # discount, lower

plotter = Plotter(f"tasks-didact-12.json", "train",
                  utt_obs_key=[OBS_LANGUAGE, OBS_TARGET_DESCRIPTION])
plotter.prepare_environment(agent_role, speaker, follower,
                            hparams=take_env("didact",
                                             "full",
                                             True,
                                             False,
                                             False),
                            shuffle=False, task_index=None)
plotter.fn_key_to_action = None  # overwrite
plotter.run_interactive()
