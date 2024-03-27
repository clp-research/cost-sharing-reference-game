import argparse

from neumad import CautiousPlanner
from neumad.agents import Role
from neumad.agents.eval import Evaluator
from neumad.agents.follower import HeuristicFollowerAgent
from neumad.agents.speaker.adapters import HeuristicSpeakerAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.hparams import take_env

parser = argparse.ArgumentParser()
parser.add_argument("split_name", type=str, help="[val,test,holdout]")
parser.add_argument("target_map_size", type=int, choices=[12, 21, 27])
args = parser.parse_args()

env_follower = HeuristicFollowerAgent(heuristic_follower=CautiousPlanner(confidence=(0.9, 0.5)))
env_speaker = HeuristicSpeakerAgent(speaker=HeuristicSpeaker(distance_threshold=3, time_threshold=3))

evaluator = Evaluator(args.split_name, args.target_map_size)
evaluator.prepare_environment(Role.follower, env_speaker, env_follower,
                              take_env("didact", "full",
                                       center_spawn=True, only_center_pos=False, show_progress=True))
evaluator.evaluate(env_follower)
