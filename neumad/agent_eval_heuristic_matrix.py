import argparse

from tqdm import tqdm

from neumad import Setting
from neumad.agents import Role
from neumad.agents.eval import Evaluator
from neumad.agents.follower import HeuristicFollowerAgent
from neumad.agents.follower.planning_cautious import CautiousPlanner
from neumad.agents.speaker.adapters import HeuristicSpeakerAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.hparams import take_env_eval

parser = argparse.ArgumentParser("Usage: val")
parser.add_argument("split_name", type=str, help="[val,test,holdout]")
args = parser.parse_args()

map_sizes = [12, 21, 27]
reactivities = [1, 2, 3, 4]
uppers = [.99]
with tqdm(total=len(map_sizes) * len(reactivities) * len(uppers)) as pbar:
    for map_size in map_sizes:
        print("map:", map_size)
        evaluator = Evaluator(args.split_name, map_size)
        for reactivity in reactivities:
            print("speaker:", reactivity)
            tt = reactivity
            td = reactivity
            for upper in uppers:
                print("follower:", upper)
                env_follower = HeuristicFollowerAgent(heuristic_follower=CautiousPlanner(confidence=(upper, 0.5)))
                env_speaker = HeuristicSpeakerAgent(speaker=HeuristicSpeaker(distance_threshold=tt, time_threshold=td))
                setting = Setting(env_follower, env_follower, env_speaker, Role.follower)
                evaluator.prepare_environment(setting, take_env_eval(map_size, show_progress=False))
                evaluator.evaluate(env_follower)
                pbar.update(1)
