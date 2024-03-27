import argparse

from neumad import NeuralSpeakerAgent, Role, Setting
from neumad.agents.eval import Evaluator
from neumad.hparams import take_env

parser = argparse.ArgumentParser()
parser.add_argument("split_name", type=str, help="[val,test,holdout]")
parser.add_argument("target_map_size", type=int, choices=[12, 21, 27])
parser.add_argument("model_name", type=str, choices=["pretrained", "additive-rnd"])
parser.add_argument("--seed", "-R", type=int, help="Index of the random seed to use")
parser.add_argument("--gpu", "-G", type=int, help="GPU device to use. Default: None (only cpu)", default=None)

args = parser.parse_args()
split_name = args.split_name
target_map_size = args.target_map_size
seed = args.seed

model_name = args.model_name
file_name = "best_model"
# file_name="latest_eval_model"
task_name = f"tasks-didact-12-full"
ckpt_path = lambda model_path: f"saved_models/TakePieceEnv/MultiAgent/{model_path}/{file_name}.zip"

speaker_path = f"speaker-ma/{task_name}/{model_name}/follower_{model_name}/{seed}"
speaker = NeuralSpeakerAgent(speaker_path, agent_name="speaker_neural")
speaker.load(ckpt_path(speaker_path), is_multi_agent=True, device=args.gpu)

follower_path = f"follower-ma/{task_name}/{model_name}/speaker_{model_name}/{seed}"
follower = NeuralSpeakerAgent(follower_path, agent_name="follower_neural")
follower.load(ckpt_path(follower_path), is_multi_agent=True, device=args.gpu)

setting = Setting(follower, env_follower=follower, env_speaker=speaker, agent_role=Role.follower)

evaluator = Evaluator(split_name, target_map_size)
evaluator.prepare_environment(setting,
                              take_env("didact", "full",
                                       center_spawn=True, only_center_pos=False,
                                       show_progress=True))
evaluator.evaluate(follower)
