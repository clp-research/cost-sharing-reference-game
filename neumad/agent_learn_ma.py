import argparse

from neumad import set_global_seed, NeuralSpeakerAgent, Role
from neumad.agents.train_ma import MultiAgentTrainer
from neumad.hparams import take_env, model

"""
Setting is pre-determined: both agents are neural ones. We iteratively improve them keeping one side fix for N steps. 
"""
parser = argparse.ArgumentParser()
parser.add_argument("source_map_size", type=int, choices=[12, 21, 27])
parser.add_argument("data_type", type=str, choices=["naive", "didact"])
parser.add_argument("target_descriptor", type=str, choices=["minimal", "full"])
parser.add_argument("--speaker_name", "-SN", type=str, default="additive-rnd")
parser.add_argument("--follower_name", "-FN", type=str, default="additive-rnd")
parser.add_argument("--gpu", "-G", type=int, help="GPU device to use. Default: None (only cpu)", default=None)
parser.add_argument("--timesteps", "-T", type=int, help="Number in millions. Default: 1 for 1M", default=1)
parser.add_argument("--seed", "-R", type=int, help="Index of the random seed to use", required=True)
args = parser.parse_args()
set_global_seed(args.seed)

task_name = f"tasks-{args.data_type}-{args.source_map_size}-{args.target_descriptor}"
log_path = f"multi-agent/{task_name}/speaker_{args.speaker_name}/follower_{args.speaker_name}/{args.seed}"
ckpt_path = lambda model_name: f"saved_models/TakePieceEnv/v1/{model_name}/best_model.zip"

speaker_path = f"speaker-ma/{task_name}/{args.speaker_name}/follower_{args.speaker_name}/{args.seed}"
speaker = NeuralSpeakerAgent(speaker_path, agent_name=args.speaker_name,
                             hparams=model(Role.speaker, fusion_mode="additive", multi_agent=True))
if "pretrained" in args.speaker_name:
    model_path = f"speaker/tasks-didact-12-full/additive-rnd/follower_cautious_c=99/92999"
    speaker.load(ckpt_path(model_path), is_multi_agent=True)

follower_path = f"follower-ma/{task_name}/{args.follower_name}/speaker_{args.speaker_name}/{args.seed}"
follower = NeuralSpeakerAgent(follower_path, agent_name=args.follower_name,
                              hparams=model(Role.follower, fusion_mode="additive", multi_agent=True))
if "pretrained" in args.follower_name:
    model_path = f"follower/tasks-didact-12-full/additive-rnd/pretrained/92999"
    follower.load(ckpt_path(model_path), is_multi_agent=True)

env_params = take_env(args.data_type, args.target_descriptor, center_spawn=False, only_center_pos=False)

trainer = MultiAgentTrainer(args.source_map_size, speaker, follower)
trainer.prepare_environment(env_hparams=env_params)
trainer.train_agents(total_timesteps=args.timesteps * 1_000_000, log_path=log_path,
                     gpu=args.gpu, gpu_frac=0.2, seed=args.seed)
