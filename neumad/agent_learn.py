from neumad import create_training_setting, set_global_seed
from neumad.agents.train import Trainer
from neumad.hparams import take_env

args, setting = create_training_setting()
set_global_seed(args.seed)

agent = setting.agent
assert agent is not None, "No agent to be trained defined"

model_name = agent.get_name()
center_spawn = False if "rnd" in model_name else True
env_params = take_env(args.data_type, args.target_descriptor, center_spawn, only_center_pos=False)
trainer = Trainer(args.source_map_size)
trainer.prepare_environment(setting, env_hparams=env_params)
trainer.train_agent(agent, time_steps=args.timesteps * 1_000_000, gpu=args.gpu, seed=args.seed)
