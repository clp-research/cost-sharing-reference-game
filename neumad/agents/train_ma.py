from typing import Dict

import torch.cuda
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo.utils import agent_selector
from tqdm import tqdm

from cogrip.base_env import EPISODE_OUTCOME_SUCCESS
from cogrip.tasks import TaskLoader
from neumad.agents import utils, TrainableAgent
from neumad.agents.ppo_ma import MultiAgentRecurrentPPO
from neumad.agents.utils import default_save_dir, default_log_dir
from neumad.agents.callbacks import LogEpisodicEnvInfoCallback, LatestEvalCheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

from neumad.envs.take_env import SpeakerTakePieceEnv, FollowerTakePieceEnv


class MultiAgentTrainer:

    def __init__(self, source_map_size: int, speaker: TrainableAgent, follower: TrainableAgent):
        self.callbacks = dict(speaker=[], follower=[])
        self.source_map_size = source_map_size
        self.speaker = speaker
        self.follower = follower
        self.agents: Dict[str, TrainableAgent] = dict(speaker=speaker, follower=follower)
        self.agent_selector = agent_selector(agent_order=[(n, a) for n, a in self.agents.items()])
        """ init with prepare_environment()"""
        self.train_env_speaker = None
        self.train_env_follower = None
        self.val_envs = dict()
        self.val_episodes = None
        self.env_hparams = None

    def prepare_environment(self, env_hparams: Dict):
        self.env_hparams = env_hparams
        print("Env-HParams:")
        for k, v in env_hparams.items():
            print(k, v)
        num_envs = env_hparams["env.num_envs"]
        data_type = env_hparams["env.data_type"]

        task_file = f"tasks-{data_type}-{self.source_map_size}.json"
        task_loaders, _ = TaskLoader.all_from_file(file_name=task_file, do_shuffle=True)
        train_loader = task_loaders["train"]
        val_loader = task_loaders["val"]
        self.val_episodes = len(val_loader)

        self.train_env_speaker = DummyVecEnv(
            [lambda: SpeakerTakePieceEnv(train_loader, self.speaker, self.follower, hparams=env_hparams)
             for _ in range(num_envs)])
        self.train_env_follower = DummyVecEnv(
            [lambda: FollowerTakePieceEnv(train_loader.clone(), self.speaker, self.follower, hparams=env_hparams)
             for _ in range(num_envs)])

        self.val_envs["follower"] = FollowerTakePieceEnv(val_loader, self.speaker, self.follower, hparams=env_hparams)
        self.val_envs["speaker"] = SpeakerTakePieceEnv(val_loader, self.speaker, self.follower, hparams=env_hparams)

    def train_agents(self, total_timesteps, log_path,
                     eval_freq=25000, num_threads=1, gpu_frac=.5, gpu=None, seed: int = None):
        print("Training multi-agent")
        if utils.is_ubuntu():
            torch.set_num_threads(num_threads)
            if gpu is not None:
                torch.cuda.set_device(gpu)
                torch.cuda.set_per_process_memory_fraction(gpu_frac, gpu)

        self.speaker.setup(self.train_env_speaker)
        self.follower.setup(self.train_env_follower)

        if utils.is_ubuntu():
            env_name = "TakePieceEnv-MultiAgent"
            log_dir = default_log_dir(env_name, log_path)
            logger = configure_logger(tensorboard_log=log_dir, tb_log_name="MultiAgentRecurrentPPO",
                                      reset_num_timesteps=True)
            print(logger.dir)
            for name, agent in self.agents.items():
                self.callbacks[name].append(LogEpisodicEnvInfoCallback())
                save_dir = default_save_dir(env_name, agent.get_path())
                self.callbacks[name].append(EvalCallback(self.val_envs[name],
                                                         callback_after_eval=LatestEvalCheckpointCallback(save_dir),
                                                         eval_freq=eval_freq,
                                                         n_eval_episodes=self.val_episodes,
                                                         best_model_save_path=save_dir, verbose=False))
                agent.get_algorithm().set_logger(logger)  # shared logger

            hparams = dict()
            hparams.update(self.follower.hparams)
            hparams.update(self.speaker.hparams)
            hparams.update(self.env_hparams)
            logger.record("hparams", HParam(hparam_dict=prepare_hparams_for_upload(hparams),
                                            metric_dict={"eval/mean_reward": 0,
                                                         EPISODE_OUTCOME_SUCCESS: 0}))

        for name, agent in self.agents.items():
            algorithm: MultiAgentRecurrentPPO = agent.get_algorithm()
            algorithm.set_random_seed(seed)
            _, callbacks = algorithm.ma_setup_learn(total_timesteps, self.callbacks[name])
            callbacks.on_training_start(locals(), globals())
            self.callbacks[name] = callbacks  # set CallbackList

        num_timesteps = 0
        with tqdm(total=num_timesteps) as pbar:
            current_name, current_agent = self.agent_selector.reset()
            pbar.set_description(f"Train {current_name}")
            while num_timesteps < total_timesteps:
                current_algorithm = current_agent.get_algorithm()
                n_steps = current_algorithm.num_timesteps
                pbar.set_description(f"Collect rollouts {current_name}")
                continue_training = current_algorithm.ma_collect_rollouts(self.callbacks[current_name])
                n_steps = current_algorithm.num_timesteps - n_steps
                num_timesteps += n_steps
                pbar.update(n_steps)
                if continue_training is False:
                    break
                current_algorithm.ma_update_current_progress_remaining(total_timesteps)
                current_algorithm.logger.dump(step=num_timesteps)
                pbar.set_description(f"Train {current_name}")
                current_algorithm.ma_train()
                current_name, current_agent = self.agent_selector.next()

        for name, agent in self.agents.items():
            self.callbacks[name].on_training_end()


def prepare_hparams_for_upload(hparams):
    # replace classes with name (cannot upload a class)
    if "agent.extractor" in hparams:
        hparams["agent.extractor"] = hparams["agent.extractor"].get_name()
    return hparams
