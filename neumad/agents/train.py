from typing import Dict

import torch.cuda
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import DummyVecEnv

from cogrip.base_env import EPISODE_OUTCOME_SUCCESS
from cogrip.tasks import TaskLoader
from neumad import Setting
from neumad.agents import utils, Agent, Role, TrainableAgent
from neumad.agents.utils import default_save_dir, default_log_dir
from neumad.agents.callbacks import LogEpisodicEnvInfoCallback, RecordVideoCallback, SaveLatestCheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from neumad.envs.take_env import SpeakerTakePieceEnv, FollowerTakePieceEnv

ENV_NAME = "TakePieceEnv-v1"
class Trainer:

    def __init__(self, source_map_size: int):
        self.callbacks = list()
        self.source_map_size = source_map_size
        """ init with prepare_environment()"""
        self.train_env = None
        self.val_env = None
        self.val_episodes = None
        self.env_hparams = None

    def prepare_environment(self, setting: Setting, env_hparams: Dict):
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

        agent_role = setting.agen_role
        speaker = setting.env_speaker
        follower = setting.env_follower

        if agent_role.is_speaker():
            self.train_env = DummyVecEnv(
                [lambda: SpeakerTakePieceEnv(train_loader, speaker, follower, hparams=env_hparams)
                 for _ in range(num_envs)])
            self.val_env = SpeakerTakePieceEnv(val_loader, speaker, follower, hparams=env_hparams)
        elif agent_role.is_follower():
            self.train_env = DummyVecEnv(
                [lambda: FollowerTakePieceEnv(train_loader, speaker, follower, hparams=env_hparams)
                 for _ in range(num_envs)])
            self.val_env = FollowerTakePieceEnv(val_loader, speaker, follower, hparams=env_hparams)
        else:
            raise ValueError(agent_role)

    def train_agent(self, agent: TrainableAgent, time_steps, eval_freq=25000,
                    num_threads=1, gpu_frac=.5, gpu=None, seed: int = None):
        print("Training agent  : ", agent.get_name())
        if utils.is_ubuntu():
            torch.set_num_threads(num_threads)
            if gpu is not None:
                torch.cuda.set_device(gpu)
                torch.cuda.set_per_process_memory_fraction(gpu_frac, gpu)
        agent.setup(self.train_env)
        if utils.is_ubuntu():
            self.add_loginfo_callback()
            save_dir = default_save_dir(ENV_NAME, agent.get_path())

            self.add_save_latest_checkpoint_callback(save_dir)
            self.add_eval_callback(self.val_env,
                                   eval_video_env=None,  # create_env(loaders["val"].clone()),
                                   save_dir=save_dir,
                                   episodes_per_env=self.val_episodes, eval_freq=eval_freq)
            logger = agent.set_tb_logger(default_log_dir(ENV_NAME, agent.get_path()))
            hparams = dict()
            hparams.update(agent.hparams)
            hparams.update(self.env_hparams)
            logger.record("hparams", HParam(hparam_dict=prepare_hparams_for_upload(hparams),
                                            metric_dict={"eval/mean_reward": 0,
                                                         EPISODE_OUTCOME_SUCCESS: 0}))
        algorithm = agent.get_algorithm()
        algorithm.set_random_seed(seed)
        algorithm.learn(total_timesteps=time_steps,
                        progress_bar=True,
                        log_interval=-1,
                        callback=list(self.callbacks))

    def add_loginfo_callback(self):
        self.callbacks.append(LogEpisodicEnvInfoCallback())
        return self

    def add_train_video_callback(self, env, every_n_rollouts=100):
        self.callbacks.append(RecordVideoCallback(env, phase="train", n_freq=every_n_rollouts))
        return self

    def add_save_freq_checkpoint_callback(self, save_dir, save_freq=None):
        self.callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir, name_prefix="model"))
        return self

    def add_save_latest_checkpoint_callback(self, save_dir, only_on_training_end=False):
        self.callbacks.append(SaveLatestCheckpointCallback(save_dir, only_on_training_end))
        return self

    def add_eval_callback(self, eval_env, eval_video_env, save_dir: str,
                          episodes_per_env=5, eval_freq=1):
        # use a separate env for eval
        video_callback = None
        if eval_video_env is not None:
            video_callback = RecordVideoCallback(eval_video_env, phase="eval")
        print("Eval every:", eval_freq)
        self.callbacks.append(EvalCallback(eval_env,
                                           eval_freq=eval_freq,
                                           n_eval_episodes=episodes_per_env,
                                           best_model_save_path=save_dir, verbose=False,
                                           callback_on_new_best=video_callback))
        return self


def prepare_hparams_for_upload(hparams):
    # replace classes with name (cannot upload a class)
    if "agent.extractor" in hparams:
        hparams["agent.extractor"] = hparams["agent.extractor"].get_name()
    return hparams
