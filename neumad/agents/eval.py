from typing import Dict

import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import make_output_format, Logger
from stable_baselines3.common.monitor import Monitor

from cogrip.tasks import TaskLoader
from neumad import Setting
from neumad.agents import Agent, Role, TestableAgent
from neumad.agents.callbacks import LogEpisodicEnvInfoCallback
from neumad.envs.take_env import SpeakerTakePieceEnv, FollowerTakePieceEnv


class Evaluator:

    def __init__(self, split_name: str, target_map_size: int, data_type: str = "didact"):
        """ New evaluator for a specific task set and environment"""
        self.split_name = split_name
        self.target_map_size = target_map_size
        self.results_root = "results/TakePieceEnv"
        """ load tasks """
        task_file = f"tasks-{data_type}-{target_map_size}.json"
        self.task_loader = TaskLoader.from_file(split_name, file_name=task_file)
        self.n_episodes = len(self.task_loader)
        """ init with prepare_environment()"""
        self.eval_env = None
        self.file_suffix = None

    def prepare_environment(self, setting: Setting, env_hparams: Dict, verbose: bool = False):
        """
        :param agent_role: the agent is trained for. Determines the env.
        :param speaker: within the environment. Might be NoopAgent.
        :param follower: within the environment. Might be NoopAgent.
        :param num_envs: to create for training
        """
        self.task_loader.reset()  # reset in case of several calls
        self.file_suffix = f"_{self.split_name}_{self.target_map_size}"  # reset in case of several calls

        if verbose:
            print("Env-HParams:")
            for k, v in env_hparams.items():
                print(k, v)

        agent_role = setting.agen_role
        speaker = setting.env_speaker
        follower = setting.env_follower

        if agent_role.is_speaker():
            self.eval_env = SpeakerTakePieceEnv(self.task_loader, speaker, follower, hparams=env_hparams)
            self.file_suffix += f"_{follower.get_name()}"
        elif agent_role.is_follower():
            self.eval_env = FollowerTakePieceEnv(self.task_loader, speaker, follower, hparams=env_hparams)
            self.file_suffix += f"_{speaker.get_name()}"
        else:
            raise ValueError(agent_role)

    def evaluate(self, agent: TestableAgent):
        print("Evaluate agent  : ", agent.get_name())
        torch.set_num_threads(1)

        agent_path = agent.get_path()
        results_dir = f"{self.results_root}/{agent_path}"
        logger = Logger(results_dir, output_formats=[
            make_output_format("log", results_dir, self.file_suffix),
            make_output_format("json", results_dir, self.file_suffix)
        ])
        agent.set_logger(logger)
        agent.set_env(self.eval_env)  # make env available to callback (similar to setup() for train)

        metric_cb = LogEpisodicEnvInfoCallback()
        # todo: this might be simplified avoiding the use of algorithm (only to carry env and logger)
        algorithm = agent.get_algorithm()
        metric_cb.init_callback(algorithm)  # requires that env is set before

        monitor_env = Monitor(env=self.eval_env, filename=f"{results_dir}/{self.split_name}", override_existing=True)
        metric_cb.on_rollout_start()
        evaluate_policy(algorithm, monitor_env, n_eval_episodes=self.n_episodes, deterministic=True,
                        callback=metric_cb.on_eval_step)
        metric_cb.on_rollout_end()
        logger.dump()  # write the metrics to the logger output
