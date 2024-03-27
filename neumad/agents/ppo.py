import os
from abc import ABC
from typing import Dict

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import configure_logger

from neumad.agents import TrainableAgent, TestableAgent
from neumad.agents.ppo_ma import MultiAgentRecurrentPPO
from neumad.agents.utils import count_parameters


class PPOAgent(TrainableAgent, TestableAgent, ABC):

    def __init__(self, agent_path: str, agent_name: str, hparams: Dict):
        super().__init__(agent_path, agent_name, hparams)
        self.algorithm: OnPolicyAlgorithm = None

    def setup(self, env):
        if self.algorithm is None:
            print("Setup PPOAgent: Initialize new algorithm")
            self.algorithm = self.create_algorithm(env)
        else:
            print("Setup PPOAgent: Algorithm already loaded. Setting env.")
            self.set_env(env)  # make env available to callback (similar to setup() for train)
        print("Trainable parameters:", count_parameters(self.algorithm.policy, only_trainable=True))
        return self

    def set_tb_logger(self, log_dir):
        """ For training """
        tb_log_name = self.algorithm.__class__.__name__
        logger = configure_logger(tensorboard_log=log_dir, tb_log_name=tb_log_name, reset_num_timesteps=True)
        self.algorithm.set_logger(logger)
        self.algorithm.policy.features_extractor.logger = logger
        return logger

    def get_algorithm(self) -> BaseAlgorithm:
        return self.algorithm

    def load(self, ckpt_path: str = None, is_recurrent=True, is_multi_agent: bool = False,
             device: int = None, zip_file: str = "best_model"):
        if ckpt_path is None:  # try default
            print(f"Auto-detect agent checkpoint")
            ckpt_path = self.get_ckpt_path(zip_file=zip_file)
        print(f"Load agent from {ckpt_path}")
        if is_recurrent:
            if is_multi_agent:
                self.algorithm = MultiAgentRecurrentPPO.load(ckpt_path, device=f"cuda:{device}" if device else "auto")
            else:
                # if an env is given here, some unuseful code is called
                self.algorithm = RecurrentPPO.load(ckpt_path, device=f"cuda:{device}" if device else "auto")
        else:
            raise NotImplementedError("PPO not supported")
            #    self.algorithm = PPO.load(ckpt_path)  # if an env is given here, some unuseful code is called
        return self

    def set_logger(self, logger):
        self.algorithm.set_logger(logger)

    def set_env(self, env):
        self.algorithm.env = env  # not using set_env() which wraps this one

    def create_algorithm(self, env):
        policy_dims = self.hparams["policy.size"]
        policy_kwargs = {
            "features_extractor_class": self.hparams["agent.extractor"],
            "features_extractor_kwargs": {"hparams": self.hparams},
            "net_arch": dict(pi=[policy_dims, policy_dims], vf=[policy_dims, policy_dims]),
            "normalize_images": True
        }
        if self.hparams["agent.recurrent"]:
            policy_kwargs.update({
                "shared_lstm": True,
                "enable_critic_lstm": False,
                "n_lstm_layers": 1,
                "lstm_hidden_size": self.hparams["policy.memory.size"],
            })
            if self.hparams["agent.multi_agent"]:
                return MultiAgentRecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs)
            return RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=policy_kwargs)
        raise NotImplementedError("PPO not supported")
        # return PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
