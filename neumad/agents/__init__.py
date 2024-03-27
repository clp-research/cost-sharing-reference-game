import abc
from enum import Enum
from typing import Optional, Union, Dict, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger


class Role(Enum):
    """ An agent can be in the role of the follower or speaker """
    follower = 0
    speaker = 1

    def is_speaker(self):
        return self == Role.speaker

    def is_follower(self):
        return self == Role.follower

    @staticmethod
    def from_string(role: str):
        for _role in list(Role):
            if str(_role.name) == role:
                return _role
        raise ValueError(role)


class Agent(abc.ABC):
    """ Agents can act either outside of an environment or from within"""

    def __init__(self, agent_path: str, agent_name: str):
        self.agent_path = agent_path
        self.agent_name = agent_name

    def get_path(self) -> str:
        """ Relative path of the agents location under a root directory """
        return self.agent_path

    def get_ckpt_path(self, model_root="saved_models/TakePieceEnv/v1", zip_file="best_model") -> str:
        model_path = self.get_path()
        return f"{model_root}/{model_path}/{zip_file}"  # .zip will be automatically appended

    def get_name(self) -> str:
        return self.agent_name

    @abc.abstractmethod
    def get_algorithm(self) -> BaseAlgorithm:
        """
        An action predictor, here BaseAlgorithm, because it allows direct use in callbacks.
        :return the algorithm used for training or evaluation (as a predictor)
        """
        pass

    @abc.abstractmethod
    def on_reset(self, current_task, current_board, current_gripper_coords, info) -> None:
        """ Called from within an environment, when the agent acts as part of an env """
        pass

    @abc.abstractmethod
    def on_step(self, obs, action=None) -> Union[np.array, int, str]:
        """ Called from within an environment, when the agent acts as part of an env """
        pass


class TrainableAgent(Agent, abc.ABC):

    def __init__(self, agent_path: str, agent_name: str, hparams: Dict):
        super().__init__(agent_path, agent_name)
        self.hparams = hparams
        if hparams is None:
            print("HParams: <not given> (possibly before loading an agent)")
        else:
            print("HParams:")
            for k, v in hparams.items():
                print(k, v)

    @abc.abstractmethod
    def setup(self, env):
        pass

    @abc.abstractmethod
    def set_tb_logger(self, log_dir):
        pass


class TestableAgent(Agent, abc.ABC):

    def __init__(self, agent_path: str, agent_name: str):
        super().__init__(agent_path, agent_name)

    @abc.abstractmethod
    def set_env(self, env):
        """ For evaluation """
        pass

    @abc.abstractmethod
    def set_logger(self, logger: Logger):
        """ For evaluation """
        pass


class HeuristicAgent(TestableAgent, abc.ABC):
    """ No need to store weights; we actually never train this agent. The agent only predicts. """

    def __init__(self, agent_name):
        super().__init__(agent_path=f"heuristic/{agent_name}", agent_name=agent_name)
        """ Helpers for evaluation (parts of BaseAlgorithm) """
        self.env = None
        self.logger = None  # for compatibility with callback during eval
        self.num_timesteps = 0  # for compatibility with callback during eval

    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]],
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """ Integration with the evaluation:
                - for neural we return the algorithm
                - for heuristic we implement only the relevant parts (this method)
        """
        actions = []
        if isinstance(observation, dict):  # unpack batches of values to batch of obs (during eval)
            obss = []
            for key, values in observation.items():
                for idx, value in enumerate(values):
                    if len(obss) < idx + 1:
                        obss.append(dict())
                    obss[idx][key] = value  # heuristic follower works with numpy
            observation = obss
        for obs in observation:  # batch of observations: this is what we get during trajectory collection
            action = self.on_step(obs)
            actions.append(action)
        return np.array(actions), None

    def get_algorithm(self) -> BaseAlgorithm:
        return self

    def set_logger(self, logger: Logger):
        self.logger = logger

    def set_env(self, env):
        self.env = env

    def get_env(self):
        """ Integration with the callbacks on evaluation """
        return self.env


class NoopAgent(Agent, abc.ABC):

    def __init__(self, agent_name):
        super().__init__(agent_path=f"noop/{agent_name}", agent_name=agent_name)

    def get_algorithm(self) -> BaseAlgorithm:
        raise NotImplementedError()

    def set_logger(self, logger: Logger):
        raise NotImplementedError()

    def set_env(self, env):
        raise NotImplementedError()

    def on_reset(self, current_task, current_board, current_gripper_coords, info):
        pass  # nothing to reset
