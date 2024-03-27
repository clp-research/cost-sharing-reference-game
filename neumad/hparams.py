from enum import Enum

from neumad.agents import Role
from neumad.agents.extractors.custom import VLFeaturesExtractor
from neumad.envs.take_env import OBS_RGB_PARTIAL, OBS_LANGUAGE, OBS_TARGET_DESCRIPTION, OBS_POS_FULL_CURRENT, \
    OBS_POS_FULL_TARGET

DEBUG = False


def model(role: Role, fusion_mode: str = "concat", is_recurrent: bool = True,
          feature_dims: int = 128, multi_agent: bool = False):
    assert isinstance(role, Role)
    if role.is_speaker():
        text_obs = OBS_TARGET_DESCRIPTION
        pos_obs = OBS_POS_FULL_TARGET
    if role.is_follower():
        text_obs = OBS_LANGUAGE
        pos_obs = OBS_POS_FULL_CURRENT
    memory_size = feature_dims
    if fusion_mode == "concat":
        memory_size = 2 * memory_size
    return {
        "agent.recurrent": is_recurrent,
        "agent.extractor": VLFeaturesExtractor,
        "agent.obs.vision": OBS_RGB_PARTIAL,
        "agent.obs.text": text_obs,
        "agent.obs.pos": pos_obs,
        "agent.features": feature_dims,
        "word_embedding_dims": 32,
        "policy.memory.size": memory_size,
        "policy.size": 64,
        "agent.fusion": fusion_mode,
        "agent.multi_agent": multi_agent
    }


def take_env_eval(target_map_size: int, show_progress: bool = True):
    return take_env("didact", "full", num_env=1,
                    center_spawn=True, only_center_pos=False, show_progress=show_progress)


def take_env(data_type, target_descriptor,  center_spawn: bool, only_center_pos: bool,
             show_progress: bool = False, num_env=4):
    return {
        "env.num_envs": num_env,
        "env.data_type": data_type,
        "env.trg_desc": target_descriptor,
        "env.center_spawn": center_spawn,
        "env.show_progress": show_progress,
        "env.only_center_pos": only_center_pos
    }
