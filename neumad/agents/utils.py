import os
import time
import sys

from stable_baselines3.common.utils import set_random_seed

SEEDS = [49184, 72611, 12784, 98506, 92999, 61680, 25011, 55166, 65337, 53893]


def debug_grid(grid):
    print()
    for idx, row in enumerate(grid[0]):
        print(idx, row)
    print()
    for idx, row in enumerate(grid[-1]):
        print(idx, row)


def default_seed():
    return SEEDS[0]


def get_all_seeds():
    return SEEDS


def set_global_seed(seed: int = None):
    assert seed in SEEDS, f"{seed} not in {SEEDS}"
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_random_seed(seed)
    return seed


def default_dir(basedir, env_name, model_name, use_timestamp=False, sub_dir=None):
    env_name = env_name.replace("-", "/")
    path = f"{basedir}/{env_name}/{model_name}"
    if use_timestamp:
        timestamp = int(time.time() * 1000)
        path = path + f"/{timestamp}"
    if sub_dir is not None:
        path = path + f"/{sub_dir}"
    return path


def default_log_dir(env_name, model_name, logdir="/cache/tensorboard-logdir", use_timestamp=False, sub_dir=None):
    return default_dir(logdir, env_name, model_name, use_timestamp, sub_dir)


def default_save_dir(env_name, model_path, use_timestamp=False, sub_dir=None):
    return default_dir("saved_models", env_name, model_path, use_timestamp, sub_dir)


def default_ckpts_path(env_name, model_path, file_name="best_model", use_timestamp=False, sub_dir=None):
    save_dir = default_dir("saved_models", env_name, model_path, use_timestamp, sub_dir)
    save_path = save_dir + f"/{file_name}"
    return save_path


def default_results_dir(env_name, model_name, use_timestamp=False, sub_dir=None):
    results_dir = default_dir("results", env_name, model_name, use_timestamp, sub_dir)
    return results_dir


def default_results_path(env_name, model_name, file_name="best_model", use_timestamp=False, sub_dir=None):
    results_dir = default_dir("results", env_name, model_name, use_timestamp, sub_dir)
    results_path = results_dir + f"/{file_name}"
    return results_path


def is_ubuntu():
    return sys.platform == "linux"


def is_mac():
    return sys.platform == "darwin"


def get_current_timestamp():
    return int(time.time() * 1000)


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
