import os
import traceback

from neumad import Setting, create_eval_setting, setting_from_args
from neumad.agents.eval import Evaluator
from neumad.hparams import take_env


def evaluate_agent(setting: Setting):
    agent = setting.agent
    assert agent is not None, "No agent to be evaluated defined"
    print("Evaluate: ", agent.get_ckpt_path())
    agent.load()
    evaluator = Evaluator(args.split_name, args.target_map_size)
    evaluator.prepare_environment(setting,
                                  take_env("didact", "full",
                                           center_spawn=True, only_center_pos=False, show_progress=True))
    evaluator.evaluate(agent)


if __name__ == "__main__":
    args, setting = create_eval_setting()
    if args.seeds_all:
        ckpt_path = setting.agent.get_ckpt_path()
        model_dir = ckpt_path.split("<SEED>")[0]
        for seed in os.listdir(model_dir):
            try:
                setting = setting_from_args(args, seed)
                evaluate_agent(setting)
            except Exception as e:
                # maybe specific seeds were not stored
                traceback.print_exception(e)
    else:
        evaluate_agent(setting)
