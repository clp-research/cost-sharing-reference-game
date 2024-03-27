import argparse
from typing import Tuple, Optional

from neumad.agents import Agent
from neumad.agents.follower import NoopFollower, HeuristicFollowerAgent
from neumad.agents.follower.adapters import NeuralFollowerAgent
from neumad.agents.follower.planning_cautious import CautiousPlanner
from neumad.agents.speaker import NoopSpeaker
from neumad.agents.speaker.adapters import NeuralSpeakerAgent, HeuristicSpeakerAgent
from neumad.agents.speaker.heuristic import HeuristicSpeaker
from neumad.agents.utils import set_global_seed
from neumad.hparams import Role, model

OPT_NEURAL = "neural"
OPT_HEURISTIC = "heuristic"
OPT_SPEAKER = "speaker"
OPT_FOLLOWER = "follower"


class Setting:

    def __init__(self, agent: Agent, env_follower: Agent, env_speaker: Agent, agent_role: Role):
        self.agent: Agent = agent
        self.env_speaker: Agent = env_speaker
        self.env_follower: Agent = env_follower
        self.agen_role: Role = agent_role


def create_training_setting() -> Tuple[argparse.Namespace, Setting]:
    parser = argparse.ArgumentParser()
    add_setting_args(parser)
    parser.add_argument("--timesteps", "-T", type=int, help="Number in millions. Default: 1 for 1M", default=1)
    parser.add_argument("--seed", "-R", type=int, help="Index of the random seed to use", required=True)
    args = parser.parse_args()
    setting = setting_from_args(args, args.seed)
    return args, setting


def create_eval_setting() -> Tuple[argparse.Namespace, Optional[Setting]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_name", type=str, help="[val,test,holdout]")
    parser.add_argument("target_map_size", type=int, choices=[12, 21, 27])
    parser.add_argument("--seed", "-R", type=int, help="Index of the random seed to use")
    parser.add_argument("--seeds_all", "-A", action="store_true",
                        help="Evaluate agents for all available seeds. "
                             "Useful if an agents has been trained on all seeds.")
    add_setting_args(parser)
    args = parser.parse_args()
    # call this anyway, also with seed=None to create an initial setting
    setting = setting_from_args(args, args.seed)
    return args, setting


def add_setting_args(parser: argparse.ArgumentParser):
    parser.add_argument("source_map_size", type=int, choices=[12, 21, 27])
    parser.add_argument("data_type", type=str, choices=["naive", "didact"])
    parser.add_argument("target_descriptor", type=str, choices=["minimal", "full"])
    parser.add_argument("role", type=str, choices=[OPT_FOLLOWER, OPT_SPEAKER],
                        help=f"The agent that serves the step action.")
    parser.add_argument("--speaker_type", "-S", type=str, choices=[OPT_NEURAL, OPT_HEURISTIC])
    parser.add_argument("--speaker_name", "-SN", type=str, default="additive-rnd")
    parser.add_argument("--follower_type", "-F", type=str, choices=[OPT_NEURAL, OPT_HEURISTIC])
    parser.add_argument("--follower_name", "-FN", type=str, default="additive-rnd")
    parser.add_argument("--environment", "-E", type=str, help="Force an environment. Default: env == role")
    parser.add_argument("--gpu", "-G", type=int, help="GPU device to use. Default: None (only cpu)", default=None)


def setting_from_args(args, seed):
    agent_role = Role.from_string(args.role)
    agent = None  # to be trained
    task_name = f"tasks-{args.data_type}-{args.source_map_size}-{args.target_descriptor}"

    if args.follower_type == OPT_NEURAL and agent_role.is_follower():
        # init other partner first, so we can use the name in the model_path
        if args.speaker_type == OPT_NEURAL:
            # preset the model to use here for now
            model_name = args.speaker_name
            assert model_name == "pretrained"
            model_path = f"speaker/tasks-didact-12-full/additive-rnd/follower_cautious_c=99/92999"
            fusion_mode = "concat" if "concat" in model_path else "additive"
            env_speaker = NeuralSpeakerAgent(model_path, model_name, hparams=model(Role.speaker, fusion_mode))
            env_speaker.load(device=args.gpu)
        elif args.speaker_type == OPT_HEURISTIC:
            # for example tt=2
            speaker_name = args.speaker_name
            reactivity = int(speaker_name.split("=")[1])
            env_speaker = HeuristicSpeakerAgent(
                HeuristicSpeaker(distance_threshold=reactivity, time_threshold=reactivity))
        else:
            env_speaker = NoopSpeaker()
        # for example additive-rnd
        model_name = args.follower_name
        fusion_mode = "concat" if "concat" in model_name else "additive"
        if seed is None:
            # define a placeholder
            model_path = f"follower/{task_name}/{model_name}/{env_speaker.get_name()}/<SEED>"
        else:
            model_path = f"follower/{task_name}/{model_name}/{env_speaker.get_name()}/{seed}"
        agent = NeuralFollowerAgent(model_path, model_name, hparams=model(agent_role, fusion_mode))
        env_follower = agent  # assigned, but "does nothing"

    if args.speaker_type == OPT_NEURAL and agent_role.is_speaker():
        # init other partner first, so we can use the name in the model_path
        if args.follower_type == OPT_NEURAL:
            NotImplementedError()
        elif args.follower_type == OPT_HEURISTIC:
            # for example cautious_c=85
            follower_name = args.follower_name
            strategy, conf = follower_name.split("_")
            upper = int(conf.split("=")[1]) / 100
            assert strategy in ["cautious"]
            planner = CautiousPlanner(confidence=(upper, 0.5))
            env_follower = HeuristicFollowerAgent(planner)
        else:
            env_follower = NoopFollower()
        # for example additive-rnd
        model_name = args.speaker_name
        fusion_mode = "concat" if "concat" in model_name else "additive"
        if seed is None:
            model_path = f"speaker/{task_name}/{model_name}/{env_follower.get_name()}/<SEED>"
        else:
            model_path = f"speaker/{task_name}/{model_name}/{env_follower.get_name()}/{seed}"
        agent = NeuralSpeakerAgent(model_path, model_name, hparams=model(agent_role, fusion_mode))
        env_speaker = agent  # agent is itself the verbalizer

    if args.environment is not None:
        # special case where we train a speaker acting as the follower (on the follower env)
        # both follower and speaker receive the same obs, but speaker uses the target descriptor
        agent_role = Role.from_string(args.environment)
        print("Force environment: ", agent_role)

    assert agent is not None, "No agent defined"
    return Setting(agent, env_follower, env_speaker, agent_role)
