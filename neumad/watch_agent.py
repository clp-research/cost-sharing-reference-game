from neumad import create_eval_setting
from neumad.envs.manual_recording import Recorder
from neumad.envs.take_env import OBS_TARGET_DESCRIPTION, MAX_SIZE, OBS_LANGUAGE
from neumad.hparams import take_env

args, setting = create_eval_setting()

# if setting.agen_role.is_speaker():
#    utt_obs_key = OBS_TARGET_DESCRIPTION
# else:
utt_obs_key = OBS_LANGUAGE

agent = setting.agent
agent.load()
recorder = Recorder(args.split_name, args.target_map_size, utt_obs_key=utt_obs_key, shuffle=True, full_view=True)
recorder.prepare_environment(setting,
                             take_env("didact", "full",
                                      center_spawn=False, show_progress=True, only_center_pos=False))
recorder.record_episodes(agent, max_episodes=20, gif_duration=60,
                         record_name=f"{setting.agen_role.name}_{agent.get_name()}_{args.split_name}_{args.data_type}_{args.source_map_size}_{args.target_map_size}_{args.seed}",
                         record_dir="gifs")
