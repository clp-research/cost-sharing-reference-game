from neumad import NeuralSpeakerAgent, Role, Setting
from neumad.envs.manual_recording import Recorder
from neumad.envs.take_env import OBS_TARGET_DESCRIPTION, MAX_SIZE, OBS_LANGUAGE
from neumad.hparams import take_env

split_name = "val"
seed = 98506
model_name = "additive-rnd"
# model_name = "pretrained"
task_name = f"tasks-didact-12-full"
# file_name = "best_model"
file_name = "latest_eval_model"
ckpt_path = lambda model_name: f"saved_models/TakePieceEnv/MultiAgent/{model_name}/{file_name}.zip"

speaker_path = f"speaker-ma/{task_name}/{model_name}/follower_{model_name}/{seed}"
speaker = NeuralSpeakerAgent(speaker_path, agent_name="speaker_neural")
speaker.load(ckpt_path(speaker_path), is_multi_agent=True)

follower_path = f"follower-ma/{task_name}/{model_name}/speaker_{model_name}/{seed}"
follower = NeuralSpeakerAgent(follower_path, agent_name="follower_neural")
follower.load(ckpt_path(follower_path), is_multi_agent=True)

# if setting.agen_role.is_speaker():
#    utt_obs_key = OBS_TARGET_DESCRIPTION
# else:
utt_obs_key = OBS_LANGUAGE

setting = Setting(follower, env_follower=follower, env_speaker=speaker, agent_role=Role.follower)
recorder = Recorder(split_name, 12, utt_obs_key=utt_obs_key, shuffle=True)
recorder.prepare_environment(setting,
                             take_env("didact", "full",
                                      center_spawn=True, show_progress=True, only_center_pos=False))
recorder.record_episodes(follower, max_episodes=10, gif_duration=60,
                         record_name=f"{split_name}_multi_agent_{model_name}_{seed}",
                         record_dir="gifs")
