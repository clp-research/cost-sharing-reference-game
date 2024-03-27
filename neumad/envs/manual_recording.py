from typing import Tuple, Dict

import numpy as np
import imageio
from PIL import ImageFont, Image, ImageDraw

from cogrip.base_env import EPISODE_OUTCOME_SUCCESS, EPISODE_OUTCOME_ABORT
from cogrip.tasks import TaskLoader
from neumad import Setting
from neumad.agents import utils, Agent
from neumad.envs.take_env import SpeakerTakePieceEnv, FollowerTakePieceEnv, trans_obs, overview_to_greyscale

font = ImageFont.load_default()
if utils.is_ubuntu():
    # with open("/usr/share/fonts/truetype/freefont/FreeMono.ttf", "rb") as f:
    #    font = ImageFont.truetype(font=BytesIO(f.read()), size=10)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
if utils.is_mac():
    font = ImageFont.truetype("Keyboard.ttf", 20)


def _highlight_follower_view(vision_obs, max_size: int, agent_view_size: int, current_gripper_coords: Tuple[int, int]):
    x, y = current_gripper_coords
    context_size = int((agent_view_size - 1) / 2)
    topx = x - context_size
    topy = y - context_size
    for offy in range(agent_view_size):
        for offx in range(agent_view_size):
            vx = topx + offx
            vy = topy + offy
            if (vx >= 0 and vy >= 0) and (vx < max_size and vy < max_size):
                coord = vision_obs[vy, vx]
                if np.all(coord == 255):  # only affect the white pixels
                    vision_obs[vy, vx] = (235, 235, 235)


def cache_image(gif_images, partial, overview, utterance, step=None, success=None, abort=None, full_view=False):
    if full_view:
        bigger_img = overview.repeat(10, axis=0).repeat(10, axis=1)
    else:
        partial = np.moveaxis(partial, 0, -1)
        bigger_partial = partial.repeat(12, axis=0).repeat(12, axis=1)  # 35 * 6 = 210
        grey_overview = overview_to_greyscale(overview)  # MAX_SIZE * 5 = 210
        bigger_overview = grey_overview.repeat(7, axis=0).repeat(7, axis=1)  # 30 * 7 = 210
        bigger_img = np.concatenate((bigger_partial, bigger_overview), axis=1)

    if success is None:
        text_bg = (200, 200, 200)
    if success is True:
        text_bg = (0, 200, 0)
    if success is False:
        text_bg = (200, 0, 0)
    if abort:
        text_bg = (255, 165, 0)

    mission_img = Image.new("RGB", (bigger_img.shape[1], 40), text_bg)
    draw = ImageDraw.Draw(mission_img)
    if not utterance:
        utterance = "<silence>"
    if utterance == "<empty>":
        utterance = ""
    if step is not None:
        utterance = f"({step})  {utterance}"
    draw.text((5, 5), utterance, (0, 0, 0), font=font)
    mission_img = np.array(mission_img)

    image_with_text = np.concatenate([mission_img, bigger_img], axis=0)
    image_with_text = image_with_text.astype(np.uint8)
    gif_images.append(image_with_text)


class Recorder:

    def __init__(self, split_name: str, target_map_size: int, utt_obs_key: str, shuffle: bool, full_view: bool = False):
        """ New evaluator for a specific task set and environment"""
        self.split_name = split_name
        self.target_map_size = target_map_size
        self.utt_obs_key = utt_obs_key
        self.shuffle = shuffle
        """ init with prepare_environment()"""
        self.env = None
        self.full_view = full_view

    def prepare_environment(self, setting: Setting, env_hparams: Dict):
        print("Env-HParams:")
        for k, v in env_hparams.items():
            print(k, v)
        data_type = env_hparams["env.data_type"]
        task_file = f"tasks-{data_type}-{self.target_map_size}.json"
        task_loader = TaskLoader.from_file(self.split_name, file_name=task_file,
                                           do_shuffle=self.shuffle, force_shuffle=self.shuffle)

        agent_role = setting.agen_role
        speaker = setting.env_speaker
        follower = setting.env_follower

        if agent_role.is_speaker():
            self.env = SpeakerTakePieceEnv(task_loader, speaker, follower, hparams=env_hparams)
        elif agent_role.is_follower():
            self.env = FollowerTakePieceEnv(task_loader, speaker, follower, hparams=env_hparams)
        else:
            raise ValueError(agent_role)

    def record_episodes(self, agent: Agent, max_episodes: int, gif_duration: int,  # in seconds
                        record_name: str, record_dir: str):
        gif_images = []
        n_episodes = 0

        steps = 0
        total_reward = 0

        obs, _ = self.env.reset()

        utt, partial, overview = trans_obs(obs, self.utt_obs_key)
        if self.full_view:
            overview = self.env.render()
        prev_partial = partial
        prev_overview = overview
        state = None
        episode_start = np.ones((1,), dtype=bool)
        algorithm = agent.get_algorithm()
        print(f"Task: {self.env.current_task.idx}")
        while n_episodes < max_episodes:
            cache_image(gif_images, prev_partial, prev_overview, utt, step=steps, full_view=self.full_view)

            action, state = algorithm.predict(obs, deterministic=True, state=state, episode_start=episode_start)
            obs, reward, done, _, info = self.env.step(action)
            episode_start = done

            steps += 1
            total_reward += reward
            utt, partial, overview = trans_obs(obs, self.utt_obs_key)
            if self.full_view:
                overview = self.env.render()

            #cache_image(gif_images, partial, overview, "<empty>", step=steps, full_view=self.full_view)
            prev_partial = partial
            prev_overview = overview

            if done:
                n_episodes += 1
                cache_image(gif_images, partial, overview, utt, step=steps,
                            success=info[EPISODE_OUTCOME_SUCCESS] == 1,
                            abort=info[EPISODE_OUTCOME_ABORT] == 1,
                            full_view=self.full_view)
                self.env.reset()
                steps = 0
                total_reward = 0
                print(f"Task: {self.env.current_task.idx}")

        print(f"Save gif with {len(gif_images)} frames")
        file_name = f'{record_dir}/{record_name}.gif'
        print("Save git to", file_name)
        imageio.mimsave(file_name, gif_images, duration=1000 * gif_duration / len(gif_images), loop=0)
