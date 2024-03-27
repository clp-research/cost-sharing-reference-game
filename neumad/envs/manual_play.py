from typing import Dict

import numpy as np

from cogrip.tasks import TaskLoader
from neumad.agents import Role, Agent
from neumad.envs import FollowerAction, SpeakerAction
from neumad.envs.take_env import SpeakerTakePieceEnv, FollowerTakePieceEnv, trans_obs, overview_to_greyscale


def prep_ax(ax, utt, partial, overview, step=None, reward=None, total_reward=None, context=None, done=None, idx=None):
    partial = np.moveaxis(partial, 0, -1)
    bigger_img = partial.repeat(12, axis=0).repeat(12, axis=1)  # 35 * 6 = 420
    grey_overview = overview_to_greyscale(overview)  # MAX_SIZE * 5 = 210
    bigger_overview = grey_overview.repeat(7, axis=0).repeat(7, axis=1)  # 30 * 7 = 420
    image = np.concatenate((bigger_img, bigger_overview), axis=1)
    if context is None:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticks([])
    title_msg = ""
    if idx:
        title_msg += f"idx: {idx}\n"
    title_msg += f"utt: {utt}\n"
    if step:
        title_msg += f"step: {step}"
    if reward:
        title_msg += f" reward: {round(reward, 2)}"
    if total_reward:
        title_msg += f" total: {round(total_reward, 2)}\n"
    if done is not None:
        title_msg += "SUCCESS" if done else "FAILURE"
    ax.set_title(title_msg, loc='left')
    if context:
        context.set_data(image)
    else:
        context = ax.imshow(image)
    return context


class Plotter:

    def __init__(self, task_file: str, split_name: str, utt_obs_key: str) -> object:
        """ New evaluator for a specific task set and environment"""
        self.split_name = split_name
        self.task_file = task_file
        self.utt_obs_key = utt_obs_key
        """ init with prepare_environment()"""
        self.env = None
        self.n_episodes = None
        self.fn_key_to_action = None

    def prepare_environment(self, agent_role: Role, speaker: Agent, follower: Agent, hparams: Dict,
                            shuffle=True, task_index=None):
        """
        :param agent_role: the agent is trained for. Determines the env.
        :param speaker: within the environment. Might be NoopAgent.
        :param follower: within the environment. Might be NoopAgent.
        :param num_envs: to create for training
        """
        task_loader = TaskLoader.from_file(self.split_name, file_name=self.task_file, do_shuffle=shuffle,
                                           force_shuffle=shuffle, task_index=task_index)
        self.n_episodes = len(task_loader)
        if agent_role.is_speaker():
            self.env = SpeakerTakePieceEnv(task_loader, speaker, follower, hparams)

            def speaker_event_handling(event):
                key = event.key.lower()
                action = SpeakerAction.silence  # zero
                directions = ["a", "d", "w", "s"]
                if key == "q":
                    action = SpeakerAction.confirmation
                if key == "e":
                    action = SpeakerAction.correction
                if key in directions:
                    action = SpeakerAction.from_object(directions.index(key) + 3)
                if key == " ":
                    action = SpeakerAction.take
                references = ["1", "2", "3", "4", "5", "6"]
                if key in references:
                    action = SpeakerAction.from_object(references.index(key) + 8)
                print(action)
                return action

            self.fn_key_to_action = speaker_event_handling
        elif agent_role.is_follower():
            self.env = FollowerTakePieceEnv(task_loader, speaker, follower, hparams)

            def follower_event_handling(event):
                key = event.key.lower()
                action = FollowerAction.wait  # zero
                directions = ["a", "d", "w", "s"]
                if key in directions:
                    action = directions.index(key) + 1
                if key == " ":
                    action = FollowerAction.take
                return action

            self.fn_key_to_action = follower_event_handling
        else:
            raise ValueError(agent_role)

    def run_interactive(self):
        from matplotlib import pyplot as plt
        # see https://matplotlib.org/stable/tutorials/introductory/customizing.html
        plt.rcParams['keymap.save'].remove('s')
        plt.rcParams['keymap.fullscreen'].remove('f')
        plt.rcParams['keymap.quit'].remove('q')

        global context
        global steps
        global total_reward
        steps = 0
        total_reward = 0

        obs, info = self.env.reset()

        fig, ax = plt.subplots(1, 1)
        utt, partial, overview = trans_obs(obs, self.utt_obs_key)
        context = prep_ax(ax, utt, partial, overview, step=0, reward=0, total_reward=0, idx=info["task/idx"])

        def on_press(event):
            if self.fn_key_to_action is not None:
                action = self.fn_key_to_action(event)
            else:
                action = "auto"
            obs, reward, done, _, info = self.env.step(action)
            global context, steps, total_reward
            steps += 1
            total_reward += reward
            utt, partial, overview = trans_obs(obs, self.utt_obs_key)
            context = prep_ax(ax, utt, partial, overview, step=steps, reward=reward, total_reward=total_reward,
                              context=context,
                              done=info["episode/outcome/success"] if done else None, idx=info["task/idx"])
            if done:
                self.env.reset()
                steps = 0
                total_reward = 0
            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_press)
        plt.show()
