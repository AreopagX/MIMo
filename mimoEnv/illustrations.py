""" Training script for the demonstration experiments.

This script allows simple training and testing of RL algorithms in the demo environments with a command line
interface. A selection of RL algorithms from the Stable Baselines3 library can be selected.
Interactive rendering is disabled during training to speed up computation, but enabled during testing, so the behaviour
of the model can be observed directly.

Trained models are saved into the "models/<scenario>" directory, i.e. if you train a reach model and name it
"my_model", it will be saved under "models/reach/my_model".

To train a given algorithm for some number of time steps::

    python illustrations.py --env=reach --train_for=200000 --test_for=1000 --algorithm=PPO --save_model=<model_suffix>

To review a trained model::

    python illustrations.py --env=reach --test_for=1000 --load_model=<your_model_suffix>

The available algorithms are ``PPO, SAC, TD3, DDPG, A2C``.
"""
import csv
import os
from pathlib import Path

import gymnasium as gym
import time
import argparse
import cv2
import numpy as np
import stable_baselines3.common.logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel


class InfoWriter:
    def __init__(self, path):
        self.file = Path(path)
        self.buffer = []

    def flush(self):
        if len(self.buffer) == 0:
            return
        write_header = not self.file.exists()
        with open(self.file, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=self.buffer[0].keys())
            if write_header:
                csv_writer.writeheader()
            csv_writer.writerows(self.buffer)
            self.buffer = []

    def append(self, data):
        self.buffer.append(data)
        if len(self.buffer) > 1000:
            self.flush()

    def __del__(self):
        self.flush()


"""class SummaryWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def __init__(self, global_step=0):
        self.global_step = global_step
        super().__init__()

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        data = self.locals["infos"][0]["logging"]
        data = {f"train.{key}": value for key, value in data.items()}
        self.tb_formatter.writer.add_scalars("info", data, self.global_step)
        self.global_step += 1
        return True"""


class InfoWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def __init__(self, file, global_step=0):
        self.writer = InfoWriter(file)
        self.global_step = global_step
        super().__init__()

    def _on_step(self) -> bool:
        data = self.locals["infos"][0]["logging"]
        data = {f"{key}": value for key, value in data.items()}
        data[f"global_step"] = self.global_step
        self.writer.append(data)
        self.global_step += 1
        return True


def test(env, save_dir, test_for=1000, model=None, render_video=False):
    """ Testing function to view the behaviour of a model.

    Args:
        env (MIMoEnv): The environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        save_dir (str): The directory in which any rendered videos will be saved.
        test_for (int): The number of timesteps the testing runs in total. This will be broken into multiple episodes
            if necessary.
        model:  The stable baselines model object. If ``None`` we take random actions instead. Default ``None``.
        render_video (bool): If ``True``, all episodes during testing will be recorded and saved as videos in
            `save_dir`.
    """
    obs, _ = env.reset()
    images = []
    im_counter = 0

    if model is None:
        print("No model loaded, taking random actions")

    #tb_formatter = next(formatter for formatter in model.logger.output_formats if isinstance(formatter,
    #                                                                                         TensorBoardOutputFormat))
    info_writer = InfoWriter(f"{save_dir}/info.csv")

    for idx in range(test_for):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, trunc, info = env.step(action)

        #for key, value in info["logging"].items():
        #tb_formatter.write(info["logging"], {}, idx)
        data = info["logging"]
        data = {f"{key}": value for key, value in data.items()}
        data["global_step"] = idx
        #tb_formatter.writer.add_scalars("info", data, idx)
        info_writer.append(data)

        if render_video:
            img = env.mujoco_renderer.render(render_mode="rgb_array")
            images.append(img)
        if done or trunc or idx == (test_for - 1):
            time.sleep(1)
            if render_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(os.path.join(save_dir, 'episode_{}.avi'.format(im_counter)), fourcc, 50,
                                        (500, 500))
                for img in images:
                    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.destroyAllWindows()
                video.release()
                images = []
                im_counter += 1
            obs, _ = env.reset()

    env.reset()


def main():
    """ CLI for the demonstration environments.

    Command line interface that can train and load models for the standup scenario. Possible parameters are:

    - ``--env``: The demonstration environment to use. Must be one of ``reach, standup, selfbody, catch``.
    - ``--train_for``: The number of time steps to train. No training takes place if this is 0. Default 0.
    - ``--test_for``: The number of time steps to test. Testing renders the environment to an interactive window, so
      the trained behaviour can be observed. Default 1000.
    - ``--save_every``: The number of time steps between model saves. This can be larger than the total training time,
      in which case we save once when training completes. Default 100000.
    - ``--algorithm``: The algorithm to train. This argument must be provided if you train. Must be one of
      ``PPO, SAC, TD3, DDPG, A2C, HER``.
    - ``--load_model``: The path to the model to load.
    - ``--save_model``: The directory name where the trained model will be saved. An input of "my_model", will lead to
        the model being saved under "models/<env>/my_model".
    - ``--use_muscles``: This flag switches between actuation models. By default, the spring-damper model is used. If
        this flag is set, the muscle model is used instead.
    - ``--render_video``: If this flag is set, each testing episode is recorded and saved as a video in the same
        directory as the models.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True,
                        choices=['reach', 'standup', 'selfbody', 'catch', 'fall', "pain"],
                        help='The demonstration environment to use. Must be one of "reach", "standup", "selfbody", '
                             '"catch"')
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str,
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='model', type=str,
                        help='Name of model to save')
    parser.add_argument('--render_video', action='store_true',
                        help='Renders a video for each episode during the test run.')
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')

    args = parser.parse_args()
    env_name = args.env
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for
    render = args.render_video
    use_muscle = args.use_muscle

    save_dir = os.path.join("models", env_name, save_model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    actuation_model = MuscleModel if use_muscle else SpringDamperModel

    env_names = {"reach": "MIMoReach-v0",
                 "standup": "MIMoStandup-v0",
                 "selfbody": "MIMoSelfBody-v0",
                 "catch": "MIMoCatch-v0",
                 "pain": "MIMoSelfBodyPain-v0",
                 "fall": "MIMoFall-v0"}

    env = gym.make(env_names[env_name], actuation_model=actuation_model)
    env.reset()

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    # load pretrained model or create new one
    if algorithm is None:
        model = None
    elif load_model:
        model = RL.load(load_model, env)
    else:
        model = RL("MultiInputPolicy", env,
                   tensorboard_log=save_dir,
                   verbose=1)

    # train model
    model.learn(
        total_timesteps=train_for, reset_num_timesteps=False,
        callback=[
            InfoWriterCallback(f"{save_dir}/train_info.csv", 0),
            CheckpointCallback(
                save_freq=save_every,
                save_path=save_dir,
                name_prefix=save_model,
            )
        ]
    )

    model.save(os.path.join(save_dir, "model_final"))

    # test(env, save_dir, model=model, test_for=test_for, render_video=render)


if __name__ == '__main__':
    main()
