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
from functools import partial
from pathlib import Path

import gymnasium as gym
import time
import argparse
import cv2
import numpy as np
import stable_baselines3.common.logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
import types


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
        self.buffer.append(data.copy())
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
        data = {f"train.{key}": value for key, value in data.items()}
        data[f"train.global_step"] = self.global_step
        self.writer.append(data)
        self.global_step += 1
        return True

def overwrite_sample_goal(self, target_geom_idx = None):
    """Samples a new goal and returns it.

    The goal consists of a target geom that we try to touch, returned as a one-hot encoding.
    We also populate :attr:`.target_geom` and :attr:`.target_body`. which are used by other functions.

    Returns:
        numpy.ndarray: The target geom in a one hot encoding.
    """
    # randomly select geom as target (except for 2 latest geoms that correspond to fingers)
    active_geom_codes = list(self.touch.sensor_outputs.keys())
    if target_geom_idx is None:
        target_geom_idx = np.random.randint(len(active_geom_codes) - 2)
    self.target_geom = active_geom_codes[int(target_geom_idx)]
    # We want the output of the desired goal as a one hot encoding,
    # rather than the raw index
    target_geom_onehot = np.zeros(37)  # 36 geoms in MIMo
    if isinstance(self.target_geom, int):
        target_geom_onehot[self.target_geom] = 1

    self.target_body = self.model.body(self.model.geom(self.target_geom).bodyid).name
    self.logging_values["target_name"] = self.target_body
    self.logging_values["target_index"] = self.target_geom
    return target_geom_onehot


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
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')
    parser.add_argument('--algorithm', default=None, type=str,
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--render_video', action='store_true',
                        help='Renders a video for each episode during the test run.')
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')
    parser.add_argument('--info', default="info", type=str,
                        help='Filename used to save the evaluation info.')
    parser.add_argument("--deterministic", action="store_true",
                        help="Make the policy prediction deterministic.")

    args = parser.parse_args()
    env_name = args.env
    algorithm = args.algorithm
    load_model = args.load_model
    test_for = args.test_for
    render = args.render_video
    use_muscle = args.use_muscle
    info_filename = args.info
    deterministic = args.deterministic

    save_dir = os.path.join("models", env_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    actuation_model = MuscleModel if use_muscle else SpringDamperModel

    env_names = {"selfbody": "MIMoSelfBody-v0",
                 "pain": "MIMoSelfBodyPain-v0"}

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

    # load pretrained model
    model = RL.load(load_model, env)

    right_arm_joints = [
        "robot:right_shoulder_horizontal", "robot:right_shoulder_ad_ab",
        "robot:right_shoulder_rotation", "robot:right_elbow",
        "robot:right_hand1", "robot:right_hand2", "robot:right_hand3",
        "robot:right_fingers"
    ]
    right_arm_indices = [env.data.joint(joint).id + 6 for joint in right_arm_joints]
    init_qpos = np.array([env.data.qpos[id] for id in right_arm_indices])

    info_writer = InfoWriter(f"{save_dir}/{info_filename}.csv")

    for target_geom_idx in range(11):
        print(f"target_geom_idx: {target_geom_idx}")

        images = []
        im_counter = 0

        func = types.MethodType(partial(overwrite_sample_goal, target_geom_idx=target_geom_idx), env)
        env.unwrapped.sample_goal = func

        for initdx in range(20):
            np.random.seed(initdx)
            mod_qpos = init_qpos + 0.1 * np.random.randn(*init_qpos.shape)


            qpos = env.init_sitting_qpos
            qpos[right_arm_indices] = mod_qpos
            env.reset()
            obs = env.reset_model(qpos)

            for idx in range(500):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, done, trunc, info = env.step(action)

                data = info["logging"]
                data["success"] = int(info["is_success"])
                data["global_step"] = idx
                data["seed"] = initdx
                info_writer.append(data)

                if deterministic and (done or trunc):
                    break

                if render:
                    img = env.mujoco_renderer.render(render_mode="rgb_array")
                    images.append(img)
                if done or trunc or idx == (test_for - 1):
                    time.sleep(1)
                    if render:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video = cv2.VideoWriter(
                            os.path.join(save_dir, f"episode_{info_filename}_{target_geom_idx}_{im_counter}.avi"),
                            fourcc, 50, (500, 500)
                        )
                        for img in images:
                            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.destroyAllWindows()
                        video.release()
                        images = []
                        im_counter += 1

    info_writer.flush()

if __name__ == '__main__':
    main()
