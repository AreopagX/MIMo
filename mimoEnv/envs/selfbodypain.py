""" This module contains a simple experiment where MIMo is tasked with touching parts of his own body.

The scene is empty except for MIMo, who is sitting on the ground. The task is for MIMo to touch a randomized target
body part with his right arm. MIMo is fixed in the initial sitting position and can only move his right arm.
Sensory inputs consist of touch and proprioception. Proprioception uses the default settings, but touch excludes
several body parts and uses a lowered resolution to improve runtime.
The body part can be any of the geoms constituting MIMo.

MIMos initial position is constant in all episodes. The target body part is randomized. An episode is completed
successfully if MIMo touches the target body part with his right arm.

The reward structure consists of a large fixed reward for touching the right body part, a shaping reward for touching
another body part, depending on the distance between the contact and the target body part, and a penalty for each time
step.

The class with the environment is :class:`~mimoEnv.envs.selfbody.MIMoSelfBodyEnv` while the path to the scene XML is
defined in :data:`SELFBODY_XML`.
"""

import os
import numpy as np
from gymnasium import spaces

from mimoEnv.envs import MIMoSelfBodyEnv
from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS, SCENE_DIRECTORY
import mimoEnv.utils as env_utils
from mimoActuation.actuation import SpringDamperModel
from mimoEnv.envs.pain_touch import PainTouch

TOUCH_PARAMS = {
    "scales": {
        "left_foot": 0.05,
        "right_foot": 0.05,
        "left_lower_leg": 0.1,
        "right_lower_leg": 0.1,
        "left_upper_leg": 0.1,
        "right_upper_leg": 0.1,
        "hip": 0.1,
        "lower_body": 0.1,
        "upper_body": 0.1,
        "head": 0.1,
        "left_upper_arm": 0.01,
        "left_lower_arm": 0.01,
        "right_fingers": 0.01
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" List of possible target bodies.

:meta hide-value:
"""

PAIN_PARAMS = {
    "scales": {
        "left_foot": 0.05,
        "right_foot": 0.05,
        "left_lower_leg": 0.1,
        "right_lower_leg": 0.1,
        "left_upper_leg": 0.1,
        "right_upper_leg": 0.1,
        "hip": 0.1,
        "lower_body": 0.1,
        "upper_body": 0.1,
        "head": 0.1,
        "left_upper_arm": 0.01,
        "left_lower_arm": 0.01,
    },
    "touch_function": "normal",
    "response_function": "spread_linear",
    "extend_observation_space": False
}
""" List of possible target bodies.

:meta hide-value:
"""

SITTING_POSITION = {
    "robot:hip_lean1": np.array([0.039088]), "robot:hip_rot1": np.array([0.113112]),
    "robot:hip_bend1": np.array([0.5323]), "robot:hip_lean2": np.array([0]), "robot:hip_rot2": np.array([0]),
    "robot:hip_bend2": np.array([0.5323]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:left_shoulder_horizontal": np.array([0.683242]), "robot:left_shoulder_ad_ab": np.array([0.3747]),
    "robot:left_shoulder_rotation": np.array([-0.62714]), "robot:left_elbow": np.array([-0.756016]),
    "robot:left_hand1": np.array([0.28278]), "robot:left_hand2": np.array([0]), "robot:left_hand3": np.array([0]),
    "robot:left_fingers": np.array([-0.461583]),
    "robot:right_hip1": np.array([-1.51997]), "robot:right_hip2": np.array([-0.397578]),
    "robot:right_hip3": np.array([0.0976615]), "robot:right_knee": np.array([-1.85479]),
    "robot:right_foot1": np.array([-0.585865]), "robot:right_foot2": np.array([-0.358165]),
    "robot:right_foot3": np.array([0]), "robot:right_toes": np.array([0]),
    "robot:left_hip1": np.array([-1.23961]), "robot:left_hip2": np.array([-0.8901]),
    "robot:left_hip3": np.array([0.7156]), "robot:left_knee": np.array([-2.531]),
    "robot:left_foot1": np.array([-0.63562]), "robot:left_foot2": np.array([0.5411]),
    "robot:left_foot3": np.array([0.366514]), "robot:left_toes": np.array([0.24424]),
}
""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position (excluding the right arm) each step.

:meta hide-value:
"""


SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "selfbody_scene.xml")
""" Path to the scene for this experiment.

:meta hide-value:
"""


class MIMoSelfBodyPainEnv(MIMoSelfBodyEnv):
    """ MIMo learns about his own body.

    MIMo is tasked with touching a given part of his body using his right arm.
    Attributes and parameters are mostly identical to the base class, but there are two changes.
    The constructor takes two arguments less, `goals_in_observation` and `done_active`, which are both permanently
    set to ``True``.
    Finally, there are two extra attributes for handling the goal state. The :attr:`.goal` attribute stores the target
    geom in a one hot encoding, while :attr:`.target_geom` and :attr:`.target_body` store the geom and its associated
    body as an index. For more information on geoms and bodies please see the MuJoCo documentation.

    Attributes:
        target_geom (int): The body part MIMo should try to touch, as a MuJoCo geom.
        target_body (str): The name of the kinematic body that the target geom is a part of.
        init_sitting_qpos (numpy.ndarray): The initial position.
    """

    touch_observations = {}
    pain_observations = {}

    def __init__(self,
                 model_path=SELFBODY_XML,
                 initial_qpos=SITTING_POSITION,
                 frame_skip=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 pain_params=PAIN_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=SpringDamperModel,
                 goals_in_observation=True,
                 done_active=True,
                 randomize_qpos=False,
                 **kwargs,
                 ):

        self.pain_params = pain_params

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         randomize_qpos=randomize_qpos,
                         **kwargs)

    def _env_setup(self):
        super()._env_setup()
        if self.pain_params is not None:
            self.pain = PainTouch(self, self.pain_params)

    def get_pain_obs(self):
        obs = self.pain.get_touch_obs()
        return obs

    def _set_observation_space(self):
        super()._set_observation_space()
        obs = self._get_obs()
        if self.pain_params["extend_observation_space"]:
            self.observation_space["pain"] = spaces.Box(-np.inf, np.inf, shape=obs["pain"].shape, dtype=np.float32)

    def _get_obs(self):
        """ Adds the size of the ball to the observations.

        Returns:
            Dict: The altered observation dictionary.
        """
        obs = super()._get_obs()
        if self.pain_params["extend_observation_space"]:
            obs["pain"] = self.get_pain_obs()
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward each step.

        Three different rewards can be returned:

        - If we touched the target geom, the reward is 500.
        - If we touched a geom, but not the target, the reward is the negative of the distance between the touch
          contact and the target body.
        - Otherwise the reward is -1.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        active_geom_codes = list(self.touch.sensor_outputs.keys())

        fingers_touch_max = max(
            np.max(self.touch.sensor_outputs[active_geom_codes[-1]]),
            np.max(self.touch.sensor_outputs[active_geom_codes[-2]])
        )
        contact_with_fingers = (fingers_touch_max > 0)

        pain_penalty = self.get_pain_obs().max()

        # compute reward:
        if info["is_success"]:
            reward = 500
        elif contact_with_fingers:
            target_body_pos = self.data.body(self.target_body).xpos
            fingers_pos = self.data.body("right_fingers").xpos
            distance = np.linalg.norm(fingers_pos - target_body_pos)
            reward = - distance
        else:
            reward = -1
        self.logging_values["reward.without_pain"] = reward
        self.logging_values["reward.pain_penalty"] = pain_penalty
        reward -= pain_penalty
        self.logging_values["reward"] = reward

        return reward

    def reset_model(self, *args, **kwargs):
        """ Reset to the initial sitting position.

        Returns:
            Dict: Observations after reset.
        """
        self.touch_observations.clear()
        self.pain_observations.clear()
        return super().reset_model(*args, **kwargs)