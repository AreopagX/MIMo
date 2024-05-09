""" This module contains a simple reaching experiment in which MIMo tries to catch a falling ball.

The scene consists of MIMo with his right arm outstretched and his palm open. A ball is located just above MIMos palm.
The task is for him to catch the falling ball.
MIMo is fixed in position and can only move his right hand.
Sensory input consists of the full proprioceptive inputs and touch input.

An episode is completed successfully if MIMo holds onto the ball continuously for 1 second. An episode fails when the
ball drops some distance below MIMos hand or is bounced into the distance.

There is a small negative reward for each step without touching the ball, a larger positive reward for each step in
contact with the ball and then a large fixed reward on success.
"""
import os
import random

import mujoco
import numpy as np

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_TOUCH_PARAMS
from mimoActuation.muscle import MuscleModel
import mimoEnv.utils as env_utils

FALL_XML = os.path.join(SCENE_DIRECTORY, "fall_scene.xml")
""" Path to the scene.

:meta hide-value:
"""

FALL_INITIAL_QPOS = np.array([0.19,0.0013,0.051,0.65,-0.0022,-0.76,-0.0019,-0.0016,0.00023,0.12,-0.0045,-0.0049,0.13,0.097,-0.15,-0.091,-0.018,0.026,-0.011,-0.044,0.028,-0.0022,-0.36,0.065,-0.11,-0.22,-0.35,0.071,-0.065,-0.22,-0.39,0.058,-0.2,-0.3,-0.43,0.023,-0.033,-0.16,-0.083,0.003,0.0029,-0.47,-0.34,0.0012,-0.0022,-0.014,-0.084,0.0024,0.0022,-0.47,-0.34,0.0009,-0.0017,-0.014,0.0,0.0,0.5,0.0,0.0,0.0,0.0])


class MIMoFallEnv(MIMoEnv):
    """ MIMo tries to catch a falling ball.

    MIMo is tasked with catching a falling ball and holding onto it for one second. MIMo's head and eyes automatically
    track the ball. The position of the ball is slightly randomized each episode.
    The constructor takes three additional arguments over the base environment.

    Args:
        action_penalty (bool): If ``True``, an action penalty based on the cost function of the actuation model is
            applied to the reward. Default ``True``.
        jitter (bool): If ``True``, the input actions are multiplied with a perturbation array which is randomized
            every 10-50 time steps. Default ``False``.
        position_inaccurate (bool): If ``True``, the position tracked by the head is offset by a small random distance
            from the true position of the ball. Default ``False``.

    Attributes:
        action_penalty (bool): If ``True``, an action penalty based on the cost function of the actuation model is
            applied to the reward. Default ``True``.
        jitter (bool): If ``True``, the input actions are multiplied with a perturbation array which is randomized
            every 10-50 time steps. Default ``False``.
        use_position_inaccuracy (bool): If ``True``, the position tracked by the head is offset by a small random
            distance from the true position of the ball. Default ``False``.
        position_limits (np.ndarray): Maximum distances away from the default ball position for the randomization.
        position_inaccuracy_limits (np.ndarray): Maximum distances for the head tracking offset.
        position_offset (np.ndarray): The actual inaccuracy of the head tracking. This is randomized each episode.
        size_limits (Tuple[float, float]): Minimum and maximum size of the ball.
        ball_size (float): Current ball size. Changes each episode.
        mass_limits (Tuple[float, float]): Minimum and maximum mass of the ball.
        ball_mass (float): Current ball mass. Changes each episode.
        jitter_array (np.ndarray): Control inputs are multiplied by this array before being passed to MuJoCo. This is
            randomized every so often.
        jitter_period (int): The number of steps the current jitter array is used for before being randomized again.
        steps_in_contact_for_success (int): For how many steps MIMo must hold onto the ball.
        in_contact_past (List[bool]): A list storing which past steps we were in contact for. This list works by
            modulo, i.e. to determine if MIMo held the ball on step `i`, do
            ``in_contact_past[i % steps_in_contact_for_success]``.
    """
    def __init__(self,
                 model_path=FALL_XML,
                 initial_qpos=None,
                 frame_skip=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=MuscleModel,
                 goals_in_observation=False,
                 done_active=True,
                 action_penalty=True,
                 **kwargs):

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
                         default_camera_config=None,
                         **kwargs)
        
        self.init_qpos = FALL_INITIAL_QPOS

        self.steps_in_contact_for_success = 100
        self.in_contact_past = [False for _ in range(self.steps_in_contact_for_success)]
        self.steps = 0
        self.target_id = self.model.body("target").id
        self.hand_id = self.model.body("right_hand").id
        self.head_id = self.model.body("head").id
        self.target_geoms = env_utils.get_geoms_for_body(self.model, self.target_id)
        self.own_geoms = []
        for body_name in touch_params["scales"]:
            body_id = self.model.body(body_name).id
            self.own_geoms.extend(env_utils.get_geoms_for_body(self.model, body_id))
        self.action_penalty = action_penalty
        print("Action penalty: ", self.action_penalty)

        # Info required to randomize ball position
        target_joint = "target_joint"
        self.target_joint_id = self.model.joint(target_joint).id
        self.target_joint_qpos = env_utils.get_joint_qpos_addr(self.model, self.target_joint_id)
        self.target_joint_qvel = env_utils.get_joint_qvel_addr(self.model, self.target_joint_id)


    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        MIMo is rewarded for each time step in contact with the target. Completing an episode successfully awards +100,
        while failing leads to a -100 penalty. Additionally, there is an action penalty based on the cost function of
        the actuation model.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        reward = 0
        return reward

    def _get_obs(self):
        """ Adds the size of the ball to the observations.

        Returns:
            Dict: The altered observation dictionary.
        """
        obs = super()._get_obs()
        return obs

    def is_success(self, achieved_goal, desired_goal):
        """ Returns true if MIMo touches the object continuously for 1 second.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``True`` if MIMo has been touching the ball for the last second, ``False`` otherwise.
        """
        return False

    def is_failure(self, achieved_goal, desired_goal):
        """ Returns ``True`` if the ball drops below MIMo's hand.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``True`` if the ball drops below MIMo's hand, ``False`` otherwise.
        """
        return False

    def is_truncated(self):
        """Dummy function.

        Returns:
            bool: Always returns ``False``.
        """
        return False

    def sample_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def get_achieved_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def reset_model(self):
        """ Resets the simulation.

        We reset the simulation and then slightly move both MIMos arm and the ball randomly. The randomization is
        limited such that MIMo can always reach the ball.

        Returns:
            bool: Always returns ``True``.
        """
        self.set_state(self.init_qpos, self.init_qvel)

        self._step_callback()
        self.steps = 0
    
        return self._get_obs()

    def _step_callback(self):
        """ Checks if MIMo is touching the ball and performs head tracking.
        """
        self.steps += 1