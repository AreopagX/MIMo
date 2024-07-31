import mujoco
import numpy as np
import mimoEnv.utils as env_utils
from mimoTouch.touch import TrimeshTouch


class PainTouch(TrimeshTouch):
    """ A pain sensor based on the :class:`mimoTouch.touch.TrimeshTouch`.

    See :class:`mimoTouch.touch.TrimeshTouch` for an overview how the touch sensor works.

    The pain sensations are computed using the magnitudes of normal forces that are sensed at parts of MIMo's body.
    The magnitudes are first scaled by :attr:`pain_slope` and then optionally by a material softness score as defined in
    :attr:`softness_by_name` (see :func:`get_raw_force`).

    The :attr:`pain_threshold` is then subtracted from the absolute magnitude, and only values greater than zero are
    kept. The resulting values are fed through a decaying mechanism, which is calculated as the maximum of the scaled
    old and new sensations. The parameters :attr:`pain_decay_old_values` and :attr:`pain_decay_new_values` are used for
    that matter. (See :func:`normal` and :func:`get_touch_obs`)

    The following attributes are provided in addition to those of :class:`mimoTouch.touch.TrimeshTouch`.

    Attributes:
        pain_threshold (float): Only touch sensations greater than this threshold will create painful sensations.
        pain_upper_bound (float): Optionally limits the pain sensations to this upper bound.
        pain_slope (float): A number which scales the pain sensations.
        pain_decay_old_values (float): A factor by which the old sensations are weighted in the decay.
        pain_decay_new_values (float): A factor by which the new sensations are weighted in the decay.
        softness_by_name (Dict[str, float]): A dictionary that maps softness coefficients to geometry names.
            These coefficients influence the calculation of the forces :func:`.get_raw_force`.
            If no value for a body is provided a scale of 1.0 is used.
        sensitivity_by_name (Dict[str, float]): A dictionary that maps sensitivity coefficients to body names.
            These coefficients influence the calculation of the forces :func:`.get_raw_force`.
            If no value for a body is provided a scale of 1.0 is used.
        """

    VALID_TOUCH_TYPES = {"normal": 1}

    def __init__(
            self,
            *args,
            pain_threshold=20,
            pain_upper_bound=float("inf"),
            pain_slope=0.1,
            pain_decay_old_values=0.0,
            pain_decay_new_values=1.0,
            softness_by_name={},
            sensitivity_by_name={},
            **kwargs):
        self.old_sensor_obs = None

        self.pain_threshold = pain_threshold
        self.pain_upper_bound = pain_upper_bound
        self.pain_slope = pain_slope
        self.pain_decay_old_values = pain_decay_old_values
        self.pain_decay_new_values = pain_decay_new_values
        self.softness_by_name = softness_by_name
        self.softness_by_geom_id = {}
        self.sensitivity_by_name = sensitivity_by_name
        self.sensitivity_by_geom_id = {}

        super().__init__(*args, **kwargs)

        for name in self.softness_by_name.keys():
            ids = env_utils.get_geoms_for_body(self.m_model, body_id=self.m_data.body(name).id)
            for id in ids:
                self.softness_by_geom_id[id] = self.softness_by_name[name]

        for name in self.sensitivity_by_name.keys():
            ids = env_utils.get_geoms_for_body(self.m_model, body_id=self.m_data.body(name).id)
            for id in ids:
                self.sensitivity_by_geom_id[id] = self.sensitivity_by_name[name]

    def get_raw_force(self, contact_id, body_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

        The forces are scaled using the :attr:`PAIN_SLOPE` attribute. Additionally, different material softnesses can
        be emulated by providing scaling factors in :attr:`softness_by_name` during initialization.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The relevant body in the contact. One of the geoms belonging to this body must be involved
                in the contact!

        Returns:
            np.ndarray: An array with shape (3,) with the normal force and the two tangential friction forces.
        """
        forces = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.m_model, self.m_data, contact_id, forces)
        contact = self.m_data.contact[contact_id]
        if contact.geom1 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            forces *= -self.pain_slope  # convention is that normal points away from geom1
            forces *= self.softness_by_geom_id.get(contact.geom2, 1.0)  # multiply with a softness-based factor
            forces *= self.sensitivity_by_geom_id.get(contact.geom1, 1.0)  # account for body sensitivity
        elif contact.geom2 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            forces *= self.pain_slope
            forces *= self.softness_by_geom_id.get(contact.geom1, 1.0)  # multiply with a softness-based factor
            forces *= self.sensitivity_by_geom_id.get(contact.geom2, 1.0)  # account for body sensitivity
        else:
            RuntimeError("Mismatch between contact and body")
        return forces[:3]

    def normal(self, contact_id, body_id):
        """ Pain function. Returns pain sensation of the body.

        Pain sensations are the absolute magnitude of the normal forces on the contact body.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array of shape (1,) with the normal force.
        """
        normal_forces = self.normal_force(contact_id, body_id)
        normal = np.sqrt(np.power(normal_forces, 2).sum()).reshape((1,))
        return normal

    def get_touch_obs(self):
        """ Produces the current pain sensor outputs.

        Does the full contact getting-processing process, such that we get the forces, as determined by
        :attr:`.touch_type` and :attr:`.response_type`, for each sensor. :attr:`.touch_function` is called to compute
        the raw output force, which is then distributed over the sensors using :attr:`.response_function`.

        The indices of the output dictionary :attr:`~mimoTouch.touch.TrimeshTouch.sensor_outputs` and the sensor
        dictionary :attr:`.sensor_positions` are aligned, such that the ith sensor on `body` has position
        ``.sensor_positions[body][i]`` and output in ``.sensor_outputs[body][i]``.

        This method first thresholds the forces using :attr:`.PAIN_THRESHOLD`, then simulates a decaying pain using old
        and current pain sensations.

        Returns:
            np.ndarray: An array containing all the pain sensations.
        """
        # get touch sensations
        touch_obs = super().get_touch_obs()

        # needed during initialization
        if self.old_sensor_obs is None:
            self.old_sensor_obs = np.zeros_like(touch_obs)

        # apply threshold and decay
        pain_obs = np.maximum(touch_obs - self.pain_threshold, 0.0)
        pain_obs = np.minimum(pain_obs, self.pain_upper_bound)
        pain_obs = np.maximum(
            self.pain_decay_old_values * self.old_sensor_obs,
            self.pain_decay_new_values * pain_obs
        )

        # store pain observations for the next step
        self.old_sensor_obs = pain_obs
        return pain_obs
