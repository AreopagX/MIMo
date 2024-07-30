import mujoco
import numpy as np
import mimoEnv.utils as env_utils
from mimoTouch.touch import TrimeshTouch


class PainTouch(TrimeshTouch):
    PAIN_THRESHOLD = 200.0
    PAIN_SLOPE = 0.1
    PAIN_OLD_VALUES_FACTOR = 0.0
    PAIN_NEW_VALUES_FACTOR = 1.0

    VALID_TOUCH_TYPES = {"normal": 1}
    softness_by_name = {
    }
    softness_by_geom_id = {}

    def __init__(self, *args, **kwargs):
        self.old_sensor_obs = {}
        super().__init__(*args, **kwargs)
        self.old_sensor_obs = self.get_empty_sensor_dict(self.touch_size)

        """text_start = self.m_model.text_adr
        text_end = self.m_model.text_size
        self.m_model.text_data[text_start[0]:text_end[0]]"""

        #env.model.name_textadr
        #self.m_model.numeric(0).name

        for name in self.softness_by_name.keys():
            ids = env_utils.get_geoms_for_body(self.m_model, body_id=self.m_data.body(name).id)
            for id in ids:
                self.softness_by_geom_id[id] = self.softness_by_name[name]

    def get_raw_force(self, contact_id, body_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

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
            forces *= -self.PAIN_SLOPE  # Convention is that normal points away from geom1
            forces *= self.softness_by_geom_id.get(contact.geom2, 1.0)  # multiply with a softness-based factor
        elif contact.geom2 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            forces *= self.PAIN_SLOPE
            forces *= self.softness_by_geom_id.get(contact.geom1, 1.0)  # multiply with a softness-based factor
        else:
            RuntimeError("Mismatch between contact and body")
        return forces[:3]

    def normal(self, contact_id, body_id):
        normal_forces = self.normal_force(contact_id, body_id)
        normal = np.sqrt(np.power(normal_forces, 2).sum()).reshape((1,))
        return normal

    def get_touch_obs(self):
        if len(self.old_sensor_obs) == 0:
            return super().get_touch_obs()
        super().get_touch_obs()
        for key in self.sensor_outputs.keys():
            #self.sensor_outputs[key] = \
            #    np.maximum(self.sensor_outputs[key] - self.PAIN_THRESHOLD, 0.0) * np.ones_like(self.sensor_outputs[key])

            #self.old_sensor_obs[key] = (1 - self.alpha) * self.old_sensor_obs[key] + self.alpha * self.sensor_outputs[key]
            new_value = np.maximum(
                self.PAIN_OLD_VALUES_FACTOR * self.old_sensor_obs[key],
                self.PAIN_NEW_VALUES_FACTOR * self.sensor_outputs[key]
            )
            self.old_sensor_obs[key] = new_value
            self.sensor_outputs[key] = self.old_sensor_obs[key]
        sensor_obs = self.flatten_sensor_dict(self.old_sensor_obs)
        return sensor_obs
