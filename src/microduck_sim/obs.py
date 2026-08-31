"""The 61-dim observation contract shared by all Microduck policies (Lab 5).

Layout (verified against the official browser simulator's glue code and the
ONNX metadata — "new-cmd-obs" flavor):

    [ base_ang_vel(3),        # imu_ang_vel gyro, rad/s, trunk frame
      projected_gravity(3),   # world -z rotated into the trunk frame
      joint_pos(14),          # qpos - DEFAULT_POSE, actuator order
      joint_vel(14),          # qvel, actuator order
      last_action(14),        # previous raw policy output
      command(13) ]           # [vx, vy, wz, neck_pitch, head_pitch, head_yaw,
                              #  head_roll, 0...0] (7 used + zero padding);
                              # sit-stand uses cmd[0] as posture flag,
                              # one-shots put [cos φ, sin φ] in cmd[0:2]

Normalization is baked into the ONNX files — feed raw physical values.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Actuator order — identical in every model variant and every policy.
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
NUM_JOINTS = 14
OBS_SIZE = 61
CMD_SIZE = 13
ACTION_SCALE = 1.0

# STAND keyframe ctrl from scene_walk.xml — the pose actions are relative to.
DEFAULT_POSE = np.array([
    0.0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
    0.3490658503988659, 0.3490658503988659, 0.0, 0.0,
    0.0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
])

# Velocity command limits (from the robot runtime's keyboard mapping).
VEL_FWD, VEL_BACK, VEL_ANG = 0.25, -0.2, 1.0


class ObsBuilder:
    """Builds the 61-dim observation vector from live MjData."""

    def __init__(self, model: mujoco.MjModel):
        self.qpos_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES]
        )
        self.dof_adr = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES]
        )
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(model.sensor_adr[gyro_id])
        self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    def projected_gravity(self, data: mujoco.MjData) -> np.ndarray:
        """World -z expressed in the trunk frame ([0,0,-1] when upright)."""
        quat = data.xquat[self.trunk_id]  # wxyz
        gravity = np.zeros(3)
        mujoco.mju_rotVecQuat(gravity, np.array([0.0, 0.0, -1.0]), _conj(quat))
        return gravity

    def build(
        self,
        data: mujoco.MjData,
        last_action: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        obs = np.empty(OBS_SIZE, dtype=np.float32)
        obs[0:3] = data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        obs[3:6] = self.projected_gravity(data)
        obs[6:20] = data.qpos[self.qpos_adr] - DEFAULT_POSE
        obs[20:34] = data.qvel[self.dof_adr]
        obs[34:48] = last_action
        obs[48:61] = command
        return obs


def _conj(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([w, -x, -y, -z])
