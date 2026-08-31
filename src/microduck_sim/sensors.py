"""Sensor access and a realistic IMU noise model (Lab 4).

The MJCF defines clean sensors at the trunk ``imu`` site (`imu_ang_vel` gyro,
`imu_accel` accelerometer, plus a ground-truth `orientation` framequat). MuJoCo
does not inject noise itself — `NoisyIMU` adds Gaussian noise and gyro bias in
software, which lets labs dial realism up and down.

Pitfall baked into the API on purpose: right after a reset the accelerometer
reads ≈ 0 — sensordata isn't populated until the first step, AND the duck
spawns a hair above the floor, so its first ~30 ms are free fall, where a
*perfect* accelerometer also reads zero specific force. Either way the reading
contains no gravity direction. `NoisyIMU.read()` therefore returns a `valid`
flag (``|accel| > 1 m/s²``); estimators must handle invalid samples.
"""

from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np


def sensor_adr(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    """(address, dim) of a named sensor in ``data.sensordata``."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"sensor {name!r} not found")
    return int(model.sensor_adr[sid]), int(model.sensor_dim[sid])


def read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Copy of a named sensor's current reading."""
    adr, dim = sensor_adr(model, name)
    return data.sensordata[adr : adr + dim].copy()


class IMUSample(NamedTuple):
    gyro: np.ndarray  # rad/s, body frame (y = pitch rate)
    accel: np.ndarray  # m/s², specific force incl. gravity
    valid: bool  # False when the accel reading can't contain gravity


class NoisyIMU:
    """Trunk IMU with configurable Gaussian noise and constant gyro bias.

    Typical lab values: gyro_noise=0.02 rad/s, accel_noise=0.4 m/s²,
    gyro_bias=0.05 rad/s (a lot, on purpose — it makes drift visible fast).
    Deterministic per seed.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gyro_noise: float = 0.0,
        accel_noise: float = 0.0,
        gyro_bias: float = 0.0,
        seed: int = 0,
    ):
        self.model, self.data = model, data
        self.gyro_noise, self.accel_noise = gyro_noise, accel_noise
        self.gyro_bias = gyro_bias
        self.rng = np.random.default_rng(seed)
        self._gyro_adr = sensor_adr(model, "imu_ang_vel")[0]
        self._accel_adr = sensor_adr(model, "imu_accel")[0]

    def read(self) -> IMUSample:
        d = self.data.sensordata
        gyro = (
            d[self._gyro_adr : self._gyro_adr + 3]
            + self.rng.normal(0.0, self.gyro_noise, 3)
            + self.gyro_bias
        )
        accel = d[self._accel_adr : self._accel_adr + 3] + self.rng.normal(
            0.0, self.accel_noise, 3
        )
        valid = bool(np.linalg.norm(accel) > 1.0)
        return IMUSample(gyro=gyro, accel=accel, valid=valid)
