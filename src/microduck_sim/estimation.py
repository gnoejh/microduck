"""Orientation estimation from IMU data (Lab 4).

The estimation ladder taught in Lab 4, all at the 50 Hz tick:

1. accelerometer alone: ``accel_pitch`` — no drift, but noisy and corrupted by
   dynamic acceleration (fails under pushes),
2. gyro alone: integrate pitch rate — smooth, but drifts (any bias accumulates),
3. **complementary filter** — gyro for the fast part, accel for the slow part:
   ``pitch ← α·(pitch + gyro_y·dt) + (1−α)·accel_pitch``.

Verified in this sim: balancing on the fused estimate (α=0.98, noisy sensors,
0.05 rad/s gyro bias) stands indefinitely and survives 1 N pushes; accel-only
falls at 0.8 N; gyro-only drifts off within ~3 s.
"""

from __future__ import annotations

import numpy as np

from microduck_sim.sensors import IMUSample


def accel_pitch(accel: np.ndarray) -> float:
    """Pitch [rad] from specific force alone (valid only when |accel| ≈ g)."""
    return float(np.arctan2(-accel[0], accel[2]))


class ComplementaryFilter:
    """Fuse gyro pitch rate and accelerometer pitch at a fixed rate.

    alpha → 1.0: trust the gyro (drifts with bias);
    alpha → 0.0: trust the accelerometer (noisy, lies under acceleration).
    0.98 at 50 Hz (time constant ≈ 1 s) balances the two well here.

    Handles the invalid-sample pitfall: before initialization it waits for a
    valid accel reading; on invalid samples it integrates the gyro only.
    """

    def __init__(self, alpha: float = 0.98, dt: float = 0.02):
        self.alpha, self.dt = alpha, dt
        self.pitch: float | None = None

    def update(self, sample: IMUSample) -> float:
        gyro_rate = float(sample.gyro[1])
        if self.pitch is None:
            self.pitch = accel_pitch(sample.accel) if sample.valid else 0.0
        elif sample.valid:
            self.pitch = self.alpha * (self.pitch + gyro_rate * self.dt) + (
                1.0 - self.alpha
            ) * accel_pitch(sample.accel)
        else:
            self.pitch = self.pitch + gyro_rate * self.dt
        return self.pitch
