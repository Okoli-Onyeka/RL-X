import numpy as np


class VelocityPID:

    def __init__(self, kp, ki, kd, dt):
        self.kp = np.asarray(kp, dtype=np.float32)
        self.ki = np.asarray(ki, dtype=np.float32)
        self.kd = np.asarray(kd, dtype=np.float32)

        self.dt = dt

        self.integral = np.zeros(3, dtype=np.float32)
        self.previous_error = np.zeros(3, dtype=np.float32)

        self.integral_limit = 1.0
        self.first_step = True

    def reset(self):
        self.integral[:] = 0.0
        self.previous_error[:] = 0.0
        self.first_step = True

    def update(self, desired_velocity, actual_velocity):

        desired_velocity = np.asarray(
            desired_velocity,
            dtype=np.float32
        )
        actual_velocity = np.asarray(
            actual_velocity,
            dtype=np.float32
        )

        error = desired_velocity - actual_velocity

        self.integral += error * self.dt

        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )

        if self.first_step:
            derivative = np.zeros(3, dtype=np.float32)
            self.first_step = False
        else:
            derivative = (
                error - self.previous_error
            ) / self.dt

        correction = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = error.copy()

        return correction