import numpy as np

class ResidualTrackingReward:
    def __init__(self, env, tracking_temperature = 0.01, residual_action_coeff=0.001, residual_rate_coeff=0.001):
        self.env = env
        self.tracking_temperature = tracking_temperature
        self.residual_action_coeff = residual_action_coeff
        self.residual_rate_coeff = residual_rate_coeff

    def init(self):
        pass

    def setup(self):
        self.previous_residual_action = np.zeros(
            self.env.model.nu,
            dtype=np.float32
        )

    def step(self, residual_action):
        self.previous_residual_action = residual_action.copy()

    def reward_and_info(self, info, done):
        #actual robpot velocity in the robots local frame
        local_linear_velocity = self.env.orientation_quat_inv.apply(
            self.env.data.qvel[:3]
        )

        actual_velocity = np.array([
            local_linear_velocity[0],
            local_linear_velocity[1],
            self.env.data.qvel[5]
        ])

        target_velocity = np.array([
            self.env.goal_x_velocity,
            self.env.goal_y_velocity,
            self.env.goal_yaw_velocity
        ])

        #command-tracking reward
        tracking_error = np.sum(
            np.square(target_velocity - actual_velocity)
        )

        tracking_reward = np.exp(
            -tracking_error/self.tracking_temperature
        )

        #prefer small residual corrections
        residual_action = self.env.applied_residual_action

        residual_magnitude_penalty = (
            self.residual_action_coeff * np.sum(np.square(residual_action))
        )

        #prefer smooth residual corrections
        residual_change = (
            residual_action - self.previous_residual_action
        )

        residual_rate_penalty = (
            self.residual_rate_coeff * np.sum(np.square(residual_change))
        )

        reward = (tracking_reward
                  -residual_magnitude_penalty
                  -residual_rate_penalty
        )

        info["residual/tracking_reward"] = tracking_reward
        info["residual/tracking_error"] = tracking_error
        info["residual/action_penalty"] = residual_magnitude_penalty
        info["residual/action_rate_penalty"] = residual_rate_penalty

        return reward, info