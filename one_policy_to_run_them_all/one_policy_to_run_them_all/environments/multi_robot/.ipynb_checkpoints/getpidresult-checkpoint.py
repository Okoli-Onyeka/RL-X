import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*to get variables from other wrappers is deprecated.*"
)

import os
import sys
import numpy as np
import torch
from absl import app
from ml_collections import config_dict, config_flags

os.chdir(
    "/workspace/RL-X/one_policy_to_run_them_all/experiments"
)

sys.path.insert(
    0,
    "/workspace/RL-X/one_policy_to_run_them_all/one_policy_to_run_them_all/"
)

from environments.multi_robot.create_env import create_env
from environments.multi_robot.default_config import get_config as get_env_config
from algorithms.uni_ppo.ppo.default_config import get_config as get_algo_config
from rl_x.runner.default_config import get_config as get_runner_config
from algorithms.uni_ppo.ppo.ppo import PPO


default_env_config = get_env_config("multi_robot")
env_config_flag = config_flags.DEFINE_config_dict(
    "environment",
    default_env_config,
)

default_algo_config = get_algo_config("uni_ppo.ppo")
algo_config_flag = config_flags.DEFINE_config_dict(
    "algorithm",
    default_algo_config,
)

default_runner_config = get_runner_config("test")
runner_config_flag = config_flags.DEFINE_config_dict(
    "runner",
    default_runner_config,
)


class URMATestRunner:
    def __init__(self):
        self.config = None
        self.env = None
        self.vectorenv = None
        self.robot_env = None
        self.urma = None

    def create_config(self):
        self.config = config_dict.ConfigDict()
        self.config.environment = env_config_flag.value
        self.config.algorithm = algo_config_flag.value
        self.config.runner = runner_config_flag.value

        self.config.environment.mode = "test"

    def create_environment(self):
        self.env = create_env(config=self.config)

        # Same unwrapping pattern as your residual-RL test script.
        self.vectorenv = self.env.train_env.env
        active_id = self.vectorenv.active_env_id
        self.robot_env = self.vectorenv.envs[active_id].env

        print(
            f"Loaded robot environment: "
            f"{getattr(self.robot_env, 'LONG_NAME', type(self.robot_env).__name__)}"
        )

    def load_urma(self):
        run_path = (
            f"runs/{self.config.runner.project_name}/"
            f"{self.config.runner.exp_name}/"
            f"{self.config.runner.run_name}"
        )

        explicitly_set_algorithm_params = [
            param_name
            for param_name in algo_config_flag._flagvalues
            if param_name.startswith("algorithm.")
        ]

        self.urma = PPO.load(
            self.config,
            self.env,
            run_path,
            None,
            explicitly_set_algorithm_params,
        )

        print(f"Loaded URMA checkpoint from: {run_path}")

    def run_one_episode(self):
        observation, info = self.vectorenv.reset()

        done = False
        step_count = 0
        terminated_flag = False
        truncated_flag = False

        print("\nRunning one test episode...\n")

        try:
            while not done:
                with torch.no_grad():
                    urma_action = self.urma.get_interference_action_multi_render(
                        observation,
                        self.vectorenv.active_env_id,
                    )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.vectorenv.step(urma_action)

                observation = next_observation
                step_count += 1

                terminated_flag = bool(np.asarray(terminated).any())
                truncated_flag = bool(np.asarray(truncated).any())
                done = terminated_flag or truncated_flag

            print(f"\nEpisode finished after {step_count} control steps.")

            if terminated_flag:
                print("Termination reason: environment termination.")
            elif truncated_flag:
                print("Termination reason: episode horizon reached.")

        finally:
            # Close only after natural episode completion so your environment's
            # existing metrics code can print/save at done.
            if self.env is not None:
                self.env.close()

    def main(self, argv):
        del argv

        self.create_config()
        self.create_environment()
        self.load_urma()
        self.run_one_episode()


runner = URMATestRunner()

if __name__ == "__main__":
    app.run(runner.main)
