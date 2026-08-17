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


# ============================================================
# COMMAND BENCHMARK
# ============================================================

MAGNITUDES = [0.3, 0.6, 0.9]


def create_test_commands():
    commands = []

    # --------------------------------------------------------
    # Excite each axis independently:
    # x, y, yaw
    # at ±0.3, ±0.6, ±0.9
    # --------------------------------------------------------
    for axis in range(3):
        for magnitude in MAGNITUDES:
            for sign in [1, -1]:

                command = [0.0, 0.0, 0.0]
                command[axis] = sign * magnitude

                commands.append(command)

    # --------------------------------------------------------
    # Stationary test
    # --------------------------------------------------------
    commands.append([
        0.0,
        0.0,
        0.0,
    ])

    # --------------------------------------------------------
    # Combined commands
    # --------------------------------------------------------
    commands.extend([
        [0.6,  0.6,  0.0],
        [0.6, -0.6,  0.0],
        [0.6,  0.0,  0.6],
        [0.6,  0.0, -0.6],
    ])

    return commands


COMMANDS = create_test_commands()


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# TEST RUNNER
# ============================================================

class URMATestRunner:

    def __init__(self):
        self.config = None
        self.env = None
        self.vectorenv = None
        self.robot_env = None
        self.urma = None

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    def create_config(self):

        self.config = config_dict.ConfigDict()

        self.config.environment = env_config_flag.value
        self.config.algorithm = algo_config_flag.value
        self.config.runner = runner_config_flag.value

        self.config.environment.mode = "test"

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    def create_environment(self):

        self.env = create_env(
            config=self.config
        )

        self.vectorenv = self.env.train_env.env

        active_id = self.vectorenv.active_env_id

        self.robot_env = (
            self.vectorenv
            .envs[active_id]
            .env
        )

        print(
            f"Loaded robot environment: "
            f"{getattr(self.robot_env, 'LONG_NAME', type(self.robot_env).__name__)}")

    # --------------------------------------------------------
    # URMA
    # --------------------------------------------------------

    def load_urma(self):

        run_path = (
            f"runs/"
            f"{self.config.runner.project_name}/"
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

        print(
            f"Loaded URMA checkpoint from: "
            f"{run_path}"
        )

    # --------------------------------------------------------
    # ONE COMMAND / ONE EPISODE
    # --------------------------------------------------------

    def run_one_episode(self, command):

        command = np.asarray(
            command,
            dtype=np.float32,
        )

        # Set the desired PID target BEFORE reset.
        self.robot_env.set_desired_command(
            command
        )

        observation, info = self.vectorenv.reset()

        done = False
        step_count = 0

        terminated_flag = False
        truncated_flag = False

        command_string = (
            f"{command[0]:.1f}_"
            f"{command[1]:.1f}_"
            f"{command[2]:.1f}"
        )

        print("\n" + "=" * 70)
        print(
            f"PID TEST COMMAND: "
            f"[{command[0]:+.1f}, "
            f"{command[1]:+.1f}, "
            f"{command[2]:+.1f}]"
        )
        print(
            f"Results: "
            f"/workspace/pid_results/"
            f"{command_string}"
        )
        print("=" * 70 + "\n")

        while not done:

            with torch.no_grad():

                urma_action = (
                    self.urma
                    .get_interference_action_multi_render(
                        observation,
                        self.vectorenv.active_env_id,
                    )
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self.vectorenv.step(
                urma_action
            )

            observation = next_observation

            step_count += 1

            terminated_flag = bool(
                np.asarray(terminated).any()
            )

            truncated_flag = bool(
                np.asarray(truncated).any()
            )

            done = (
                terminated_flag
                or truncated_flag
            )

        print(
            f"\nEpisode finished after "
            f"{step_count} control steps."
        )

        if terminated_flag:

            print(
                "Termination reason: "
                "environment termination."
            )

        elif truncated_flag:

            print(
                "Termination reason: "
                "episode horizon reached."
            )

    # --------------------------------------------------------
    # ALL COMMANDS
    # --------------------------------------------------------

    def run_all_commands(self):

        print("\n")
        print("=" * 70)
        print("PID COMMAND BENCHMARK")
        print("=" * 70)

        print(
            f"Number of commands: "
            f"{len(COMMANDS)}"
        )

        print("=" * 70)

        for command_index, command in enumerate(
            COMMANDS,
            start=1,
        ):

            print(
                f"\nRunning command "
                f"{command_index}/"
                f"{len(COMMANDS)}"
            )

            self.run_one_episode(
                command
            )

        # Disable external command after benchmark.
        self.robot_env.set_desired_command(
            None
        )

        print("\n" + "=" * 70)
        print("ALL PID TESTS COMPLETE")
        print(
            "Results saved in: "
            "/workspace/pid_results/"
        )
        print("=" * 70 + "\n")

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------

    def main(self, argv):

        del argv

        try:

            self.create_config()

            self.create_environment()

            self.load_urma()

            self.run_all_commands()

        finally:

            # IMPORTANT:
            # do not close after each individual episode.
            # The same environment is reused for all commands.
            if self.env is not None:
                self.env.close()


runner = URMATestRunner()


if __name__ == "__main__":
    app.run(
        runner.main
    )