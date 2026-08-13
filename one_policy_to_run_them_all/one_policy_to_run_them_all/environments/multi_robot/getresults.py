import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*to get variables from other wrappers is deprecated.*"
)

from pathlib import Path
import torch
import torch.optim.adam

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/workspace/RL-X/one_policy_to_run_them_all/one_policy_to_run_them_all/")

checkpoint_dir = Path("/workspace/residual_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

import csv 
from default_config import get_config
from environments.multi_robot.create_env import create_env
from absl import app
from ml_collections import config_dict, config_flags
from algorithms.uni_ppo.ppo.ppo import PPO
from residualactorcritic import ResidualActorCritic

from environments.multi_robot.default_config import get_config as get_env_config
from algorithms.uni_ppo.ppo.default_config import get_config as get_algo_config
from rl_x.runner.default_config import get_config as get_runner_config




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


#define flags
default_env_config = get_env_config("multi_robot")

env_config_flag = config_flags.DEFINE_config_dict("environment", default_env_config,)

default_algo_config = get_algo_config("uni_ppo.ppo")

algo_config_flag = config_flags.DEFINE_config_dict("algorithm", default_algo_config)

default_runner_config = get_runner_config("test")

runner_config_flag = config_flags.DEFINE_config_dict("runner", default_runner_config)





class TestModel():
    def __init__(self):
        pass

# load saved model

    def load_residual_model(self, final_model: bool, update_number: None|int = None):

        if not final_model:
            checkpoint_path = checkpoint_dir / f"residual_policy_update_{update_number}.pt"
        else:
            checkpoint_path = checkpoint_dir / "residual_policy_final.pt"

        checkpoint = torch.load(checkpoint_path, map_location=device)

        obs_size = checkpoint["observation_size"]
        action_size = checkpoint["action_size"]
                
        residual_policy = ResidualActorCritic(obs_size, action_size).to(device)
        residual_policy.load_state_dict(checkpoint["model_state_dict"])

        return residual_policy

#load urma
    def get_urma_model(self):
        run_path = f"runs/{self.config.runner.project_name}/{self.config.runner.exp_name}/{self.config.runner.run_name}"
        explicitly_set_algorithm_params = [param_name for param_name in algo_config_flag._flagvalues if param_name.startswith("algorithm.")]
        urma = PPO.load(self.config, self.env, run_path, None, explicitly_set_algorithm_params)
        return urma

    def create_config(self):
        self.config = config_dict.ConfigDict()

        self.config.environment = env_config_flag.value

        self.config.environment.render = False
        self.config.environment.mode = "test"

        self.config.algorithm = algo_config_flag.value
        self.config.runner = runner_config_flag.value


    def create_environment(self):
        self.env = create_env(config=self.config)

        self.vectorenv = self.env.train_env.env

        active_id = self.vectorenv.active_env_id
        self.robot_env = self.vectorenv.envs[active_id].env

    def run(self):
        app.run(self.main)

    def main(self, argv):
        del argv

        self.create_config()
        self.create_environment()

        self.residual_policy = self.load_residual_model(True)
        self.residual_policy.eval()

        device = next(self.residual_policy.parameters()).device
        self.urma = self.get_urma_model()
        self.compare_models(self.urma, self.residual_policy)

        

    def compare_models(self, urma, residual_policy):
        urma_results = self.run_test_episode(False)

        residual_results = self.run_test_episode(True)

        self.save_trajectory(
            urma_results,
            "/workspace/urma_only_tracking.csv",
        )

        self.save_trajectory(
            residual_results,
            "/workspace/residual_tracking.csv",
        )

        for axis in ["x", "y", "yaw"]:
            self.plot_tracking_comparison(urma_results, residual_results, axis)

            urma_only_metrics = self.calculate_metrics(urma_results, axis)

            residual_metrics = self.calculate_metrics(residual_results, axis)

            print(f"\n{axis.upper()} tracking")
            print("URMA only:", urma_only_metrics)
            print("Residual:", residual_metrics)

            if urma_only_metrics["MAE"] > 0:
                improvement = (
                    urma_only_metrics["MAE"]
                    - residual_metrics["MAE"]
                ) / urma_only_metrics["MAE"] * 100

                print(
                    f"MAE improvement: {improvement:.2f}%"
                )

    def run_test_episode(self, use_residual:bool):
        
        observation, info = self.vectorenv.reset()

        observation = self.set_goal_velocity()

        done = False

        trajectory = []

        episode_step = 0
        try:
            while not done:
                if use_residual:
                    residual_observation = self.robot_env.get_residual_observation()

                    residual_observation_tensor = torch.tensor(residual_observation, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        residual_action_tensor = self.residual_policy.actor(
                            residual_observation_tensor
                        )
                    
                    residual_action = (residual_action_tensor.squeeze(0).cpu().numpy())

                else:
                    residual_action = np.zeros(self.robot_env.model.nu, dtype=np.float32)
                
                
                self.robot_env.set_residual_action(residual_action)

                with torch.no_grad():
                    base_action = self.urma.get_interference_action_multi_render(
                        observation, 
                        self.vectorenv.active_env_id
                    )

                next_observation, urma_reward, terminated, truncated, info = self.vectorenv.step(base_action)

                observation = next_observation

                terminated_flag = bool(np.asarray(terminated).any())
                truncated_flag = bool(np.asarray(truncated).any())

                done = terminated_flag or truncated_flag
                if done: 
                    break

                episode_step += 1

                tracking = self.robot_env.get_displacement_tracking(episode_step)

                trajectory.append(tracking)
        finally:
            self.robot_env.close()

        return trajectory
    
    def save_trajectory(self, trajectory, filename):

        fieldnames = [
            "time",

            "desired_x_displacement",
            "actual_x_displacement",
            "x_displacement_error",
            "absolute_x_displacement_error",

            "desired_y_displacement",
            "actual_y_displacement",
            "y_displacement_error",
            "absolute_y_displacement_error",

            "desired_yaw_displacement",
            "actual_yaw_displacement",
            "yaw_displacement_error",
            "absolute_yaw_displacement_error",
        ]

        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(trajectory)

    def plot_tracking_comparison(
        self,
        urma_only_results,
        residual_results,
        axis,
    ):
        error_key = f"absolute_{axis}_displacement_error"

        urma_time = [
            row["time"]
            for row in urma_only_results
        ]

        urma_error = [
            row[error_key]
            for row in urma_only_results
        ]

        residual_time = [
            row["time"]
            for row in residual_results
        ]

        residual_error = [
            row[error_key]
            for row in residual_results
        ]

        plt.figure(figsize=(10, 5))

        plt.plot(
            urma_time,
            urma_error,
            label="URMA only",
        )

        plt.plot(
            residual_time,
            residual_error,
            label="URMA + residual policy",
        )
        
        if axis == "yaw":
            ylabel = (
                "Absolute yaw tracking error (rad)"
            )
            title_axis = "Yaw"
        else:
            ylabel = (
                f"Absolute {axis} "
                "displacement error (m)"
            )
            title_axis = axis.upper()

        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)

        plt.title(
            "URMA versus residual-policy "
            f"{title_axis} tracking"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            f"/workspace/{axis}_displacement_comparison.png",
            dpi=300,
        )

        plt.close()

    def calculate_metrics(self, results, axis,):

        error_key = (f"{axis}_displacement_error")
        errors = np.asarray([row[error_key] for row in results], dtype=float,)

        absolute_errors = np.abs(errors)

        return {
            "MAE": float(np.mean(absolute_errors)),
            "RMSE": float(
                np.sqrt(np.mean(errors ** 2))
            ),
            "final_absolute_error": float(
                absolute_errors[-1]
            ),
            "maximum_absolute_error": float(
                np.max(absolute_errors)
            ),
        }
    
    def set_goal_velocity(self):
        self.robot_env.goal_x_velocity = 1.0
        self.robot_env.goal_y_velocity = 0.5
        self.robot_env.goal_yaw_velocity = 0.5

        observation = self.robot_env.get_observation()

        return np.array([observation])
        


testrunner = TestModel()
testrunner.run()

