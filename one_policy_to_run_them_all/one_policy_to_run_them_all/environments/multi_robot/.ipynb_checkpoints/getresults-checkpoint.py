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
        magnitudes = [0.3, 0.6, 0.9]

        commands = []
        
        # Excite each axis independently
        for axis in range(3):
            for mag in magnitudes:
                for sign in [1, -1]:
                    command = [0.0, 0.0, 0.0]
                    command[axis] = sign * mag
                    commands.append(command)
        
        # Zero command
        commands.append([0.0, 0.0, 0.0])
        
        # Combined commands
        commands += [
            [0.6,  0.6,  0.0],  # forward + left/right strafe
            [0.6, -0.6,  0.0],
            [0.6,  0.0,  0.6],  # forward + yaw
            [0.6,  0.0, -0.6],
        ]

        for command in commands:
        
            urma_results = self.run_test_episode(False, command[0], command[1], command[2])
    
            residual_results = self.run_test_episode(True, command[0], command[1], command[2])
    
            self.save_trajectory(
                urma_results,
                f"/workspace/urma_only_tracking_{command[0]}_{command[1]}_{command[2]}.csv",
            )
    
            self.save_trajectory(
                residual_results,
                f"/workspace/residual_tracking_{command[0]}_{command[1]}_{command[2]}.csv",
            )
    
            for axis in ["x", "y", "yaw"]:
                self.plot_tracking_comparison(urma_results, residual_results, axis, command)
    
            self.print_pid_style_results(
                urma_results,
                label=f"URMA ONLY TRACKING RESULTS {command[0]}_{command[1]}_{command[2]}",
            )
    
            self.print_pid_style_results(
                residual_results,
                label=f"RESIDUAL RL TRACKING RESULTS {command[0]}_{command[1]}_{command[2]}",
            )

    def run_test_episode(self, use_residual:bool, x, y, z):
        
        observation, info = self.vectorenv.reset()

        observation = self.set_goal_velocity(x, y, z)

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

                linear_velocity = self.robot_env.orientation_quat_inv.apply(
                    self.robot_env.data.qvel[:3]
                )

                desired_velocity = np.array([
                    self.robot_env.goal_x_velocity,
                    self.robot_env.goal_y_velocity,
                    self.robot_env.goal_yaw_velocity,
                ], dtype=float)

                actual_velocity = np.array([
                    linear_velocity[0],
                    linear_velocity[1],
                    self.robot_env.data.qvel[5],
                ], dtype=float)

                velocity_error = desired_velocity - actual_velocity

                tracking.update({
                    "desired_x_velocity": float(desired_velocity[0]),
                    "actual_x_velocity": float(actual_velocity[0]),
                    "x_velocity_error": float(velocity_error[0]),
                    "desired_y_velocity": float(desired_velocity[1]),
                    "actual_y_velocity": float(actual_velocity[1]),
                    "y_velocity_error": float(velocity_error[1]),
                    "desired_yaw_velocity": float(desired_velocity[2]),
                    "actual_yaw_velocity": float(actual_velocity[2]),
                    "yaw_velocity_error": float(velocity_error[2]),
                })

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

            "desired_x_velocity",
            "actual_x_velocity",
            "x_velocity_error",
            "desired_y_velocity",
            "actual_y_velocity",
            "y_velocity_error",
            "desired_yaw_velocity",
            "actual_yaw_velocity",
            "yaw_velocity_error",
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
        command,
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
            f"URMA versus residual-policy_({command[0]}_{command[1]}_{command[2]})"
            f"{title_axis} tracking"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            f"/workspace/residual_images/{axis}_displacement_comparison_({command[0]}_{command[1]}_{command[2]}).png",
            dpi=300,
        )

        plt.close()

    def calculate_velocity_metrics(self, results, axis):
        errors = np.asarray(
            [row[f"{axis}_velocity_error"] for row in results],
            dtype=float,
        )

        mse = float(np.mean(errors ** 2))

        return {
            "MAE": float(np.mean(np.abs(errors))),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "bias": float(np.mean(errors)),
            "max_absolute_error": float(np.max(np.abs(errors))),
        }

    def calculate_displacement_metrics(self, results, axis):
        errors = np.asarray(
            [row[f"{axis}_displacement_error"] for row in results],
            dtype=float,
        )

        return {
            "final_error": float(errors[-1]),
            "maximum_absolute_error": float(np.max(np.abs(errors))),
        }

    def print_pid_style_results(self, results, label):
        if not results:
            print(f"\n{label}: no tracking samples recorded.")
            return

        velocity_metrics = {
            axis: self.calculate_velocity_metrics(results, axis)
            for axis in ["x", "y", "yaw"]
        }
        displacement_metrics = {
            axis: self.calculate_displacement_metrics(results, axis)
            for axis in ["x", "y", "yaw"]
        }

        all_errors = np.concatenate([
            np.asarray([row[f"{axis}_velocity_error"] for row in results], dtype=float)
            for axis in ["x", "y", "yaw"]
        ])
        overall_mse = float(np.mean(all_errors ** 2))
        overall_mae = float(np.mean(np.abs(all_errors)))
        overall_rmse = float(np.sqrt(overall_mse))

        print("\n" + "=" * 68)
        print(label)
        print("=" * 68)
        print(
            f"Overall velocity: MAE={overall_mae:.6f}, "
            f"MSE={overall_mse:.6f}, RMSE={overall_rmse:.6f}"
        )

        for axis, unit, prefix in [
            ("x", "m/s", "  x velocity"),
            ("y", "m/s", "  y velocity"),
            ("yaw", "rad/s", "yaw velocity"),
        ]:
            m = velocity_metrics[axis]
            print(
                f"{prefix} [{unit}]: "
                f"MAE={m['MAE']:.6f} | "
                f"MSE={m['MSE']:.6f} | "
                f"RMSE={m['RMSE']:.6f} | "
                f"bias={m['bias']:+.6f} | "
                f"max|e|={m['max_absolute_error']:.6f}"
            )

        print("Final cumulative tracking error:")
        print(
            f"  x [m]: final={displacement_metrics['x']['final_error']:+.6f} | "
            f"max|e|={displacement_metrics['x']['maximum_absolute_error']:.6f}"
        )
        print(
            f"  y [m]: final={displacement_metrics['y']['final_error']:+.6f} | "
            f"max|e|={displacement_metrics['y']['maximum_absolute_error']:.6f}"
        )
        print(
            f"yaw [rad]: final={displacement_metrics['yaw']['final_error']:+.6f} | "
            f"max|e|={displacement_metrics['yaw']['maximum_absolute_error']:.6f}"
        )
        print("=" * 68)

    def set_goal_velocity(self, x, y, z):
        self.robot_env.goal_x_velocity = x
        self.robot_env.goal_y_velocity = y
        self.robot_env.goal_yaw_velocity = z


        observation = self.robot_env.get_observation()

        return np.array([observation])
        


testrunner = TestModel()
testrunner.run()

