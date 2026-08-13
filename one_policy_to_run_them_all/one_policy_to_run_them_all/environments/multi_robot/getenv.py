import warnings
import torch
import torch.optim.adam

warnings.filterwarnings(
    "ignore",
    message=r".*to get variables from other wrappers is deprecated.*"
)

from pathlib import Path

checkpoint_dir = Path("/workspace/residual_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)


from absl import app
import numpy as np

import sys
sys.path.insert(0,"/workspace/RL-X/one_policy_to_run_them_all/one_policy_to_run_them_all/")

from environments.multi_robot.create_env import create_env
from ml_collections import config_dict, config_flags
from environments.multi_robot.default_config import get_config as get_env_config
from algorithms.uni_ppo.ppo.default_config import get_config as get_algo_config
from rl_x.runner.default_config import get_config as get_runner_config
from algorithms.uni_ppo.ppo.ppo import PPO
from residualactorcritic import ResidualActorCritic


_environment_default_config = get_env_config("multi_robot")

_environment_config_flag = config_flags.DEFINE_config_dict(
    "environment",
    _environment_default_config,
)

_algorithm_default_config = get_algo_config("uni_ppo.ppo")

_algorithm_config_flag = config_flags.DEFINE_config_dict(
    "algorithm",
    _algorithm_default_config
)

_runner_default_config = get_runner_config("test")

_runner_config_flag = config_flags.DEFINE_config_dict(
    "runner",
    _runner_default_config
)

def main(argv):
    del argv  # We do not need the remaining command-line arguments.

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = config_dict.ConfigDict()
    config.environment = _environment_config_flag.value
    config.algorithm = _algorithm_config_flag.value
    config.runner = _runner_config_flag.value

    #create environment
    env = create_env(config=config)
    run_path = f"runs/{config.runner.project_name}/{config.runner.exp_name}/{config.runner.run_name}"

    vectorenv = env.train_env.env
    observation, info = vectorenv.reset()

    active_id = vectorenv.active_env_id
    robot_env = vectorenv.envs[active_id].env

    #load urma model
    explicitly_set_algorithm_params = [param_name for param_name in _algorithm_config_flag._flagvalues if param_name.startswith("algorithm.")]
    urma = PPO.load(config, env, run_path, None, explicitly_set_algorithm_params)

    residual_policy = ResidualActorCritic(
        observation_size=9 + (robot_env.model.nu * 2) + 3,
        action_size=robot_env.model.nu
    ).to(device)

    optimizer = torch.optim.Adam(
        residual_policy.parameters(),
        lr=1e-4,
    )


    device = next(residual_policy.parameters()).device

    rollout_steps = 256

    ppo_epochs = 3
    minibatch_size = 64
    clip_coefficient = 0.2
    value_coefficient = 0.5
    entropy_coefficient = 0.01
    max_grad_norm = 0.5

    gamma = 0.99
    gae_lambda = 0.95


    number_of_updates = 3
    episode_count = 0
    episode_residual_return = 0.0

    for update in range(number_of_updates):
        rollout_buffer = []


        for step in range(rollout_steps):

            residual_observation = robot_env.get_residual_observation()

            residual_observation_tensor = torch.tensor(
                residual_observation,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                residual_action_tensor, log_prob, _, value = residual_policy.get_action_and_value(
                    residual_observation_tensor
                )

            residual_action = (
                residual_action_tensor
                .squeeze(0)
                .cpu()
                .numpy()
            )

            robot_env.set_residual_action(residual_action)

            with torch.no_grad():
                base_action = urma.get_interference_action_multi_render(
                    observation, 
                    vectorenv.active_env_id
                )
            
            next_observation, urma_reward, terminated, truncated, info = vectorenv.step(base_action)

            terminated_flag = bool(np.asarray(terminated).any())
            truncated_flag = bool(np.asarray(truncated).any())

            done = terminated_flag or truncated_flag

            residual_reward = extract_residual_reward(info)
            episode_residual_return += residual_reward

            transition = {
                "observation": (residual_observation_tensor.squeeze(0).detach().cpu()),
                "action": (residual_action_tensor.squeeze(0).detach().cpu()),
                "log_prob": (log_prob.squeeze(0).detach().cpu()),
                "value": (value.squeeze(0).detach().cpu()),
                "reward": float(residual_reward),
                "terminated": terminated_flag,
                "truncated": truncated_flag,
                "done": done,
            }
            rollout_buffer.append(transition)
            observation = next_observation

            if done:
                episode_count += 1

                print(
                    f"Episode {episode_count} residual return: "
                    f"{episode_residual_return:.3f}"
                )

                episode_residual_return = 0.0
        
        #computing final value
        final_residual_observation = robot_env.get_residual_observation()

        final_residual_observation_tensor = torch.as_tensor(
            final_residual_observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        #get the final critic value
        with torch.no_grad():
            final_value = residual_policy.critic(
                final_residual_observation_tensor
            ).squeeze()


        if rollout_buffer[-1]["done"]:
            final_value = torch.tensor(
                0.0,
                device=device,
            )

    
    
        #convertting rollout to tensors
        observations, actions, old_log_probs, values, rewards, dones = convert_transition_to_tensors(rollout_buffer, device)


        #computing gae and returns
        advantages = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        for step in reversed(range(len(rollout_buffer))):

            #use final value for last rollout step
            if step == len(rollout_buffer) - 1:
                next_value = final_value
            else:
                next_value = values[step + 1]

            next_non_terminal = 1.0 - dones[step]

            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]

            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            advantages[step] = last_gae

        returns = advantages + values

        #normalise advantages
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)

        batch_size = len(rollout_buffer)

        #run ppo updates
        for epoch in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = indices[start:end]

                minibatch_observations = observations[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]

                _, new_log_probs, entropy, new_values = residual_policy.get_action_and_value(minibatch_observations, action=minibatch_actions,)

                log_ratio = new_log_probs - minibatch_old_log_probs

                ratio = log_ratio.exp()

                unclipped_policy_loss = -minibatch_advantages * ratio

                clipped_policy_loss = (
                    -minibatch_advantages * torch.clamp(ratio, 1.0 - clip_coefficient, 1.0 + clip_coefficient,)
                )

                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

                value_loss = 0.5 * (new_values - minibatch_returns).pow(2).mean()

                entropy_loss = entropy.mean()

                total_loss = (policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_loss)

                #check if total loss is finite
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite PPO loss detected: {total_loss.item()}")

                optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(residual_policy.parameters(), max_grad_norm,)

                optimizer.step()
                with torch.no_grad():
                    residual_policy.log_std.clamp_(-3.0, -1.5)
        mean_reward = rewards.mean().item()
        mean_action_magnitude = actions.abs().mean().item()

        print(
            f"Update {update + 1}/{number_of_updates} | "
            f"policy loss: {policy_loss.item():.4f} | "
            f"value loss: {value_loss.item():.4f} | "
            f"entropy: {entropy_loss.item():.4f} | "
            f"log_std: {residual_policy.log_std.mean().item():.4f}"
            f"mean reward: {mean_reward:.4f} | "
            f"mean residual magnitude: "
            f"{mean_action_magnitude:.4f}"
        )

        print(
           f"ratio mean: {ratio.mean().item():.4f}",
            f"ration min: {ratio.min().item():.4f}",
            f"ration max: {ratio.max().item():.4f}"
        )
            
        #save after each update
        if (update + 1) % 10 == 0:
            checkpoint_path = (
                checkpoint_dir
                / f"residual_policy_update_{update + 1}.pt"
            )

            torch.save(
                {
                    "update": update + 1,
                    "model_state_dict": residual_policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "observation_size": 9 + 13 + 13+3,
                    "action_size": robot_env.model.nu,
                },
                checkpoint_path,
            )

            print(f"Saved checkpoint: {checkpoint_path}")
    #save final update
    torch.save(
        {
            "update": number_of_updates,
            "model_state_dict": residual_policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "observation_size": 9 + 13 + 13 + 3,
            "action_size": robot_env.model.nu,
        },
        checkpoint_dir / "residual_policy_final.pt",
    )



def convert_transition_to_tensors(rollout_buffer, device):
    observations = torch.stack([
        transition["observation"]
        for transition in rollout_buffer
    ]).to(device)

    actions = torch.stack([
        transition["action"]
        for transition in rollout_buffer
    ]).to(device)

    old_log_probs = torch.stack([
        transition["log_prob"]
        for transition in rollout_buffer
    ]).to(device)

    values = torch.stack([
        transition["value"]
        for transition in rollout_buffer
    ]).to(device)

    rewards = torch.tensor(
        [
            transition["reward"]
            for transition in rollout_buffer
        ],
        dtype=torch.float32,
        device=device,
    )

    dones = torch.tensor(
        [
            transition["done"]
            for transition in rollout_buffer
        ],
        dtype=torch.float32,
        device=device,
    )

    return observations, actions, old_log_probs, values, rewards, dones


def extract_residual_reward(info):
    if "residual_reward" in info:
        values = np.asarray(info["residual_reward"])

        if "_residual_reward" in info:
            mask = np.asarray(
                info["_residual_reward"],
                dtype=bool
            )
            values = values[mask]

        return float(values.reshape(-1)[0])

    if "final_info" in info:
        final_info = np.asarray(
            info["final_info"],
            dtype=object
        )

        final_mask = np.asarray(
            info["_final_info"],
            dtype=bool
        )

        valid_info = final_info[final_mask]

        for env_info in valid_info:
            if (
                env_info is not None
                and "residual_reward" in env_info
            ):
                return float(
                    np.asarray(
                        env_info["residual_reward"]
                    ).squeeze()
                )

    raise KeyError(
        "Could not find residual_reward. "
        f"Available keys: {list(info.keys())}"
    )

if __name__ == "__main__":
    app.run(main)