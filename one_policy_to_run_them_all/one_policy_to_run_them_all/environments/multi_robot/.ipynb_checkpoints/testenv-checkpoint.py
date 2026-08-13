import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import csv

import numpy as np
import torch
from absl import app
from ml_collections import config_dict, config_flags

warnings.filterwarnings(
    "ignore",
    message=r".*to get variables from other wrappers is deprecated.*"
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
from residualactorcritic import ResidualActorCritic


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CHECKPOINT_DIR = Path("/workspace/residual_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------

_environment_default_config = get_env_config("multi_robot")
_environment_config_flag = config_flags.DEFINE_config_dict(
    "environment",
    _environment_default_config,
)

_algorithm_default_config = get_algo_config("uni_ppo.ppo")
_algorithm_config_flag = config_flags.DEFINE_config_dict(
    "algorithm",
    _algorithm_default_config,
)

_runner_default_config = get_runner_config("test")
_runner_config_flag = config_flags.DEFINE_config_dict(
    "runner",
    _runner_default_config,
)

# checkpoints
checkpoint_dir = Path("/workspace/residual_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------

ROLLOUT_STEPS = 2048

PPO_EPOCHS = 3
MINIBATCH_SIZE = 32
CLIP_COEFFICIENT = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
MAX_GRAD_NORM = 0.5

GAMMA = 0.99
GAE_LAMBDA = 0.95

NUMBER_OF_UPDATES = 1500
LEARNING_RATE = 1e-4


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------

def create_config():
    config = config_dict.ConfigDict()
    config.environment = _environment_config_flag.value
    config.algorithm = _algorithm_config_flag.value
    config.runner = _runner_config_flag.value
    return config


def setup_environment(config):
    env = create_env(config=config)

    run_path = (
        f"runs/{config.runner.project_name}/"
        f"{config.runner.exp_name}/"
        f"{config.runner.run_name}"
    )

    vectorenv = env.train_env.env
    observation, info = vectorenv.reset()

    active_id = vectorenv.active_env_id
    robot_env = vectorenv.envs[active_id].env

    return env, vectorenv, robot_env, observation, info, run_path


def load_urma(config, env, run_path):
    explicitly_set_algorithm_params = [param_name for param_name in _algorithm_config_flag._flagvalues if param_name.startswith("algorithm.")]

    urma = PPO.load(config, env, run_path, None, explicitly_set_algorithm_params,)
    return urma


def create_residual_policy(robot_env, device):
    start_update = 0
    observation_size = 9 + (robot_env.model.nu * 2)
    action_size = robot_env.model.nu

    residual_policy = ResidualActorCritic(
        observation_size=observation_size,
        action_size=action_size,
    ).to(device)

    optimizer = torch.optim.Adam(residual_policy.parameters(), lr=LEARNING_RATE,)

    return residual_policy, optimizer, observation_size, action_size, start_update


def load_residual_policy(final_model, device, update_number: int|None = None):

    if not final_model:
        checkpoint_path = checkpoint_dir / f"residual_policy_update_{update_number}.pt"
    else:
        checkpoint_path = checkpoint_dir / "residual_policy_final.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_size = checkpoint["observation_size"]
    action_size = checkpoint["action_size"]
            
    residual_policy = ResidualActorCritic(obs_size, action_size).to(device)
    residual_policy.load_state_dict(checkpoint["model_state_dict"])

    model_device = next(residual_policy.parameters()).device

    optimizer = torch.optim.Adam(residual_policy.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Make sure Adam's state is on same GPU
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(model_device)

    start_update = checkpoint["update"]

    return residual_policy, optimizer, obs_size, action_size, start_update


def get_residual_action(residual_policy, robot_env):

    device = next(residual_policy.parameters()).device

    residual_observation = robot_env.get_residual_observation()

    residual_observation_tensor = torch.tensor(residual_observation, dtype=torch.float32, device=device,).unsqueeze(0)

    with torch.no_grad():
        residual_action_tensor, log_prob, _, value = (residual_policy.get_action_and_value(residual_observation_tensor))

    residual_action = (residual_action_tensor.squeeze(0).cpu().numpy())

    return (
        residual_observation_tensor,
        residual_action_tensor,
        residual_action,
        log_prob,
        value,
    )


def collect_rollout(rollout_steps, observation,vectorenv,robot_env, urma, residual_policy, device, episode_count, episode_residual_return,
                    episode_tracking_error_sum, episode_steps, episode_numbers, episode_returns, episode_tracking_errors,):
    rollout_buffer = []

    for _ in range(rollout_steps):
        (residual_observation_tensor,
            residual_action_tensor,
            residual_action,
            log_prob,
            value,
        ) = get_residual_action(residual_policy,robot_env,)

        robot_env.set_residual_action(residual_action)

        with torch.no_grad():
            base_action = urma.get_interference_action_multi_render(
                observation,
                vectorenv.active_env_id,
            )

        (
            next_observation,
            urma_reward,
            terminated,
            truncated,
            info,
        ) = vectorenv.step(base_action)

        terminated_flag = bool(np.asarray(terminated).any())
        truncated_flag = bool(np.asarray(truncated).any())
        done = terminated_flag or truncated_flag

        residual_reward = extract_residual_reward(info)
        episode_residual_return += residual_reward

        tracking_error = extract_tracking_error(info)

        if np.isfinite(tracking_error):
            episode_tracking_error_sum += tracking_error

        episode_steps += 1

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

            mean_tracking_error = (
                episode_tracking_error_sum / episode_steps
                if episode_steps > 0
                else np.nan
            )

            episode_numbers.append(episode_count)
            episode_returns.append(episode_residual_return)
            episode_tracking_errors.append(mean_tracking_error)

            print(
                f"Episode {episode_count} | "
                f"return: {episode_residual_return:.3f} | "
                f"mean tracking error: {mean_tracking_error:.5f}"
            )

            episode_residual_return = 0.0
            episode_tracking_error_sum = 0.0
            episode_steps = 0

    return (rollout_buffer,observation,episode_count,episode_residual_return, episode_tracking_error_sum, episode_steps)


def get_final_value(residual_policy,robot_env,rollout_buffer,):

    device = next(residual_policy.parameters()).device
    final_residual_observation = robot_env.get_residual_observation()

    final_residual_observation_tensor = torch.as_tensor(
        final_residual_observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        final_value = residual_policy.critic(final_residual_observation_tensor).squeeze()

    if rollout_buffer[-1]["done"]:
        final_value = torch.tensor(0.0,device=device,)

    return final_value


def compute_gae_and_returns(values,rewards,dones,final_value,gamma,gae_lambda,):

    advantages = torch.zeros_like(rewards)

    last_gae = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=rewards.device,
    )

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = final_value
        else:
            next_value = values[step + 1]

        next_non_terminal = 1.0 - dones[step]

        delta = (
            rewards[step]
            + gamma * next_value * next_non_terminal
            - values[step]
        )

        last_gae = (delta + gamma * gae_lambda * next_non_terminal * last_gae)

        advantages[step] = last_gae

    returns = advantages + values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def run_ppo_update(
    residual_policy,
    optimizer,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
):
    batch_size = len(observations)

    policy_loss = None
    value_loss = None
    entropy_loss = None
    ratio = None

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(
            batch_size,
            device=observations.device,
        )

        for start in range(0, batch_size, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            minibatch_indices = indices[start:end]

            minibatch_observations = observations[minibatch_indices]
            minibatch_actions = actions[minibatch_indices]
            minibatch_old_log_probs = old_log_probs[minibatch_indices]
            minibatch_advantages = advantages[minibatch_indices]
            minibatch_returns = returns[minibatch_indices]

            (
                _,
                new_log_probs,
                entropy,
                new_values,
            ) = residual_policy.get_action_and_value(
                minibatch_observations,
                action=minibatch_actions,
            )

            log_ratio = (
                new_log_probs
                - minibatch_old_log_probs
            )

            ratio = log_ratio.exp()

            unclipped_policy_loss = (
                -minibatch_advantages * ratio
            )

            clipped_policy_loss = (
                -minibatch_advantages
                * torch.clamp(
                    ratio,
                    1.0 - CLIP_COEFFICIENT,
                    1.0 + CLIP_COEFFICIENT,
                )
            )

            policy_loss = torch.max(
                unclipped_policy_loss,
                clipped_policy_loss,
            ).mean()

            value_loss = (
                0.5
                * (new_values - minibatch_returns)
                .pow(2)
                .mean()
            )

            entropy_loss = entropy.mean()

            total_loss = (
                policy_loss
                + VALUE_COEFFICIENT * value_loss
                - ENTROPY_COEFFICIENT * entropy_loss
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "Non-finite PPO loss detected: "
                    f"{total_loss.item()}"
                )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                residual_policy.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            with torch.no_grad():
                residual_policy.log_std.clamp_(
                    -4.0,
                    -2.5,
                )

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------
# Logging / checkpoint helpers
# ---------------------------------------------------------------------

def print_update_metrics(
    update,
    residual_policy,
    rewards,
    actions,
    metrics,
    start_update
):
    policy_loss = metrics["policy_loss"]
    value_loss = metrics["value_loss"]
    entropy_loss = metrics["entropy_loss"]
    ratio = metrics["ratio"]

    mean_reward = rewards.mean().item()
    mean_action_magnitude = actions.abs().mean().item()

    print(
        f"Update {update + 1}/{NUMBER_OF_UPDATES+start_update} | "
        f"policy loss: {policy_loss.item():.4f} | "
        f"value loss: {value_loss.item():.4f} | "
        f"entropy: {entropy_loss.item():.4f} | "
        f"log_std: {residual_policy.log_std.mean().item():.4f} | "
        f"mean reward: {mean_reward:.4f} | "
        f"mean residual magnitude: "
        f"{mean_action_magnitude:.4f}"
    )

    print(
        f"ratio mean: {ratio.mean().item():.4f}",
        f"ratio min: {ratio.min().item():.4f}",
        f"ratio max: {ratio.max().item():.4f}",
    )


def save_checkpoint(
    checkpoint_path,
    update,
    residual_policy,
    optimizer,
    observation_size,
    action_size,
):
    torch.save(
        {
            "update": update,
            "model_state_dict": residual_policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "observation_size": observation_size,
            "action_size": action_size,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


def save_periodic_checkpoint(
    update,
    residual_policy,
    optimizer,
    observation_size,
    action_size,
):
    if (update + 1) % 10 != 0:
        return

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"residual_policy_update_{update + 1}.pt"
    )
    

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        update=update + 1,
        residual_policy=residual_policy,
        optimizer=optimizer,
        observation_size=observation_size,
        action_size=action_size,
    )

def extract_tracking_error(info):
    key = "residual/tracking_error"

    if key in info:
        values = np.asarray(info[key])

        mask_key = "_" + key

        if mask_key in info:
            mask = np.asarray(info[mask_key], dtype=bool)
            values = values[mask]

        return float(values.reshape(-1)[0])

    return np.nan

def save_training_plots(episode_numbers, episode_returns, episode_tracking_errors):
    if len(episode_numbers) == 0:
        return

    # Episode return
    plt.figure(figsize=(8, 5))
    plt.plot(episode_numbers, episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Residual PPO Training: Episode Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "episode_return.png", dpi=300)
    plt.close()

    # Tracking error
    plt.figure(figsize=(8, 5))
    plt.plot(episode_numbers, episode_tracking_errors)
    plt.xlabel("Episode")
    plt.ylabel("Mean Tracking Error")
    plt.title("Residual PPO Training: Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "episode_tracking_error.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Existing utility helpers
# ---------------------------------------------------------------------

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

    return (
        observations,
        actions,
        old_log_probs,
        values,
        rewards,
        dones,
    )


def extract_residual_reward(info):
    if "residual_reward" in info:
        values = np.asarray(info["residual_reward"])

        if "_residual_reward" in info:
            mask = np.asarray(
                info["_residual_reward"],
                dtype=bool,
            )
            values = values[mask]

        return float(values.reshape(-1)[0])

    if "final_info" in info:
        final_info = np.asarray(
            info["final_info"],
            dtype=object,
        )

        final_mask = np.asarray(
            info["_final_info"],
            dtype=bool,
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


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train(
    vectorenv,
    robot_env,
    urma,
    residual_policy,
    optimizer,
    observation,
    device,
    observation_size,
    action_size,
    start_update
):
    episode_count = 0
    episode_residual_return = 0.0
    episode_tracking_error_sum = 0.0
    episode_steps = 0

    episode_numbers = []
    episode_returns = []
    episode_tracking_errors = []

    for update in range(start_update, NUMBER_OF_UPDATES+start_update):
        (
            rollout_buffer,
            observation,
            episode_count,
            episode_residual_return,
            episode_tracking_error_sum,
            episode_steps,
        ) = collect_rollout(
            rollout_steps=ROLLOUT_STEPS,
            observation=observation,
            vectorenv=vectorenv,
            robot_env=robot_env,
            urma=urma,
            residual_policy=residual_policy,
            device=device,
            episode_count=episode_count,
            episode_residual_return=episode_residual_return,
            episode_tracking_error_sum=episode_tracking_error_sum,
            episode_steps=episode_steps,
            episode_numbers=episode_numbers,
            episode_returns=episode_returns,
            episode_tracking_errors=episode_tracking_errors,
        )

        final_value = get_final_value(
            residual_policy=residual_policy,
            robot_env=robot_env,
            rollout_buffer=rollout_buffer,
        )

        (
            observations,
            actions,
            old_log_probs,
            values,
            rewards,
            dones,
        ) = convert_transition_to_tensors(
            rollout_buffer,
            device,
        )

        advantages, returns = compute_gae_and_returns(
            values=values,
            rewards=rewards,
            dones=dones,
            final_value=final_value,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        metrics = run_ppo_update(
            residual_policy=residual_policy,
            optimizer=optimizer,
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
        )

        print_update_metrics(
            update=update,
            residual_policy=residual_policy,
            rewards=rewards,
            actions=actions,
            metrics=metrics,
            start_update = start_update
        )

        save_periodic_checkpoint(
            update=update,
            residual_policy=residual_policy,
            optimizer=optimizer,
            observation_size=observation_size,
            action_size=action_size,
        )

        save_training_plots(
            episode_numbers,
            episode_returns,
            episode_tracking_errors,
        )

    save_checkpoint(
        checkpoint_path=CHECKPOINT_DIR / "residual_policy_final.pt",
        update=NUMBER_OF_UPDATES+start_update,
        residual_policy=residual_policy,
        optimizer=optimizer,
        observation_size=observation_size,
        action_size=action_size,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv):
    del argv

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    config = create_config()

    (
        env,
        vectorenv,
        robot_env,
        observation,
        info,
        run_path,
    ) = setup_environment(config)

    urma = load_urma(
        config=config,
        env=env,
        run_path=run_path,
    )

    (
        residual_policy,
        optimizer,
        observation_size,
        action_size,
        start_update
    ) = create_residual_policy(
        robot_env=robot_env,
        device=device,
    )

    train(
        vectorenv=vectorenv,
        robot_env=robot_env,
        urma=urma,
        residual_policy=residual_policy,
        optimizer=optimizer,
        observation=observation,
        device=device,
        observation_size=observation_size,
        action_size=action_size,
        start_update=start_update,
    )


if __name__ == "__main__":
    app.run(main)
