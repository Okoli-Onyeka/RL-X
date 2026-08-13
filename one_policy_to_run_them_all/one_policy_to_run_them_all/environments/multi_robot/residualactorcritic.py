import torch
import torch.nn as nn
from torch.distributions import Normal

class ResidualActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.log_std = nn.Parameter(
            torch.full((action_size,), -2.0)
        )

    def get_action_and_value(
            self,
            observation: torch.Tensor,
            action: torch.Tensor | None = None,
    ):
        mean = self.actor(observation)
        std = torch.exp(self.log_std).expand_as(mean)

        distribution = Normal(mean, std)

        if action is None:
            action = distribution.sample()

        log_probability = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observation).squeeze(-1)

        return action, log_probability, entropy, value

