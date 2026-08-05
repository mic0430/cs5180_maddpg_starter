from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DDPGBatch:
    """A batch of single-agent transitions used by DDPG."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

    def to(self, device: torch.device) -> "DDPGBatch":
        """Return a copy of the batch moved to the requested device."""
        return DDPGBatch(
            observations=self.observations.to(
                device=device,
                dtype=torch.float32,
            ),
            actions=self.actions.to(
                device=device,
                dtype=torch.float32,
            ),
            rewards=self.rewards.to(
                device=device,
                dtype=torch.float32,
            ),
            next_observations=self.next_observations.to(
                device=device,
                dtype=torch.float32,
            ),
            dones=self.dones.to(
                device=device,
                dtype=torch.float32,
            ),
        )


class DDPGAgent:
    """
    Generic deterministic actor-critic learner.

    Expected neural-network interfaces:

        actor(observations) -> actions

        critic(observations, actions) -> q_values
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
    ) -> None:
        if actor_lr <= 0.0:
            raise ValueError("actor_lr must be positive.")

        if critic_lr <= 0.0:
            raise ValueError("critic_lr must be positive.")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be between 0 and 1.")

        # Use the device that the actor is already on.
        try:
            self.device = next(actor.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Target networks begin as exact, independent copies.
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        # Target networks are updated manually, not by gradient descent.
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.gamma = gamma
        self.tau = tau

        # These can be scalars or vectors such as the values from
        # env.action_space(agent).low and .high.
        self.action_low = torch.as_tensor(
            action_low,
            dtype=torch.float32,
            device=self.device,
        )

        self.action_high = torch.as_tensor(
            action_high,
            dtype=torch.float32,
            device=self.device,
        )

        if torch.any(self.action_low >= self.action_high):
            raise ValueError(
                "Every action_low value must be below action_high."
            )

    def _bound_actions(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp actions to the environment's valid range."""
        low = self.action_low.to(
            device=actions.device,
            dtype=actions.dtype,
        )

        high = self.action_high.to(
            device=actions.device,
            dtype=actions.dtype,
        )

        return torch.maximum(
            torch.minimum(actions, high),
            low,
        )

    @staticmethod
    def _as_column(
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a tensor shaped [batch_size] into [batch_size, 1].

        Replay buffers sometimes return rewards and dones as one-dimensional
        tensors, while critics generally return [batch_size, 1].
        """
        if values.ndim == 1:
            return values.unsqueeze(-1)

        return values

    def act(
        self,
        observation: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Select an action using the actor.

        Gaussian exploration noise is added when noise_std is greater than 0.
        """
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        # The actor expects a batch dimension.
        single_observation = observation.ndim == 1

        if single_observation:
            observation = observation.unsqueeze(0)

        was_training = self.actor.training
        self.actor.eval()

        with torch.no_grad():
            action = self.actor(observation)

            if noise_std > 0.0:
                noise = noise_std * torch.randn_like(action)
                action = action + noise

            action = self._bound_actions(action)

        # Restore the actor's previous mode.
        self.actor.train(was_training)

        if single_observation:
            action = action.squeeze(0)

        return action

    def compute_critic_targets(
        self,
        batch: DDPGBatch,
    ) -> torch.Tensor:
        """
        Calculate the critic's TD targets.

        target = reward
                 + gamma
                 * (1 - done)
                 * target_critic(next_state, target_actor(next_state))
        """
        batch = batch.to(self.device)

        rewards = self._as_column(batch.rewards)
        dones = self._as_column(batch.dones).clamp(0.0, 1.0)

        with torch.no_grad():
            next_actions = self.target_actor(
                batch.next_observations
            )

            next_actions = self._bound_actions(next_actions)

            next_q_values = self.target_critic(
                batch.next_observations,
                next_actions,
            )

            targets = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_q_values
            )

        return targets

    def update(
        self,
        batch: DDPGBatch,
    ) -> dict[str, float]:
        """
        Perform one complete DDPG update.

        The order is:

        1. Update the critic.
        2. Update the actor.
        3. Soft-update the target networks.
        """
        batch = batch.to(self.device)

        # ------------------------------------------------------------
        # Critic update
        # ------------------------------------------------------------

        critic_targets = self.compute_critic_targets(batch)

        predicted_q_values = self.critic(
            batch.observations,
            batch.actions,
        )

        critic_loss = F.mse_loss(
            predicted_q_values,
            critic_targets,
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------
        # Actor update
        # ------------------------------------------------------------

        # We need gradients through the critic with respect to the actions,
        # but we do not want to update the critic's parameters here.
        self.critic.requires_grad_(False)

        predicted_actions = self.actor(batch.observations)
        predicted_actions = self._bound_actions(predicted_actions)

        actor_loss = -self.critic(
            batch.observations,
            predicted_actions,
        ).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic.requires_grad_(True)

        # ------------------------------------------------------------
        # Target-network update
        # ------------------------------------------------------------

        self.update_target_networks()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "mean_q": float(
                predicted_q_values.detach().mean().cpu()
            ),
            "mean_target_q": float(
                critic_targets.detach().mean().cpu()
            ),
        }

    @torch.no_grad()
    def update_target_networks(self) -> None:
        """
        Soft-update both target networks.

        target = (1 - tau) * target + tau * online
        """
        for target_parameter, parameter in zip(
            self.target_actor.parameters(),
            self.actor.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - self.tau)
            target_parameter.add_(
                parameter,
                alpha=self.tau,
            )

        for target_parameter, parameter in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
            strict=True,
        ):
            target_parameter.mul_(1.0 - self.tau)
            target_parameter.add_(
                parameter,
                alpha=self.tau,
            )