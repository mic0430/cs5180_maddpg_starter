
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.networks import Actor, Critic
from src.common.device import resolve_device
from src.common.replay_buffer import ReplayBatch
from src.common.target_updates import hard_update, soft_update


class IndependentDDPG:
    """Independent DDPG learners for a multi-agent environment.

    Each agent owns a local actor and local critic. Unlike MADDPG,
    each critic receives only that agent's observation and action.
    """

    def __init__(
        self,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        exploration_noise_std: float = 0.1,
        device: str | None = "auto",
    ) -> None:
        self._validate_positive_integer(
            "num_agents",
            num_agents,
        )
        self._validate_positive_integer(
            "observation_dim",
            observation_dim,
        )
        self._validate_positive_integer(
            "action_dim",
            action_dim,
        )
        self._validate_positive_float(
            "actor_lr",
            actor_lr,
        )
        self._validate_positive_float(
            "critic_lr",
            critic_lr,
        )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "gamma must be between 0 and 1."
            )

        if not 0.0 <= tau <= 1.0:
            raise ValueError(
                "tau must be between 0 and 1."
            )

        if exploration_noise_std < 0.0:
            raise ValueError(
                "exploration_noise_std must be non-negative."
            )

        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.exploration_noise_std = float(
            exploration_noise_std
        )
        self.device = resolve_device(device)

        self.actors = [
            Actor(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        self.critics = [
            Critic(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        self.target_actors = [
            Actor(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        self.target_critics = [
            Critic(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        for agent_index in range(self.num_agents):
            hard_update(
                target=self.target_actors[agent_index],
                source=self.actors[agent_index],
            )
            hard_update(
                target=self.target_critics[agent_index],
                source=self.critics[agent_index],
            )

            self.target_actors[agent_index].eval()
            self.target_critics[agent_index].eval()

        self.actor_optimizers = [
            torch.optim.Adam(
                actor.parameters(),
                lr=actor_lr,
            )
            for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(
                critic.parameters(),
                lr=critic_lr,
            )
            for critic in self.critics
        ]

    def select_actions(
        self,
        observations: np.ndarray,
        explore: bool = False,
        noise_std: float | None = None,
    ) -> np.ndarray:
        """Select one local action for each agent."""
        observation_array = np.ascontiguousarray(
            observations,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_agents,
            self.observation_dim,
        )

        if observation_array.shape != expected_shape:
            raise ValueError(
                f"Expected observations with shape "
                f"{expected_shape}, but received "
                f"{observation_array.shape}."
            )

        if not np.all(np.isfinite(observation_array)):
            raise ValueError(
                "Observations contain NaN or infinite values."
            )

        selected_noise_std = (
            self.exploration_noise_std
            if noise_std is None
            else float(noise_std)
        )

        if selected_noise_std < 0.0:
            raise ValueError(
                "noise_std must be non-negative."
            )

        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=self.device,
        )

        actions: list[torch.Tensor] = []

        with torch.no_grad():
            for agent_index, actor in enumerate(
                self.actors
            ):
                action = actor(
                    observation_tensor[
                        agent_index
                    ].unsqueeze(0)
                ).squeeze(0)

                if explore and selected_noise_std > 0.0:
                    action = (
                        action
                        + selected_noise_std
                        * torch.randn_like(action)
                    )

                actions.append(
                    torch.clamp(
                        action,
                        min=-1.0,
                        max=1.0,
                    )
                )

        action_tensor = torch.stack(
            actions,
            dim=0,
        )

        return np.ascontiguousarray(
            action_tensor.cpu().numpy(),
            dtype=np.float32,
        )

    def update(
        self,
        batch: ReplayBatch,
    ) -> dict[str, float]:
        """Perform one local DDPG update for each agent."""
        (
            observations,
            actions,
            rewards,
            next_observations,
            terminations,
        ) = self._batch_to_tensors(batch)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        mean_q_values: list[float] = []
        mean_target_q_values: list[float] = []

        for agent_index in range(self.num_agents):
            local_observations = observations[
                :,
                agent_index,
                :,
            ]
            local_actions = actions[
                :,
                agent_index,
                :,
            ]
            local_next_observations = next_observations[
                :,
                agent_index,
                :,
            ]
            local_rewards = rewards[
                :,
                agent_index,
            ].unsqueeze(-1)
            local_terminations = terminations[
                :,
                agent_index,
            ].to(torch.float32).unsqueeze(-1)

            with torch.no_grad():
                target_actions = self.target_actors[
                    agent_index
                ](
                    local_next_observations
                )

                next_q = self.target_critics[
                    agent_index
                ](
                    local_next_observations,
                    target_actions,
                )

                target_q = (
                    local_rewards
                    + self.gamma
                    * (1.0 - local_terminations)
                    * next_q
                )

            critic = self.critics[agent_index]
            current_q = critic(
                local_observations,
                local_actions,
            )

            critic_loss = F.mse_loss(
                current_q,
                target_q,
            )

            critic_optimizer = (
                self.critic_optimizers[agent_index]
            )
            critic_optimizer.zero_grad(
                set_to_none=True
            )
            critic_loss.backward()
            critic_optimizer.step()

            for parameter in critic.parameters():
                parameter.requires_grad_(False)

            try:
                predicted_actions = self.actors[
                    agent_index
                ](
                    local_observations
                )

                actor_loss = -critic(
                    local_observations,
                    predicted_actions,
                ).mean()

                actor_optimizer = (
                    self.actor_optimizers[agent_index]
                )
                actor_optimizer.zero_grad(
                    set_to_none=True
                )
                actor_loss.backward()
                actor_optimizer.step()
            finally:
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

            actor_losses.append(
                float(actor_loss.detach().cpu())
            )
            critic_losses.append(
                float(critic_loss.detach().cpu())
            )
            mean_q_values.append(
                float(current_q.detach().mean().cpu())
            )
            mean_target_q_values.append(
                float(target_q.detach().mean().cpu())
            )

        for agent_index in range(self.num_agents):
            soft_update(
                target=self.target_actors[agent_index],
                source=self.actors[agent_index],
                tau=self.tau,
            )
            soft_update(
                target=self.target_critics[agent_index],
                source=self.critics[agent_index],
                tau=self.tau,
            )

        statistics = {
            "actor_loss": float(
                np.mean(actor_losses)
            ),
            "critic_loss": float(
                np.mean(critic_losses)
            ),
            "mean_q": float(
                np.mean(mean_q_values)
            ),
            "mean_target_q": float(
                np.mean(mean_target_q_values)
            ),
        }

        if not all(
            np.isfinite(value)
            for value in statistics.values()
        ):
            raise FloatingPointError(
                "Independent DDPG update produced "
                "non-finite statistics."
            )

        return statistics

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save all learners and optimizers."""
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "num_agents": self.num_agents,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "gamma": self.gamma,
                "tau": self.tau,
                "exploration_noise_std": (
                    self.exploration_noise_std
                ),
                "actors": [
                    actor.state_dict()
                    for actor in self.actors
                ],
                "critics": [
                    critic.state_dict()
                    for critic in self.critics
                ],
                "target_actors": [
                    actor.state_dict()
                    for actor in self.target_actors
                ],
                "target_critics": [
                    critic.state_dict()
                    for critic in self.target_critics
                ],
                "actor_optimizers": [
                    optimizer.state_dict()
                    for optimizer
                    in self.actor_optimizers
                ],
                "critic_optimizers": [
                    optimizer.state_dict()
                    for optimizer
                    in self.critic_optimizers
                ],
            },
            destination,
        )

    def load(
        self,
        path: str | Path,
        load_optimizers: bool = True,
    ) -> None:
        """Load a checkpoint created by :meth:`save`."""
        checkpoint: dict[str, Any] = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )

        for name in (
            "num_agents",
            "observation_dim",
            "action_dim",
        ):
            if checkpoint[name] != getattr(self, name):
                raise ValueError(
                    f"Checkpoint {name}={checkpoint[name]} "
                    f"does not match current {name}="
                    f"{getattr(self, name)}."
                )

        for agent_index in range(self.num_agents):
            self.actors[agent_index].load_state_dict(
                checkpoint["actors"][agent_index]
            )
            self.critics[agent_index].load_state_dict(
                checkpoint["critics"][agent_index]
            )
            self.target_actors[
                agent_index
            ].load_state_dict(
                checkpoint["target_actors"][
                    agent_index
                ]
            )
            self.target_critics[
                agent_index
            ].load_state_dict(
                checkpoint["target_critics"][
                    agent_index
                ]
            )

            if load_optimizers:
                self.actor_optimizers[
                    agent_index
                ].load_state_dict(
                    checkpoint[
                        "actor_optimizers"
                    ][agent_index]
                )
                self.critic_optimizers[
                    agent_index
                ].load_state_dict(
                    checkpoint[
                        "critic_optimizers"
                    ][agent_index]
                )

    def _batch_to_tensors(
        self,
        batch: ReplayBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        observations = np.asarray(
            batch.observations
        )
        actions = np.asarray(batch.actions)
        rewards = np.asarray(batch.rewards)
        next_observations = np.asarray(
            batch.next_observations
        )
        terminations = np.asarray(
            batch.terminations
        )

        if observations.ndim != 3:
            raise ValueError(
                "Batch observations must have three dimensions."
            )

        batch_size = observations.shape[0]

        expected_shapes = {
            "observations": (
                batch_size,
                self.num_agents,
                self.observation_dim,
            ),
            "actions": (
                batch_size,
                self.num_agents,
                self.action_dim,
            ),
            "rewards": (
                batch_size,
                self.num_agents,
            ),
            "next_observations": (
                batch_size,
                self.num_agents,
                self.observation_dim,
            ),
            "terminations": (
                batch_size,
                self.num_agents,
            ),
        }

        arrays = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminations": terminations,
        }

        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"Batch {name} must have shape "
                    f"{expected_shape}, but received "
                    f"{arrays[name].shape}."
                )

        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
        ):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(
                    f"Batch {name} contains NaN or "
                    "infinite values."
                )

        return (
            torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                next_observations,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                terminations,
                dtype=torch.bool,
                device=self.device,
            ),
        )

    @staticmethod
    def _validate_positive_integer(
        name: str,
        value: int,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 1:
            raise ValueError(
                f"{name} must be at least 1."
            )

    @staticmethod
    def _validate_positive_float(
        name: str,
        value: float,
    ) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be numeric."
            )

        if float(value) <= 0.0:
            raise ValueError(
                f"{name} must be positive."
            )
