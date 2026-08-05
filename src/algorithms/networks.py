
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _validate_dimension(
    name: str,
    value: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 1:
        raise ValueError(
            f"{name} must be at least 1."
        )


def _validate_hidden_sizes(
    hidden_sizes: Sequence[int],
) -> tuple[int, ...]:
    sizes = tuple(hidden_sizes)

    for index, size in enumerate(sizes):
        _validate_dimension(
            f"hidden_sizes[{index}]",
            size,
        )

    return sizes


def _build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a feed-forward ReLU multilayer perceptron."""
    layer_sizes = [
        input_dim,
        *hidden_sizes,
        output_dim,
    ]

    layers: list[nn.Module] = []

    for input_size, output_size in zip(
        layer_sizes[:-2],
        layer_sizes[1:-1],
    ):
        layers.append(
            nn.Linear(
                input_size,
                output_size,
            )
        )
        layers.append(nn.ReLU())

    layers.append(
        nn.Linear(
            layer_sizes[-2],
            layer_sizes[-1],
        )
    )

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic actor using one agent's local observation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()

        _validate_dimension(
            "observation_dim",
            observation_dim,
        )
        _validate_dimension(
            "action_dim",
            action_dim,
        )

        validated_hidden_sizes = (
            _validate_hidden_sizes(hidden_sizes)
        )

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = validated_hidden_sizes

        self.network = _build_mlp(
            input_dim=observation_dim,
            hidden_sizes=validated_hidden_sizes,
            output_dim=action_dim,
            final_activation=nn.Tanh(),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(
                "observations must be a torch.Tensor."
            )

        if observations.ndim not in {1, 2}:
            raise ValueError(
                "Actor observations must have shape "
                "(observation_dim,) or "
                "(batch_size, observation_dim)."
            )

        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "Actor expected observation dimension "
                f"{self.observation_dim}, but received "
                f"{observations.shape[-1]}."
            )

        return self.network(observations)


class Critic(nn.Module):
    """Q-network for observation-action pairs.

    For MADDPG, pass flattened joint observations and
    flattened joint actions.

    For independent DDPG, pass one agent's local observation
    and action.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()

        _validate_dimension(
            "observation_dim",
            observation_dim,
        )
        _validate_dimension(
            "action_dim",
            action_dim,
        )

        validated_hidden_sizes = (
            _validate_hidden_sizes(hidden_sizes)
        )

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = validated_hidden_sizes

        self.network = _build_mlp(
            input_dim=observation_dim + action_dim,
            hidden_sizes=validated_hidden_sizes,
            output_dim=1,
            final_activation=None,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(
                "observations must be a torch.Tensor."
            )

        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                "actions must be a torch.Tensor."
            )

        if observations.ndim == 1:
            observations = observations.unsqueeze(0)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if observations.ndim != 2:
            raise ValueError(
                "Critic observations must have shape "
                "(batch_size, observation_dim)."
            )

        if actions.ndim != 2:
            raise ValueError(
                "Critic actions must have shape "
                "(batch_size, action_dim)."
            )

        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "Observation and action batch sizes must match."
            )

        if observations.shape[1] != self.observation_dim:
            raise ValueError(
                "Critic expected observation dimension "
                f"{self.observation_dim}, but received "
                f"{observations.shape[1]}."
            )

        if actions.shape[1] != self.action_dim:
            raise ValueError(
                "Critic expected action dimension "
                f"{self.action_dim}, but received "
                f"{actions.shape[1]}."
            )

        critic_input = torch.cat(
            (
                observations,
                actions,
            ),
            dim=-1,
        )

        return self.network(critic_input)
