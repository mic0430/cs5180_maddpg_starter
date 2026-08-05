from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ReplayBatch:
    """A batch of multi-agent transitions sampled from replay memory."""

    observations: FloatArray
    actions: FloatArray
    rewards: FloatArray
    next_observations: FloatArray
    terminations: BoolArray
    truncations: BoolArray

    @property
    def dones(self) -> BoolArray:
        """Return termination-or-truncation masks.

        Use this to decide whether an episode interaction loop should stop.
        For Bellman bootstrapping, many modern implementations use
        `terminations` rather than this combined mask.
        """
        return np.logical_or(
            self.terminations,
            self.truncations,
        )


class MultiAgentReplayBuffer:
    """Fixed-size replay buffer for multi-agent continuous control.

    Each stored transition has the following shapes:

        observations:      (num_agents, observation_dim)
        actions:           (num_agents, action_dim)
        rewards:           (num_agents,)
        next_observations: (num_agents, observation_dim)
        terminations:      (num_agents,)
        truncations:       (num_agents,)

    Sampling returns the same data with a batch dimension added first.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        seed: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")

        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")

        if observation_dim < 1:
            raise ValueError("observation_dim must be at least 1.")

        if action_dim < 1:
            raise ValueError("action_dim must be at least 1.")

        self.capacity = capacity
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self._observations = np.empty(
            (
                capacity,
                num_agents,
                observation_dim,
            ),
            dtype=np.float32,
        )

        self._actions = np.empty(
            (
                capacity,
                num_agents,
                action_dim,
            ),
            dtype=np.float32,
        )

        self._rewards = np.empty(
            (
                capacity,
                num_agents,
            ),
            dtype=np.float32,
        )

        self._next_observations = np.empty(
            (
                capacity,
                num_agents,
                observation_dim,
            ),
            dtype=np.float32,
        )

        self._terminations = np.empty(
            (
                capacity,
                num_agents,
            ),
            dtype=np.bool_,
        )

        self._truncations = np.empty(
            (
                capacity,
                num_agents,
            ),
            dtype=np.bool_,
        )

        self._position = 0
        self._size = 0

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Return whether the buffer has reached its capacity."""
        return self._size == self.capacity

    @property
    def position(self) -> int:
        """Return the index where the next transition will be written."""
        return self._position

    def add(
        self,
        observations: FloatArray,
        actions: FloatArray,
        rewards: FloatArray,
        next_observations: FloatArray,
        terminations: BoolArray,
        truncations: BoolArray,
    ) -> None:
        """Store one multi-agent transition.

        When capacity is reached, the oldest transitions are overwritten
        using a circular-buffer strategy.
        """
        observations = self._prepare_float_array(
            name="observations",
            value=observations,
            expected_shape=(
                self.num_agents,
                self.observation_dim,
            ),
        )

        actions = self._prepare_float_array(
            name="actions",
            value=actions,
            expected_shape=(
                self.num_agents,
                self.action_dim,
            ),
        )

        rewards = self._prepare_float_array(
            name="rewards",
            value=rewards,
            expected_shape=(self.num_agents,),
        )

        next_observations = self._prepare_float_array(
            name="next_observations",
            value=next_observations,
            expected_shape=(
                self.num_agents,
                self.observation_dim,
            ),
        )

        terminations = self._prepare_bool_array(
            name="terminations",
            value=terminations,
            expected_shape=(self.num_agents,),
        )

        truncations = self._prepare_bool_array(
            name="truncations",
            value=truncations,
            expected_shape=(self.num_agents,),
        )

        index = self._position

        self._observations[index] = observations
        self._actions[index] = actions
        self._rewards[index] = rewards
        self._next_observations[index] = (
            next_observations
        )
        self._terminations[index] = terminations
        self._truncations[index] = truncations

        self._position = (
            self._position + 1
        ) % self.capacity

        self._size = min(
            self._size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
        replace: bool = False,
    ) -> ReplayBatch:
        """Sample a random batch of transitions.

        Args:
            batch_size:
                Number of transitions to sample.
            replace:
                Whether the same transition may be selected more than
                once in a batch.

        Returns:
            A ReplayBatch containing contiguous NumPy arrays.
        """
        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if self._size == 0:
            raise ValueError(
                "Cannot sample from an empty replay buffer."
            )

        if not replace and batch_size > self._size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} "
                f"without replacement from a buffer "
                f"containing only {self._size} transitions."
            )

        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=replace,
        )

        return ReplayBatch(
            observations=np.ascontiguousarray(
                self._observations[indices],
                dtype=np.float32,
            ),
            actions=np.ascontiguousarray(
                self._actions[indices],
                dtype=np.float32,
            ),
            rewards=np.ascontiguousarray(
                self._rewards[indices],
                dtype=np.float32,
            ),
            next_observations=np.ascontiguousarray(
                self._next_observations[indices],
                dtype=np.float32,
            ),
            terminations=np.ascontiguousarray(
                self._terminations[indices],
                dtype=np.bool_,
            ),
            truncations=np.ascontiguousarray(
                self._truncations[indices],
                dtype=np.bool_,
            ),
        )

    def clear(self) -> None:
        """Remove all stored transitions without reallocating memory."""
        self._position = 0
        self._size = 0

    def reseed(
        self,
        seed: int | None,
    ) -> None:
        """Reset the random number generator used for sampling."""
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _prepare_float_array(
        name: str,
        value: NDArray[np.generic],
        expected_shape: tuple[int, ...],
    ) -> FloatArray:
        """Validate and convert an input to contiguous float32."""
        array = np.ascontiguousarray(
            value,
            dtype=np.float32,
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, but received "
                f"{array.shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return array

    @staticmethod
    def _prepare_bool_array(
        name: str,
        value: NDArray[np.generic],
        expected_shape: tuple[int, ...],
    ) -> BoolArray:
        """Validate and convert an input to contiguous bool values."""
        array = np.ascontiguousarray(
            value,
            dtype=np.bool_,
        )

        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, but received "
                f"{array.shape}."
            )

        return array