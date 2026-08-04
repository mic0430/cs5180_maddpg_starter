from __future__ import annotations

from typing import Any

import numpy as np
from mpe2 import simple_spread_v3
from numpy.typing import NDArray
from pettingzoo.utils.env import ParallelEnv


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


class SimpleSpreadWrapper:
    """MADDPG-friendly wrapper around MPE2 simple_spread_v3.

    The wrapper:
    - uses PettingZoo's Parallel API;
    - enables continuous actions;
    - preserves a fixed agent ordering;
    - converts observation/reward dictionaries into arrays;
    - optionally converts 2D force actions into MPE2's 5D actions;
    - maps tanh actor outputs from [-1, 1] to the environment action bounds.
    """

    def __init__(
        self,
        num_agents: int = 3,
        max_cycles: int = 25,
        local_ratio: float = 0.5,
        use_2d_actions: bool = True,
        render_mode: str | None = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")

        if not 0.0 <= local_ratio <= 1.0:
            raise ValueError("local_ratio must be between 0 and 1.")

        self.use_2d_actions = use_2d_actions

        self.env: ParallelEnv = simple_spread_v3.parallel_env(
            N=num_agents,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=True,
            render_mode=render_mode,
            curriculum=False,
            terminate_on_success=False,
        )

        # Fixed ordering used by the wrapper and replay buffer.
        self.agent_names = list(self.env.possible_agents)
        self.num_agents = len(self.agent_names)

        if self.num_agents == 0:
            raise ValueError("The environment contains no possible agents.")

        self.agent_to_index = {
            agent_name: index
            for index, agent_name in enumerate(self.agent_names)
        }

        first_agent = self.agent_names[0]

        observation_space = self.env.observation_space(first_agent)
        action_space = self.env.action_space(first_agent)

        if observation_space.shape is None:
            raise ValueError(
                "SimpleSpreadWrapper requires an observation space "
                "with a fixed shape."
            )

        if action_space.shape is None:
            raise ValueError(
                "SimpleSpreadWrapper requires a continuous action space "
                "with a fixed shape."
            )

        self.observation_dim = int(
            np.prod(observation_space.shape)
        )

        self.environment_action_dim = int(
            np.prod(action_space.shape)
        )

        # The actor outputs either:
        # - [ux, uy] in [-1, 1], or
        # - the original MPE2 action dimension, also assumed to come
        #   from a tanh output in [-1, 1].
        self.action_dim = (
            2
            if self.use_2d_actions
            else self.environment_action_dim
        )

        if self.use_2d_actions and self.environment_action_dim != 5:
            raise ValueError(
                "use_2d_actions=True requires an MPE2-style "
                f"5D action space, but received "
                f"{self.environment_action_dim} dimensions."
            )

        # Copy the environment bounds so they remain stable and float32.
        self.env_act_low = np.ascontiguousarray(
            action_space.low,
            dtype=np.float32,
        ).reshape(-1)

        self.env_act_high = np.ascontiguousarray(
            action_space.high,
            dtype=np.float32,
        ).reshape(-1)

        if self.env_act_low.shape != (
            self.environment_action_dim,
        ):
            raise ValueError(
                "Unexpected action-space lower-bound shape: "
                f"{self.env_act_low.shape}."
            )

        if self.env_act_high.shape != (
            self.environment_action_dim,
        ):
            raise ValueError(
                "Unexpected action-space upper-bound shape: "
                f"{self.env_act_high.shape}."
            )

        self.global_observation_dim = (
            self.num_agents * self.observation_dim
        )

        self.joint_action_dim = (
            self.num_agents * self.action_dim
        )

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        observations, infos = self.env.reset(seed=seed)

        observation_array = self._observations_to_array(
            observations
        )

        return observation_array, infos

    def step(
        self,
        actions: FloatArray,
    ) -> tuple[
        FloatArray,
        FloatArray,
        BoolArray,
        BoolArray,
        dict[str, Any],
    ]:
        actions = np.ascontiguousarray(
            actions,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_agents,
            self.action_dim,
        )

        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape}, "
                f"but received {actions.shape}."
            )

        if not np.all(np.isfinite(actions)):
            raise ValueError(
                "Actions contain NaN or infinite values."
            )

        if self.use_2d_actions:
            env_actions = self._convert_2d_actions(
                actions
            )
        else:
            env_actions = self._scale_actor_actions(
                actions
            )

        # PettingZoo expects actions only for agents that are
        # currently active in the environment.
        action_dict = {
            agent_name: env_actions[
                self.agent_to_index[agent_name]
            ]
            for agent_name in self.env.agents
        }

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action_dict)

        next_observation_array = (
            self._observations_to_array(
                next_observations
            )
        )

        reward_array = np.ascontiguousarray(
            [
                rewards.get(agent_name, 0.0)
                for agent_name in self.agent_names
            ],
            dtype=np.float32,
        )

        termination_array = np.ascontiguousarray(
            [
                terminations.get(agent_name, True)
                for agent_name in self.agent_names
            ],
            dtype=np.bool_,
        )

        truncation_array = np.ascontiguousarray(
            [
                truncations.get(agent_name, True)
                for agent_name in self.agent_names
            ],
            dtype=np.bool_,
        )

        return (
            next_observation_array,
            reward_array,
            termination_array,
            truncation_array,
            infos,
        )

    def _observations_to_array(
        self,
        observations: dict[str, np.ndarray],
    ) -> FloatArray:
        """Return observations in fixed agent order.

        Missing observations are represented by zero vectors.
        Every observation is flattened to shape
        ``(observation_dim,)``.
        """
        result: list[FloatArray] = []

        for agent_name in self.agent_names:
            observation = observations.get(agent_name)

            if observation is None:
                observation_array = np.zeros(
                    self.observation_dim,
                    dtype=np.float32,
                )
            else:
                observation_array = np.asarray(
                    observation,
                    dtype=np.float32,
                ).reshape(-1)

                if (
                    observation_array.size
                    != self.observation_dim
                ):
                    raise ValueError(
                        f"Observation for {agent_name} has "
                        f"{observation_array.size} elements, "
                        f"but expected "
                        f"{self.observation_dim}."
                    )

            result.append(observation_array)

        stacked = np.stack(result, axis=0)

        return np.ascontiguousarray(
            stacked,
            dtype=np.float32,
        )

    def _scale_actor_actions(
        self,
        actions: FloatArray,
    ) -> FloatArray:
        """Map tanh actor outputs from [-1, 1] to env bounds.

        For an environment range [low, high], the mapping is:

            low + 0.5 * (action + 1) * (high - low)

        For MPE2's [0, 1] action space:

            -1 -> 0
             0 -> 0.5
             1 -> 1
        """
        clipped_actions = np.clip(
            actions,
            -1.0,
            1.0,
        )

        env_actions = (
            self.env_act_low
            + 0.5
            * (clipped_actions + 1.0)
            * (
                self.env_act_high
                - self.env_act_low
            )
        )

        # A final clip protects against tiny floating-point
        # overshoots near the boundaries.
        env_actions = np.clip(
            env_actions,
            self.env_act_low,
            self.env_act_high,
        )

        return np.ascontiguousarray(
            env_actions,
            dtype=np.float32,
        )

    def _convert_2d_actions(
        self,
        actions: FloatArray,
    ) -> FloatArray:
        """Convert all [ux, uy] actions to MPE2's 5D format.

        Input shape:
            (num_agents, 2)

        Output shape:
            (num_agents, 5)

        Output order:
            [no_action, left, right, down, up]
        """
        clipped_actions = np.clip(
            actions,
            -1.0,
            1.0,
        )

        ux = clipped_actions[:, 0]
        uy = clipped_actions[:, 1]

        env_actions = np.zeros(
            (
                self.num_agents,
                self.environment_action_dim,
            ),
            dtype=np.float32,
        )

        env_actions[:, 0] = 0.0
        env_actions[:, 1] = np.maximum(-ux, 0.0)
        env_actions[:, 2] = np.maximum(ux, 0.0)
        env_actions[:, 3] = np.maximum(-uy, 0.0)
        env_actions[:, 4] = np.maximum(uy, 0.0)

        return np.ascontiguousarray(
            env_actions,
            dtype=np.float32,
        )

    def global_state(
        self,
        observations: FloatArray,
    ) -> FloatArray:
        """Flatten all agents' observations for a centralized critic."""
        observations = np.ascontiguousarray(
            observations,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_agents,
            self.observation_dim,
        )

        if observations.shape != expected_shape:
            raise ValueError(
                f"Expected observations with shape "
                f"{expected_shape}, but received "
                f"{observations.shape}."
            )

        return np.ascontiguousarray(
            observations.reshape(-1),
            dtype=np.float32,
        )

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()