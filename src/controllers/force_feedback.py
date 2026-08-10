"""Decentralized hand-crafted force-feedback controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceFeedbackGains:
    """Simulation-specific gains for the hand-crafted controller."""

    goal: float = 2.00
    attachment: float = 0.35
    relative_velocity: float = 0.20
    coupling_force: float = 0.15
    agent_damping: float = 0.10
    action_limit: float = 1.0


class ForceFeedbackController:
    """
    Decentralized controller using one agent's local observation only.

    Expected observation layout:

        0:2   agent velocity
        2:4   attachment position - agent position
        4:6   payload velocity - agent velocity
        6:8   target position - payload position
        8:10  coupling force acting on this agent
        10    payload orientation
        11    payload angular velocity

    The first controller version intentionally does not use the final
    two rotational terms. Rotation stabilization can be added after the
    basic transport behavior is validated.
    """

    observation_dim = 12
    action_dim = 2

    def __init__(
        self,
        gains: ForceFeedbackGains | None = None,
    ) -> None:
        self.gains = gains or ForceFeedbackGains()

        if self.gains.action_limit <= 0.0:
            raise ValueError(
                "action_limit must be positive."
            )

    def select_action(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return one bounded 2-D action from one local observation."""

        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        if observation.shape != (self.observation_dim,):
            raise ValueError(
                "Expected observation shape "
                f"({self.observation_dim},), "
                f"got {observation.shape}."
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError(
                "Observation must contain only finite values."
            )

        agent_velocity = observation[0:2]
        attachment_relative_position = observation[2:4]
        payload_relative_velocity = observation[4:6]
        goal_relative_to_payload = observation[6:8]
        coupling_force = observation[8:10]

        action = (
            self.gains.goal
            * goal_relative_to_payload
            + self.gains.attachment
            * attachment_relative_position
            + self.gains.relative_velocity
            * payload_relative_velocity
            + self.gains.coupling_force
            * coupling_force
            - self.gains.agent_damping
            * agent_velocity
        )

        return np.clip(
            action,
            -self.gains.action_limit,
            self.gains.action_limit,
        ).astype(
            np.float64,
            copy=False,
        )

    def select_actions(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Return one decentralized action for every agent."""

        observations = np.asarray(
            observations,
            dtype=np.float64,
        )

        if observations.ndim != 2:
            raise ValueError(
                "Expected observations to have shape "
                "(num_agents, observation_dim)."
            )

        if observations.shape[1] != self.observation_dim:
            raise ValueError(
                "Expected observation dimension "
                f"{self.observation_dim}, "
                f"got {observations.shape[1]}."
            )

        return np.stack(
            [
                self.select_action(observation)
                for observation in observations
            ],
            axis=0,
        )
