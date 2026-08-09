from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import Geometry, build_geometry
from .payload import Payload


class CooperativeTransportEnv:
    """
    Cooperative multi-agent transport environment.

    Designed for Stage 2 experiments:
        - MADDPG
        - Independent DDPG
        - Hand-crafted force-feedback controller
        - Robustness evaluation

    Core properties
    ---------------
    * N configurable agents.
    * Continuous 2-D actions.
    * One shared payload.
    * Configurable attachment locations.
    * Spring-damper coupling.
    * Configurable payload geometry.
    * Translation + rotation for non-point payloads.
    * Optional obstacles.
    * Shared cooperative reward.
    * Local actor observations.
    * Global state for centralized critics.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        self._load_config()
        self._validate_config()

        self.rng = np.random.default_rng(self.seed)

        # ====================================================
        # Payload
        # ====================================================

        self.payload = Payload(
            config=self.payload_cfg,
            initial_state=self.initial_state_cfg,
        )

        # ====================================================
        # Agents
        # ====================================================

        self.agent_positions = np.zeros(
            (self.num_agents, 2),
            dtype=np.float64,
        )

        self.agent_velocities = np.zeros(
            (self.num_agents, 2),
            dtype=np.float64,
        )

        # Force acting ON each agent from its coupling.
        self.coupling_forces = np.zeros(
            (self.num_agents, 2),
            dtype=np.float64,
        )

        # ====================================================
        # Target
        # ====================================================

        self.target_position = self._vec2(
            self.target_cfg["position"]
        )

        self.target_radius = float(
            self.target_cfg["radius"]
        )

        # ====================================================
        # Obstacles
        # ====================================================

        self.obstacles = self._build_obstacles()

        # ====================================================
        # Episode state
        # ====================================================

        self.step_count = 0
        self.episode_return = 0.0

        self.previous_goal_distance = 0.0

        self.success = False
        self.success_reward_given = False

        self.collision_count = 0

        self.last_actions = np.zeros(
            (self.num_agents, self.action_dim),
            dtype=np.float64,
        )

        # Metric accumulators.
        self.total_control_effort = 0.0
        self.total_force_disagreement = 0.0

        self.reset()

    # ========================================================
    # Configuration
    # ========================================================

    def _load_config(self) -> None:
        self.env_cfg = self.config["env"]
        self.world_cfg = self.config["world"]
        self.agent_cfg = self.config["agent"]
        self.payload_cfg = self.config["payload"]
        self.coupling_cfg = self.config["coupling"]
        self.target_cfg = self.config["target"]
        self.obstacles_cfg = self.config["obstacles"]
        self.initial_state_cfg = self.config["initial_state"]
        self.action_cfg = self.config["action"]
        self.observation_cfg = self.config["observation"]
        self.reward_cfg = self.config["reward"]
        self.termination_cfg = self.config["termination"]
        self.safety_cfg = self.config["numerical_safety"]
        self.evaluation_cfg = self.config.get(
            "evaluation",
            {},
        )

        # Environment.
        self.seed = int(
            self.env_cfg.get("seed", 0)
        )

        self.num_agents = int(
            self.env_cfg["num_agents"]
        )

        self.dt = float(
            self.env_cfg["dt"]
        )

        self.max_steps = int(
            self.env_cfg["max_steps"]
        )

        # World.
        self.x_min = float(
            self.world_cfg["x_min"]
        )
        self.x_max = float(
            self.world_cfg["x_max"]
        )
        self.y_min = float(
            self.world_cfg["y_min"]
        )
        self.y_max = float(
            self.world_cfg["y_max"]
        )

        self.boundary_mode = self.world_cfg.get(
            "boundary_mode",
            "clip",
        )

        # Agent.
        self.agent_mass = float(
            self.agent_cfg["mass"]
        )

        self.agent_max_control_force = float(
            self.agent_cfg["max_control_force"]
        )

        self.agent_max_speed = float(
            self.agent_cfg["max_speed"]
        )

        self.agent_linear_damping = float(
            self.agent_cfg["linear_damping"]
        )

        self.max_attachment_distance = float(
            self.agent_cfg["max_attachment_distance"]
        )

        # Coupling.
        self.spring_constant = float(
            self.coupling_cfg["spring_constant"]
        )

        self.spring_damping = float(
            self.coupling_cfg["spring_damping"]
        )

        self.rest_length = float(
            self.coupling_cfg["rest_length"]
        )

        self.max_coupling_force = float(
            self.coupling_cfg["max_force"]
        )

        self.epsilon = float(
            self.coupling_cfg.get(
                "epsilon",
                1e-8,
            )
        )

        # Action.
        self.action_dim = int(
            self.action_cfg["dimension"]
        )

        self.action_low = float(
            self.action_cfg["low"]
        )

        self.action_high = float(
            self.action_cfg["high"]
        )

    def _validate_config(self) -> None:
        if self.num_agents <= 0:
            raise ValueError(
                "env.num_agents must be positive."
            )

        if self.dt <= 0:
            raise ValueError(
                "env.dt must be positive."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "env.max_steps must be positive."
            )

        if self.x_min >= self.x_max:
            raise ValueError(
                "world.x_min must be smaller than world.x_max."
            )

        if self.y_min >= self.y_max:
            raise ValueError(
                "world.y_min must be smaller than world.y_max."
            )

        if self.boundary_mode != "clip":
            raise ValueError(
                "Current implementation supports only "
                "world.boundary_mode='clip'."
            )

        if self.agent_mass <= 0:
            raise ValueError(
                "agent.mass must be positive."
            )

        if self.agent_max_control_force <= 0:
            raise ValueError(
                "agent.max_control_force must be positive."
            )

        if self.agent_max_speed <= 0:
            raise ValueError(
                "agent.max_speed must be positive."
            )

        if self.agent_linear_damping < 0:
            raise ValueError(
                "agent.linear_damping must be non-negative."
            )

        if self.max_attachment_distance <= 0:
            raise ValueError(
                "agent.max_attachment_distance must be positive."
            )

        if self.spring_constant < 0:
            raise ValueError(
                "coupling.spring_constant must be non-negative."
            )

        if self.spring_damping < 0:
            raise ValueError(
                "coupling.spring_damping must be non-negative."
            )

        if not (
            0.0
            <= self.rest_length
            < self.max_attachment_distance
        ):
            raise ValueError(
                "coupling.rest_length must satisfy "
                "0 <= rest_length < "
                "agent.max_attachment_distance."
            )

        if self.max_coupling_force <= 0:
            raise ValueError(
                "coupling.max_force must be positive."
            )

        if self.action_dim != 2:
            raise ValueError(
                "Current transport environment requires "
                "action.dimension = 2."
            )

        if self.action_low >= self.action_high:
            raise ValueError(
                "action.low must be smaller than action.high."
            )

        if float(self.target_cfg["radius"]) <= 0:
            raise ValueError(
                "target.radius must be positive."
            )

        attachment_offsets = (
            self.payload_cfg
            .get("attachments", {})
            .get("offsets", [])
        )

        if len(attachment_offsets) != self.num_agents:
            raise ValueError(
                "Number of payload attachment offsets must "
                "equal env.num_agents."
            )

        if (
            len(
                self.initial_state_cfg[
                    "agent_positions"
                ]
            )
            != self.num_agents
        ):
            raise ValueError(
                "Number of initial agent positions must "
                "equal env.num_agents."
            )

        if (
            len(
                self.initial_state_cfg[
                    "agent_velocities"
                ]
            )
            != self.num_agents
        ):
            raise ValueError(
                "Number of initial agent velocities must "
                "equal env.num_agents."
            )

        geometry = build_geometry(
            self.payload_cfg["shape"],
            self.payload_cfg.get(
                "dimensions",
                [],
            ),
        )

        attachment_cfg = self.payload_cfg.get(
            "attachments",
            {},
        )

        if attachment_cfg.get(
            "enforce_clearance",
            False,
        ):
            geometry.validate_attachment_clearance(
                attachment_offsets=attachment_offsets,
                max_attachment_distance=(
                    self.max_attachment_distance
                ),
                clearance_margin=float(
                    attachment_cfg.get(
                        "clearance_margin",
                        0.0,
                    )
                ),
            )

    # ========================================================
    # General helpers
    # ========================================================

    @staticmethod
    def _vec2(
        value: Sequence[float],
    ) -> np.ndarray:
        arr = np.asarray(
            value,
            dtype=np.float64,
        )

        if arr.shape != (2,):
            raise ValueError(
                f"Expected 2-D vector, got shape {arr.shape}."
            )

        return arr.copy()

    @staticmethod
    def _clip_vector_norm(
        vector: np.ndarray,
        max_norm: float,
    ) -> np.ndarray:
        norm = float(
            np.linalg.norm(vector)
        )

        if norm > max_norm and norm > 0.0:
            return (
                vector
                * max_norm
                / norm
            )

        return vector

    # ========================================================
    # Obstacles
    # ========================================================

    def _build_obstacles(self) -> list[dict]:
        if not self.obstacles_cfg.get(
            "enabled",
            False,
        ):
            return []

        obstacles = []

        for item in self.obstacles_cfg.get(
            "items",
            [],
        ):
            geometry = build_geometry(
                item["shape"],
                item.get(
                    "dimensions",
                    [],
                ),
            )

            obstacles.append(
                {
                    "geometry": geometry,
                    "position": self._vec2(
                        item["position"]
                    ),
                    "orientation": float(
                        item.get(
                            "orientation",
                            0.0,
                        )
                    ),
                }
            )

        return obstacles

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed: int | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(
                seed
            )

        self.step_count = 0
        self.episode_return = 0.0

        self.success = False
        self.success_reward_given = False

        self.collision_count = 0

        self.total_control_effort = 0.0
        self.total_force_disagreement = 0.0

        self.last_actions[:] = 0.0
        self.coupling_forces[:] = 0.0

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload_position = self._vec2(
            self.initial_state_cfg[
                "payload_position"
            ]
        )

        payload_velocity = self._vec2(
            self.initial_state_cfg[
                "payload_velocity"
            ]
        )

        payload_position += self._sample_vec_noise(
            self.initial_state_cfg.get(
                "payload_position_noise",
                0.0,
            )
        )

        payload_velocity += self._sample_vec_noise(
            self.initial_state_cfg.get(
                "payload_velocity_noise",
                0.0,
            )
        )

        orientation = float(
            self.initial_state_cfg.get(
                "payload_orientation",
                0.0,
            )
        )

        angular_velocity = float(
            self.initial_state_cfg.get(
                "payload_angular_velocity",
                0.0,
            )
        )

        if self.payload.geometry.supports_rotation:
            orientation += self._sample_scalar_noise(
                self.initial_state_cfg.get(
                    "orientation_noise",
                    0.0,
                )
            )

            angular_velocity += (
                self._sample_scalar_noise(
                    self.initial_state_cfg.get(
                        "angular_velocity_noise",
                        0.0,
                    )
                )
            )

        self.payload.reset(
            position=payload_position,
            velocity=payload_velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
        )

        # ----------------------------------------------------
        # Agents
        # ----------------------------------------------------

        self.agent_positions = np.asarray(
            self.initial_state_cfg[
                "agent_positions"
            ],
            dtype=np.float64,
        ).copy()

        self.agent_velocities = np.asarray(
            self.initial_state_cfg[
                "agent_velocities"
            ],
            dtype=np.float64,
        ).copy()

        agent_pos_noise = float(
            self.initial_state_cfg.get(
                "agent_position_noise",
                0.0,
            )
        )

        if agent_pos_noise > 0:
            self.agent_positions += (
                self.rng.uniform(
                    -agent_pos_noise,
                    agent_pos_noise,
                    size=(self.num_agents, 2),
                )
            )

        agent_vel_noise = float(
            self.initial_state_cfg.get(
                "agent_velocity_noise",
                0.0,
            )
        )

        if agent_vel_noise > 0:
            self.agent_velocities += (
                self.rng.uniform(
                    -agent_vel_noise,
                    agent_vel_noise,
                    size=(self.num_agents, 2),
                )
            )

        self._handle_world_boundaries()
        self._enforce_all_attachment_constraints()

        self.previous_goal_distance = (
            self._goal_distance()
        )

        self.success = self._check_success()

        return (
            self._get_all_observations(),
            self._build_info(),
        )

    def _sample_vec_noise(
        self,
        magnitude: float,
    ) -> np.ndarray:
        magnitude = float(magnitude)

        if magnitude <= 0:
            return np.zeros(
                2,
                dtype=np.float64,
            )

        return self.rng.uniform(
            -magnitude,
            magnitude,
            size=2,
        )

    def _sample_scalar_noise(
        self,
        magnitude: float,
    ) -> float:
        magnitude = float(magnitude)

        if magnitude <= 0:
            return 0.0

        return float(
            self.rng.uniform(
                -magnitude,
                magnitude,
            )
        )

    # ========================================================
    # Main environment step
    # ========================================================

    def step(
        self,
        actions: Sequence[Sequence[float]],
    ):
        actions = np.asarray(
            actions,
            dtype=np.float64,
        )

        expected_shape = (
            self.num_agents,
            self.action_dim,
        )

        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape}, "
                f"got {actions.shape}."
            )

        actions = np.clip(
            actions,
            self.action_low,
            self.action_high,
        )

        self.last_actions = actions.copy()

        # Save pre-step state in case obstacle collision
        # requires payload motion rollback.
        payload_snapshot = self._payload_snapshot()

        # ----------------------------------------------------
        # 1. Coupling force at current state
        # ----------------------------------------------------

        self.coupling_forces = (
            self._compute_coupling_forces()
        )

        # ----------------------------------------------------
        # 2. Apply reaction forces to payload
        # ----------------------------------------------------

        attachment_positions = (
            self.payload.get_attachment_positions()
        )

        for i in range(self.num_agents):
            force_on_payload = (
                -self.coupling_forces[i]
            )

            self.payload.apply_force_at_point(
                force=force_on_payload,
                world_point=attachment_positions[i],
            )

        # ----------------------------------------------------
        # 3. Integrate agent dynamics
        # ----------------------------------------------------

        self._integrate_agents(
            actions
        )

        # ----------------------------------------------------
        # 4. Integrate payload dynamics
        # ----------------------------------------------------

        self.payload.integrate(
            dt=self.dt,
            clip_linear_speed=self.safety_cfg.get(
                "clip_payload_speed",
                True,
            ),
            clip_angular_speed=self.safety_cfg.get(
                "clip_angular_speed",
                True,
            ),
        )

        # ----------------------------------------------------
        # 5. Boundaries
        # ----------------------------------------------------

        self._handle_world_boundaries()

        # ----------------------------------------------------
        # 6. Obstacle collision
        # ----------------------------------------------------

        collision = (
            self._payload_collides_with_obstacle()
        )

        if collision:
            self.collision_count += 1

            if (
                self.obstacles_cfg.get(
                    "collision_mode",
                    "reject_motion",
                )
                == "reject_motion"
            ):
                self._restore_payload_snapshot(
                    payload_snapshot
                )

        # Attachment positions may have moved because
        # payload translated/rotated or was rolled back.
        self._enforce_all_attachment_constraints()

        # Re-apply world limits after projection.
        self._handle_world_boundaries()

        # ----------------------------------------------------
        # 7. Advance episode counter
        # ----------------------------------------------------

        self.step_count += 1

        # ----------------------------------------------------
        # 8. Success
        # ----------------------------------------------------

        self.success = self._check_success()

        # ----------------------------------------------------
        # 9. Reward
        # ----------------------------------------------------

        team_reward = self._compute_reward(
            actions=actions,
        )

        self.episode_return += team_reward

        # ----------------------------------------------------
        # 10. Metrics
        # ----------------------------------------------------

        action_effort = float(
            np.sum(
                np.square(actions)
            )
        )

        self.total_control_effort += (
            action_effort
        )

        disagreement = (
            self._force_disagreement()
        )

        self.total_force_disagreement += (
            disagreement
        )

        # ----------------------------------------------------
        # 11. Termination
        # ----------------------------------------------------

        terminated = False

        if (
            self.success
            and self.termination_cfg.get(
                "terminate_on_success",
                True,
            )
        ):
            terminated = True

        if (
            collision
            and self.termination_cfg.get(
                "terminate_on_collision",
                False,
            )
        ):
            terminated = True

        finite_state = (
            self._state_is_finite()
        )

        if (
            not finite_state
            and self.termination_cfg.get(
                "terminate_on_nonfinite_state",
                True,
            )
        ):
            terminated = True

        truncated = bool(
            self.step_count >= self.max_steps
            and not terminated
        )

        # ----------------------------------------------------
        # 12. Output
        # ----------------------------------------------------

        observations = (
            self._get_all_observations()
        )

        rewards = np.full(
            self.num_agents,
            team_reward,
            dtype=np.float64,
        )

        info = self._build_info()

        info["collision"] = bool(collision)
        info["finite_state"] = bool(
            finite_state
        )

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # Payload snapshot
    # ========================================================

    def _payload_snapshot(self) -> dict:
        return {
            "position":
                self.payload.position.copy(),
            "velocity":
                self.payload.velocity.copy(),
            "orientation":
                float(
                    self.payload.orientation
                ),
            "angular_velocity":
                float(
                    self.payload.angular_velocity
                ),
        }

    def _restore_payload_snapshot(
        self,
        snapshot: dict,
    ) -> None:
        self.payload.position = (
            snapshot["position"].copy()
        )

        self.payload.velocity = (
            snapshot["velocity"].copy()
        )

        self.payload.orientation = float(
            snapshot["orientation"]
        )

        self.payload.angular_velocity = float(
            snapshot["angular_velocity"]
        )

        self.payload.clear_forces()

    # ========================================================
    # Coupling force
    # ========================================================

    def _compute_coupling_forces(
        self,
    ) -> np.ndarray:
        """
        Return spring-damper forces acting ON THE AGENTS.

        d = p_agent - p_attachment
        L = ||d||
        n = d / L

        Spring:
            F_s = -k (L - L0) n

        Damping:
            F_d = -c ((v_agent - v_attachment) · n) n
        """

        attachments = (
            self.payload.get_attachment_positions()
        )

        forces = np.zeros(
            (self.num_agents, 2),
            dtype=np.float64,
        )

        for i in range(self.num_agents):
            displacement = (
                self.agent_positions[i]
                - attachments[i]
            )

            length = float(
                np.linalg.norm(
                    displacement
                )
            )

            if length <= self.epsilon:
                direction = np.zeros(
                    2,
                    dtype=np.float64,
                )
            else:
                direction = (
                    displacement / length
                )

            extension = (
                length - self.rest_length
            )

            spring_force = (
                -self.spring_constant
                * extension
                * direction
            )

            attachment_velocity = (
                self._attachment_velocity(i)
            )

            relative_velocity = (
                self.agent_velocities[i]
                - attachment_velocity
            )

            radial_relative_speed = float(
                np.dot(
                    relative_velocity,
                    direction,
                )
            )

            damping_force = (
                -self.spring_damping
                * radial_relative_speed
                * direction
            )

            total_force = (
                spring_force
                + damping_force
            )

            forces[i] = (
                self._clip_vector_norm(
                    total_force,
                    self.max_coupling_force,
                )
            )

        return forces

    def _attachment_velocity(
        self,
        index: int,
    ) -> np.ndarray:
        attachment = (
            self.payload.get_attachment_position(
                index
            )
        )

        r = (
            attachment
            - self.payload.position
        )

        omega = (
            self.payload.angular_velocity
        )

        rotational_velocity = np.array(
            [
                -omega * r[1],
                omega * r[0],
            ],
            dtype=np.float64,
        )

        return (
            self.payload.velocity
            + rotational_velocity
        )

    # ========================================================
    # Agent integration
    # ========================================================

    def _integrate_agents(
        self,
        actions: np.ndarray,
    ) -> None:
        for i in range(self.num_agents):
            control_force = (
                actions[i]
                * self.agent_max_control_force
            )

            damping_force = (
                -self.agent_linear_damping
                * self.agent_velocities[i]
            )

            total_force = (
                control_force
                + self.coupling_forces[i]
                + damping_force
            )

            acceleration = (
                total_force
                / self.agent_mass
            )

            # Semi-implicit Euler.
            self.agent_velocities[i] += (
                acceleration
                * self.dt
            )

            if self.safety_cfg.get(
                "clip_agent_speed",
                True,
            ):
                self.agent_velocities[i] = (
                    self._clip_vector_norm(
                        self.agent_velocities[i],
                        self.agent_max_speed,
                    )
                )

            self.agent_positions[i] += (
                self.agent_velocities[i]
                * self.dt
            )

    # ========================================================
    # Attachment constraint
    # ========================================================

    def _enforce_all_attachment_constraints(
        self,
    ) -> None:
        attachments = (
            self.payload.get_attachment_positions()
        )

        for i in range(self.num_agents):
            delta = (
                self.agent_positions[i]
                - attachments[i]
            )

            distance = float(
                np.linalg.norm(delta)
            )

            if (
                distance
                <= self.max_attachment_distance
            ):
                continue

            if distance <= self.epsilon:
                continue

            direction = (
                delta / distance
            )

            # Project agent onto allowed circle.
            self.agent_positions[i] = (
                attachments[i]
                + direction
                * self.max_attachment_distance
            )

            # Remove outward radial velocity.
            radial_speed = float(
                np.dot(
                    self.agent_velocities[i],
                    direction,
                )
            )

            if radial_speed > 0:
                self.agent_velocities[i] -= (
                    radial_speed
                    * direction
                )

    # ========================================================
    # Boundaries
    # ========================================================

    def _handle_world_boundaries(
        self,
    ) -> None:
        self.payload.position[0] = np.clip(
            self.payload.position[0],
            self.x_min,
            self.x_max,
        )

        self.payload.position[1] = np.clip(
            self.payload.position[1],
            self.y_min,
            self.y_max,
        )

        self.agent_positions[:, 0] = np.clip(
            self.agent_positions[:, 0],
            self.x_min,
            self.x_max,
        )

        self.agent_positions[:, 1] = np.clip(
            self.agent_positions[:, 1],
            self.y_min,
            self.y_max,
        )

    # ========================================================
    # Collision
    # ========================================================

    def _payload_collides_with_obstacle(
        self,
    ) -> bool:
        for obstacle in self.obstacles:
            if self.payload.collides_with(
                other_geometry=obstacle[
                    "geometry"
                ],
                other_position=obstacle[
                    "position"
                ],
                other_orientation=obstacle[
                    "orientation"
                ],
            ):
                return True

        return False

    # ========================================================
    # Target / reward
    # ========================================================

    def _goal_distance(self) -> float:
        return float(
            np.linalg.norm(
                self.payload.position
                - self.target_position
            )
        )

    def _check_success(self) -> bool:
        return bool(
            self._goal_distance()
            <= self.target_radius
        )

    def _compute_reward(
        self,
        actions: np.ndarray,
    ) -> float:
        current_distance = (
            self._goal_distance()
        )

        progress = (
            self.previous_goal_distance
            - current_distance
        )

        reward = (
            float(
                self.reward_cfg.get(
                    "progress_weight",
                    1.0,
                )
            )
            * progress
        )

        # Success bonus only once.
        if (
            self.success
            and not self.success_reward_given
        ):
            reward += float(
                self.reward_cfg.get(
                    "success_bonus",
                    0.0,
                )
            )

            self.success_reward_given = True

        action_penalty_weight = float(
            self.reward_cfg.get(
                "action_penalty_weight",
                0.0,
            )
        )

        if action_penalty_weight > 0:
            reward -= (
                action_penalty_weight
                * float(
                    np.sum(
                        np.square(actions)
                    )
                )
            )

        self.previous_goal_distance = (
            current_distance
        )

        return float(reward)

    # ========================================================
    # Local observations
    # ========================================================

    def _get_observation(
        self,
        agent_index: int,
    ) -> np.ndarray:
        if not (
            0
            <= agent_index
            < self.num_agents
        ):
            raise IndexError(
                "Invalid agent index."
            )

        parts = []

        attachment = (
            self.payload.get_attachment_position(
                agent_index
            )
        )

        if self.observation_cfg.get(
            "include_agent_velocity",
            True,
        ):
            parts.append(
                self.agent_velocities[
                    agent_index
                ]
            )

        if self.observation_cfg.get(
            "include_attachment_relative_position",
            True,
        ):
            parts.append(
                attachment
                - self.agent_positions[
                    agent_index
                ]
            )

        if self.observation_cfg.get(
            "include_payload_relative_velocity",
            True,
        ):
            parts.append(
                self.payload.velocity
                - self.agent_velocities[
                    agent_index
                ]
            )

        if self.observation_cfg.get(
            "include_goal_relative_to_payload",
            True,
        ):
            parts.append(
                self.target_position
                - self.payload.position
            )

        if self.observation_cfg.get(
            "include_coupling_force",
            True,
        ):
            parts.append(
                self.coupling_forces[
                    agent_index
                ]
            )

        if self.observation_cfg.get(
            "include_payload_orientation",
            True,
        ):
            parts.append(
                np.array(
                    [
                        self.payload.orientation
                    ],
                    dtype=np.float64,
                )
            )

        if self.observation_cfg.get(
            "include_payload_angular_velocity",
            True,
        ):
            parts.append(
                np.array(
                    [
                        self.payload.angular_velocity
                    ],
                    dtype=np.float64,
                )
            )

        if not parts:
            return np.empty(
                0,
                dtype=np.float64,
            )

        return np.concatenate(
            parts
        ).astype(
            np.float64,
            copy=False,
        )

    def _get_all_observations(
        self,
    ) -> np.ndarray:
        return np.stack(
            [
                self._get_observation(i)
                for i in range(
                    self.num_agents
                )
            ],
            axis=0,
        )

    # ========================================================
    # Centralized state
    # ========================================================

    def global_state_vector(
        self,
    ) -> np.ndarray:
        """
        Flat state suitable for centralized critics.

        Contains:
            all agent positions
            all agent velocities
            payload position
            payload velocity
            payload orientation
            payload angular velocity
            target relative position

        Point payload keeps orientation/angular velocity at zero,
        preserving state dimension consistency.
        """

        return np.concatenate(
            [
                self.agent_positions.reshape(-1),
                self.agent_velocities.reshape(-1),
                self.payload.position,
                self.payload.velocity,
                np.array(
                    [
                        self.payload.orientation,
                        self.payload.angular_velocity,
                    ],
                    dtype=np.float64,
                ),
                (
                    self.target_position
                    - self.payload.position
                ),
            ]
        )

    def state(self) -> dict:
        return {
            "agent_positions":
                self.agent_positions.copy(),
            "agent_velocities":
                self.agent_velocities.copy(),
            "payload":
                self.payload.state_dict(),
            "target_position":
                self.target_position.copy(),
            "coupling_forces":
                self.coupling_forces.copy(),
        }

    # ========================================================
    # Metrics
    # ========================================================

    def _force_disagreement(
        self,
    ) -> float:
        if self.num_agents <= 1:
            return 0.0

        mean_force = np.mean(
            self.coupling_forces,
            axis=0,
        )

        return float(
            np.mean(
                np.linalg.norm(
                    self.coupling_forces
                    - mean_force,
                    axis=1,
                )
            )
        )

    def _mean_attachment_distance(
        self,
    ) -> float:
        attachments = (
            self.payload.get_attachment_positions()
        )

        distances = np.linalg.norm(
            self.agent_positions
            - attachments,
            axis=1,
        )

        return float(
            np.mean(distances)
        )

    def _state_is_finite(
        self,
    ) -> bool:
        if not self.payload.is_finite():
            return False

        if not np.all(
            np.isfinite(
                self.agent_positions
            )
        ):
            return False

        if not np.all(
            np.isfinite(
                self.agent_velocities
            )
        ):
            return False

        if not np.all(
            np.isfinite(
                self.coupling_forces
            )
        ):
            return False

        return True

    def _build_info(self) -> dict:
        mean_force_disagreement = (
            self.total_force_disagreement
            / max(self.step_count, 1)
        )

        return {
            "success": bool(
                self.success
            ),
            "step_count": int(
                self.step_count
            ),
            "episode_return": float(
                self.episode_return
            ),
            "payload_distance_to_goal":
                self._goal_distance(),
            "payload_position":
                self.payload.position.copy(),
            "payload_velocity":
                self.payload.velocity.copy(),
            "payload_orientation":
                float(
                    self.payload.orientation
                ),
            "payload_angular_velocity":
                float(
                    self.payload.angular_velocity
                ),
            "coupling_forces":
                self.coupling_forces.copy(),
            "force_disagreement":
                self._force_disagreement(),
            "mean_force_disagreement":
                float(
                    mean_force_disagreement
                ),
            "total_control_effort":
                float(
                    self.total_control_effort
                ),
            "mean_attachment_distance":
                self._mean_attachment_distance(),
            "collision_count":
                int(
                    self.collision_count
                ),
        }

    # ========================================================
    # Dimensions
    # ========================================================

    @property
    def observation_dim(
        self,
    ) -> int:
        return int(
            self._get_observation(
                0
            ).shape[0]
        )

    @property
    def global_state_dim(
        self,
    ) -> int:
        return int(
            self.global_state_vector().shape[0]
        )

    @property
    def joint_action_dim(
        self,
    ) -> int:
        return (
            self.num_agents
            * self.action_dim
        )

    # ========================================================
    # Render / close
    # ========================================================

    def render(self):
        """
        Rendering intentionally kept separate from training
        and physics.

        Can later be implemented with matplotlib / pygame.
        """
        raise NotImplementedError(
            "Rendering has not been implemented."
        )

    def close(self) -> None:
        pass
