from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import Geometry, build_geometry, cross_2d


class Payload:
    """
    Physical payload used by the cooperative transport environment.

    Responsibilities:
        - store translational state
        - store rotational state
        - own payload mass and damping parameters
        - own payload geometry
        - compute attachment world positions
        - integrate force / torque over time
        - delegate collision checks to Geometry

    Payload does NOT handle:
        - agents
        - spring/coupling force calculation
        - reward
        - observation
        - termination
        - target logic
    """

    def __init__(
        self,
        config: dict,
        initial_state: dict,
    ) -> None:
        """
        Parameters
        ----------
        config:
            payload section of the environment config.

            Expected fields include:
                shape
                dimensions
                mass
                linear_damping
                max_speed
                angular_damping
                max_angular_speed
                attachments.offsets

        initial_state:
            initial_state section of the environment config.

            Expected fields include:
                payload_position
                payload_velocity
                payload_orientation
                payload_angular_velocity
        """

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        self.geometry: Geometry = build_geometry(
            shape=config["shape"],
            dimensions=config.get("dimensions", []),
        )

        # ----------------------------------------------------
        # Physical parameters
        # ----------------------------------------------------

        self.mass = float(config["mass"])

        if self.mass <= 0:
            raise ValueError("Payload mass must be positive.")

        self.linear_damping = float(
            config.get("linear_damping", 0.0)
        )

        self.max_speed = float(
            config.get("max_speed", np.inf)
        )

        self.angular_damping = float(
            config.get("angular_damping", 0.0)
        )

        self.max_angular_speed = float(
            config.get("max_angular_speed", np.inf)
        )

        if self.linear_damping < 0:
            raise ValueError(
                "Payload linear_damping must be non-negative."
            )

        if self.angular_damping < 0:
            raise ValueError(
                "Payload angular_damping must be non-negative."
            )

        if self.max_speed <= 0:
            raise ValueError(
                "Payload max_speed must be positive."
            )

        if self.max_angular_speed <= 0:
            raise ValueError(
                "Payload max_angular_speed must be positive."
            )

        # ----------------------------------------------------
        # Moment of inertia
        # ----------------------------------------------------

        if self.geometry.supports_rotation:
            self.inertia = self.geometry.compute_inertia(
                self.mass
            )
        else:
            self.inertia = None

        # ----------------------------------------------------
        # Attachments
        # ----------------------------------------------------

        attachments_cfg = config.get(
            "attachments",
            {},
        )

        self.attachment_offsets = np.asarray(
            attachments_cfg.get("offsets", []),
            dtype=np.float64,
        )

        if self.attachment_offsets.ndim != 2:
            raise ValueError(
                "attachment_offsets must be a 2-D array."
            )

        if (
            len(self.attachment_offsets) > 0
            and self.attachment_offsets.shape[1] != 2
        ):
            raise ValueError(
                "Each attachment offset must have exactly "
                "two coordinates [x, y]."
            )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.position = self._vec2(
            initial_state.get(
                "payload_position",
                [0.0, 0.0],
            )
        )

        self.velocity = self._vec2(
            initial_state.get(
                "payload_velocity",
                [0.0, 0.0],
            )
        )

        if self.geometry.supports_rotation:
            self.orientation = float(
                initial_state.get(
                    "payload_orientation",
                    0.0,
                )
            )

            self.angular_velocity = float(
                initial_state.get(
                    "payload_angular_velocity",
                    0.0,
                )
            )

        else:
            # Point payload has no rotational dynamics.
            self.orientation = 0.0
            self.angular_velocity = 0.0

        # ----------------------------------------------------
        # Force accumulators
        # ----------------------------------------------------

        self.force_accumulator = np.zeros(
            2,
            dtype=np.float64,
        )

        self.torque_accumulator = 0.0

    # ========================================================
    # Utility
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
                f"Expected shape (2,), got {arr.shape}."
            )

        return arr.copy()

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        position: Sequence[float],
        velocity: Sequence[float] = (0.0, 0.0),
        orientation: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> None:
        """
        Reset payload physical state.
        """

        self.position = self._vec2(position)
        self.velocity = self._vec2(velocity)

        if self.geometry.supports_rotation:
            self.orientation = float(orientation)
            self.angular_velocity = float(
                angular_velocity
            )
        else:
            self.orientation = 0.0
            self.angular_velocity = 0.0

        self.clear_forces()

    # ========================================================
    # Force / torque accumulation
    # ========================================================

    def clear_forces(self) -> None:
        """
        Clear accumulated external force and torque.
        """

        self.force_accumulator[:] = 0.0
        self.torque_accumulator = 0.0

    def apply_force(
        self,
        force: Sequence[float],
    ) -> None:
        """
        Apply a force at the payload center.

        A force applied at the center contributes no torque.
        """

        force = self._vec2(force)

        self.force_accumulator += force

    def apply_force_at_point(
        self,
        force: Sequence[float],
        world_point: Sequence[float],
    ) -> None:
        """
        Apply a force at a world-coordinate point.

        This contributes:
            - linear force
            - torque about the payload center

        Torque in 2-D:
            tau = r x F

        where:
            r = world_point - payload_center
        """

        force = self._vec2(force)
        world_point = self._vec2(world_point)

        self.force_accumulator += force

        if self.geometry.supports_rotation:
            lever_arm = (
                world_point - self.position
            )

            self.torque_accumulator += cross_2d(
                lever_arm,
                force,
            )

    def apply_torque(
        self,
        torque: float,
    ) -> None:
        """
        Apply a direct torque to the payload.

        Ignored by point payloads.
        """

        if not self.geometry.supports_rotation:
            return

        self.torque_accumulator += float(torque)

    # ========================================================
    # Attachment positions
    # ========================================================

    @property
    def num_attachments(self) -> int:
        return len(self.attachment_offsets)

    def get_attachment_position(
        self,
        index: int,
    ) -> np.ndarray:
        """
        Return one attachment position in world coordinates.
        """

        if not 0 <= index < self.num_attachments:
            raise IndexError(
                f"Attachment index {index} is out of range."
            )

        return self.geometry.attachment_world_position(
            attachment_offset=self.attachment_offsets[index],
            payload_position=self.position,
            payload_orientation=self.orientation,
        )

    def get_attachment_positions(
        self,
    ) -> np.ndarray:
        """
        Return all attachment positions.

        Shape:
            (num_attachments, 2)
        """

        if self.num_attachments == 0:
            return np.empty(
                (0, 2),
                dtype=np.float64,
            )

        return np.stack(
            [
                self.get_attachment_position(i)
                for i in range(
                    self.num_attachments
                )
            ],
            axis=0,
        )

    # ========================================================
    # Dynamics
    # ========================================================

    def integrate(
        self,
        dt: float,
        clip_linear_speed: bool = True,
        clip_angular_speed: bool = True,
    ) -> None:
        """
        Advance payload dynamics by one simulation step.

        Semi-implicit Euler integration:

            a = F / m
            v_(t+1) = v_t + a * dt
            p_(t+1) = p_t + v_(t+1) * dt

        Rotation:

            alpha = tau / I
            omega_(t+1) = omega_t + alpha * dt
            theta_(t+1) = theta_t + omega_(t+1) * dt
        """

        dt = float(dt)

        if dt <= 0:
            raise ValueError(
                "dt must be positive."
            )

        # ----------------------------------------------------
        # Linear dynamics
        # ----------------------------------------------------

        damping_force = (
            -self.linear_damping
            * self.velocity
        )

        total_force = (
            self.force_accumulator
            + damping_force
        )

        acceleration = (
            total_force / self.mass
        )

        self.velocity += acceleration * dt

        if clip_linear_speed:
            self._clip_linear_speed()

        self.position += (
            self.velocity * dt
        )

        # ----------------------------------------------------
        # Angular dynamics
        # ----------------------------------------------------

        if self.geometry.supports_rotation:
            damping_torque = (
                -self.angular_damping
                * self.angular_velocity
            )

            total_torque = (
                self.torque_accumulator
                + damping_torque
            )

            angular_acceleration = (
                total_torque / self.inertia
            )

            self.angular_velocity += (
                angular_acceleration * dt
            )

            if clip_angular_speed:
                self._clip_angular_speed()

            self.orientation += (
                self.angular_velocity * dt
            )

            self.orientation = (
                self._wrap_angle(
                    self.orientation
                )
            )

        else:
            self.orientation = 0.0
            self.angular_velocity = 0.0

        # Forces apply only for one step.
        self.clear_forces()

    # ========================================================
    # Speed clipping
    # ========================================================

    def _clip_linear_speed(self) -> None:
        speed = float(
            np.linalg.norm(
                self.velocity
            )
        )

        if (
            speed > self.max_speed
            and speed > 0.0
        ):
            self.velocity *= (
                self.max_speed / speed
            )

    def _clip_angular_speed(self) -> None:
        self.angular_velocity = float(
            np.clip(
                self.angular_velocity,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
        )

    @staticmethod
    def _wrap_angle(
        angle: float,
    ) -> float:
        """
        Wrap angle to [-pi, pi).
        """

        return float(
            (angle + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )

    # ========================================================
    # Collision
    # ========================================================

    def collides_with(
        self,
        other_geometry: Geometry,
        other_position: Sequence[float],
        other_orientation: float = 0.0,
        *,
        position: Sequence[float] | None = None,
        orientation: float | None = None,
    ) -> bool:
        """
        Check collision against another geometry.

        Optional position/orientation arguments allow collision
        checking for a candidate future payload pose without
        modifying the real payload state.

        This is useful for collision_mode="reject_motion".
        """

        check_position = (
            self.position
            if position is None
            else self._vec2(position)
        )

        check_orientation = (
            self.orientation
            if orientation is None
            else float(orientation)
        )

        if not self.geometry.supports_rotation:
            check_orientation = 0.0

        return self.geometry.collides_with(
            position=check_position,
            orientation=check_orientation,
            other=other_geometry,
            other_position=other_position,
            other_orientation=other_orientation,
        )

    # ========================================================
    # State helpers
    # ========================================================

    def is_finite(self) -> bool:
        """
        Return False if the payload contains NaN or Inf.
        """

        values = [
            *self.position,
            *self.velocity,
            self.orientation,
            self.angular_velocity,
        ]

        return bool(
            np.all(
                np.isfinite(values)
            )
        )

    def state_dict(self) -> dict:
        """
        Return a copy of payload state for logging / debugging.
        """

        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": float(
                self.orientation
            ),
            "angular_velocity": float(
                self.angular_velocity
            ),
            "mass": float(
                self.mass
            ),
            "inertia": (
                None
                if self.inertia is None
                else float(self.inertia)
            ),
            "shape": self.geometry.shape_name,
            "attachment_positions":
                self.get_attachment_positions(),
        }
