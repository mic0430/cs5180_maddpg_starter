import math

import numpy as np
import pytest

from src.envs.payload import Payload
from src.envs.geometry import RectangleGeometry


# ============================================================
# Helpers
# ============================================================


def make_rectangle_config():
    """
    Basic rotating rectangular payload configuration.
    """
    return {
        "shape": "rectangle",
        "dimensions": [0.60, 0.30],
        "mass": 3.0,
        "linear_damping": 0.0,
        "max_speed": 2.0,
        "angular_damping": 0.0,
        "max_angular_speed": 4.0,
        "attachments": {
            "offsets": [
                [-0.70, 0.00],
                [0.70, 0.00],
            ]
        },
    }


def make_point_config():
    """
    Basic point payload configuration.
    """
    return {
        "shape": "point",
        "dimensions": [],
        "mass": 3.0,
        "linear_damping": 0.0,
        "max_speed": 2.0,
        "angular_damping": 0.0,
        "max_angular_speed": 4.0,
        "attachments": {
            "offsets": [
                [-0.70, 0.00],
                [0.70, 0.00],
            ]
        },
    }


def make_initial_state():
    return {
        "payload_position": [0.0, 0.0],
        "payload_velocity": [0.0, 0.0],
        "payload_orientation": 0.0,
        "payload_angular_velocity": 0.0,
    }


# ============================================================
# Initialization
# ============================================================


def test_rectangle_payload_initialization():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    assert payload.mass == pytest.approx(3.0)

    np.testing.assert_allclose(
        payload.position,
        [0.0, 0.0],
    )

    np.testing.assert_allclose(
        payload.velocity,
        [0.0, 0.0],
    )

    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)

    assert payload.geometry.shape_name == "rectangle"
    assert payload.geometry.supports_rotation is True

    assert payload.inertia is not None


def test_point_payload_initialization():
    state = make_initial_state()

    state["payload_orientation"] = 1.0
    state["payload_angular_velocity"] = 2.0

    payload = Payload(
        make_point_config(),
        state,
    )

    assert payload.geometry.shape_name == "point"
    assert payload.geometry.supports_rotation is False

    assert payload.inertia is None

    # Point payload must ignore rotational initial state.
    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)


def test_invalid_mass_raises():
    config = make_rectangle_config()
    config["mass"] = 0.0

    with pytest.raises(ValueError):
        Payload(
            config,
            make_initial_state(),
        )


def test_invalid_negative_damping_raises():
    config = make_rectangle_config()
    config["linear_damping"] = -0.1

    with pytest.raises(ValueError):
        Payload(
            config,
            make_initial_state(),
        )


# ============================================================
# Moment of inertia
# ============================================================


def test_rectangle_inertia_is_computed_from_geometry():
    config = make_rectangle_config()

    payload = Payload(
        config,
        make_initial_state(),
    )

    mass = 3.0
    width = 0.60
    height = 0.30

    expected = (
        mass
        * (width**2 + height**2)
        / 12.0
    )

    assert payload.inertia == pytest.approx(expected)


# ============================================================
# Attachments
# ============================================================


def test_number_of_attachments():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    assert payload.num_attachments == 2


def test_attachment_positions_without_rotation():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    positions = payload.get_attachment_positions()

    expected = np.array(
        [
            [-0.70, 0.00],
            [0.70, 0.00],
        ]
    )

    np.testing.assert_allclose(
        positions,
        expected,
        atol=1e-8,
    )


def test_attachments_follow_payload_translation():
    state = make_initial_state()
    state["payload_position"] = [1.0, 2.0]

    payload = Payload(
        make_rectangle_config(),
        state,
    )

    positions = payload.get_attachment_positions()

    expected = np.array(
        [
            [0.30, 2.00],
            [1.70, 2.00],
        ]
    )

    np.testing.assert_allclose(
        positions,
        expected,
        atol=1e-8,
    )


def test_rectangle_attachments_rotate_with_payload():
    state = make_initial_state()

    state["payload_orientation"] = math.pi / 2

    payload = Payload(
        make_rectangle_config(),
        state,
    )

    positions = payload.get_attachment_positions()

    expected = np.array(
        [
            [0.0, -0.70],
            [0.0, 0.70],
        ]
    )

    np.testing.assert_allclose(
        positions,
        expected,
        atol=1e-8,
    )


def test_point_attachments_do_not_rotate():
    state = make_initial_state()

    state["payload_orientation"] = math.pi / 2

    payload = Payload(
        make_point_config(),
        state,
    )

    positions = payload.get_attachment_positions()

    expected = np.array(
        [
            [-0.70, 0.00],
            [0.70, 0.00],
        ]
    )

    np.testing.assert_allclose(
        positions,
        expected,
        atol=1e-8,
    )


def test_invalid_attachment_index_raises():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    with pytest.raises(IndexError):
        payload.get_attachment_position(2)


# ============================================================
# No-force dynamics
# ============================================================


def test_stationary_payload_remains_stationary():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.integrate(dt=0.05)

    np.testing.assert_allclose(
        payload.position,
        [0.0, 0.0],
        atol=1e-8,
    )

    np.testing.assert_allclose(
        payload.velocity,
        [0.0, 0.0],
        atol=1e-8,
    )

    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)


# ============================================================
# Linear force
# ============================================================


def test_center_force_causes_translation():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.apply_force([3.0, 0.0])

    payload.integrate(
        dt=0.1,
        clip_linear_speed=False,
    )

    # m = 3
    # F = 3
    #
    # acceleration = 1
    #
    # v_new = 0 + 1 * 0.1 = 0.1
    #
    # semi-implicit Euler:
    # x_new = 0 + 0.1 * 0.1 = 0.01

    np.testing.assert_allclose(
        payload.velocity,
        [0.1, 0.0],
        atol=1e-8,
    )

    np.testing.assert_allclose(
        payload.position,
        [0.01, 0.0],
        atol=1e-8,
    )


def test_center_force_does_not_create_rotation():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.apply_force([3.0, 1.0])

    payload.integrate(dt=0.1)

    assert payload.angular_velocity == pytest.approx(0.0)
    assert payload.orientation == pytest.approx(0.0)


# ============================================================
# Torque / force at attachment
# ============================================================


def test_force_at_right_attachment_creates_positive_torque():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    attachment = payload.get_attachment_position(1)

    # Right attachment:
    #
    # r = [0.7, 0]
    #
    # upward force:
    # F = [0, 1]
    #
    # tau = rx * Fy - ry * Fx = 0.7

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=attachment,
    )

    assert payload.force_accumulator[0] == pytest.approx(0.0)
    assert payload.force_accumulator[1] == pytest.approx(1.0)

    assert payload.torque_accumulator == pytest.approx(0.7)


def test_force_at_left_attachment_creates_negative_torque():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    attachment = payload.get_attachment_position(0)

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=attachment,
    )

    # r = [-0.7, 0]
    # F = [0, 1]
    #
    # tau = -0.7

    assert payload.torque_accumulator == pytest.approx(-0.7)


def test_equal_upward_forces_at_both_sides_cancel_torque():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    left = payload.get_attachment_position(0)
    right = payload.get_attachment_position(1)

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=left,
    )

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=right,
    )

    # Equal forces create:
    #
    # net force = [0, 2]
    # net torque = 0

    np.testing.assert_allclose(
        payload.force_accumulator,
        [0.0, 2.0],
        atol=1e-8,
    )

    assert payload.torque_accumulator == pytest.approx(0.0)


def test_unbalanced_attachment_force_changes_orientation():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    right = payload.get_attachment_position(1)

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=right,
    )

    payload.integrate(
        dt=0.1,
        clip_angular_speed=False,
    )

    assert payload.angular_velocity > 0.0
    assert payload.orientation > 0.0


# ============================================================
# Point payload rotation
# ============================================================


def test_point_payload_does_not_rotate_from_offcenter_force():
    payload = Payload(
        make_point_config(),
        make_initial_state(),
    )

    payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=[0.7, 0.0],
    )

    payload.integrate(dt=0.1)

    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)

    # Linear motion should still occur.
    assert payload.velocity[1] > 0.0


def test_direct_torque_is_ignored_by_point_payload():
    payload = Payload(
        make_point_config(),
        make_initial_state(),
    )

    payload.apply_torque(100.0)

    payload.integrate(dt=0.1)

    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)


# ============================================================
# Damping
# ============================================================


def test_linear_damping_reduces_velocity():
    config = make_rectangle_config()

    config["linear_damping"] = 1.0

    state = make_initial_state()

    state["payload_velocity"] = [1.0, 0.0]

    payload = Payload(
        config,
        state,
    )

    payload.integrate(
        dt=0.1,
        clip_linear_speed=False,
    )

    assert payload.velocity[0] < 1.0
    assert payload.velocity[0] > 0.0


def test_angular_damping_reduces_angular_velocity():
    config = make_rectangle_config()

    config["angular_damping"] = 1.0

    state = make_initial_state()

    state["payload_angular_velocity"] = 1.0

    payload = Payload(
        config,
        state,
    )

    payload.integrate(
        dt=0.1,
        clip_angular_speed=False,
    )

    assert payload.angular_velocity < 1.0
    assert payload.angular_velocity > 0.0


# ============================================================
# Speed clipping
# ============================================================


def test_linear_speed_is_clipped():
    config = make_rectangle_config()

    config["max_speed"] = 0.5

    payload = Payload(
        config,
        make_initial_state(),
    )

    payload.apply_force([100.0, 0.0])

    payload.integrate(
        dt=1.0,
        clip_linear_speed=True,
    )

    speed = np.linalg.norm(payload.velocity)

    assert speed == pytest.approx(0.5)


def test_linear_speed_can_skip_clipping():
    config = make_rectangle_config()

    config["max_speed"] = 0.5

    payload = Payload(
        config,
        make_initial_state(),
    )

    payload.apply_force([100.0, 0.0])

    payload.integrate(
        dt=1.0,
        clip_linear_speed=False,
    )

    assert np.linalg.norm(payload.velocity) > 0.5


def test_angular_speed_is_clipped():
    config = make_rectangle_config()

    config["max_angular_speed"] = 0.5

    payload = Payload(
        config,
        make_initial_state(),
    )

    payload.apply_torque(100.0)

    payload.integrate(
        dt=1.0,
        clip_angular_speed=True,
    )

    assert abs(payload.angular_velocity) == pytest.approx(0.5)


# ============================================================
# Force clearing
# ============================================================


def test_force_accumulators_are_cleared_after_integrate():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.apply_force([3.0, 1.0])
    payload.apply_torque(2.0)

    payload.integrate(dt=0.1)

    np.testing.assert_allclose(
        payload.force_accumulator,
        [0.0, 0.0],
        atol=1e-8,
    )

    assert payload.torque_accumulator == pytest.approx(0.0)


def test_clear_forces():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.apply_force([5.0, 3.0])
    payload.apply_torque(2.0)

    payload.clear_forces()

    np.testing.assert_allclose(
        payload.force_accumulator,
        [0.0, 0.0],
    )

    assert payload.torque_accumulator == pytest.approx(0.0)


# ============================================================
# Reset
# ============================================================


def test_reset_payload():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.apply_force([3.0, 0.0])
    payload.integrate(dt=0.1)

    payload.reset(
        position=[1.0, 2.0],
        velocity=[0.2, 0.3],
        orientation=0.5,
        angular_velocity=0.4,
    )

    np.testing.assert_allclose(
        payload.position,
        [1.0, 2.0],
    )

    np.testing.assert_allclose(
        payload.velocity,
        [0.2, 0.3],
    )

    assert payload.orientation == pytest.approx(0.5)
    assert payload.angular_velocity == pytest.approx(0.4)

    np.testing.assert_allclose(
        payload.force_accumulator,
        [0.0, 0.0],
    )

    assert payload.torque_accumulator == pytest.approx(0.0)


def test_point_reset_ignores_rotation():
    payload = Payload(
        make_point_config(),
        make_initial_state(),
    )

    payload.reset(
        position=[1.0, 2.0],
        velocity=[0.0, 0.0],
        orientation=1.5,
        angular_velocity=3.0,
    )

    assert payload.orientation == pytest.approx(0.0)
    assert payload.angular_velocity == pytest.approx(0.0)


# ============================================================
# Angle wrapping
# ============================================================


def test_orientation_is_wrapped():
    state = make_initial_state()

    state["payload_orientation"] = math.pi - 0.01
    state["payload_angular_velocity"] = 1.0

    payload = Payload(
        make_rectangle_config(),
        state,
    )

    payload.integrate(
        dt=0.1,
        clip_angular_speed=False,
    )

    assert -math.pi <= payload.orientation < math.pi


# ============================================================
# Collision
# ============================================================


def test_payload_collides_with_obstacle():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    obstacle = RectangleGeometry(
        width=0.5,
        height=0.5,
    )

    assert payload.collides_with(
        other_geometry=obstacle,
        other_position=[0.0, 0.0],
        other_orientation=0.0,
    )


def test_payload_does_not_collide_with_distant_obstacle():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    obstacle = RectangleGeometry(
        width=0.5,
        height=0.5,
    )

    assert not payload.collides_with(
        other_geometry=obstacle,
        other_position=[2.0, 2.0],
        other_orientation=0.0,
    )


def test_candidate_collision_does_not_modify_payload_state():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    obstacle = RectangleGeometry(
        width=0.5,
        height=0.5,
    )

    old_position = payload.position.copy()
    old_orientation = payload.orientation

    collision = payload.collides_with(
        other_geometry=obstacle,
        other_position=[1.0, 0.0],
        other_orientation=0.0,

        # Test hypothetical future pose.
        position=[1.0, 0.0],
        orientation=math.pi / 4,
    )

    assert collision

    # Collision query must not change real payload state.
    np.testing.assert_allclose(
        payload.position,
        old_position,
    )

    assert payload.orientation == pytest.approx(
        old_orientation
    )


# ============================================================
# Numerical validity
# ============================================================


def test_normal_payload_state_is_finite():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    assert payload.is_finite()


def test_nan_position_is_not_finite():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.position[0] = np.nan

    assert not payload.is_finite()


def test_inf_velocity_is_not_finite():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    payload.velocity[1] = np.inf

    assert not payload.is_finite()


# ============================================================
# state_dict
# ============================================================


def test_state_dict_contains_expected_fields():
    payload = Payload(
        make_rectangle_config(),
        make_initial_state(),
    )

    state = payload.state_dict()

    assert "position" in state
    assert "velocity" in state
    assert "orientation" in state
    assert "angular_velocity" in state
    assert "mass" in state
    assert "inertia" in state
    assert "shape" in state
    assert "attachment_positions" in state

    assert state["shape"] == "rectangle"

    assert state["attachment_positions"].shape == (2, 2)

    