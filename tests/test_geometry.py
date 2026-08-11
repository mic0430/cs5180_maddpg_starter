import math

import numpy as np
import pytest

from src.envs.geometry import (
    PointGeometry,
    CircleGeometry,
    SegmentGeometry,
    RectangleGeometry,
    build_geometry,
    rotation_matrix,
)


# ============================================================
# Geometry factory
# ============================================================


def test_build_point_geometry():
    geometry = build_geometry("point", [])

    assert isinstance(geometry, PointGeometry)
    assert geometry.shape_name == "point"
    assert geometry.supports_rotation is False


def test_build_circle_geometry():
    geometry = build_geometry("circle", [0.3])

    assert isinstance(geometry, CircleGeometry)
    assert geometry.radius == pytest.approx(0.3)


def test_build_segment_geometry():
    geometry = build_geometry("segment", [0.8])

    assert isinstance(geometry, SegmentGeometry)
    assert geometry.length == pytest.approx(0.8)


def test_build_rectangle_geometry():
    geometry = build_geometry("rectangle", [0.6, 0.3])

    assert isinstance(geometry, RectangleGeometry)
    assert geometry.width == pytest.approx(0.6)
    assert geometry.height == pytest.approx(0.3)


@pytest.mark.parametrize(
    "shape, dimensions",
    [
        ("point", [1.0]),
        ("circle", []),
        ("circle", [0.2, 0.3]),
        ("segment", []),
        ("rectangle", [0.6]),
        ("rectangle", [0.6, 0.3, 0.1]),
    ],
)
def test_invalid_dimensions_raise(shape, dimensions):
    with pytest.raises(ValueError):
        build_geometry(shape, dimensions)


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        build_geometry("triangle", [1.0, 1.0])


# ============================================================
# Rotation / coordinate transforms
# ============================================================


def test_rotation_matrix_90_degrees():
    R = rotation_matrix(math.pi / 2)

    vector = np.array([1.0, 0.0])

    rotated = R @ vector

    np.testing.assert_allclose(
        rotated,
        np.array([0.0, 1.0]),
        atol=1e-8,
    )


def test_local_to_world_without_rotation():
    geometry = RectangleGeometry(0.6, 0.3)

    local_point = np.array([0.2, -0.1])
    position = np.array([1.0, 2.0])

    world = geometry.local_to_world(
        local_point,
        position,
        orientation=0.0,
    )

    np.testing.assert_allclose(
        world,
        np.array([1.2, 1.9]),
        atol=1e-8,
    )


def test_local_to_world_with_rotation():
    geometry = RectangleGeometry(0.6, 0.3)

    local_point = np.array([1.0, 0.0])

    world = geometry.local_to_world(
        local_point,
        position=np.array([2.0, 3.0]),
        orientation=math.pi / 2,
    )

    np.testing.assert_allclose(
        world,
        np.array([2.0, 4.0]),
        atol=1e-8,
    )


def test_world_to_local_inverse():
    geometry = RectangleGeometry(0.6, 0.3)

    local = np.array([0.2, 0.1])
    position = np.array([1.5, -0.5])
    orientation = 0.73

    world = geometry.local_to_world(
        local,
        position,
        orientation,
    )

    recovered = geometry.world_to_local(
        world,
        position,
        orientation,
    )

    np.testing.assert_allclose(
        recovered,
        local,
        atol=1e-8,
    )


def test_point_geometry_ignores_rotation():
    geometry = PointGeometry()

    local = np.array([0.4, 0.2])
    position = np.array([1.0, 1.0])

    world_0 = geometry.local_to_world(
        local,
        position,
        orientation=0.0,
    )

    world_rotated = geometry.local_to_world(
        local,
        position,
        orientation=math.pi / 2,
    )

    np.testing.assert_allclose(
        world_0,
        world_rotated,
        atol=1e-8,
    )


# ============================================================
# Attachment positions
# ============================================================


def test_rectangle_attachment_rotates_with_payload():
    geometry = RectangleGeometry(
        width=0.6,
        height=0.3,
    )

    attachment_offset = np.array([0.7, 0.0])

    position = np.array([0.0, 0.0])

    attachment_world = geometry.attachment_world_position(
        attachment_offset,
        position,
        payload_orientation=math.pi / 2,
    )

    np.testing.assert_allclose(
        attachment_world,
        np.array([0.0, 0.7]),
        atol=1e-8,
    )


def test_point_attachment_does_not_rotate():
    geometry = PointGeometry()

    attachment_offset = np.array([0.7, 0.0])

    attachment_world = geometry.attachment_world_position(
        attachment_offset,
        payload_position=np.array([1.0, 2.0]),
        payload_orientation=math.pi / 2,
    )

    np.testing.assert_allclose(
        attachment_world,
        np.array([1.7, 2.0]),
        atol=1e-8,
    )


# ============================================================
# Moment of inertia
# ============================================================


def test_point_inertia_is_none():
    geometry = PointGeometry()

    assert geometry.compute_inertia(3.0) is None


def test_circle_inertia():
    geometry = CircleGeometry(radius=0.3)

    mass = 3.0

    expected = 0.5 * mass * 0.3**2

    assert geometry.compute_inertia(mass) == pytest.approx(expected)


def test_segment_inertia():
    geometry = SegmentGeometry(length=0.8)

    mass = 3.0

    expected = mass * 0.8**2 / 12.0

    assert geometry.compute_inertia(mass) == pytest.approx(expected)


def test_rectangle_inertia():
    geometry = RectangleGeometry(
        width=0.6,
        height=0.3,
    )

    mass = 3.0

    expected = (
        mass
        * (0.6**2 + 0.3**2)
        / 12.0
    )

    assert geometry.compute_inertia(mass) == pytest.approx(expected)


@pytest.mark.parametrize(
    "geometry",
    [
        PointGeometry(),
        CircleGeometry(0.3),
        SegmentGeometry(0.8),
        RectangleGeometry(0.6, 0.3),
    ],
)
def test_nonpositive_mass_raises(geometry):
    with pytest.raises(ValueError):
        geometry.compute_inertia(0.0)


# ============================================================
# Distance to geometry
# ============================================================


def test_rectangle_distance_outside_right_edge():
    geometry = RectangleGeometry(
        width=0.6,
        height=0.3,
    )

    # Rectangle extends from x=-0.3 to x=+0.3.
    # Point is at x=0.7.
    # Expected clearance = 0.4.
    distance = geometry.distance_to_boundary(
        [0.7, 0.0]
    )

    assert distance == pytest.approx(0.4)


def test_rectangle_distance_inside_is_zero():
    geometry = RectangleGeometry(0.6, 0.3)

    distance = geometry.distance_to_boundary(
        [0.1, 0.05]
    )

    assert distance == pytest.approx(0.0)


def test_circle_distance_outside():
    geometry = CircleGeometry(radius=0.3)

    distance = geometry.distance_to_boundary(
        [0.5, 0.0]
    )

    assert distance == pytest.approx(0.2)


def test_circle_distance_inside_is_zero():
    geometry = CircleGeometry(radius=0.3)

    distance = geometry.distance_to_boundary(
        [0.1, 0.0]
    )

    assert distance == pytest.approx(0.0)


def test_segment_distance():
    geometry = SegmentGeometry(length=1.0)

    # Segment is [-0.5, 0.5] on x axis.
    distance = geometry.distance_to_boundary(
        [0.0, 0.3]
    )

    assert distance == pytest.approx(0.3)


# ============================================================
# Attachment clearance
# ============================================================


def test_valid_rectangle_attachment_clearance():
    geometry = RectangleGeometry(
        width=0.6,
        height=0.3,
    )

    offsets = [
        [-0.7, 0.0],
        [0.7, 0.0],
    ]

    # Distance to rectangle = 0.4.
    # Required > 0.35 + 0.02 = 0.37.
    geometry.validate_attachment_clearance(
        attachment_offsets=offsets,
        max_attachment_distance=0.35,
        clearance_margin=0.02,
    )


def test_invalid_rectangle_attachment_clearance():
    geometry = RectangleGeometry(
        width=0.6,
        height=0.3,
    )

    offsets = [
        [-0.7, 0.0],
        [0.7, 0.0],
    ]

    # Clearance = 0.4.
    # Required > 0.39 + 0.02 = 0.41.
    with pytest.raises(ValueError):
        geometry.validate_attachment_clearance(
            attachment_offsets=offsets,
            max_attachment_distance=0.39,
            clearance_margin=0.02,
        )


def test_point_payload_skips_clearance_constraint():
    geometry = PointGeometry()

    geometry.validate_attachment_clearance(
        attachment_offsets=[
            [0.0, 0.0],
            [0.1, 0.0],
        ],
        max_attachment_distance=100.0,
        clearance_margin=10.0,
    )


# ============================================================
# Point collisions
# ============================================================


def test_point_inside_rectangle_collision():
    point = PointGeometry()
    rectangle = RectangleGeometry(1.0, 0.5)

    assert point.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=rectangle,
        other_position=[0.0, 0.0],
        other_orientation=0.0,
    )


def test_point_outside_rectangle_no_collision():
    point = PointGeometry()
    rectangle = RectangleGeometry(1.0, 0.5)

    assert not point.collides_with(
        position=[2.0, 0.0],
        orientation=0.0,
        other=rectangle,
        other_position=[0.0, 0.0],
        other_orientation=0.0,
    )


# ============================================================
# Circle collisions
# ============================================================


def test_circle_circle_overlap():
    a = CircleGeometry(0.5)
    b = CircleGeometry(0.5)

    assert a.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=b,
        other_position=[0.8, 0.0],
        other_orientation=0.0,
    )


def test_circle_circle_separated():
    a = CircleGeometry(0.5)
    b = CircleGeometry(0.5)

    assert not a.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=b,
        other_position=[1.1, 0.0],
        other_orientation=0.0,
    )


def test_circle_touching_rectangle():
    circle = CircleGeometry(radius=0.2)

    rectangle = RectangleGeometry(
        width=1.0,
        height=1.0,
    )

    # Rectangle right edge = x=0.5.
    # Circle center at 0.7 with r=0.2 -> touching.
    assert circle.collides_with(
        position=[0.7, 0.0],
        orientation=0.0,
        other=rectangle,
        other_position=[0.0, 0.0],
        other_orientation=0.0,
    )


# ============================================================
# Segment collisions
# ============================================================


def test_crossing_segments_collide():
    horizontal = SegmentGeometry(length=2.0)
    vertical = SegmentGeometry(length=2.0)

    assert horizontal.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=vertical,
        other_position=[0.0, 0.0],
        other_orientation=math.pi / 2,
    )


def test_parallel_segments_do_not_collide():
    a = SegmentGeometry(length=2.0)
    b = SegmentGeometry(length=2.0)

    assert not a.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=b,
        other_position=[0.0, 1.0],
        other_orientation=0.0,
    )


# ============================================================
# Rectangle collisions
# ============================================================


def test_axis_aligned_rectangles_overlap():
    a = RectangleGeometry(1.0, 1.0)
    b = RectangleGeometry(1.0, 1.0)

    assert a.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=b,
        other_position=[0.5, 0.0],
        other_orientation=0.0,
    )


def test_axis_aligned_rectangles_separated():
    a = RectangleGeometry(1.0, 1.0)
    b = RectangleGeometry(1.0, 1.0)

    assert not a.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=b,
        other_position=[2.0, 0.0],
        other_orientation=0.0,
    )


def test_rotated_rectangles_overlap():
    a = RectangleGeometry(1.0, 0.3)
    b = RectangleGeometry(1.0, 0.3)

    assert a.collides_with(
        position=[0.0, 0.0],
        orientation=math.pi / 4,
        other=b,
        other_position=[0.2, 0.0],
        other_orientation=-math.pi / 4,
    )


def test_rotated_rectangles_separated():
    a = RectangleGeometry(1.0, 0.3)
    b = RectangleGeometry(1.0, 0.3)

    assert not a.collides_with(
        position=[0.0, 0.0],
        orientation=math.pi / 4,
        other=b,
        other_position=[2.0, 0.0],
        other_orientation=-math.pi / 4,
    )


# ============================================================
# Narrow-gap behavior
# ============================================================


def test_long_payload_cannot_fit_horizontal_through_narrow_gap():
    """
    Two rectangular obstacles form a vertical passage.

    The payload is 1.0 long and 0.2 thick.

    When horizontal, its vertical thickness is only 0.2,
    so this specific configuration SHOULD fit.

    This test establishes the "fit" orientation first.
    """

    payload = RectangleGeometry(
        width=1.0,
        height=0.2,
    )

    top_obstacle = RectangleGeometry(
        width=2.0,
        height=0.4,
    )

    bottom_obstacle = RectangleGeometry(
        width=2.0,
        height=0.4,
    )

    # Obstacles leave a vertical gap of 0.30:
    #
    # top bottom edge = +0.15
    # bottom top edge = -0.15
    #
    # gap = 0.30
    #
    # horizontal payload vertical thickness = 0.20 -> fits.

    assert not payload.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=top_obstacle,
        other_position=[0.0, 0.35],
        other_orientation=0.0,
    )

    assert not payload.collides_with(
        position=[0.0, 0.0],
        orientation=0.0,
        other=bottom_obstacle,
        other_position=[0.0, -0.35],
        other_orientation=0.0,
    )


def test_long_payload_collides_after_rotation_in_same_gap():
    """
    Same passage as previous test.

    After 90-degree rotation, payload vertical extent becomes 1.0,
    which is larger than the 0.30 gap.

    It must collide with the obstacles.
    """

    payload = RectangleGeometry(
        width=1.0,
        height=0.2,
    )

    top_obstacle = RectangleGeometry(
        width=2.0,
        height=0.4,
    )

    bottom_obstacle = RectangleGeometry(
        width=2.0,
        height=0.4,
    )

    collides_top = payload.collides_with(
        position=[0.0, 0.0],
        orientation=math.pi / 2,
        other=top_obstacle,
        other_position=[0.0, 0.35],
        other_orientation=0.0,
    )

    collides_bottom = payload.collides_with(
        position=[0.0, 0.0],
        orientation=math.pi / 2,
        other=bottom_obstacle,
        other_position=[0.0, -0.35],
        other_orientation=0.0,
    )

    assert collides_top or collides_bottom
