"""
Geometry definitions for the cooperative transport environment.

Supported payload / obstacle shapes:
    - point
    - circle
    - segment
    - rectangle

Geometry is responsible for:
    1. shape dimensions
    2. rotation support
    3. moment-of-inertia calculation
    4. local/world coordinate transforms
    5. attachment world positions
    6. distance from a point to the geometry
    7. collision detection

Geometry does NOT store:
    - payload position
    - payload velocity
    - payload mass
    - payload angular velocity

Those belong to Payload / Environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


EPS = 1e-10


# ============================================================
# Helper functions
# ============================================================


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Return the 2-D rotation matrix for angle theta.

    Parameters
    ----------
    theta:
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (2, 2)
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )


def as_vec2(value: Sequence[float]) -> np.ndarray:
    """
    Convert input to a 2-D float vector.
    """
    arr = np.asarray(value, dtype=np.float64)

    if arr.shape != (2,):
        raise ValueError(
            f"Expected a 2-D vector with shape (2,), got {arr.shape}."
        )

    return arr


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """
    2-D cross product.

    For vectors:
        a = [ax, ay]
        b = [bx, by]

    returns:
        ax * by - ay * bx

    This scalar is useful for torque calculations.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """
    Minimum Euclidean distance from a point to a line segment.
    """
    point = as_vec2(point)
    start = as_vec2(start)
    end = as_vec2(end)

    segment = end - start
    length_sq = float(np.dot(segment, segment))

    if length_sq <= EPS:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / length_sq)
    t = np.clip(t, 0.0, 1.0)

    closest = start + t * segment

    return float(np.linalg.norm(point - closest))


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Signed orientation of ordered points (a, b, c).

    > 0 : counter-clockwise
    < 0 : clockwise
    = 0 : collinear
    """
    return cross_2d(b - a, c - a)


def point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    eps: float = EPS,
) -> bool:
    """
    Return True if point lies on the closed segment [start, end].
    """
    if abs(orientation(start, end, point)) > eps:
        return False

    return (
        min(start[0], end[0]) - eps
        <= point[0]
        <= max(start[0], end[0]) + eps
        and min(start[1], end[1]) - eps
        <= point[1]
        <= max(start[1], end[1]) + eps
    )


def segments_intersect(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> bool:
    """
    Return True if two closed 2-D line segments intersect.
    """
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    # General intersection.
    if (
        ((o1 > EPS and o2 < -EPS) or (o1 < -EPS and o2 > EPS))
        and
        ((o3 > EPS and o4 < -EPS) or (o3 < -EPS and o4 > EPS))
    ):
        return True

    # Collinear / touching cases.
    if abs(o1) <= EPS and point_on_segment(b1, a1, a2):
        return True

    if abs(o2) <= EPS and point_on_segment(b2, a1, a2):
        return True

    if abs(o3) <= EPS and point_on_segment(a1, b1, b2):
        return True

    if abs(o4) <= EPS and point_on_segment(a2, b1, b2):
        return True

    return False


def segment_to_segment_distance(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> float:
    """
    Minimum distance between two line segments.
    """
    if segments_intersect(a1, a2, b1, b2):
        return 0.0

    return min(
        point_to_segment_distance(a1, b1, b2),
        point_to_segment_distance(a2, b1, b2),
        point_to_segment_distance(b1, a1, a2),
        point_to_segment_distance(b2, a1, a2),
    )


# ============================================================
# Base class
# ============================================================


class Geometry(ABC):
    """
    Base class for payload and obstacle geometry.

    Geometry coordinates are defined relative to the entity center.

    Position and orientation are supplied externally because Geometry
    does not own physical state.
    """

    shape_name: str
    supports_rotation: bool = True

    @abstractmethod
    def compute_inertia(self, mass: float) -> float | None:
        """
        Compute moment of inertia about the geometry center.

        Point geometry returns None because rotational dynamics
        are disabled.
        """
        raise NotImplementedError

    @abstractmethod
    def distance_to_boundary(
        self,
        local_point: Sequence[float],
    ) -> float:
        """
        Minimum distance from a point to the geometry itself.

        The input point must be expressed in payload-local coordinates.

        Returns
        -------
        float
            0 if the point is on or inside the geometry.
            Positive distance if it is outside.
        """
        raise NotImplementedError

    @abstractmethod
    def contains_local_point(
        self,
        local_point: Sequence[float],
    ) -> bool:
        """
        Return True if a local point lies inside / on the geometry.
        """
        raise NotImplementedError

    def local_to_world(
        self,
        local_point: Sequence[float],
        position: Sequence[float],
        orientation: float = 0.0,
    ) -> np.ndarray:
        """
        Convert a geometry-local point to world coordinates.
        """
        local_point = as_vec2(local_point)
        position = as_vec2(position)

        if self.supports_rotation:
            return position + rotation_matrix(orientation) @ local_point

        return position + local_point

    def world_to_local(
        self,
        world_point: Sequence[float],
        position: Sequence[float],
        orientation: float = 0.0,
    ) -> np.ndarray:
        """
        Convert a world point to geometry-local coordinates.
        """
        world_point = as_vec2(world_point)
        position = as_vec2(position)

        relative = world_point - position

        if self.supports_rotation:
            return rotation_matrix(-orientation) @ relative

        return relative

    def attachment_world_position(
        self,
        attachment_offset: Sequence[float],
        payload_position: Sequence[float],
        payload_orientation: float = 0.0,
    ) -> np.ndarray:
        """
        Convert a payload-local attachment offset to world coordinates.
        """
        return self.local_to_world(
            attachment_offset,
            payload_position,
            payload_orientation,
        )

    def validate_attachment_clearance(
        self,
        attachment_offsets: Sequence[Sequence[float]],
        max_attachment_distance: float,
        clearance_margin: float = 0.0,
    ) -> None:
        """
        Validate that an agent constrained around each attachment
        cannot enter the payload geometry.

        Requirement:

            distance(attachment, payload geometry)
                >
            max_attachment_distance + clearance_margin
        """
        if max_attachment_distance < 0:
            raise ValueError(
                "max_attachment_distance must be non-negative."
            )

        if clearance_margin < 0:
            raise ValueError(
                "clearance_margin must be non-negative."
            )

        # Point payload has no physical interior.
        if self.shape_name == "point":
            return

        required = max_attachment_distance + clearance_margin

        for i, offset in enumerate(attachment_offsets):
            distance = self.distance_to_boundary(offset)

            if distance <= required:
                raise ValueError(
                    f"Attachment {i} has insufficient payload clearance: "
                    f"distance={distance:.6f}, "
                    f"required>{required:.6f}."
                )

    def collides_with(
        self,
        position: Sequence[float],
        orientation: float,
        other: "Geometry",
        other_position: Sequence[float],
        other_orientation: float = 0.0,
    ) -> bool:
        """
        Generic collision dispatcher.
        """
        return geometries_collide(
            self,
            as_vec2(position),
            orientation,
            other,
            as_vec2(other_position),
            other_orientation,
        )


# ============================================================
# Point
# ============================================================


class PointGeometry(Geometry):
    shape_name = "point"
    supports_rotation = False

    def __init__(self) -> None:
        pass

    def compute_inertia(self, mass: float) -> None:
        if mass <= 0:
            raise ValueError("mass must be positive.")

        return None

    def distance_to_boundary(
        self,
        local_point: Sequence[float],
    ) -> float:
        point = as_vec2(local_point)

        # A point payload has zero physical size.
        return float(np.linalg.norm(point))

    def contains_local_point(
        self,
        local_point: Sequence[float],
    ) -> bool:
        point = as_vec2(local_point)

        return bool(np.linalg.norm(point) <= EPS)


# ============================================================
# Circle
# ============================================================


class CircleGeometry(Geometry):
    shape_name = "circle"
    supports_rotation = True

    def __init__(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError("Circle radius must be positive.")

        self.radius = float(radius)

    def compute_inertia(self, mass: float) -> float:
        if mass <= 0:
            raise ValueError("mass must be positive.")

        return 0.5 * mass * self.radius**2

    def distance_to_boundary(
        self,
        local_point: Sequence[float],
    ) -> float:
        point = as_vec2(local_point)

        radial_distance = float(np.linalg.norm(point))

        return max(0.0, radial_distance - self.radius)

    def contains_local_point(
        self,
        local_point: Sequence[float],
    ) -> bool:
        point = as_vec2(local_point)

        return bool(
            np.linalg.norm(point) <= self.radius + EPS
        )


# ============================================================
# Segment
# ============================================================


class SegmentGeometry(Geometry):
    shape_name = "segment"
    supports_rotation = True

    def __init__(self, length: float) -> None:
        if length <= 0:
            raise ValueError("Segment length must be positive.")

        self.length = float(length)
        self.half_length = self.length / 2.0

    def compute_inertia(self, mass: float) -> float:
        if mass <= 0:
            raise ValueError("mass must be positive.")

        return mass * self.length**2 / 12.0

    def local_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-self.half_length, 0.0]),
            np.array([self.half_length, 0.0]),
        )

    def world_endpoints(
        self,
        position: Sequence[float],
        orientation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.local_endpoints()

        return (
            self.local_to_world(a, position, orientation),
            self.local_to_world(b, position, orientation),
        )

    def distance_to_boundary(
        self,
        local_point: Sequence[float],
    ) -> float:
        point = as_vec2(local_point)

        a, b = self.local_endpoints()

        return point_to_segment_distance(point, a, b)

    def contains_local_point(
        self,
        local_point: Sequence[float],
    ) -> bool:
        return self.distance_to_boundary(local_point) <= EPS


# ============================================================
# Rectangle
# ============================================================


class RectangleGeometry(Geometry):
    shape_name = "rectangle"
    supports_rotation = True

    def __init__(self, width: float, height: float) -> None:
        if width <= 0:
            raise ValueError("Rectangle width must be positive.")

        if height <= 0:
            raise ValueError("Rectangle height must be positive.")

        self.width = float(width)
        self.height = float(height)

        self.half_width = self.width / 2.0
        self.half_height = self.height / 2.0

    def compute_inertia(self, mass: float) -> float:
        if mass <= 0:
            raise ValueError("mass must be positive.")

        return (
            mass
            * (self.width**2 + self.height**2)
            / 12.0
        )

    def local_vertices(self) -> np.ndarray:
        """
        Return rectangle vertices counter-clockwise.
        """
        return np.array(
            [
                [-self.half_width, -self.half_height],
                [ self.half_width, -self.half_height],
                [ self.half_width,  self.half_height],
                [-self.half_width,  self.half_height],
            ],
            dtype=np.float64,
        )

    def world_vertices(
        self,
        position: Sequence[float],
        orientation: float,
    ) -> np.ndarray:
        vertices = self.local_vertices()

        R = rotation_matrix(orientation)
        position = as_vec2(position)

        return vertices @ R.T + position

    def contains_local_point(
        self,
        local_point: Sequence[float],
    ) -> bool:
        x, y = as_vec2(local_point)

        return bool(
            abs(x) <= self.half_width + EPS
            and abs(y) <= self.half_height + EPS
        )

    def distance_to_boundary(
        self,
        local_point: Sequence[float],
    ) -> float:
        """
        Distance from point to rectangle geometry.

        Returns zero for points inside/on the rectangle.
        """
        x, y = as_vec2(local_point)

        dx = max(abs(x) - self.half_width, 0.0)
        dy = max(abs(y) - self.half_height, 0.0)

        return float(np.hypot(dx, dy))


# ============================================================
# Collision helpers
# ============================================================


def point_inside_geometry(
    point: np.ndarray,
    geometry: Geometry,
    position: np.ndarray,
    orientation: float,
) -> bool:
    local = geometry.world_to_local(
        point,
        position,
        orientation,
    )

    return geometry.contains_local_point(local)


def rectangle_edges(
    rectangle: RectangleGeometry,
    position: np.ndarray,
    orientation: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    vertices = rectangle.world_vertices(
        position,
        orientation,
    )

    return [
        (vertices[i], vertices[(i + 1) % 4])
        for i in range(4)
    ]


def circle_circle_collision(
    a: CircleGeometry,
    pos_a: np.ndarray,
    b: CircleGeometry,
    pos_b: np.ndarray,
) -> bool:
    distance = np.linalg.norm(pos_a - pos_b)

    return bool(
        distance <= a.radius + b.radius + EPS
    )


def point_circle_collision(
    point_pos: np.ndarray,
    circle: CircleGeometry,
    circle_pos: np.ndarray,
) -> bool:
    return bool(
        np.linalg.norm(point_pos - circle_pos)
        <= circle.radius + EPS
    )


def point_rectangle_collision(
    point_pos: np.ndarray,
    rectangle: RectangleGeometry,
    rect_pos: np.ndarray,
    rect_orientation: float,
) -> bool:
    return point_inside_geometry(
        point_pos,
        rectangle,
        rect_pos,
        rect_orientation,
    )


def point_segment_collision(
    point_pos: np.ndarray,
    segment: SegmentGeometry,
    segment_pos: np.ndarray,
    segment_orientation: float,
) -> bool:
    a, b = segment.world_endpoints(
        segment_pos,
        segment_orientation,
    )

    return (
        point_to_segment_distance(point_pos, a, b)
        <= EPS
    )


def circle_segment_collision(
    circle: CircleGeometry,
    circle_pos: np.ndarray,
    segment: SegmentGeometry,
    segment_pos: np.ndarray,
    segment_orientation: float,
) -> bool:
    a, b = segment.world_endpoints(
        segment_pos,
        segment_orientation,
    )

    distance = point_to_segment_distance(
        circle_pos,
        a,
        b,
    )

    return bool(distance <= circle.radius + EPS)


def circle_rectangle_collision(
    circle: CircleGeometry,
    circle_pos: np.ndarray,
    rectangle: RectangleGeometry,
    rect_pos: np.ndarray,
    rect_orientation: float,
) -> bool:
    """
    Circle vs oriented rectangle collision.

    Convert circle center into rectangle-local coordinates,
    then find closest point on axis-aligned local rectangle.
    """
    local_circle = rectangle.world_to_local(
        circle_pos,
        rect_pos,
        rect_orientation,
    )

    closest = np.array(
        [
            np.clip(
                local_circle[0],
                -rectangle.half_width,
                rectangle.half_width,
            ),
            np.clip(
                local_circle[1],
                -rectangle.half_height,
                rectangle.half_height,
            ),
        ]
    )

    distance = np.linalg.norm(local_circle - closest)

    return bool(distance <= circle.radius + EPS)


def segment_segment_collision(
    a: SegmentGeometry,
    pos_a: np.ndarray,
    orient_a: float,
    b: SegmentGeometry,
    pos_b: np.ndarray,
    orient_b: float,
) -> bool:
    a1, a2 = a.world_endpoints(pos_a, orient_a)
    b1, b2 = b.world_endpoints(pos_b, orient_b)

    return segments_intersect(a1, a2, b1, b2)


def segment_rectangle_collision(
    segment: SegmentGeometry,
    seg_pos: np.ndarray,
    seg_orientation: float,
    rectangle: RectangleGeometry,
    rect_pos: np.ndarray,
    rect_orientation: float,
) -> bool:
    s1, s2 = segment.world_endpoints(
        seg_pos,
        seg_orientation,
    )

    # Endpoint already inside rectangle.
    if point_rectangle_collision(
        s1,
        rectangle,
        rect_pos,
        rect_orientation,
    ):
        return True

    if point_rectangle_collision(
        s2,
        rectangle,
        rect_pos,
        rect_orientation,
    ):
        return True

    for r1, r2 in rectangle_edges(
        rectangle,
        rect_pos,
        rect_orientation,
    ):
        if segments_intersect(s1, s2, r1, r2):
            return True

    return False


def rectangle_rectangle_collision(
    a: RectangleGeometry,
    pos_a: np.ndarray,
    orient_a: float,
    b: RectangleGeometry,
    pos_b: np.ndarray,
    orient_b: float,
) -> bool:
    """
    Oriented rectangle vs oriented rectangle collision
    using the Separating Axis Theorem (SAT).
    """
    verts_a = a.world_vertices(pos_a, orient_a)
    verts_b = b.world_vertices(pos_b, orient_b)

    def axes_from_vertices(vertices: np.ndarray) -> list[np.ndarray]:
        axes = []

        for i in range(4):
            edge = vertices[(i + 1) % 4] - vertices[i]

            normal = np.array(
                [-edge[1], edge[0]],
                dtype=np.float64,
            )

            norm = np.linalg.norm(normal)

            if norm > EPS:
                axes.append(normal / norm)

        # Rectangles technically only need two unique axes,
        # but keeping four makes this function simpler and clear.
        return axes

    axes = (
        axes_from_vertices(verts_a)
        + axes_from_vertices(verts_b)
    )

    for axis in axes:
        proj_a = verts_a @ axis
        proj_b = verts_b @ axis

        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        # Separating axis exists => no collision.
        if (
            max_a < min_b - EPS
            or max_b < min_a - EPS
        ):
            return False

    return True


# ============================================================
# Generic collision dispatcher
# ============================================================


def geometries_collide(
    a: Geometry,
    pos_a: np.ndarray,
    orient_a: float,
    b: Geometry,
    pos_b: np.ndarray,
    orient_b: float,
) -> bool:
    """
    Collision dispatcher for all supported shape pairs.
    """

    # --------------------------------------------------------
    # Point
    # --------------------------------------------------------

    if isinstance(a, PointGeometry):
        if isinstance(b, PointGeometry):
            return bool(
                np.linalg.norm(pos_a - pos_b) <= EPS
            )

        if isinstance(b, CircleGeometry):
            return point_circle_collision(
                pos_a,
                b,
                pos_b,
            )

        if isinstance(b, SegmentGeometry):
            return point_segment_collision(
                pos_a,
                b,
                pos_b,
                orient_b,
            )

        if isinstance(b, RectangleGeometry):
            return point_rectangle_collision(
                pos_a,
                b,
                pos_b,
                orient_b,
            )

    # Make dispatcher symmetric.
    if isinstance(b, PointGeometry):
        return geometries_collide(
            b,
            pos_b,
            orient_b,
            a,
            pos_a,
            orient_a,
        )

    # --------------------------------------------------------
    # Circle
    # --------------------------------------------------------

    if isinstance(a, CircleGeometry):
        if isinstance(b, CircleGeometry):
            return circle_circle_collision(
                a,
                pos_a,
                b,
                pos_b,
            )

        if isinstance(b, SegmentGeometry):
            return circle_segment_collision(
                a,
                pos_a,
                b,
                pos_b,
                orient_b,
            )

        if isinstance(b, RectangleGeometry):
            return circle_rectangle_collision(
                a,
                pos_a,
                b,
                pos_b,
                orient_b,
            )

    if isinstance(b, CircleGeometry):
        return geometries_collide(
            b,
            pos_b,
            orient_b,
            a,
            pos_a,
            orient_a,
        )

    # --------------------------------------------------------
    # Segment
    # --------------------------------------------------------

    if isinstance(a, SegmentGeometry):
        if isinstance(b, SegmentGeometry):
            return segment_segment_collision(
                a,
                pos_a,
                orient_a,
                b,
                pos_b,
                orient_b,
            )

        if isinstance(b, RectangleGeometry):
            return segment_rectangle_collision(
                a,
                pos_a,
                orient_a,
                b,
                pos_b,
                orient_b,
            )

    if isinstance(b, SegmentGeometry):
        return geometries_collide(
            b,
            pos_b,
            orient_b,
            a,
            pos_a,
            orient_a,
        )

    # --------------------------------------------------------
    # Rectangle
    # --------------------------------------------------------

    if (
        isinstance(a, RectangleGeometry)
        and isinstance(b, RectangleGeometry)
    ):
        return rectangle_rectangle_collision(
            a,
            pos_a,
            orient_a,
            b,
            pos_b,
            orient_b,
        )

    raise TypeError(
        f"Unsupported collision pair: "
        f"{type(a).__name__} vs {type(b).__name__}"
    )


# ============================================================
# Geometry factory
# ============================================================


def build_geometry(
    shape: str,
    dimensions: Sequence[float] | None = None,
) -> Geometry:
    """
    Construct Geometry from config values.

    Expected dimensions:
        point      -> []
        circle     -> [radius]
        segment    -> [length]
        rectangle  -> [width, height]
    """
    shape = str(shape).lower().strip()

    dimensions = (
        []
        if dimensions is None
        else list(dimensions)
    )

    if shape == "point":
        if len(dimensions) != 0:
            raise ValueError(
                "Point geometry requires dimensions: []."
            )

        return PointGeometry()

    if shape == "circle":
        if len(dimensions) != 1:
            raise ValueError(
                "Circle geometry requires "
                "dimensions: [radius]."
            )

        return CircleGeometry(
            radius=float(dimensions[0]),
        )

    if shape == "segment":
        if len(dimensions) != 1:
            raise ValueError(
                "Segment geometry requires "
                "dimensions: [length]."
            )

        return SegmentGeometry(
            length=float(dimensions[0]),
        )

    if shape == "rectangle":
        if len(dimensions) != 2:
            raise ValueError(
                "Rectangle geometry requires "
                "dimensions: [width, height]."
            )

        return RectangleGeometry(
            width=float(dimensions[0]),
            height=float(dimensions[1]),
        )

    raise ValueError(
        f"Unsupported geometry shape: {shape!r}. "
        "Supported shapes are: "
        "'point', 'circle', 'segment', 'rectangle'."
    )
