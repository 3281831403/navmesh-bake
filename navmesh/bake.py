"""Navigation-mesh baking contracts.

Coordinate system
-----------------
Points are 3D tuples ``(x, y, z)`` and ``+y`` is world up.  Floor polygons
are projected onto the ``xz`` plane.  Normals point out of the walkable side
of a surface, so a flat floor has the normal ``(0, 1, 0)``.

Angles
------
Every argument or field whose name ends in ``_deg`` is expressed in degrees,
never radians.  Internally an implementation may convert with
``math.radians`` and must not compare a degree value against a radian value.

Tolerance
---------
Geometric equality, coplanarity, shared-edge detection and angle comparisons
use an absolute tolerance of ``1e-9`` (degrees for angles, metres for lengths)
unless a function documents a stricter/looser threshold.

Error handling
--------------
Malformed input, invalid parameters and thresholds that cannot produce a
valid result raise :class:`BakeError`.  The stubs deliberately raise
``NotImplementedError`` until implemented.
"""

from __future__ import annotations


class BakeError(ValueError):
    """Raised when navmesh input or a baking parameter violates the contract."""


def merge_triangles(
    triangles: list[dict],
    max_angle_deg: float,
    max_deviation: float,
) -> list[list[int]]:
    """Merge adjacent coplanar triangles into triangle groups.

    Parameters
    ----------
    triangles:
        List of triangle dictionaries.  Every item must contain
        ``"vertices"`` (three ``(x, y, z)`` points) and may contain
        ``"normal"`` (a 3D vector); a missing normal is derived from the
        right-handed vertex winding.
    max_angle_deg:
        Maximum angle between the two face normals for a merge, in degrees.
        The value is compared in degrees; it is not a radian value.
    max_deviation:
        Maximum distance, in metres, from a candidate triangle's vertices to
        the adjacent triangle's plane.  Non-negative.

    Returns
    -------
    list[list[int]]
        Groups of input triangle indices.  Only triangles that share a full
        edge may be merged.  Every group is sorted ascending and the groups
        are ordered by their smallest index.  A triangle with no merge
        partner is returned as a singleton group.

    Tolerance
    ---------
    The angle comparison allows ``1e-9`` degrees of slack and distance
    comparisons allow ``1e-9`` metres of slack.

    Raises
    ------
    BakeError
        If a triangle is malformed, contains non-finite coordinates, or if
        ``max_angle_deg`` is outside ``[0, 180]`` or ``max_deviation`` is
        negative/non-finite.
    """
    raise NotImplementedError


def shrink_boundary(
    polygon: list[tuple[float, float, float]],
    agent_radius: float,
    step_height: float,
    max_slope_deg: float,
) -> list[tuple[float, float, float]]:
    """Inset a walkable boundary by an agent radius.

    Parameters
    ----------
    polygon:
        Ordered boundary vertices ``(x, y, z)`` of one polygon, with ``+y``
        up.  At least three vertices are required; the polygon must be simple
        in the ``xz`` projection.
    agent_radius:
        Positive radius of the agent, in metres.  The boundary is offset
        horizontally.  Offsets use each **edge normal**, not vertex normals.

        * ``0`` raises :class:`BakeError` (no agent can have zero radius).
        * A negative value also raises :class:`BakeError`.
    step_height:
        Maximum allowed absolute height change between adjacent input
        vertices, in metres.  It is an independent gate; it must never be
        compared with ``max_slope_deg``.
    max_slope_deg:
        Maximum allowed edge inclination from horizontal, in degrees.  The
        edge slope is ``degrees(atan2(abs(dy), horizontal_edge_length))``.
        It is an independent gate.

    Returns
    -------
    list[tuple[float, float, float]]
        The inset boundary with the same winding and the same number/order of
        vertices as the input.  Concave corners are handled by intersecting
        offset edge lines, so the result is simple and must not self-intersect.

    Tolerance
    ---------
    Heights, lengths and slope angles compare with ``1e-9`` slack.  If the
    radius is too large for the polygon, or an edge exceeds either
    ``step_height`` or ``max_slope_deg``, a :class:`BakeError` is raised
    instead of returning a collapsed/self-intersecting boundary.

    Raises
    ------
    BakeError
        For invalid parameters, malformed/non-simple polygons, a boundary
        that cannot be inset, or an edge that violates either threshold.
    """
    raise NotImplementedError


def classify_slopes(triangles: list[dict], max_slope_deg: float) -> dict:
    """Split triangle indices into walkable and blocked surfaces by normal.

    The slope of a face is the angle between its outward unit normal and the
    world-up vector ``(0, 1, 0)``::

        slope_deg = degrees(acos(clamp(dot(normal, up) / |normal|, -1, 1)))

    A flat floor therefore has slope ``0`` and a vertical wall has slope
    ``90``.  The classification must use the supplied/recomputed face normal,
    not a difference between face-center heights.

    Parameters
    ----------
    triangles:
        Triangle dictionaries with ``"vertices"`` and optionally ``"normal"``.
    max_slope_deg:
        Walkable threshold in degrees, within ``[0, 90]``.  A face is
        walkable when ``slope_deg <= max_slope_deg`` (inclusive).

    Returns
    -------
    dict
        Exactly ``{"walkable": [...], "blocked": [...]}``.  Both lists contain
        input indices, sorted ascending, and every input index appears in
        exactly one list.

    Tolerance
    ---------
    The threshold comparison allows ``1e-9`` degrees of slack.  A downward
    normal is blocked even if its unsigned angle were within the threshold.

    Raises
    ------
    BakeError
        For malformed triangles, zero-length normals, non-finite values, or a
        threshold outside ``[0, 90]`` degrees.
    """
    raise NotImplementedError


def split_islands(polygons: list[dict]) -> list[list[int]]:
    """Group polygons into connected components by shared edges.

    Two polygons are connected only when they share a full edge, i.e. an
    endpoint pair that is equal within tolerance in either orientation.
    Sharing a single vertex is not enough.  This function defines island
    membership from topology; it does not use distance between centroids.

    Parameters
    ----------
    polygons:
        Polygon dictionaries with ``"vertices"`` containing at least three
        3D points in boundary order.

    Returns
    -------
    list[list[int]]
        One sorted list of input polygon indices per connected component.
        Components are ordered by their smallest index.  An empty input
        returns ``[]``.

    Angle unit
    ----------
    This function does not use angles.

    Tolerance
    ---------
    Endpoint equality uses ``1e-9`` metres of absolute slack.

    Raises
    ------
    BakeError
        For malformed polygons, non-finite coordinates, or polygons without a
        full edge.
    """
    raise NotImplementedError


def apply_obstacles(navmesh: dict, obstacles: list[dict]) -> dict:
    """Apply dynamic obstacles and recompute topology and agent clearance.

    Parameters
    ----------
    navmesh:
        Dictionary with ``"polygons"`` (each has ``"id"`` and ``"vertices"``)
        and ``"agents"`` (each has ``"id"`` and a positive ``"radius"`` in
        metres).  Polygon normals, costs and island ids may be present.
    obstacles:
        Inline list of obstacle dictionaries.  Each has ``"id"``,
        ``"vertices"`` (a footprint of at least three 3D points) and
        ``"height"`` in metres.  The footprint is extruded upward from its
        base ``y``; it removes/occludes floor area it intersects.

    Returns
    -------
    dict
        A new navmesh dictionary (the input is not mutated) containing at
        least:

        * ``"polygons"`` -- the remaining/updated polygons, with fresh
          ``"island"`` ids assigned after the obstacles are applied;
        * ``"islands"`` -- components as lists of polygon indices, ordered by
          their smallest index;
        * ``"agent_clearance"`` -- ``{agent_id: bool}`` for **every** agent.
          A large radius must be evaluated even when a smaller agent appears
          first in the list.

        Connectivity and island ownership are recomputed after obstacle
        insertion.  A path may never cross an obstacle wall.

    Angle unit
    ----------
    The function does not take an angle argument.  Any internal slope check
    uses degrees.

    Tolerance
    ---------
    Obstacle/polygon intersections and shared-edge tests use ``1e-9`` metres
    or the corresponding angular slack of ``1e-9`` degrees.

    Raises
    ------
    BakeError
        For empty/invalid polygons, duplicate agent ids, non-positive agent
        radii, malformed or non-finite obstacle footprints, or non-finite
        heights.
    """
    raise NotImplementedError


def assign_costs(navmesh: dict, area_costs: dict) -> dict:
    """Assign a traversal cost to every polygon.

    Parameters
    ----------
    navmesh:
        Dictionary whose ``"polygons"`` each contain a unique ``"id"``.
    area_costs:
        Mapping from polygon id to a positive numeric cost.  A missing
        polygon id is allowed and defaults to ``1.0``.  A present value that
        is zero, negative, non-finite, or within ``1e-9`` of zero is invalid.

    Returns
    -------
    dict
        A new navmesh dictionary (the input is not mutated) whose polygons
        each carry a ``"cost"`` field.  Costs supplied by ``area_costs`` are
        preserved; only missing ids receive ``1.0``.

    Angle unit
    ----------
    This function does not use angles.

    Tolerance
    ---------
    Cost validation and comparisons use an absolute tolerance of ``1e-9``;
    in particular, ``1.0 + 1e-10`` remains a valid positive cost.

    Raises
    ------
    BakeError
        If a polygon has no usable id, ``area_costs`` references an unknown
        polygon, or a supplied cost is zero, negative or non-finite.
    """
    raise NotImplementedError


def validate(navmesh: dict, params: dict) -> list[str]:
    """Return human-readable/stably coded problems with a baked navmesh.

    Parameters
    ----------
    navmesh:
        Baked navmesh with ``"polygons"``.  Each polygon may contain
        ``"id"``, ``"vertices"``, ``"normal"``, ``"cost"`` and ``"island"``.
    params:
        Baking parameters.  Recognised keys include ``"max_slope_deg"``
        (degrees), ``"agent_radius"`` (metres), and ``"min_passage_width"``
        (metres).  Missing keys use conservative defaults.

    Returns
    -------
    list[str]
        Sorted issue codes; an empty list means the navmesh passes.  At a
        minimum the following issues are reported:

        * ``"island_missing:<polygon id>"`` -- an island id is absent or does
          not match the shared-edge components;
        * ``"cost_missing:<polygon id>"`` -- no finite positive cost;
        * ``"normal_down:<polygon id>"`` -- the face normal points downward;
        * ``"narrow_passage:<polygon id>"`` -- a passage is narrower than the
          configured agent width and has effectively been swallowed.

    Angle unit
    ----------
    ``params["max_slope_deg"]`` is in degrees.

    Tolerance
    ---------
    Normal, width and angle checks use ``1e-9`` of slack (degrees for angles,
    metres for lengths).

    Raises
    ------
    BakeError
        For malformed navmesh data or non-finite/out-of-range parameters.
    """
    raise NotImplementedError


def bake_batch(tiles: list[dict], params: dict) -> dict:
    """Bake a batch of map tiles atomically.

    Each tile is a dictionary with a unique ``"id"`` plus the geometry keys
    accepted by the individual baking functions (for example
    ``"triangles"``, ``"obstacles"``, and ``"area_costs"``).  Tiles are baked
    in input order with a fresh result object per tile.

    Parameters
    ----------
    tiles:
        Inline list of tile dictionaries.
    params:
        Shared baking parameters, including ``max_angle_deg``,
        ``max_deviation``, ``agent_radius``, ``step_height`` and
        ``max_slope_deg``.  Angle values are in degrees.

    Returns
    -------
    dict
        On complete success::

            {"ok": True, "failed": [], "baked": [{"id": ..., "navmesh": ...}, ...]}

        where ``baked`` follows input tile order.  If any tile fails, all
        work is rolled back and exactly this shape is returned::

            {"ok": False, "failed": [failed tile ids in input order], "baked": []}

        A partially baked prefix must never be returned or left behind.

    Angle unit
    ----------
    All ``*_deg`` fields in ``params`` are in degrees.

    Tolerance
    ---------
    Results are deterministic for the same input and use ``1e-9`` geometric
    tolerance, so repeated bakes produce field-for-field equal dictionaries.

    Raises
    ------
    BakeError
        Only for an invalid batch envelope (for example non-list ``tiles`` or
        duplicate/missing tile ids).  Per-tile failures are reported in the
        returned ``"failed"`` list and do not leak partial output.
    """
    raise NotImplementedError
