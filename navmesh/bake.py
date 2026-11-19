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
valid result raise :class:`BakeError`.
"""

from __future__ import annotations

import math
from copy import deepcopy

EPS = 1e-9
UP = (0.0, 1.0, 0.0)


class BakeError(ValueError):
    """Raised when navmesh input or a baking parameter violates the contract."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _as_point(point) -> tuple[float, float, float]:
    if not isinstance(point, (tuple, list)) or len(point) != 3:
        raise BakeError("a point must be a sequence of three coordinates")
    if not all(_is_number(c) for c in point):
        raise BakeError("point contains a non-finite coordinate")
    return (float(point[0]), float(point[1]), float(point[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    length = _norm(v)
    if length <= EPS:
        raise BakeError("zero-length vector cannot be normalized")
    return (v[0] / length, v[1] / length, v[2] / length)


def _distance3(a, b):
    return _norm(_sub(a, b))


def _triangle_vertices(triangle) -> list[tuple[float, float, float]]:
    if not isinstance(triangle, dict) or "vertices" not in triangle:
        raise BakeError("triangle is missing its vertices")
    vertices = triangle["vertices"]
    if not isinstance(vertices, (tuple, list)) or len(vertices) != 3:
        raise BakeError("a triangle must have exactly three vertices")
    return [_as_point(p) for p in vertices]


def _face_normal(vertices, supplied=None):
    """Return the unit face normal, preferring a validated supplied normal."""
    if supplied is not None:
        if not isinstance(supplied, (tuple, list)) or len(supplied) != 3:
            raise BakeError("normal must be a three-component vector")
        if not all(_is_number(c) for c in supplied):
            raise BakeError("normal contains a non-finite component")
        n = (float(supplied[0]), float(supplied[1]), float(supplied[2]))
        if _norm(n) <= EPS:
            raise BakeError("zero-length normal")
        return _normalize(n)
    a, b, c = vertices
    n = _cross(_sub(b, a), _sub(c, a))
    if _norm(n) <= EPS:
        raise BakeError("degenerate triangle has no computable normal")
    return _normalize(n)


def _canonical_ids(points) -> list[int]:
    """Cluster points equal within EPS and return a stable id per point."""
    representatives: list[tuple] = []
    ids: list[int] = []
    for point in points:
        found = -1
        for idx, other in enumerate(representatives):
            if _distance3(point, other) <= EPS:
                found = idx
                break
        if found < 0:
            found = len(representatives)
            representatives.append(point)
        ids.append(found)
    return ids


def _edge_set(vertex_ids: list[int]) -> set[frozenset[int]]:
    edges = set()
    count = len(vertex_ids)
    for i in range(count):
        a = vertex_ids[i]
        b = vertex_ids[(i + 1) % count]
        if a == b:
            raise BakeError("polygon has a zero-length edge")
        edges.add(frozenset((a, b)))
    return edges


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Always attach the larger root id under the smaller one so
            # component structure stays deterministic.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _components(size: int, adjacency: list[list[int]]) -> list[list[int]]:
    uf = _UnionFind(size)
    for i, neighbours in enumerate(adjacency):
        for j in neighbours:
            uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(size):
        groups.setdefault(uf.find(i), []).append(i)
    ordered = [sorted(members) for members in groups.values()]
    ordered.sort(key=lambda members: members[0])
    return ordered


# ---------------------------------------------------------------------------
# 2D helpers (work in the xz projection; a 2D point is (x, z))
# ---------------------------------------------------------------------------

def _p2(point) -> tuple[float, float]:
    return (float(point[0]), float(point[2]))


def _orient2(a, b, c):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if value > EPS:
        return 1
    if value < -EPS:
        return -1
    return 0


def _on_segment2(p, a, b) -> bool:
    if _orient2(a, b, p) != 0:
        return False
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def _segments_intersect2(a, b, c, d) -> bool:
    o1, o2 = _orient2(a, b, c), _orient2(a, b, d)
    o3, o4 = _orient2(c, d, a), _orient2(c, d, b)
    if o1 != 0 and o2 != 0 and o1 != o2 and o3 != 0 and o4 != 0 and o3 != o4:
        return True
    return (
        _on_segment2(c, a, b)
        or _on_segment2(d, a, b)
        or _on_segment2(a, c, d)
        or _on_segment2(b, c, d)
    )


def _segment_intersection_point2(a, b, c, d):
    """Return the proper crossing point of two segments, else None."""
    direction = (b[0] - a[0], b[1] - a[1])
    other = (d[0] - c[0], d[1] - c[1])
    hit = _line_intersect2(a, direction, c, other)
    if hit is None:
        return None
    t, u = hit
    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        return (a[0] + t * direction[0], a[1] + t * direction[1])
    return None


def _line_intersect2(p, d, q, e):
    det = d[0] * e[1] - d[1] * e[0]
    if abs(det) <= EPS:
        return None
    diff = (q[0] - p[0], q[1] - p[1])
    t = (diff[0] * e[1] - diff[1] * e[0]) / det
    u = (diff[0] * d[1] - diff[1] * d[0]) / det
    return t, u


def _signed_area2(poly) -> float:
    total = 0.0
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        total += a[0] * b[1] - b[0] * a[1]
    return total


def _point_in_poly2(p, poly) -> bool:
    """Boundary-inclusive point-in-polygon test in the xz projection."""
    count = len(poly)
    for i in range(count):
        if _on_segment2(p, poly[i], poly[(i + 1) % count]):
            return True
    inside = False
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        if (a[1] > p[1]) != (b[1] > p[1]):
            cross_x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if cross_x > p[0]:
                inside = not inside
    return inside


def _strictly_inside2(p, poly) -> bool:
    if not _point_in_poly2(p, poly):
        return False
    count = len(poly)
    for i in range(count):
        if _on_segment2(p, poly[i], poly[(i + 1) % count]):
            return False
    return True


def _is_simple2(poly) -> bool:
    count = len(poly)
    for i in range(count):
        for j in range(i + 1, count):
            if _distance2(poly[i], poly[j]) <= EPS:
                return False
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        for j in range(i + 1, count):
            c = poly[j]
            d = poly[(j + 1) % count]
            adjacent = (j == i + 1) or (i == 0 and j == count - 1)
            if adjacent:
                continue
            if _segments_intersect2(a, b, c, d):
                return False
    return True


def _distance2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _offset_polygon2d(poly, radius):
    """Offset a simple polygon by ``radius`` along each edge normal.

    Positive radius insets (moves edges toward the interior), negative radius
    outsets.  Concave corners are produced by intersecting the two adjacent
    offset edge lines, so the result stays simple.  Raises BakeError when the
    offset collapses, reverses orientation or self-intersects.
    """
    count = len(poly)
    if count < 3:
        raise BakeError("a polygon needs at least three vertices")
    if not _is_simple2(poly):
        raise BakeError("polygon is not simple in the xz projection")
    area = _signed_area2(poly)
    if abs(area) <= EPS:
        raise BakeError("polygon has zero projected area")
    orientation = 1.0 if area > 0 else -1.0

    lines = []
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        dx, dz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dz)
        if length <= EPS:
            raise BakeError("polygon has a zero-length edge")
        # Interior normal: left of the directed edge for CCW, right for CW.
        inward = (orientation * (-dz) / length, orientation * dx / length)
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        anchor = (
            midpoint[0] + inward[0] * radius,
            midpoint[1] + inward[1] * radius,
        )
        direction = (dx / length, dz / length)
        lines.append((anchor, direction, inward, a, b))

    result = []
    for i in range(count):
        prev_anchor, prev_dir, _, _, _ = lines[(i - 1) % count]
        cur_anchor, cur_dir, inward, vertex, _ = lines[i]
        hit = _line_intersect2(prev_anchor, prev_dir, cur_anchor, cur_dir)
        if hit is None:
            # Collinear consecutive edges: shift the vertex straight inward.
            result.append(
                (vertex[0] + inward[0] * radius, vertex[1] + inward[1] * radius)
            )
        else:
            t, _ = hit
            result.append(
                (
                    prev_anchor[0] + t * prev_dir[0],
                    prev_anchor[1] + t * prev_dir[1],
                )
            )

    new_area = _signed_area2(result)
    if abs(new_area) <= EPS or (1.0 if new_area > 0 else -1.0) != orientation:
        raise BakeError("boundary offset collapsed or reversed orientation")
    if not _is_simple2(result):
        raise BakeError("offset boundary self-intersects")
    if radius > 0:
        # Erosion invariant: every offset vertex is at least ``radius`` away
        # from the original boundary.  Past the collapse radius the offset
        # lines re-intersect into a small polygon whose vertices are too close
        # to (or on the wrong side of) original edges; winding alone cannot
        # catch this, so test the actual minimum boundary distance.
        for point in result:
            min_distance = min(
                _point_segment_distance2(point, poly[i], poly[(i + 1) % count])
                for i in range(count)
            )
            if min_distance < radius - EPS:
                raise BakeError("boundary offset collapsed past an edge")
    elif radius < 0:
        for point in poly:
            if not _point_in_poly2(point, result):
                raise BakeError("outset boundary failed to contain the polygon")
    return result, lines


def _point_segment_distance2(p, a, b) -> float:
    dx, dz = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dz * dz
    if length_sq <= EPS * EPS:
        return _distance2(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / length_sq
    t = max(0.0, min(1.0, t))
    nearest = (a[0] + t * dx, a[1] + t * dz)
    return _distance2(p, nearest)


# ---------------------------------------------------------------------------
# merge_triangles
# ---------------------------------------------------------------------------

def merge_triangles(
    triangles: list[dict],
    max_angle_deg: float,
    max_deviation: float,
) -> list[list[int]]:
    """Merge adjacent coplanar triangles into triangle groups."""
    if not isinstance(triangles, (tuple, list)):
        raise BakeError("triangles must be a list")
    if not _is_number(max_angle_deg) or max_angle_deg < 0 or max_angle_deg > 180:
        raise BakeError("max_angle_deg must be a finite value in [0, 180] degrees")
    if not _is_number(max_deviation) or max_deviation < 0:
        raise BakeError("max_deviation must be a finite non-negative number")

    count = len(triangles)
    if count == 0:
        return []

    prepared = []
    all_points = []
    for triangle in triangles:
        vertices = _triangle_vertices(triangle)
        normal = _face_normal(vertices, triangle.get("normal") if isinstance(triangle, dict) else None)
        prepared.append((vertices, normal))
        all_points.extend(vertices)

    point_ids = _canonical_ids(all_points)
    triangle_edges = []
    for i in range(count):
        ids = point_ids[3 * i:3 * (i + 1)]
        triangle_edges.append(_edge_set(ids))

    uf = _UnionFind(count)
    limit_rad = math.radians(max_angle_deg)
    for i in range(count):
        vertices_i, normal_i = prepared[i]
        for j in range(i + 1, count):
            vertices_j, normal_j = prepared[j]
            if not (triangle_edges[i] & triangle_edges[j]):
                continue
            # Angle threshold is degrees in, converted to radians only here.
            cos_angle = max(-1.0, min(1.0, _dot(normal_i, normal_j)))
            if math.acos(cos_angle) > limit_rad + math.radians(EPS):
                continue
            if not _within_plane(vertices_i, vertices_j[0], normal_j, max_deviation):
                continue
            if not _within_plane(vertices_j, vertices_i[0], normal_i, max_deviation):
                continue
            uf.union(i, j)

    return _groups_from_uf(uf, count)


def _groups_from_uf(uf, count):
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(uf.find(i), []).append(i)
    ordered = [sorted(members) for members in groups.values()]
    ordered.sort(key=lambda members: members[0])
    return ordered


def _within_plane(vertices, plane_point, plane_normal, max_deviation) -> bool:
    for vertex in vertices:
        distance = abs(_dot(_sub(vertex, plane_point), plane_normal))
        if distance > max_deviation + EPS:
            return False
    return True


# ---------------------------------------------------------------------------
# shrink_boundary
# ---------------------------------------------------------------------------

def shrink_boundary(
    polygon: list[tuple[float, float, float]],
    agent_radius: float,
    step_height: float,
    max_slope_deg: float,
) -> list[tuple[float, float, float]]:
    """Inset a walkable boundary by an agent radius."""
    if not isinstance(polygon, (tuple, list)) or len(polygon) < 3:
        raise BakeError("a boundary needs at least three vertices")
    vertices = [_as_point(p) for p in polygon]
    if not _is_number(agent_radius) or agent_radius <= 0:
        raise BakeError("agent_radius must be a finite positive number")
    if not _is_number(step_height) or step_height < 0:
        raise BakeError("step_height must be a finite non-negative number")
    if not _is_number(max_slope_deg) or not (0 <= max_slope_deg <= 90):
        raise BakeError("max_slope_deg must be a finite value in [0, 90] degrees")

    count = len(vertices)
    for i in range(count):
        a = vertices[i]
        b = vertices[(i + 1) % count]
        delta = _sub(b, a)
        horizontal = math.hypot(delta[0], delta[2])
        if horizontal <= EPS and abs(delta[1]) <= EPS:
            raise BakeError("polygon has a zero-length edge")
        # Independent gate 1: absolute height change vs step height.
        if abs(delta[1]) > step_height + EPS:
            raise BakeError("edge height change exceeds the step-height limit")
        # Independent gate 2: inclination from horizontal in degrees.
        slope_deg = math.degrees(math.atan2(abs(delta[1]), horizontal))
        if slope_deg > max_slope_deg + EPS:
            raise BakeError("edge inclination exceeds the max-slope limit")

    poly2d = [_p2(p) for p in vertices]
    inset2d, _ = _offset_polygon2d(poly2d, agent_radius)

    inset = []
    for i in range(count):
        x, z = inset2d[i]
        # Reconstruct y by projecting the new vertex onto each adjacent edge's
        # line and averaging the two edge-local heights.
        height_estimates = []
        for edge_end in (i, (i + 1) % count):
            a3 = vertices[edge_end]
            b3 = vertices[(edge_end + 1) % count]
            a2 = _p2(a3)
            b2 = _p2(b3)
            edge_2d = (b2[0] - a2[0], b2[1] - a2[1])
            edge_len = math.hypot(edge_2d[0], edge_2d[1])
            if edge_len <= EPS:
                continue
            t = ((x - a2[0]) * edge_2d[0] + (z - a2[1]) * edge_2d[1]) / edge_len**2
            dy = b3[1] - a3[1]
            height_estimates.append(a3[1] + t * dy)
        y = sum(height_estimates) / len(height_estimates) if height_estimates else 0.0
        inset.append((x, y, z))
    return inset


def classify_slopes(triangles: list[dict], max_slope_deg: float) -> dict:
    """Split triangle indices into walkable and blocked surfaces by normal."""
    if not isinstance(triangles, (tuple, list)):
        raise BakeError("triangles must be a list")
    if not _is_number(max_slope_deg) or not (0 <= max_slope_deg <= 90):
        raise BakeError("max_slope_deg must be a finite value in [0, 90] degrees")

    walkable = []
    blocked = []
    for index, triangle in enumerate(triangles):
        vertices = _triangle_vertices(triangle)
        normal = _face_normal(
            vertices, triangle.get("normal") if isinstance(triangle, dict) else None
        )
        # A downward-facing surface is never walkable, even if its unsigned
        # inclination would fit the threshold.
        if normal[1] < -EPS:
            blocked.append(index)
            continue
        cosine = max(-1.0, min(1.0, _dot(normal, UP)))
        slope_deg = math.degrees(math.acos(cosine))
        if slope_deg <= max_slope_deg + EPS:
            walkable.append(index)
        else:
            blocked.append(index)
    return {"walkable": sorted(walkable), "blocked": sorted(blocked)}


# ---------------------------------------------------------------------------
# split_islands
# ---------------------------------------------------------------------------

def split_islands(polygons: list[dict]) -> list[list[int]]:
    """Group polygons into connected components by shared edges."""
    if not isinstance(polygons, (tuple, list)):
        raise BakeError("polygons must be a list")
    if len(polygons) == 0:
        return []

    all_points = []
    polygon_edges = []
    for polygon in polygons:
        if not isinstance(polygon, dict) or "vertices" not in polygon:
            raise BakeError("polygon is missing its vertices")
        raw = polygon["vertices"]
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise BakeError("polygon needs at least three vertices")
        vertices = [_as_point(p) for p in raw]
        all_points.extend(vertices)
        polygon_edges.append(vertices)

    point_ids = _canonical_ids(all_points)
    edge_sets = []
    offset = 0
    adjacency = [[] for _ in polygons]
    edge_owners: dict[frozenset[int], list[int]] = {}
    for index, vertices in enumerate(polygon_edges):
        ids = point_ids[offset:offset + len(vertices)]
        offset += len(vertices)
        edges = _edge_set(ids)
        edge_sets.append(edges)
        for edge in edges:
            owners = edge_owners.setdefault(edge, [])
            for owner in owners:
                adjacency[index].append(owner)
                adjacency[owner].append(index)
            owners.append(index)
    return _components(len(polygons), adjacency)


# ---------------------------------------------------------------------------
# apply_obstacles
# ---------------------------------------------------------------------------

def apply_obstacles(navmesh: dict, obstacles: list[dict]) -> dict:
    """Apply dynamic obstacles and recompute topology and agent clearance."""
    if not isinstance(navmesh, dict) or not isinstance(navmesh.get("polygons"), list):
        raise BakeError("navmesh must contain a polygon list")
    raw_polygons = navmesh["polygons"]
    if len(raw_polygons) == 0:
        raise BakeError("navmesh has no polygons")
    if not isinstance(obstacles, list):
        raise BakeError("obstacles must be a list")

    polygons = []
    all_points = []
    for polygon in raw_polygons:
        if not isinstance(polygon, dict) or "id" not in polygon or "vertices" not in polygon:
            raise BakeError("polygon requires id and vertices")
        raw = polygon["vertices"]
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise BakeError("polygon needs at least three vertices")
        vertices = [_as_point(p) for p in raw]
        new_polygon = deepcopy(polygon)
        new_polygon["vertices"] = vertices
        polygons.append(new_polygon)
        all_points.extend(vertices)

    agents = navmesh.get("agents", [])
    if not isinstance(agents, list):
        raise BakeError("agents must be a list")
    seen_agent_ids = set()
    for agent in agents:
        if not isinstance(agent, dict) or "id" not in agent or "radius" not in agent:
            raise BakeError("agent requires id and radius")
        if agent["id"] in seen_agent_ids:
            raise BakeError(f"duplicate agent id: {agent['id']!r}")
        seen_agent_ids.add(agent["id"])
        if not _is_number(agent["radius"]) or agent["radius"] <= 0:
            raise BakeError("agent radius must be a finite positive number")

    prepared_obstacles = []
    for obstacle in obstacles:
        if not isinstance(obstacle, dict) or "id" not in obstacle:
            raise BakeError("obstacle requires an id")
        if "height" not in obstacle or not _is_number(obstacle["height"]):
            raise BakeError("obstacle height must be finite")
        raw = obstacle.get("vertices")
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise BakeError("obstacle footprint needs at least three vertices")
        footprint = [_as_point(p) for p in raw]
        base_y = min(p[1] for p in footprint)
        prepared_obstacles.append(
            {
                "id": obstacle["id"],
                "footprint": footprint,
                "base_y": base_y,
                "top_y": base_y + float(obstacle["height"]),
            }
        )

    adjacency = _shared_edge_adjacency(polygons, all_points)
    adjacency = _sever_blocked_edges(polygons, adjacency, prepared_obstacles)

    components = _components(len(polygons), adjacency)
    for island_id, component in enumerate(components):
        for index in component:
            polygons[index]["island"] = island_id

    clearance = {}
    for agent in agents:
        clearance[agent["id"]] = _agent_fits(
            agent["radius"], polygons, prepared_obstacles
        )

    return {
        "polygons": polygons,
        "islands": components,
        "agent_clearance": clearance,
        "agents": deepcopy(agents),
    }


def _shared_edge_adjacency(polygons, all_points):
    point_ids = _canonical_ids(all_points)
    edge_owners: dict[frozenset[int], list[int]] = {}
    adjacency = [[] for _ in polygons]
    offset = 0
    for index, polygon in enumerate(polygons):
        ids = point_ids[offset:offset + len(polygon["vertices"])]
        offset += len(polygon["vertices"])
        for edge in _edge_set(ids):
            owners = edge_owners.setdefault(edge, [])
            for owner in owners:
                adjacency[index].append(owner)
                adjacency[owner].append(index)
            owners.append(index)
    return adjacency


def _y_range_overlaps(polygon, obstacle) -> bool:
    low = min(p[1] for p in polygon["vertices"])
    high = max(p[1] for p in polygon["vertices"])
    return not (
        high < obstacle["base_y"] - EPS or low > obstacle["top_y"] + EPS
    )


def _sever_blocked_edges(polygons, adjacency, obstacles):
    point_ids = _canonical_ids(
        [point for polygon in polygons for point in polygon["vertices"]]
    )
    edge_owners: dict[frozenset[int], list[int]] = {}
    offset = 0
    for index, polygon in enumerate(polygons):
        ids = point_ids[offset:offset + len(polygon["vertices"])]
        offset += len(polygon["vertices"])
        for edge in _edge_set(ids):
            edge_owners.setdefault(edge, []).append(index)

    blocked_pairs: set[tuple[int, int]] = set()
    for edge, owners in edge_owners.items():
        if len(owners) < 2:
            continue
        endpoint_ids = list(edge)
        segment = [
            _point_for_canonical_id(point_ids, polygons, endpoint_ids[0]),
            _point_for_canonical_id(point_ids, polygons, endpoint_ids[1]),
        ]
        for obstacle in obstacles:
            if _obstacle_cuts_segment(segment, polygons, obstacle):
                for a in owners:
                    for b in owners:
                        if a != b:
                            blocked_pairs.add((min(a, b), max(a, b)))

    filtered = []
    for i, neighbours in enumerate(adjacency):
        kept = []
        for j in neighbours:
            if (min(i, j), max(i, j)) not in blocked_pairs:
                kept.append(j)
        filtered.append(kept)
    return filtered


def _point_for_canonical_id(point_ids, polygons, canonical_id):
    offset = 0
    for polygon in polygons:
        ids = point_ids[offset:offset + len(polygon["vertices"])]
        offset += len(polygon["vertices"])
        if canonical_id in ids:
            return polygon["vertices"][ids.index(canonical_id)]
    raise BakeError("internal error: canonical edge endpoint not found")


def _obstacle_cuts_segment(segment, polygons, obstacle) -> bool:
    if not any(_y_range_overlaps(polygon, obstacle) for polygon in polygons):
        return False
    a, b = _p2(segment[0]), _p2(segment[1])
    footprint = [_p2(p) for p in obstacle["footprint"]]
    if _strictly_inside2(a, footprint) or _strictly_inside2(b, footprint):
        return True
    if _point_in_poly2(a, footprint) or _point_in_poly2(b, footprint):
        return True
    count = len(footprint)
    for i in range(count):
        c, d = footprint[i], footprint[(i + 1) % count]
        if _segments_intersect2(a, b, c, d):
            return True
        if _on_segment2(c, a, b):
            return True
    return False


def _agent_fits(radius, polygons, obstacles) -> bool:
    """True when free space contains a disk of ``radius`` somewhere."""
    for polygon in polygons:
        outer = [_p2(p) for p in polygon["vertices"]]
        try:
            eroded, _ = _offset_polygon2d(outer, radius)
        except BakeError:
            continue
        inflated_obstacles = []
        for obstacle in obstacles:
            if not _y_range_overlaps(polygon, obstacle):
                continue
            footprint = [_p2(p) for p in obstacle["footprint"]]
            try:
                inflated, _ = _offset_polygon2d(footprint, -radius)
            except BakeError:
                # If the obstacle cannot be offset exactly, conservatively
                # treat its raw footprint as excluded space.
                inflated = footprint
            inflated_obstacles.append(inflated)
        if _region_has_free_point(eroded, inflated_obstacles):
            return True
    return False


def _region_has_free_point(region, obstacles) -> bool:
    candidates = list(region)
    count = len(region)
    for i in range(count):
        a, b = region[i], region[(i + 1) % count]
        candidates.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))

    for obstacle in obstacles:
        o_count = len(obstacle)
        for vertex in obstacle:
            if _point_in_poly2(vertex, region):
                candidates.append(vertex)
        for i in range(o_count):
            a, b = obstacle[i], obstacle[(i + 1) % o_count]
            midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            if _point_in_poly2(midpoint, region):
                candidates.append(midpoint)
            for j in range(count):
                c, d = region[j], region[(j + 1) % count]
                crossing = _segment_intersection_point2(a, b, c, d)
                if crossing is not None:
                    candidates.append(crossing)

    for k, obstacle_a in enumerate(obstacles):
        for obstacle_b in obstacles[k + 1:]:
            for i in range(len(obstacle_a)):
                a = obstacle_a[i]
                b = obstacle_a[(i + 1) % len(obstacle_a)]
                for j in range(len(obstacle_b)):
                    c = obstacle_b[j]
                    d = obstacle_b[(j + 1) % len(obstacle_b)]
                    crossing = _segment_intersection_point2(a, b, c, d)
                    if crossing is not None:
                        candidates.append(crossing)

    for point in candidates:
        if not _point_in_poly2(point, region):
            continue
        if any(_strictly_inside2(point, obstacle) for obstacle in obstacles):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# assign_costs
# ---------------------------------------------------------------------------

def assign_costs(navmesh: dict, area_costs: dict) -> dict:
    """Assign a traversal cost to every polygon."""
    if not isinstance(navmesh, dict) or not isinstance(navmesh.get("polygons"), list):
        raise BakeError("navmesh must contain a polygon list")
    if not isinstance(area_costs, dict):
        raise BakeError("area_costs must be a mapping")

    polygons = navmesh["polygons"]
    known_ids = set()
    for polygon in polygons:
        if not isinstance(polygon, dict) or "id" not in polygon or polygon["id"] is None:
            raise BakeError("every polygon needs a usable id")
        if polygon["id"] in known_ids:
            raise BakeError(f"duplicate polygon id: {polygon['id']!r}")
        known_ids.add(polygon["id"])

    for key in area_costs:
        if key not in known_ids:
            raise BakeError(f"area_costs references unknown polygon id: {key!r}")
        cost = area_costs[key]
        if not _is_number(cost) or cost <= EPS:
            raise BakeError("a supplied cost must be a finite positive number")

    result = deepcopy(navmesh)
    for polygon in result["polygons"]:
        polygon_id = polygon["id"]
        if polygon_id in area_costs:
            polygon["cost"] = float(area_costs[polygon_id])
        else:
            polygon["cost"] = 1.0
    return result


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def validate(navmesh: dict, params: dict) -> list[str]:
    """Return human-readable/stably coded problems with a baked navmesh."""
    if not isinstance(navmesh, dict) or not isinstance(navmesh.get("polygons"), list):
        raise BakeError("navmesh must contain a polygon list")
    if not isinstance(params, dict):
        raise BakeError("params must be a dictionary")

    max_slope_deg = params.get("max_slope_deg", 45.0)
    agent_radius = params.get("agent_radius", 0.0)
    min_passage_width = params.get("min_passage_width", 0.0)
    if not _is_number(max_slope_deg) or not (0 <= max_slope_deg <= 90):
        raise BakeError("max_slope_deg must be a finite value in [0, 90] degrees")
    if not _is_number(agent_radius) or agent_radius < 0:
        raise BakeError("agent_radius must be a finite non-negative number")
    if not _is_number(min_passage_width) or min_passage_width < 0:
        raise BakeError("min_passage_width must be finite and non-negative")

    polygons = navmesh["polygons"]
    for polygon in polygons:
        if not isinstance(polygon, dict) or "id" not in polygon or polygon["id"] is None:
            raise BakeError("every polygon needs a usable id")
        _triangle_vertices_like(polygon)

    issues: set[str] = set()

    for polygon in polygons:
        polygon_id = polygon["id"]
        cost = polygon.get("cost")
        if not _is_number(cost) or cost <= EPS:
            issues.add(f"cost_missing:{polygon_id}")

        vertices = [_as_point(p) for p in polygon["vertices"]]
        normal = _face_normal(vertices, polygon.get("normal"))
        if normal[1] < -EPS:
            issues.add(f"normal_down:{polygon_id}")

    _validate_islands(polygons, issues)
    _validate_passages(polygons, agent_radius, min_passage_width, issues)
    return sorted(issues)


def _triangle_vertices_like(polygon):
    raw = polygon.get("vertices")
    if not isinstance(raw, (tuple, list)) or len(raw) < 3:
        raise BakeError("polygon needs at least three vertices")
    vertices = [_as_point(p) for p in raw]
    poly2d = [_p2(p) for p in vertices]
    if not _is_simple2(poly2d) or abs(_signed_area2(poly2d)) <= EPS:
        raise BakeError("polygon geometry is malformed in the xz projection")
    return vertices


def _validate_islands(polygons, issues):
    components = split_islands(polygons)
    component_values = []
    for component in components:
        values = set()
        for index in component:
            island = polygons[index].get("island")
            if not _is_number(island) or float(island).is_integer() is False:
                for member in component:
                    issues.add(f"island_missing:{polygons[member]['id']}")
                break
            values.add(int(island))
        else:
            component_values.append((component, values))

    seen_values: dict[int, list] = {}
    for component, values in component_values:
        if len(values) != 1:
            for member in component:
                issues.add(f"island_missing:{polygons[member]['id']}")
            continue
        value = next(iter(values))
        seen_values.setdefault(value, []).append(component)

    for value, sharing in seen_values.items():
        if len(sharing) > 1:
            for component in sharing:
                for member in component:
                    issues.add(f"island_missing:{polygons[member]['id']}")


def _validate_passages(polygons, agent_radius, min_passage_width, issues):
    erosion_radius = max(agent_radius, min_passage_width / 2.0)
    if erosion_radius <= EPS:
        return
    for polygon in polygons:
        poly2d = [_p2(_as_point(p)) for p in polygon["vertices"]]
        try:
            _offset_polygon2d(poly2d, erosion_radius)
        except BakeError:
            issues.add(f"narrow_passage:{polygon['id']}")


# ---------------------------------------------------------------------------
# bake_batch
# ---------------------------------------------------------------------------

def bake_batch(tiles: list[dict], params: dict) -> dict:
    """Bake a batch of map tiles atomically."""
    if not isinstance(tiles, list):
        raise BakeError("tiles must be a list")
    if not isinstance(params, dict):
        raise BakeError("params must be a dictionary")

    tile_ids = []
    for tile in tiles:
        if not isinstance(tile, dict) or "id" not in tile or tile["id"] is None:
            raise BakeError("every tile needs a unique id")
        if tile["id"] in tile_ids:
            raise BakeError(f"duplicate tile id: {tile['id']!r}")
        tile_ids.append(tile["id"])

    baked = []
    failed = []
    for tile in tiles:
        try:
            navmesh = _bake_one_tile(tile, params)
        except BakeError:
            failed.append(tile["id"])
            continue
        baked.append({"id": tile["id"], "navmesh": navmesh})

    if failed:
        # Roll back the whole prefix: nothing partially baked is returned.
        return {"ok": False, "failed": failed, "baked": []}
    return {"ok": True, "failed": [], "baked": baked}


def _bake_one_tile(tile, params):
    max_angle_deg = params.get("max_angle_deg", 30.0)
    max_deviation = params.get("max_deviation", 0.01)
    max_slope_deg = params.get("max_slope_deg", 30.0)

    triangles = tile.get("triangles", [])
    merge_triangles(triangles, max_angle_deg, max_deviation)
    slopes = classify_slopes(triangles, max_slope_deg)

    polygons = []
    for index in slopes["walkable"]:
        triangle = triangles[index]
        vertices = _triangle_vertices(triangle)
        polygons.append(
            {
                "id": index,
                "vertices": vertices,
                "normal": _face_normal(
                    vertices, triangle.get("normal") if isinstance(triangle, dict) else None
                ),
            }
        )

    components = split_islands(polygons)
    for island_id, component in enumerate(components):
        for member in component:
            polygons[member]["island"] = island_id

    navmesh = {
        "polygons": polygons,
        "islands": components,
        "agents": deepcopy(params.get("agents", [])),
    }

    if tile.get("obstacles"):
        navmesh = apply_obstacles(navmesh, tile["obstacles"])

    return assign_costs(navmesh, tile.get("area_costs", {}))
