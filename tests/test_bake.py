"""Red tests for the navmesh baking exercise.

Every test calls a contract stub in ``navmesh.bake``.  The initial repository is
expected to produce 15 failures (``NotImplementedError``), with no collection,
import, or syntax errors.
"""

import math

import pytest

from navmesh.bake import (
    BakeError,
    apply_obstacles,
    assign_costs,
    bake_batch,
    classify_slopes,
    merge_triangles,
    shrink_boundary,
    split_islands,
    validate,
)


def test_merge_triangles_uses_degrees_for_angle_threshold():
    triangles = [
        {
            "id": 0,
            "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
            "normal": (0.0, 1.0, 0.0),
        },
        {
            "id": 1,
            "vertices": [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            "normal": (0.0, 0.7071067811865476, -0.7071067811865476),
        },
    ]

    groups = merge_triangles(triangles, max_angle_deg=30.0, max_deviation=10.0)

    # 45 degrees apart must not merge at a 30 degree threshold.  Treating 30
    # as radians would wrongly merge them because 0.785 < 30.
    assert groups == [[0], [1]]


def test_shrink_boundary_uses_edge_normals_at_concave_corner():
    polygon = [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, 3.0),
        (0.0, 0.0, 3.0),
    ]
    expected = [
        (0.25, 0.0, 0.25),
        (2.75, 0.0, 0.25),
        (2.75, 0.0, 0.75),
        (0.75, 0.0, 0.75),
        (0.75, 0.0, 2.75),
        (0.25, 0.0, 2.75),
    ]

    inset = shrink_boundary(
        polygon,
        agent_radius=0.25,
        step_height=0.4,
        max_slope_deg=30.0,
    )

    assert len(inset) == len(expected)
    for got, want in zip(inset, expected):
        assert got == pytest.approx(want, abs=1e-9)
    # The inset of the concave L corner must stay simple; vertex-normal
    # offsetting pushes this corner outward and crosses the opposite arm.
    assert inset[3][0] < 1.0 and inset[3][2] < 1.0
    assert inset[4][0] < 1.0 and inset[4][2] > 1.0


def test_step_height_and_max_slope_are_independent_thresholds():
    walkable_step = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 0.4, 5.0),
        (0.0, 0.4, 5.0),
    ]
    with pytest.raises(BakeError):
        # 0.4 m step over 5 m is only 4.57 degrees, but it still exceeds the
        # 0.3 m step-height limit; it must not be accepted as a slope.
        shrink_boundary(
            walkable_step,
            agent_radius=0.2,
            step_height=0.3,
            max_slope_deg=30.0,
        )

    steep_ramp = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.2, 0.3),
        (0.0, 0.2, 0.3),
    ]
    with pytest.raises(BakeError):
        # 0.2 m is below the step limit but 33.69 degrees exceeds the slope
        # limit.  The two gates are separate.
        shrink_boundary(
            steep_ramp,
            agent_radius=0.1,
            step_height=0.3,
            max_slope_deg=30.0,
        )


def test_classify_slopes_uses_face_normal_not_height_delta():
    triangles = [
        {
            "id": 0,
            "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
            "normal": (0.0, 1.0, 0.0),
        },
        {
            "id": 1,
            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8660254037844387, -0.5)],
            "normal": (0.0, 0.5, 0.8660254037844387),
        },
    ]

    result = classify_slopes(triangles, max_slope_deg=45.0)

    assert result == {"walkable": [0], "blocked": [1]}


def test_split_islands_groups_by_shared_edge_only():
    polygons = [
        {
            "id": 0,
            "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
        },
        {
            "id": 1,
            "vertices": [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
        },
        {
            "id": 2,
            "vertices": [(10.0, 0.0, 0.0), (10.0, 0.0, 1.0), (11.0, 0.0, 0.0)],
        },
    ]

    components = split_islands(polygons)

    # Polygons 0 and 1 share the edge (0,0,1)-(1,0,0).  Triangle 2 only looks
    # close by centroid; it must remain a separate island.
    assert components == [[0, 1], [2]]


def test_apply_obstacles_recomputes_connectivity_and_islands():
    navmesh = {
        "polygons": [
            {
                "id": 0,
                "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                "island": 0,
            },
            {
                "id": 1,
                "vertices": [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                "island": 0,
            },
        ],
        "agents": [{"id": "scout", "radius": 0.2}],
    }
    obstacles = [
        {
            "id": "gate",
            "vertices": [
                (0.02, 0.0, 1.02),
                (1.02, 0.0, 0.02),
                (0.98, 0.0, -0.02),
                (-0.02, 0.0, 0.98),
            ],
            "height": 2.0,
        }
    ]

    result = apply_obstacles(navmesh, obstacles)

    assert result["islands"] == [[0], [1]]
    assert result["polygons"][0]["island"] != result["polygons"][1]["island"]


def test_apply_obstacles_considers_every_agent_radius():
    navmesh = {
        "polygons": [
            {
                "id": 0,
                "vertices": [
                    (0.0, 0.0, 0.0),
                    (4.0, 0.0, 0.0),
                    (4.0, 0.0, 4.0),
                    (0.0, 0.0, 4.0),
                ],
            }
        ],
        "agents": [
            {"id": "scout", "radius": 0.2},
            {"id": "hauler", "radius": 1.2},
        ],
    }
    obstacles = [
        {
            "id": "left-wall",
            "vertices": [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.5, 0.0, 4.0), (0.0, 0.0, 4.0)],
            "height": 2.0,
        },
        {
            "id": "right-wall",
            "vertices": [(2.5, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 0.0, 4.0), (2.5, 0.0, 4.0)],
            "height": 2.0,
        },
    ]

    result = apply_obstacles(navmesh, obstacles)

    # The passage is 1.0 m wide.  The 0.2 m scout fits, the 1.2 m hauler does
    # not.  Ignoring every radius after the first would clear the hauler.
    assert result["agent_clearance"] == {"scout": True, "hauler": False}


def test_assign_costs_defaults_missing_and_rejects_non_positive():
    navmesh = {
        "polygons": [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ]
    }

    assigned = assign_costs(navmesh, {"a": 1.5, "c": 0.75})

    assert assigned["polygons"][0]["cost"] == pytest.approx(1.5, abs=1e-9)
    assert assigned["polygons"][1]["cost"] == pytest.approx(1.0, abs=1e-9)
    assert assigned["polygons"][2]["cost"] == pytest.approx(0.75, abs=1e-9)

    with pytest.raises(BakeError):
        assign_costs(navmesh, {"a": 1.0, "b": -0.1, "c": 1.0})
    with pytest.raises(BakeError):
        assign_costs(navmesh, {"a": 1.0, "b": 0.0, "c": 1.0})


def test_bake_batch_rolls_back_after_any_tile_failure():
    tiles = [
        {
            "id": "tile-a",
            "triangles": [
                {
                    "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                    "normal": (0.0, 1.0, 0.0),
                }
            ],
        },
        {
            "id": "tile-b",
            "triangles": [
                {
                    "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                    "normal": (0.0, 1.0, 0.0),
                }
            ],
        },
        {
            "id": "tile-c",
            "triangles": [
                {
                    "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                    "normal": (0.0, 1.0, 0.0),
                }
            ],
        },
    ]

    result = bake_batch(
        tiles,
        {
            "max_angle_deg": 30.0,
            "max_deviation": 0.01,
            "agent_radius": 0.3,
            "step_height": 0.4,
            "max_slope_deg": 30.0,
        },
    )

    assert result == {"ok": False, "failed": ["tile-b"], "baked": []}


def test_validate_flags_downward_normals_after_bake():
    navmesh = {
        "polygons": [
            {
                "id": 0,
                "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
                "normal": (0.0, -1.0, 0.0),
                "cost": 1.0,
                "island": 0,
            }
        ],
        "islands": [[0]],
    }

    issues = validate(
        navmesh,
        {"max_slope_deg": 45.0, "agent_radius": 0.4, "min_passage_width": 1.0},
    )

    assert "normal_down:0" in issues


def test_baking_same_input_twice_is_field_for_field_deterministic():
    tiles = [
        {
            "id": "tile-1",
            "triangles": [
                {
                    "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                    "normal": (0.0, 1.0, 0.0),
                }
            ],
        }
    ]
    params = {
        "max_angle_deg": 30.0,
        "max_deviation": 0.01,
        "agent_radius": 0.3,
        "step_height": 0.4,
        "max_slope_deg": 30.0,
    }

    first = bake_batch(tiles, params)
    second = bake_batch(tiles, params)

    assert first == second


def test_empty_triangle_list_returns_empty_structures():
    assert merge_triangles([], max_angle_deg=30.0, max_deviation=0.01) == []
    assert classify_slopes([], max_slope_deg=30.0) == {
        "walkable": [],
        "blocked": [],
    }


def test_shrink_boundary_rejects_zero_and_negative_agent_radius():
    polygon = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 0.0, 2.0),
        (0.0, 0.0, 2.0),
    ]

    with pytest.raises(BakeError):
        shrink_boundary(polygon, agent_radius=0.0, step_height=0.4, max_slope_deg=30.0)
    with pytest.raises(BakeError):
        shrink_boundary(polygon, agent_radius=-0.1, step_height=0.4, max_slope_deg=30.0)


def test_assign_costs_uses_absolute_tolerance_of_one_e_nine():
    navmesh = {"polygons": [{"id": "tile"}]}

    assigned = assign_costs(navmesh, {"tile": 1.0 + 1e-10})

    assert math.isclose(assigned["polygons"][0]["cost"], 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_bake_batch_preserves_input_tile_order_on_success():
    tile_ids = ["tile-z", "tile-a", "tile-m"]
    tiles = [
        {
            "id": tile_id,
            "triangles": [
                {
                    "vertices": [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
                    "normal": (0.0, 1.0, 0.0),
                }
            ],
        }
        for tile_id in tile_ids
    ]

    result = bake_batch(
        tiles,
        {
            "max_angle_deg": 30.0,
            "max_deviation": 0.01,
            "agent_radius": 0.3,
            "step_height": 0.4,
            "max_slope_deg": 30.0,
        },
    )

    assert result["ok"] is True
    assert [item["id"] for item in result["baked"]] == tile_ids
