"""Skin layers, vessel geometry, laminar shells, slab path and coverage."""
from __future__ import annotations

import pytest

from geometry import PLEXUS, VesselLevel, laminar_velocities, tissue_at_depth
from tissues import DERMIS, EPIDERMIS, HYPODERMIS, MUSCLE


@pytest.mark.parametrize(
    "depth_mm, expected",
    [(0.0, EPIDERMIS), (0.1, DERMIS), (2.0, HYPODERMIS), (10.0, MUSCLE)],
)
def test_tissue_at_depth_at_each_boundary(depth_mm, expected):
    assert tissue_at_depth(depth_mm) is expected


@pytest.mark.parametrize(
    "depth_mm, expected",
    [(0.0999, EPIDERMIS), (1.999, DERMIS), (9.999, HYPODERMIS)],
)
def test_tissue_at_depth_just_above_each_boundary(depth_mm, expected):
    assert tissue_at_depth(depth_mm) is expected


def test_laminar_area_weighted_mean_is_half_centreline():
    velocities, weights = laminar_velocities(10.0, 1000)
    mean = float((velocities * weights).sum())
    assert weights.sum() == pytest.approx(1.0)
    assert mean == pytest.approx(10.0)
    assert mean == pytest.approx(0.5 * velocities[0], rel=1e-5)


def test_laminar_mean_is_exact_for_few_shells():
    velocities, weights = laminar_velocities(7.0, 3)
    assert float((velocities * weights).sum()) == pytest.approx(7.0)


# --- from test_acquisition.py ---
import ast
from pathlib import Path

import pytest

from geometry import Slab, Vessel, path_length, slabs_to_cover

MAX_PATH_MM = 50.0
ROOT = Path(__file__).resolve().parents[1]


def vessel(polar_deg: float, azimuth_deg: float = 0.0) -> Vessel:
    return Vessel(depth_mm=3.0, polar_deg=polar_deg, azimuth_deg=azimuth_deg, diameter_mm=0.3)


def test_path_equals_thickness_when_vessel_parallel_to_slab_normal():
    assert path_length(Slab(2.0, 0.0), vessel(0.0), MAX_PATH_MM) == pytest.approx(2.0)
    assert path_length(Slab(2.0, 30.0), vessel(30.0), MAX_PATH_MM) == pytest.approx(2.0)


def test_path_doubles_at_60_degrees():
    assert path_length(Slab(2.0, 0.0), vessel(60.0), MAX_PATH_MM) == pytest.approx(4.0)


def test_bound_applies_when_vessel_perpendicular_to_slab_normal():
    assert path_length(Slab(2.0, 0.0), vessel(90.0), MAX_PATH_MM) == MAX_PATH_MM


def test_azimuth_matters_once_slab_is_tilted():
    slab = Slab(2.0, 60.0)
    assert path_length(slab, vessel(90.0, 0.0), MAX_PATH_MM) == pytest.approx(2.0 / (3**0.5 / 2))
    assert path_length(slab, vessel(90.0, 90.0), MAX_PATH_MM) == MAX_PATH_MM


def test_one_slab_when_parallel_to_skin_and_thickness_equals_depth():
    assert slabs_to_cover(Slab(10.0, 0.0), target_depth_mm=10.0, target_width_mm=40.0) == 1


def test_slab_perpendicular_to_skin_tiles_the_width():
    assert slabs_to_cover(Slab(10.0, 90.0), target_depth_mm=10.0, target_width_mm=40.0) == 4


def imported_modules(name: str) -> set[str]:
    tree = ast.parse((ROOT / f"{name}.py").read_text(encoding="utf-8"))
    return {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }


def test_dependencies_run_one_way():
    """The layering holds: data knows nothing of geometry, geometry nothing of sequences."""
    layers = {"tissues": {"geometry", "sequence", "detect", "protocols"},
              "geometry": {"sequence", "detect", "protocols"},
              "affine": {"tissues", "geometry", "sequence", "detect", "protocols"},
              "sequence": {"detect", "protocols"},
              "detect": {"sequence", "protocols", "geometry"}}
    for module, forbidden in layers.items():
        leaked = imported_modules(module) & forbidden
        assert not leaked, f"{module}.py must not import {sorted(leaked)}"


@pytest.mark.parametrize("level", PLEXUS, ids=lambda v: v.name)
def test_every_plexus_level_sits_inside_the_skin_and_cites_a_source(level):
    assert tissue_at_depth(level.top_mm) is not None
    assert level.bottom_mm >= level.top_mm
    assert "doi:" in level.source or "not a measurement" in level.source


def test_plexus_is_ordered_by_depth_and_widens_with_it():
    tops = [v.top_mm for v in PLEXUS]
    assert tops == sorted(tops)
    # vessels get wider as the tree descends; compare the dermal levels, which
    # are the ones measured rather than targeted
    dermal = [v for v in PLEXUS if v.bottom_mm <= 2.6]
    assert [v.diameter_um[1] for v in dermal] == sorted(v.diameter_um[1] for v in dermal)


def test_vessel_level_rejects_inverted_depths_and_diameters():
    with pytest.raises(ValueError):
        VesselLevel("bad", 2.0, 1.0, (10.0, 20.0), "test")
    with pytest.raises(ValueError):
        VesselLevel("bad", 1.0, 2.0, (20.0, 10.0), "test")
