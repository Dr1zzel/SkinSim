import ast
from pathlib import Path

import pytest

from acquisition import Slab, path_length, slabs_to_cover
from skin import Vessel

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
    assert not imported_modules("tissues") & {"skin", "acquisition"}
    assert "acquisition" not in imported_modules("skin")
