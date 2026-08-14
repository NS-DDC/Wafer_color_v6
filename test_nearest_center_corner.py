"""Regression tests for the nearest-centre grid-corner convention."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from make_all_diagnostics import _render
from wafer_die_map_v6_single import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent


def _assert_nearest(die_map) -> None:
    # For a rectangular lattice, independently selecting the nearest X and Y
    # street minimizes the Euclidean distance to wafer centre.
    assert abs(die_map.x0 - die_map.wafer_cx) <= die_map.pitch_x / 2.0
    assert abs(die_map.y0 - die_map.wafer_cy) <= die_map.pitch_y / 2.0


def main() -> None:
    # Real supplied wafers: verify each axis is snapped after all refinement.
    for path in sorted((ROOT / "Img").glob("*.png")):
        if "_overlay" in path.stem:
            continue
        die_map, _ = build_die_map_robust(
            path, config=ColorRobustConfig(min_pitch=60, max_pitch=110), return_info=True)
        _assert_nearest(die_map)
        print(f"real {path.name}: origin=({die_map.x0}, {die_map.y0})")

    # Four known-truth cases cover left/right X and above/below Y centre.
    truth = json.loads((ROOT / "TestAssets" / "CenterCorner" / "truth.json").read_text(encoding="utf-8"))
    config = ColorRobustConfig(min_pitch=60, max_pitch=100, angle_align="none")
    for filename, expected in truth.items():
        source = ROOT / "TestAssets" / "CenterCorner" / filename
        die_map, _ = build_die_map_robust(source, config=config, return_info=True)
        ex, ey = expected["nearest_corner"]
        assert abs(die_map.pitch_x - expected["pitch_x"]) <= 1.0, filename
        assert abs(die_map.pitch_y - expected["pitch_y"]) <= 1.0, filename
        assert abs(die_map.x0 - ex) <= 3, (filename, die_map.x0, ex)
        assert abs(die_map.y0 - ey) <= 3, (filename, die_map.y0, ey)
        _assert_nearest(die_map)
        rendered, _, _ = _render(source, config)
        diagnostic = source.with_name(source.stem + "_diagnostic.png")
        assert cv2.imwrite(str(diagnostic), rendered), diagnostic
        print(f"synthetic {filename}: origin=({die_map.x0}, {die_map.y0}), expected=({ex}, {ey})")


if __name__ == "__main__":
    main()
