"""Validate the asymmetric real-camera origin on raw supplied images only."""

from __future__ import annotations

from pathlib import Path

from wafer_die_map_color_robust import ColorRobustConfig
from wafer_die_map_real_axis import build_die_map_real_axis


ROOT = Path(__file__).resolve().parent


def main() -> None:
    sources = [path for path in sorted((ROOT / "Img").glob("*.png"))
               if "_overlay" not in path.stem]
    assert sources, "No raw real-camera PNG images found in Img/."
    for source in sources:
        die_map, info = build_die_map_real_axis(
            source,
            config=ColorRobustConfig(min_pitch=60, max_pitch=110),
            return_info=True,
        )
        # X is closest to centre; Y is deliberately the street above centre.
        assert abs(die_map.x0 - die_map.wafer_cx) <= die_map.pitch_x / 2.0, source.name
        assert die_map.wafer_cy - die_map.pitch_y < die_map.y0 <= die_map.wafer_cy, source.name
        assert info["axis_reference"]["profile"] == "real_camera_x_nearest_y_upper"
        print(f"{source.name}: centre=({die_map.wafer_cx}, {die_map.wafer_cy}), "
              f"reference=({die_map.x0}, {die_map.y0}), "
              f"delta=({die_map.x0 - die_map.wafer_cx}, {die_map.y0 - die_map.wafer_cy})")


if __name__ == "__main__":
    main()
