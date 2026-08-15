"""Regression test for the natural white/brown street AI fixture."""

from pathlib import Path

from wafer_die_map_v6_single import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "TestAssets" / "NaturalColorSeries" / "white_brown_natural_streets_ai.png"


def main() -> None:
    die_map, info = build_die_map_robust(
        SOURCE,
        config=ColorRobustConfig(min_pitch=25, max_pitch=90),
        return_info=True,
    )
    # The generated source is intentionally not an exact CAD render, so use a
    # small tolerance around its observed repeating street pitch.
    assert abs(die_map.pitch_x - 35.0) <= 1.0, die_map.pitch_x
    assert abs(die_map.pitch_y - 36.0) <= 1.0, die_map.pitch_y
    assert die_map.num_dies > 800, die_map.num_dies
    # The default origin is now the detected grid crossing nearest wafer centre.
    assert abs(die_map.x0 - die_map.wafer_cx) <= die_map.pitch_x / 2.0
    assert abs(die_map.y0 - die_map.wafer_cy) <= die_map.pitch_y / 2.0
    print(f"{SOURCE.name}: {info['grid']['selected_method']}, "
          f"pitch=({die_map.pitch_x:.1f}, {die_map.pitch_y:.1f}), "
          f"dies={die_map.num_dies}")


if __name__ == "__main__":
    main()
