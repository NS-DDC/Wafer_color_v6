"""Regression check for AI-generated natural-street colour variants."""

from pathlib import Path

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent


def main() -> None:
    images = sorted((ROOT / "TestAssets" / "NaturalColorSeries").glob("natural_*.png"))
    images = [path for path in images if not path.stem.endswith("_overlay")]
    assert len(images) == 3, f"Expected three source images, found {len(images)}"
    expected_pitch = (38.0, 28.0)
    for path in images:
        die_map, info = build_die_map_robust(
            path, config=ColorRobustConfig(min_pitch=20, max_pitch=100), return_info=True)
        assert abs(die_map.pitch_x - expected_pitch[0]) <= 2.0, (path.name, die_map.pitch_x)
        assert abs(die_map.pitch_y - expected_pitch[1]) <= 2.0, (path.name, die_map.pitch_y)
        assert die_map.num_dies > 1_000, (path.name, die_map.num_dies)
        print(f"{path.name}: {info['grid']['selected_method']}, "
              f"pitch=({die_map.pitch_x:.1f}, {die_map.pitch_y:.1f}), dies={die_map.num_dies}")


if __name__ == "__main__":
    main()
