"""Regression test for the committed, real-geometry colour variants."""

from pathlib import Path

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent


def main() -> None:
    reference = build_die_map_robust(
        ROOT / "Img" / "real_mips_top_p084.png",
        config=ColorRobustConfig(min_pitch=45, max_pitch=120, angle_align="none"),
    )
    variants = sorted((ROOT / "TestAssets" / "ColorVariants").glob("[0-9][0-9]_*.jpg"))
    variants = [path for path in variants if not path.stem.endswith("_overlay")]
    assert len(variants) == 7, f"Expected 7 variants, found {len(variants)}"
    for path in variants:
        config = ColorRobustConfig(
            min_pitch=45,
            max_pitch=120,
            angle_align="robust" if "rotated" in path.name else "none",
        )
        result, info = build_die_map_robust(path, config=config, return_info=True)
        assert abs(result.pitch_x - reference.pitch_x) <= 2.0, (path.name, result.pitch_x)
        assert abs(result.pitch_y - reference.pitch_y) <= 2.0, (path.name, result.pitch_y)
        print(f"{path.name}: {info['grid']['selected_method']}, "
              f"pitch=({result.pitch_x:.1f}, {result.pitch_y:.1f}), "
              f"rotation={info['rotation_deg']:.2f}")


if __name__ == "__main__":
    main()
