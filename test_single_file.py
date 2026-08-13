"""Smoke/regression test for the copy/paste standalone module."""

from pathlib import Path

import cv2

import wafer_die_map_v6_single as single


ROOT = Path(__file__).resolve().parent


def main() -> None:
    source_text = (ROOT / "wafer_die_map_v6_single.py").read_text(encoding="utf-8")
    assert "import wafer_die_map_v5" not in source_text
    # Cover every supplied reference image.  The images intentionally span
    # monochrome, coloured, exposed and patterned wafer appearances.
    for path, expected in (
        (ROOT / "Img" / "portable_bw_overlay.png", (90.0, 90.0)),
        (ROOT / "Img" / "portable_bw_sample.png", (90.0, 90.0)),
        (ROOT / "Img" / "real_casio_top_p092.png", (92.0, 92.0)),
        (ROOT / "Img" / "real_exposed_top_p078.png", (78.0, 78.0)),
        (ROOT / "Img" / "real_mips_top_p084.png", (84.0, 84.0)),
        (ROOT / "Img" / "real_piper_top_p088.png", (88.0, 88.0)),
    ):
        die_map, info = single.build_die_map_robust(
            path,
            config=single.ColorRobustConfig(min_pitch=60, max_pitch=110),
            return_info=True,
        )
        assert abs(die_map.pitch_x - expected[0]) <= 1.0
        assert abs(die_map.pitch_y - expected[1]) <= 1.0
        assert die_map.num_dies > 100
        print(f"{path.name}: pitch=({die_map.pitch_x:.1f}, {die_map.pitch_y:.1f}), "
              f"dies={die_map.num_dies}, method={info['grid']['selected_method']}")

    # Original V5 API also remains available from this same single module.
    img = cv2.imread(str(ROOT / "Img" / "real_mips_top_p084.png"), cv2.IMREAD_COLOR)
    legacy = single.build_die_map(img, grid_method="std", angle_align_method="none")
    assert legacy.num_dies > 100
    print(f"legacy V5 API: pitch=({legacy.pitch_x:.1f}, {legacy.pitch_y:.1f})")


if __name__ == "__main__":
    main()
