"""Small regression test for the colour-robust detector.

Run from this directory:
    python test_color_robust.py
"""

from pathlib import Path

import cv2

from wafer_die_map_color_robust import (
    ColorRobustConfig,
    build_die_map_robust,
    make_grid_diagnostic,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_color_robust_diagnostics"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    images = [path for path in sorted((ROOT / "Img").glob("*.png"))
              if "_overlay" not in path.stem]
    assert images, "No input PNGs found under Img"

    for path in images:
        die_map, info = build_die_map_robust(
            path,
            config=ColorRobustConfig(angle_align="none"),
            return_info=True,
        )
        assert die_map.num_dies > 0
        assert die_map.pitch_x >= 40 and die_map.pitch_y >= 40
        diagnostic = make_grid_diagnostic(path, die_map)
        cv2.imwrite(str(OUT / f"{path.stem}_grid.png"), diagnostic)
        print(
            f"{path.name}: method={info['grid']['selected_method']}, "
            f"pitch=({die_map.pitch_x:.2f}, {die_map.pitch_y:.2f}), dies={die_map.num_dies}"
        )


if __name__ == "__main__":
    main()
