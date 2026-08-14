"""Render the real-camera X-nearest/Y-upper profile for raw Img sources only."""

from __future__ import annotations

from pathlib import Path

import cv2

from make_all_diagnostics import _render_map
from wafer_die_map_color_robust import ColorRobustConfig
from wafer_die_map_real_axis import build_die_map_real_axis


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "TestAssets" / "AllDiagnostics"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "Img").glob("*.png")):
        if "_overlay" in source.stem:
            continue
        config = ColorRobustConfig(min_pitch=60, max_pitch=110)
        die_map, info = build_die_map_real_axis(source, config=config, return_info=True)
        image, _, _ = _render_map(
            die_map, info, "BLUE: X-nearest / Y-upper reference")
        output = OUT_DIR / f"real_axis__{source.stem}__diagnostic.png"
        assert cv2.imwrite(str(output), image), output
        print(f"{source.name}: {output.name} reference=({die_map.x0}, {die_map.y0})")


if __name__ == "__main__":
    main()
