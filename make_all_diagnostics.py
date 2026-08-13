"""Render grid, wafer centre and nearest reference corner for every fixture.

The source images are never edited. All result images are written directly to
``TestAssets/AllDiagnostics`` so they can be checked in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from wafer_die_map_v6_single import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "TestAssets" / "AllDiagnostics"


def _sources() -> Iterable[tuple[str, Path, ColorRobustConfig]]:
    # These are visual grid checks, so avoid geometric warping unless the
    # fixture is explicitly rotated. It also keeps all diagnostic coordinates
    # in each source image's native pixel system.
    base = ColorRobustConfig(min_pitch=25, max_pitch=160, angle_align="none")
    for path in sorted((ROOT / "Img").glob("*.png")):
        # Match the real-image regression bounds: a wide unconstrained search
        # can choose a repeated feature inside a die as a false half-pitch.
        yield "img", path, ColorRobustConfig(min_pitch=60, max_pitch=110)
    for path in sorted((ROOT / "TestAssets" / "ColorVariants").glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and "_overlay" not in path.stem:
            align = "robust" if "rotated" in path.stem else "none"
            yield "variant", path, ColorRobustConfig(min_pitch=45, max_pitch=120, angle_align=align)
    for path in sorted((ROOT / "TestAssets" / "NaturalColorSeries").glob("natural_*.png")):
        if "_overlay" not in path.stem:
            yield "natural", path, ColorRobustConfig(min_pitch=20, max_pitch=100, angle_align="none")
    yield "natural", ROOT / "TestAssets" / "NaturalColorSeries" / "white_brown_natural_streets_ai.png", base
    for name in ("generated_multicolor_noisy_wafer.png", "generated_multicolor_natural_streets_v2.png"):
        yield "generated", ROOT / "TestAssets" / name, ColorRobustConfig(min_pitch=25, max_pitch=100, angle_align="none")


def _line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
          wafer: np.ndarray, colour: tuple[int, int, int]) -> None:
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.line(mask, p1, p2, 255, 1, cv2.LINE_AA)
    pixels = wafer & (mask > 0)
    tint = np.empty_like(image)
    tint[:] = colour
    image[pixels] = cv2.addWeighted(image, 0.25, tint, 0.75, 0)[pixels]


def _render(source: Path, config: ColorRobustConfig) -> tuple[np.ndarray, object, dict]:
    die_map, info = build_die_map_robust(source, config=config, return_info=True)
    image = die_map.aligned_image.copy()
    height, width = image.shape[:2]
    wafer_mask = np.zeros((height, width), np.uint8)
    cv2.circle(wafer_mask, (die_map.wafer_cx, die_map.wafer_cy), die_map.wafer_r, 255, -1)
    inside = wafer_mask.astype(bool)
    grid_colour = (0, 235, 255)
    x_start = die_map.x0 + np.ceil(-die_map.x0 / die_map.pitch_x) * die_map.pitch_x
    y_start = die_map.y0 + np.ceil(-die_map.y0 / die_map.pitch_y) * die_map.pitch_y
    for x in np.arange(x_start, width + die_map.pitch_x, die_map.pitch_x):
        _line(image, (int(round(x)), 0), (int(round(x)), height - 1), inside, grid_colour)
    for y in np.arange(y_start, height + die_map.pitch_y, die_map.pitch_y):
        _line(image, (0, int(round(y))), (width - 1, int(round(y))), inside, grid_colour)

    cx, cy = die_map.wafer_cx, die_map.wafer_cy
    cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 12, (0, 0, 255), 2, cv2.LINE_AA)
    x0, y0 = int(round(die_map.x0)), int(round(die_map.y0))
    diamond = np.array([[x0, y0 - 12], [x0 + 12, y0], [x0, y0 + 12], [x0 - 12, y0]], np.int32)
    cv2.fillConvexPoly(image, diamond, (255, 90, 0), cv2.LINE_AA)
    cv2.polylines(image, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "RED: wafer center | BLUE: nearest grid corner", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"pitch {die_map.pitch_x:.1f} x {die_map.pitch_y:.1f}px | {info['grid']['selected_method']}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 235, 255), 2, cv2.LINE_AA)
    return image, die_map, info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ["# All diagnostic renders", "",
                "Yellow: detected grid; red: wafer centre; blue: nearest grid reference corner.", "",
                "| Source | Method | Pitch (px) | Centre | Nearest corner | Result |",
                "| --- | --- | --- | --- | --- | --- |"]
    failures: list[str] = []
    for group, source, config in _sources():
        result_name = f"{group}__{source.stem}__diagnostic.png"
        output = OUT_DIR / result_name
        if output.exists():
            # Regenerate the small metadata record, but preserve the existing
            # full-resolution render on resumptions.
            try:
                die_map, info = build_die_map_robust(source, config=config, return_info=True)
                manifest.append(
                    f"| `{group}/{source.name}` | {info['grid']['selected_method']} | "
                    f"{die_map.pitch_x:.1f} x {die_map.pitch_y:.1f} | "
                    f"({die_map.wafer_cx}, {die_map.wafer_cy}) | "
                    f"({die_map.x0}, {die_map.y0}) | [{result_name}]({result_name}) |"
                )
                print(f"INDEX {group}/{source.name}: existing result")
                continue
            except Exception as exc:
                failures.append(f"{group}/{source.name}: {exc}")
                print(f"FAIL {failures[-1]}")
                continue
        try:
            rendered, die_map, info = _render(source, config)
            assert cv2.imwrite(str(output), rendered), output
            manifest.append(
                f"| `{group}/{source.name}` | {info['grid']['selected_method']} | "
                f"{die_map.pitch_x:.1f} x {die_map.pitch_y:.1f} | "
                f"({die_map.wafer_cx}, {die_map.wafer_cy}) | "
                f"({die_map.x0}, {die_map.y0}) | [{result_name}]({result_name}) |"
            )
            print(f"OK  {group}/{source.name}: {die_map.pitch_x:.1f} x {die_map.pitch_y:.1f}")
        except Exception as exc:
            failures.append(f"{group}/{source.name}: {exc}")
            print(f"FAIL {failures[-1]}")
    (OUT_DIR / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Diagnostic failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
