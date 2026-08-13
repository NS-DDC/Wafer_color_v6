"""Create a clear grid-and-centre diagnostic for the natural white/brown fixture."""

from pathlib import Path

import cv2
import numpy as np

from wafer_die_map_v6_single import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "TestAssets" / "NaturalColorSeries" / "white_brown_natural_streets_ai.png"
OUTPUT = ROOT / "TestAssets" / "NaturalColorSeries" / "white_brown_natural_streets_ai_grid_center.png"


def _draw_clipped_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                       mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    """Draw a line only within the wafer mask, preserving the black background."""
    line_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.line(line_mask, p1, p2, 255, 1, cv2.LINE_AA)
    pixels = mask & (line_mask > 0)
    # Blend only the stroke pixels; blending a black full-frame layer would
    # otherwise darken the entire wafer.
    tint = np.empty_like(image)
    tint[:] = colour
    blended = cv2.addWeighted(image, 0.25, tint, 0.75, 0)
    image[pixels] = blended[pixels]


def main() -> None:
    die_map, info = build_die_map_robust(
        SOURCE,
        config=ColorRobustConfig(min_pitch=25, max_pitch=90),
        return_info=True,
    )
    result = die_map.aligned_image.copy()
    height, width = result.shape[:2]
    wafer_mask = np.zeros((height, width), np.uint8)
    cv2.circle(wafer_mask, (die_map.wafer_cx, die_map.wafer_cy), die_map.wafer_r, 255, -1)
    inside = wafer_mask.astype(bool)

    # Cyan represents detected street/grid locations.  Draw lines rather than
    # every die rectangle so the underlying die texture remains readable.
    grid_colour = (0, 235, 255)
    x_start = die_map.x0 + np.ceil((0.0 - die_map.x0) / die_map.pitch_x) * die_map.pitch_x
    y_start = die_map.y0 + np.ceil((0.0 - die_map.y0) / die_map.pitch_y) * die_map.pitch_y
    for x in np.arange(x_start, width + die_map.pitch_x, die_map.pitch_x):
        _draw_clipped_line(result, (int(round(x)), 0), (int(round(x)), height - 1), inside, grid_colour)
    for y in np.arange(y_start, height + die_map.pitch_y, die_map.pitch_y):
        _draw_clipped_line(result, (0, int(round(y))), (width - 1, int(round(y))), inside, grid_colour)

    # Strong red crosshair and ring make the detected wafer centre unambiguous.
    cx, cy = die_map.wafer_cx, die_map.wafer_cy
    cv2.circle(result, (cx, cy), 14, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(result, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(result, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "Detected wafer center", (cx + 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"grid {die_map.pitch_x:.0f} x {die_map.pitch_y:.0f} px | {info['grid']['selected_method']}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 235, 255), 2, cv2.LINE_AA)
    assert cv2.imwrite(str(OUTPUT), result), f"Could not write {OUTPUT}"
    print(f"Wrote {OUTPUT.name}: center=({cx}, {cy}), "
          f"pitch=({die_map.pitch_x:.1f}, {die_map.pitch_y:.1f})")


if __name__ == "__main__":
    main()
