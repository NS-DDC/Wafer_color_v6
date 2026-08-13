"""Create reusable multi-colour wafer test images from a supplied real wafer.

The source geometry and die texture are retained.  Only the saw-street colour,
noise, contrast, and optional rotation are changed, allowing repeatable visual
inspection of colour-independent grid alignment.

Run:
    python generate_color_variants.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "TestAssets" / "ColorVariants"
RNG = np.random.default_rng(20260813)


def _street_mask(shape, grid, thickness: int = 7) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for x in np.arange(grid.x0 - 30 * grid.pitch_x, grid.x0 + 31 * grid.pitch_x, grid.pitch_x):
        cv2.line(mask, (round(x), 0), (round(x), height - 1), 255, thickness)
    for y in np.arange(grid.y0 - 30 * grid.pitch_y, grid.y0 + 31 * grid.pitch_y, grid.pitch_y):
        cv2.line(mask, (0, round(y)), (width - 1, round(y)), 255, thickness)
    return mask


def _tile_colours(mask: np.ndarray, palette: np.ndarray, tile: int, noise: float) -> np.ndarray:
    h, w = mask.shape
    choices = RNG.integers(0, len(palette), size=((h + tile - 1) // tile, (w + tile - 1) // tile))
    field = np.repeat(np.repeat(palette[choices], tile, axis=0), tile, axis=1)[:h, :w]
    return np.clip(field + RNG.normal(0, noise, size=(h, w, 1)), 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(ROOT / "Img" / "real_mips_top_p084.png"), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError("Img/real_mips_top_p084.png")
    grid = build_die_map_robust(
        source, config=ColorRobustConfig(min_pitch=45, max_pitch=120, angle_align="none"))
    mask = _street_mask(source.shape, grid)

    palettes: Dict[str, np.ndarray] = {
        "01_white_brown_noise": np.array([[35, 70, 145], [75, 120, 190], [238, 238, 238]], np.int16),
        "02_cyan_magenta_chroma": np.array([[190, 125, 25], [135, 40, 190], [45, 175, 170]], np.int16),
        "03_gold_blue_street": np.array([[185, 125, 30], [235, 190, 70], [100, 55, 15]], np.int16),
        "04_purple_green_street": np.array([[175, 45, 155], [55, 170, 70], [140, 90, 105]], np.int16),
        "05_dark_low_contrast": np.array([[72, 76, 82], [88, 91, 98], [105, 105, 110]], np.int16),
        "06_multicolor_speckled": np.array([[40, 55, 190], [180, 170, 40], [190, 60, 155],
                                              [65, 180, 75], [230, 230, 230]], np.int16),
    }
    for name, palette in palettes.items():
        image = source.copy()
        street = _tile_colours(mask, palette, tile=24, noise=20 if "speckled" in name else 12)
        image[mask > 0] = street[mask > 0]
        cv2.imwrite(str(OUT / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 96])

    rotated = cv2.imread(str(OUT / "01_white_brown_noise.jpg"))
    h, w = rotated.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
    rotated = cv2.warpAffine(rotated, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    cv2.imwrite(str(OUT / "07_white_brown_noise_rotated_5deg.jpg"), rotated,
                [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"Created {len(palettes) + 1} variants in {OUT}")


if __name__ == "__main__":
    main()
