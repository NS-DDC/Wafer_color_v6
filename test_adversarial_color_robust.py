"""Adversarial colour/noise regression tests based on the supplied real wafers.

The transformations are deterministic. They preserve real die geometry while
replacing the street appearance with hostile colour/noise patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260813)


def _street_mask(shape: Tuple[int, int], pitch_x: float, pitch_y: float,
                 x0: int, y0: int, thickness: int = 7) -> np.ndarray:
    """Paint the source grid only; the die texture remains untouched."""
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for x in np.arange(x0 - 30 * pitch_x, x0 + 31 * pitch_x, pitch_x):
        cv2.line(mask, (int(round(x)), 0), (int(round(x)), height - 1), 255, thickness)
    for y in np.arange(y0 - 30 * pitch_y, y0 + 31 * pitch_y, pitch_y):
        cv2.line(mask, (0, int(round(y))), (width - 1, int(round(y))), 255, thickness)
    return mask


def _white_brown_noisy_streets(image: np.ndarray, grid) -> np.ndarray:
    out = image.copy()
    mask = _street_mask(out.shape[:2], grid.pitch_x, grid.pitch_y, grid.x0, grid.y0)
    h, w = mask.shape
    tile_h, tile_w = (h + 23) // 24, (w + 23) // 24
    choices = RNG.integers(0, 3, size=(tile_h, tile_w))
    palette = np.array([[45, 80, 145], [80, 120, 185], [225, 225, 225]], np.int16)  # BGR
    colours = np.repeat(np.repeat(palette[choices], 24, axis=0), 24, axis=1)[:h, :w]
    noise = RNG.normal(0, 22, size=(h, w, 1))
    hostile = np.clip(colours + noise, 0, 255).astype(np.uint8)
    out[mask > 0] = hostile[mask > 0]
    return out


def _chroma_only_streets(image: np.ndarray, grid) -> np.ndarray:
    out = image.copy()
    mask = _street_mask(out.shape[:2], grid.pitch_x, grid.pitch_y, grid.x0, grid.y0, thickness=6)
    h, w = mask.shape
    checker = ((np.indices((h, w)).sum(axis=0) // 30) % 2).astype(bool)
    # Similar luminance but sharply different chroma: Lab a/b should preserve
    # this boundary even when a grayscale threshold loses contrast.
    hostile = np.where(checker[..., None], np.array([170, 105, 25], np.uint8),
                       np.array([35, 70, 180], np.uint8))
    out[mask > 0] = hostile[mask > 0]
    return out


def _global_colour_noise(image: np.ndarray, _grid) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + 74) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2 + 25, 0, 255)
    hsv[:, :, 2] = np.clip((hsv[:, :, 2] ** 2) / 255.0 + 8, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(out.astype(np.float32) + RNG.normal(0, 10, size=out.shape), 0, 255).astype(np.uint8)


def _rotated_brown_streets(image: np.ndarray, grid) -> np.ndarray:
    hostile = _white_brown_noisy_streets(image, grid)
    h, w = hostile.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
    return cv2.warpAffine(hostile, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))


def _assert_stable(name: str, transformed: np.ndarray, baseline, *, align: str) -> str:
    cfg = ColorRobustConfig(min_pitch=45, max_pitch=120, angle_align=align)
    result, info = build_die_map_robust(transformed, config=cfg, return_info=True)
    assert abs(result.pitch_x - baseline.pitch_x) <= 2.0, (name, result.pitch_x, baseline.pitch_x)
    assert abs(result.pitch_y - baseline.pitch_y) <= 2.0, (name, result.pitch_y, baseline.pitch_y)
    assert result.num_dies > 100, (name, result.num_dies)
    return (f"{name}: {info['grid']['selected_method']}, pitch="
            f"({result.pitch_x:.1f}, {result.pitch_y:.1f}), rotation={info['rotation_deg']:.2f}")


def main() -> None:
    source = cv2.imread(str(ROOT / "Img" / "real_mips_top_p084.png"), cv2.IMREAD_COLOR)
    assert source is not None
    baseline, _ = build_die_map_robust(
        source, config=ColorRobustConfig(min_pitch=45, max_pitch=120, angle_align="none"), return_info=True)
    tests: list[tuple[str, Callable[[np.ndarray, object], np.ndarray], str]] = [
        ("white_brown_noisy_streets", _white_brown_noisy_streets, "none"),
        ("chroma_only_streets", _chroma_only_streets, "none"),
        ("global_hue_gamma_noise", _global_colour_noise, "none"),
        ("rotated_brown_streets", _rotated_brown_streets, "robust"),
    ]
    print(f"baseline: pitch=({baseline.pitch_x:.1f}, {baseline.pitch_y:.1f})")
    for name, transform, align in tests:
        print(_assert_stable(name, transform(source, baseline), baseline, align=align))

    generated = cv2.imread(str(ROOT / "TestAssets" / "generated_multicolor_noisy_wafer.png"))
    assert generated is not None
    result, info = build_die_map_robust(
        generated, config=ColorRobustConfig(min_pitch=30), return_info=True)
    assert result.num_dies > 100 and result.pitch_x >= 30 and result.pitch_y >= 30
    print(f"ai_generated_multicolor: {info['grid']['selected_method']}, "
          f"pitch=({result.pitch_x:.1f}, {result.pitch_y:.1f}), dies={result.num_dies}")


if __name__ == "__main__":
    main()
