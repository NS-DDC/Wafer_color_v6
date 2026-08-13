"""Regression test for small-angle grid alignment.

Each source is rotated synthetically by +/-0.5, +/-1, +/-2 and +/-3 degrees.
The test checks expected correction relative to the source's measured baseline,
near-zero residual after correction, and unchanged die pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


ROOT = Path(__file__).resolve().parent
ANGLES = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Case:
    path: Path
    min_pitch: int
    max_pitch: int
    correction_tolerance_deg: float


CASES = (
    # Real source: its unmodified grid has a measured 1.25-degree tilt, so
    # injected rotations are evaluated relative to that baseline.
    Case(ROOT / "Img" / "real_mips_top_p084.png", 20, 120, 0.05),
    # AI natural-street source: a distinct colour/texture condition.
    Case(ROOT / "TestAssets" / "NaturalColorSeries" / "natural_teal_bluegray.png", 20, 100, 0.06),
)


def _config(case: Case) -> ColorRobustConfig:
    return ColorRobustConfig(min_pitch=case.min_pitch, max_pitch=case.max_pitch)


def main() -> None:
    for case in CASES:
        image = cv2.imread(str(case.path), cv2.IMREAD_COLOR)
        assert image is not None, case.path
        height, width = image.shape[:2]
        baseline, baseline_info = build_die_map_robust(
            image, config=_config(case), return_info=True)
        baseline_angle = float(baseline_info["rotation_deg"])
        print(f"{case.path.name}: baseline correction={baseline_angle:.3f} deg, "
              f"pitch=({baseline.pitch_x:.1f}, {baseline.pitch_y:.1f})")

        for injected in ANGLES:
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), injected, 1.0)
            rotated = cv2.warpAffine(
                image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                borderValue=(0, 0, 0),
            )
            die_map, info = build_die_map_robust(rotated, config=_config(case), return_info=True)
            expected = baseline_angle - injected
            got = float(info["rotation_deg"])
            assert abs(got - expected) <= case.correction_tolerance_deg, (
                case.path.name, injected, expected, got)
            assert abs(die_map.pitch_x - baseline.pitch_x) <= 1.0
            assert abs(die_map.pitch_y - baseline.pitch_y) <= 1.0

            # A second alignment pass on the first aligned result should find
            # essentially no residual tilt.  It also verifies the coordinates
            # and pitch are stable after the warp.
            again, residual_info = build_die_map_robust(
                die_map.aligned_image, config=_config(case), return_info=True)
            residual = float(residual_info["rotation_deg"])
            assert abs(residual) <= 0.05, (case.path.name, injected, residual)
            assert abs(again.pitch_x - baseline.pitch_x) <= 1.0
            assert abs(again.pitch_y - baseline.pitch_y) <= 1.0
            print(f"  injected={injected:+.1f} deg -> correction={got:+.3f} deg, "
                  f"residual={residual:+.3f} deg")


if __name__ == "__main__":
    main()
