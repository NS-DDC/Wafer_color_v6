"""Create known-truth fixtures for nearest-wafer-centre corner selection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from make_color_wafers_claude import WaferSpec, synth_wafer


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "TestAssets" / "CenterCorner"

# Each axis deliberately places the closest street on a different side of the
# wafer centre.  Negative offsets are represented as phase = pitch - offset.
CASES = {
    "left_top": (-19.0, -17.0),
    "right_top": (19.0, -17.0),
    "left_bottom": (-19.0, 17.0),
    "right_bottom": (19.0, 17.0),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_truth = {}
    for number, (name, (dx, dy)) in enumerate(CASES.items(), start=1):
        pitch_x, pitch_y = 80.0, 76.0
        cx, cy = 790, 810
        spec = WaferSpec(
            name=name, size=1600, radius=700, cx=cx, cy=cy,
            pitch_x=pitch_x, pitch_y=pitch_y,
            phase_x=dx % pitch_x, phase_y=dy % pitch_y,
            street_w=8.0, bg=(4, 4, 4), die=(132, 114, 94),
            street=(244, 244, 242), ink=(172, 154, 132),
            speckle=(45, 82, 138), speckle_amt=0.22, speckle_on="street",
            die_jitter=8.0, noise_sigma=3.0, illum=0.12, seed=900 + number,
        )
        image, truth = synth_wafer(spec)
        filename = f"{number:02d}_{name}.png"
        assert cv2.imwrite(str(OUT_DIR / filename), image)
        truth["nearest_corner"] = [int(round(cx + dx)), int(round(cy + dy))]
        all_truth[filename] = truth
        print(f"Wrote {filename}: nearest_corner={truth['nearest_corner']}")
    (OUT_DIR / "truth.json").write_text(json.dumps(all_truth, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
