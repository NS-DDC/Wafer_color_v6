"""Make a compact visual index of all full-resolution diagnostic renders."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "TestAssets" / "AllDiagnostics"
OUTPUT = SOURCE / "all_diagnostics_contact_sheet.jpg"


def main() -> None:
    files = sorted(SOURCE.glob("*/*/diagnostic.png"))
    assert len(files) == 19, f"Expected 19 diagnostic files, got {len(files)}"
    tile_w, tile_h, label_h, cols = 300, 300, 34, 5
    rows = int(np.ceil(len(files) / cols))
    sheet = np.zeros((rows * (tile_h + label_h), cols * tile_w, 3), np.uint8)
    for index, path in enumerate(files):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, path
        thumb = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, cols)
        y, x = row * (tile_h + label_h), col * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = thumb
        label = f"{path.parent.parent.name}/{path.parent.name}"
        cv2.putText(sheet, label[:44], (x + 5, y + tile_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(OUTPUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]), OUTPUT
    print(f"Wrote {OUTPUT.name} with {len(files)} fixtures")


if __name__ == "__main__":
    main()
