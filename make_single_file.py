"""Build the copy/paste-friendly standalone V5 + colour-robust module.

Run from repository root:
    python make_single_file.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V5 = ROOT / "wafer_die_map_v5.py"
ROBUST = ROOT / "wafer_die_map_color_robust.py"
OUT = ROOT / "wafer_die_map_v6_single.py"


def main() -> None:
    v5_text = V5.read_text(encoding="utf-8")
    robust_text = ROBUST.read_text(encoding="utf-8")
    # The standalone extension uses the current file as the V5 namespace;
    # remove its imports and annotations already provided by V5.
    robust_text = robust_text.replace("from __future__ import annotations\n\n", "", 1)
    robust_text = robust_text.replace("from dataclasses import dataclass\n", "", 1)
    robust_text = robust_text.replace("from pathlib import Path\n", "", 1)
    robust_text = robust_text.replace("from typing import Any, Dict, List, Literal, Optional, Tuple, Union\n\n", "")
    robust_text = robust_text.replace("import cv2\nimport numpy as np\n\n", "", 1)
    robust_text = robust_text.replace("import wafer_die_map_v5 as v5\n\n", "", 1)
    robust_text = robust_text.replace("v5.", "",)
    robust_text = robust_text.replace(
        "\n__all__ = [\n    \"ColorRobustConfig\",\n    \"detect_grid_color_robust\",\n    \"build_die_map_robust\",\n    \"make_grid_diagnostic\",\n]\n",
        "\n# Standalone extension public API: ColorRobustConfig, detect_grid_color_robust,\n# build_die_map_robust, make_grid_diagnostic.\n",
        1,
    )
    header = '''\n\n# =============================================================================\n# V6 COLOUR-ROBUST EXTENSION (standalone: no local-module import required)\n# =============================================================================\n# This section is generated from wafer_die_map_color_robust.py.\n# build_die_map() remains the original V5 entry point.\n# build_die_map_robust() is the colour-invariant entry point.\n\n__all__ = list(__all__) + [\n    "ColorRobustConfig",\n    "detect_grid_color_robust",\n    "build_die_map_robust",\n    "make_grid_diagnostic",\n]\n\n'''
    OUT.write_text(v5_text.rstrip() + header + robust_text.lstrip(), encoding="utf-8")
    print(f"Wrote {OUT.name} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
