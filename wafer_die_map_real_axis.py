"""Real-camera axis-reference profile for colour-robust wafer die maps.

Use this *separate* profile for the supplied real camera images when the
reference convention is intentionally asymmetric:

* X axis: the vertical street closest to wafer centre.
* Y axis: the horizontal street immediately **above** wafer centre.

The general colour-robust module keeps its symmetric ``nearest_center``
default for synthetic fixtures and installations that need the mathematically
nearest crossing.  This file only changes the reported lattice reference; it
reuses the same colour, angle, and fractional-pitch detection logic.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust
import wafer_die_map_v5 as v5

__all__ = ["RealAxisConfig", "build_die_map_real_axis"]


# [SECTOR: 10_REAL_CAMERA_AXIS_REFERENCE]
RealAxisConfig = ColorRobustConfig
"""Alias of :class:`ColorRobustConfig` for the real-camera profile.

Every usual colour/angle/pitch option is available. ``origin_mode`` is
overridden by :func:`build_die_map_real_axis` so the real-image convention
cannot accidentally drift back to the symmetric test-fixture convention.
"""


def build_die_map_real_axis(
        image: Union[str, Path, np.ndarray], *,
        config: Optional[ColorRobustConfig] = None,
        pixel_per_unit: int = v5.DEFAULT_PIXEL_PER_UNIT,
        include_edge: bool = True,
        edge_margin: float = v5.DEFAULT_EDGE_MARGIN,
        edge_mode: str = v5.DEFAULT_EDGE_MODE,
        with_crops: bool = False,
        border_mode: str = "pad",
        offset_x: int = 0, offset_y: int = 0,
        margin_x: int = 0, margin_y: int = 0,
        return_info: bool = False,
        ) -> Union[v5.WaferDieMap, Tuple[v5.WaferDieMap, Dict[str, Any]]]:
    """Build a die map using the real-camera X-nearest/Y-upper convention.

    With image coordinates growing downward, the returned reference corner is
    guaranteed to satisfy ``x0`` nearest ``wafer_cx`` and
    ``wafer_cy - pitch_y < y0 <= wafer_cy``.  In other words, it is the
    nearest vertical street and the preceding (upper) horizontal street.

    Synthetic fixtures should continue to call ``build_die_map_robust`` from
    ``wafer_die_map_color_robust.py`` to use its symmetric nearest-corner
    convention.
    """
    requested = config if config is not None else ColorRobustConfig()
    real_axis_config = replace(requested, origin_mode="upper_right")
    result = build_die_map_robust(
        image, config=real_axis_config, pixel_per_unit=pixel_per_unit,
        include_edge=include_edge, edge_margin=edge_margin, edge_mode=edge_mode,
        with_crops=with_crops, border_mode=border_mode,
        offset_x=offset_x, offset_y=offset_y,
        margin_x=margin_x, margin_y=margin_y, return_info=return_info,
    )
    if return_info:
        die_map, info = result
        info["axis_reference"] = {
            "profile": "real_camera_x_nearest_y_upper",
            "x_rule": "nearest vertical street to wafer centre",
            "y_rule": "horizontal street immediately above wafer centre",
        }
        return die_map, info
    return result
