"""Color-invariant wafer / die-grid detection, kept separate from V5.

This module deliberately does not modify ``wafer_die_map_v5.py``.  Its public
``build_die_map_robust`` function returns the same ``WaferDieMap`` object, so
existing calls to ``locate_die`` and ``crop_die`` continue to work.

The detector does *not* assume a particular street or die colour.  It combines
normalised gradients from L*, a*, b* and grayscale channels, then finds the
two repeating orthogonal signals.  Therefore a bright, dark, coloured, or
mixed/noisy street can be detected as long as it creates a repeatable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np

import wafer_die_map_v5 as v5

__all__ = [
    "ColorRobustConfig",
    "detect_grid_color_robust",
    "build_die_map_robust",
    "make_grid_diagnostic",
]


@dataclass(frozen=True)
class ColorRobustConfig:
    """Tuning values for :func:`build_die_map_robust`.

    ``mode='auto'`` evaluates gradient and V5-std candidates and selects the
    one with the stronger periodic evidence.  Use ``'gradient'`` when street
    colour/brightness varies strongly, or ``'std'`` for a known stable setup.
    ``min_pitch``/``max_pitch`` are the important manual overrides when the
    actual die pitch is known.  They limit false periods from repeated circuit
    patterns inside a die.
    """

    mode: Literal["auto", "gradient", "std"] = "auto"
    min_pitch: int = 40
    max_pitch: Optional[int] = None
    roi_ratio: float = 0.60
    blur_sigma: float = 1.2
    projection_smooth: int = 7
    angle_align: Literal["robust", "none"] = "robust"
    phase_refine: bool = True
    phase_refine_search_px: int = 18
    phase_refine_max_shift_px: int = 14
    max_angle_deg: float = 12.0
    angle_min_line_length_ratio: float = 0.16
    clean: bool = True
    origin_mode: Literal["nearest_center", "upper_right"] = "nearest_center"
    """Grid-index reference point.

    ``"nearest_center"`` (default) uses the grid street intersection with the
    shortest Euclidean distance to the detected wafer centre.  This is the
    most intuitive reference for visual inspection.  ``"upper_right"`` keeps
    the original V5 convention: choose the closest vertical street but the
    horizontal street immediately above the centre, so the die to the
    upper-right is index ``(0, 0)``.
    """


def _snap_origin_to_mode(x0: float, y0: float, pitch_x: float, pitch_y: float,
                         wafer_cx: int, wafer_cy: int,
                         origin_mode: Literal["nearest_center", "upper_right"]
                         ) -> Tuple[int, int]:
    """Choose a lattice crossing relative to wafer centre after all refinement.

    Refinement can move a valid grid phase across a half-pitch boundary.  The
    grid itself is unchanged, but its *representative* ``(x0, y0)`` must be
    reselected after that move.  This prevents the Y coordinate from silently
    remaining on the street above centre when the requested convention is the
    closest intersection.
    """
    x = float(x0) + round((wafer_cx - float(x0)) / float(pitch_x)) * float(pitch_x)
    if origin_mode == "nearest_center":
        y_step = round((wafer_cy - float(y0)) / float(pitch_y))
    else:
        y_step = np.floor((wafer_cy - float(y0)) / float(pitch_y))
    y = float(y0) + y_step * float(pitch_y)
    return int(round(x)), int(round(y))


def _normalize01(values: np.ndarray) -> np.ndarray:
    """Robustly map an image response to [0, 1] without outlier domination."""
    lo, hi = np.percentile(values, (5.0, 98.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _colour_invariant_edges(image_bgr: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return x/y boundary energy from luminance *and* chroma changes.

    Taking the maximum across normalized Lab and gray gradients retains a
    brown/white or same-luminance-but-different-colour street.  Median blur
    suppresses salt-and-pepper colour noise before differentiation.
    """
    img = cv2.medianBlur(image_bgr, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    channels = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2], gray]
    gx_all: List[np.ndarray] = []
    gy_all: List[np.ndarray] = []
    for channel in channels:
        f = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), sigmaX=sigma)
        gx_all.append(_normalize01(np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))))
        gy_all.append(_normalize01(np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))))
    return np.max(gx_all, axis=0), np.max(gy_all, axis=0)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.astype(np.float64)
    return np.convolve(values.astype(np.float64), np.ones(width) / width, mode="same")


def _periodic_phase(profile: np.ndarray, pitch: float) -> float:
    """Estimate one line phase using all local peaks, robust to line thickness."""
    peaks = v5._find_periodic_peaks(profile, pitch, min_score_ratio=0.20)
    if not peaks:
        return float(v5._best_phase(profile, max(2, int(round(pitch)))))
    return float(v5._robust_phase(peaks, pitch, profile))


def _candidate_quality(profile: np.ndarray, pitch: float, phase: float) -> float:
    """How much stronger periodic line locations are than locations between them."""
    n = np.arange(-3, int(len(profile) / max(pitch, 1.0)) + 4)
    line = np.rint(phase + n * pitch).astype(int)
    mid = np.rint(phase + (n + 0.5) * pitch).astype(int)
    line = line[(line >= 0) & (line < len(profile))]
    mid = mid[(mid >= 0) & (mid < len(profile))]
    if len(line) < 3 or len(mid) < 3:
        return 0.0
    contrast = abs(float(profile[line].mean() - profile[mid].mean()))
    noise = float(np.std(profile)) + 1e-6
    return contrast / noise


def _street_centre_response(edge_profile: np.ndarray, pitch: float) -> np.ndarray:
    """Convert paired street-edge energy into a response at street centres.

    An absolute Sobel response has two maxima for a street: one at each edge.
    A short box integration joins the pair so its maximum is their midpoint.
    The width is deliberately a modest fraction of pitch and works for both
    narrow saw streets and noisier/wider scribe lanes.
    """
    width = int(np.clip(round(float(pitch) * 0.12), 5, 15))
    kernel = np.ones(width, dtype=np.float64)
    return np.convolve(edge_profile.astype(np.float64), kernel, mode="same")


def _periodic_line_alignment(response: np.ndarray, predicted_origin: int,
                              pitch: float, center: int, search_px: int,
                              max_shift_px: int) -> Tuple[int, float, float]:
    """Refine a grid phase by maximizing *every* expected street response.

    Autocorrelation gives reliable pitch, but a peak in an edge image can be
    either side of a thick street.  This searches a small shared phase shift
    and scores all periodic line locations, not just one local maximum.  A
    conservative shift guard prevents a circuit pattern from moving the whole
    map by a large fraction of a die.
    """
    response = response.astype(np.float64)
    pitch = float(pitch)
    if pitch <= 2.0 or response.size < 3:
        return int(predicted_origin), 0.0, 0.0
    line_ids = np.arange(-int(response.size / pitch) - 2,
                         int(response.size / pitch) + 3)
    best_shift, best_score = 0, -np.inf
    for shift in range(-int(search_px), int(search_px) + 1):
        positions = np.rint(predicted_origin + shift + line_ids * pitch).astype(np.int32)
        positions = positions[(positions >= 0) & (positions < response.size)]
        if positions.size < 5:
            continue
        # A street produces an energy band; max in +-2 px makes the refinement
        # robust to subpixel sampling and line width without changing phase.
        samples = np.stack([response[np.clip(positions + delta, 0, response.size - 1)]
                            for delta in (-2, -1, 0, 1, 2)], axis=1)
        score = float(np.mean(np.max(samples, axis=1)))
        if score > best_score:
            best_shift, best_score = shift, score
    if abs(best_shift) > int(max_shift_px):
        return int(predicted_origin), 0.0, float(best_score)
    confidence = float((best_score - np.median(response)) / (np.std(response) + 1e-6))
    return int(round(predicted_origin + best_shift)), float(best_shift), confidence


def _gradient_grid_candidate(image_bgr: np.ndarray, wafer_cx: int, wafer_cy: int,
                             wafer_r: int, config: ColorRobustConfig
                             ) -> Tuple[float, float, int, int, float, Dict[str, np.ndarray]]:
    gx, gy = _colour_invariant_edges(image_bgr, config.blur_sigma)
    half = max(80, int(wafer_r * config.roi_ratio))
    x1, x2 = max(0, wafer_cx - half), min(image_bgr.shape[1], wafer_cx + half)
    y1, y2 = max(0, wafer_cy - half), min(image_bgr.shape[0], wafer_cy + half)
    col = _smooth(gx[y1:y2, x1:x2].mean(axis=0), config.projection_smooth)
    row = _smooth(gy[y1:y2, x1:x2].mean(axis=1), config.projection_smooth)
    px = float(v5._autocorr_period(col, min_lag=config.min_pitch, max_lag=config.max_pitch))
    py = float(v5._autocorr_period(row, min_lag=config.min_pitch, max_lag=config.max_pitch))
    col_centre = _street_centre_response(col, px)
    row_centre = _street_centre_response(row, py)
    phx, phy = _periodic_phase(col_centre, px), _periodic_phase(row_centre, py)

    # The default is the actual closest grid intersection to wafer centre.
    # ``upper_right`` remains available for old V5 index semantics.
    x0, y0 = _snap_origin_to_mode(x1 + phx, y1 + phy, px, py,
                                  wafer_cx, wafer_cy, config.origin_mode)
    quality = _candidate_quality(col_centre, px, phx) + _candidate_quality(row_centre, py, phy)
    width_x = int(np.clip(round(px * 0.12), 5, 15))
    width_y = int(np.clip(round(py * 0.12), 5, 15))
    gx_centre = cv2.boxFilter(gx, cv2.CV_32F, (width_x, 1), normalize=False)
    gy_centre = cv2.boxFilter(gy, cv2.CV_32F, (1, width_y), normalize=False)
    return px, py, x0, y0, quality, {"gx": gx, "gy": gy, "gx_centre": gx_centre,
                                     "gy_centre": gy_centre, "col": col_centre, "row": row_centre,
                                     "x1": x1, "y1": y1}


def _std_grid_candidate(image_bgr: np.ndarray, wafer_cx: int, wafer_cy: int,
                        wafer_r: int, config: ColorRobustConfig
                        ) -> Tuple[float, float, int, int, float]:
    px, py, x0, y0 = v5.detect_grid(
        image_bgr, wafer_cx, wafer_cy, wafer_r, method="std",
        roi_ratio=config.roi_ratio, min_pitch=config.min_pitch, max_pitch=config.max_pitch)
    x0, y0 = _snap_origin_to_mode(x0, y0, px, py, wafer_cx, wafer_cy, config.origin_mode)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    half = max(80, int(wafer_r * config.roi_ratio))
    x1, x2 = max(0, wafer_cx - half), min(gray.shape[1], wafer_cx + half)
    y1, y2 = max(0, wafer_cy - half), min(gray.shape[0], wafer_cy + half)
    col, row = v5._grid_profiles_std(gray[y1:y2, x1:x2])
    # V5's std profile is low at its expected line.  The absolute contrast
    # quality measure is intentionally polarity-free.
    quality = _candidate_quality(col, px, (x0 - x1) % px)
    quality += _candidate_quality(row, py, (y0 - y1) % py)
    return float(px), float(py), int(x0), int(y0), quality


def detect_grid_color_robust(image_bgr: np.ndarray, wafer_cx: int, wafer_cy: int,
                             wafer_r: int, config: ColorRobustConfig = ColorRobustConfig(),
                             ) -> Tuple[float, float, int, int, Dict[str, Any]]:
    """Find die pitch/origin without using fixed hue, saturation, or brightness rules."""
    candidates: List[Tuple[str, float, float, int, int, float, Optional[Dict[str, np.ndarray]]]] = []
    errors: List[str] = []
    if config.mode in ("auto", "gradient"):
        try:
            px, py, x0, y0, q, diag = _gradient_grid_candidate(
                image_bgr, wafer_cx, wafer_cy, wafer_r, config)
            candidates.append(("gradient", px, py, x0, y0, q, diag))
        except Exception as exc:  # fallback candidates are intentional
            errors.append(f"gradient: {exc}")
    if config.mode in ("auto", "std"):
        try:
            px, py, x0, y0, q = _std_grid_candidate(image_bgr, wafer_cx, wafer_cy, wafer_r, config)
            candidates.append(("std", px, py, x0, y0, q, None))
        except Exception as exc:
            errors.append(f"std: {exc}")
    if not candidates:
        raise RuntimeError("No robust grid candidate succeeded. " + "; ".join(errors))

    name, px, py, x0, y0, score, diag = max(candidates, key=lambda item: item[5])
    # ``std`` is often strongest for pitch, but its low-variance profile can
    # phase-lock to one edge of a street. When invariant gradients agree on
    # pitch, retain the selected pitch but use their street-centre phase.
    phase_diag = diag
    origin_source = name
    if name == "std":
        for cand_name, cand_px, cand_py, cand_x0, cand_y0, _, cand_diag in candidates:
            agrees = (abs(cand_px - px) <= max(2.0, px * 0.03)
                      and abs(cand_py - py) <= max(2.0, py * 0.03))
            if cand_name == "gradient" and cand_diag is not None and agrees:
                x0, y0 = _snap_origin_to_mode(cand_x0, cand_y0, px, py,
                                               wafer_cx, wafer_cy, config.origin_mode)
                phase_diag = cand_diag
                origin_source = "gradient_street_centre"
                break
    phase_info: Dict[str, float] = {"shift_x": 0.0, "shift_y": 0.0,
                                    "confidence_x": 0.0, "confidence_y": 0.0}
    if config.phase_refine and phase_diag is not None:
        # Use full-image directional edge responses for a central geometry
        # anchor; the candidate's ROI was used only for pitch estimation.
        x0, sx, cx_score = _periodic_line_alignment(
            phase_diag["gx_centre"].mean(axis=0), x0, px, wafer_cx,
            config.phase_refine_search_px, config.phase_refine_max_shift_px)
        y0, sy, cy_score = _periodic_line_alignment(
            phase_diag["gy_centre"].mean(axis=1), y0, py, wafer_cy,
            config.phase_refine_search_px, config.phase_refine_max_shift_px)
        phase_info = {"shift_x": sx, "shift_y": sy,
                      "confidence_x": cx_score, "confidence_y": cy_score}
    # Phase refinement preserves the lattice but may choose a different
    # representative period.  Always re-snap it to the requested reference.
    x0, y0 = _snap_origin_to_mode(x0, y0, px, py, wafer_cx, wafer_cy, config.origin_mode)
    return px, py, x0, y0, {
        "selected_method": name,
        "quality": float(score),
        "phase_refinement": phase_info,
        "origin_mode": config.origin_mode,
        "origin_source": origin_source,
        "origin_delta_from_center_px": {"x": int(x0 - wafer_cx), "y": int(y0 - wafer_cy)},
        "candidates": [{"method": n, "pitch_x": a, "pitch_y": b, "quality": q}
                       for n, a, b, _, _, q, _ in candidates],
        "errors": errors,
    }


def _estimate_grid_angle(image_bgr: np.ndarray, wafer_cx: int, wafer_cy: int,
                         wafer_r: int, config: ColorRobustConfig) -> Tuple[float, float]:
    """Estimate small grid tilt from invariant edges; returns (rotation, confidence)."""
    gx, gy = _colour_invariant_edges(image_bgr, config.blur_sigma)
    response = _normalize01(gx + gy)
    mask = np.zeros(response.shape, np.uint8)
    cv2.circle(mask, (wafer_cx, wafer_cy), max(1, int(wafer_r * config.roi_ratio)), 255, -1)
    threshold = float(np.percentile(response[mask > 0], 78.0))
    edges = ((response >= threshold).astype(np.uint8) * 255) & mask
    min_len = max(20, int(wafer_r * config.angle_min_line_length_ratio))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(25, min_len // 2),
                            minLineLength=min_len, maxLineGap=max(4, min_len // 5))
    if lines is None:
        return 0.0, 0.0

    deviations: List[float] = []
    weights: List[float] = []
    # OpenCV returns either (N, 1, 4) or (N, 4), depending on its build.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        angle = np.degrees(np.arctan2(dy, dx))
        # line direction has 180-degree ambiguity; express it around horizontal
        # or vertical and retain only segments plausibly belonging to the grid.
        horizontal = ((angle + 90.0) % 180.0) - 90.0
        vertical = ((angle % 180.0) - 90.0)
        dev = horizontal if abs(horizontal) <= abs(vertical) else vertical
        if abs(dev) <= config.max_angle_deg:
            deviations.append(float(dev))
            weights.append(length)
    if not deviations:
        return 0.0, 0.0
    values, weights_arr = np.asarray(deviations), np.asarray(weights)
    median = float(np.median(values))
    inliers = np.abs(values - median) <= 1.5
    if int(inliers.sum()) < 3:
        return 0.0, 0.0
    correction = float(np.average(values[inliers], weights=weights_arr[inliers]))
    confidence = min(1.0, float(inliers.sum()) / 20.0) * min(1.0, weights_arr[inliers].sum() / (wafer_r * 8.0))
    return correction, confidence


def _detect_wafer_robust(image_bgr: np.ndarray) -> Tuple[int, int, int]:
    """Use V5's fast black-background path, with Otsu fallback for other backgrounds."""
    try:
        return v5.detect_wafer(image_bgr)
    except RuntimeError:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Wafer region not found in either black-background or Otsu mode.")
        (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        return int(round(cx)), int(round(cy)), int(round(r))


def build_die_map_robust(image: Union[str, Path, np.ndarray], *,
                         config: ColorRobustConfig = ColorRobustConfig(),
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
    """Build a V5-compatible die map with colour-invariant grid detection.

    The supplied image is never recoloured.  ``aligned_image`` contains only
    optional geometric rotation/wafer cleanup, so crops retain their original
    die colours.  When ``return_info=True`` diagnostics include the selected
    detector and score for logging/acceptance thresholds.
    """
    img = v5._load_bgr(image).copy()
    if config.clean:
        try:
            img = v5.clean_wafer(img)
        except RuntimeError:
            # The robust wafer locator below still supports non-black images.
            pass
    wafer_cx, wafer_cy, wafer_r = _detect_wafer_robust(img)

    rotation_deg, angle_confidence = 0.0, 1.0
    angle_source = "none"
    if config.angle_align == "robust":
        # The V5 projection+FFT alignment is substantially more accurate than
        # a single Hough estimate for sub-degree residuals.  It does not use a
        # fixed street colour; forcing its std pre-detector keeps this new path
        # colour independent.  Hough remains a safe fallback for unusual data.
        try:
            img, rotation_deg, angle_info = v5.align_wafer_by_die_render(
                img, grid_method="std", return_info=True)
            angle_confidence = float(angle_info.get("confidence", 0.0))
            angle_source = "projection_fft"
            wafer_cx, wafer_cy, wafer_r = _detect_wafer_robust(img)
        except Exception:
            rotation_deg, angle_confidence = _estimate_grid_angle(
                img, wafer_cx, wafer_cy, wafer_r, config)
            angle_source = "hough_fallback"
            if abs(rotation_deg) >= 0.05 and angle_confidence >= 0.15:
                img = v5._rotate_wafer_keep_size(img, wafer_cx, wafer_cy, rotation_deg)
                wafer_cx, wafer_cy, wafer_r = _detect_wafer_robust(img)

    pitch_x, pitch_y, x0, y0, grid_info = detect_grid_color_robust(
        img, wafer_cx, wafer_cy, wafer_r, config)
    die_w, die_h = int(round(pitch_x)), int(round(pitch_y))
    margin = edge_margin if include_edge else 0.98
    max_ix, max_iy = int(np.ceil(wafer_r / pitch_x)) + 2, int(np.ceil(wafer_r / pitch_y)) + 2
    r_lim_sq = (wafer_r * margin) ** 2
    dies: List[Dict[str, Any]] = []
    by_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for iy in range(-max_iy, max_iy + 1):
        for ix in range(-max_ix, max_ix + 1):
            cx = int(round(x0 + ix * pitch_x + pitch_x / 2.0))
            cy = int(round(y0 - iy * pitch_y - pitch_y / 2.0))
            if (cx - wafer_cx) ** 2 + (cy - wafer_cy) ** 2 > r_lim_sq:
                continue
            rect = (cx - die_w // 2, cy - die_h // 2, cx - die_w // 2 + die_w, cy - die_h // 2 + die_h)
            entry: Dict[str, Any] = {
                "index": (ix, iy), "center_px": (cx, cy), "rect_px": rect,
                "crop_rect_px": v5._crop_rect(cx, cy, die_w, die_h, offset_x, offset_y, margin_x, margin_y),
                "real_coord": ((cx - wafer_cx) / pixel_per_unit, (wafer_cy - cy) / pixel_per_unit),
                "is_edge_partial": v5._rect_crosses_circle(*rect, wafer_cx, wafer_cy, wafer_r),
                "is_edge_ring": False, "is_edge": False,
            }
            if with_crops:
                crop = v5.crop_die(img, cx, cy, die_w, die_h, offset_x=offset_x, offset_y=offset_y,
                                   margin_x=margin_x, margin_y=margin_y, border_mode=border_mode)
                if crop is None:
                    continue
                entry["image"] = crop
            dies.append(entry)
            by_index[(ix, iy)] = entry
    emode = v5._normalize_edge_mode(edge_mode)
    present = set(by_index)
    for entry in dies:
        ix, iy = entry["index"]
        entry["is_edge_ring"] = any((ix + dx, iy + dy) not in present
                                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                    if dx or dy)
        entry["is_edge"] = v5._resolve_edge_flag(entry["is_edge_partial"], entry["is_edge_ring"], emode)
    report = v5.validate_quadrant_edges(dies, wafer_cx, wafer_cy, wafer_r)
    result = v5.WaferDieMap(
        wafer_cx=wafer_cx, wafer_cy=wafer_cy, wafer_r=wafer_r,
        pitch_x=pitch_x, pitch_y=pitch_y, x0=x0, y0=y0, die_w=die_w, die_h=die_h,
        pixel_per_unit=pixel_per_unit, dies=dies, dies_by_index=by_index, image_shape=img.shape[:2],
        rotation_deg=rotation_deg, aligned_image=img, edge_mode=emode,
        angle_confidence=angle_confidence, angle_agree=angle_confidence >= 0.15,
        quadrant_report=report,
    )
    info: Dict[str, Any] = {"grid": grid_info, "rotation_deg": rotation_deg,
                            "angle_confidence": angle_confidence,
                            "angle_source": angle_source, "config": config}
    return (result, info) if return_info else result


def make_grid_diagnostic(image: Union[str, Path, np.ndarray], die_map: v5.WaferDieMap,
                         thickness: int = 2) -> np.ndarray:
    """Return an overlay on the aligned original image for visual acceptance checks."""
    base = die_map.aligned_image if die_map.aligned_image is not None else v5._load_bgr(image)
    overlay = base.copy()
    for die in die_map.dies:
        x1, y1, x2, y2 = die["rect_px"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    cv2.circle(overlay, (die_map.wafer_cx, die_map.wafer_cy), 5, (0, 0, 255), -1)
    return overlay
