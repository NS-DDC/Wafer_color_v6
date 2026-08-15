"""
================================================================================
 Wafer Die Map V5  (Python 3.9, 단일 파일 — 통째로 복사/붙여넣기 해서 사용)
================================================================================

[V5 핵심 변경]
 · 기본 얼라인 = "die_render" : dm.dies(검출한 모든 die)를 cv2.rectangle 굵기 3으로
   그린 '이상적 격자 템플릿'에, 실제 sawline 엣지를 회전 정합시켜 기울기를 찾는다.
   실제 sawline 모양이 굵기 3 사각 테두리와 일치하므로, 모든 die·양축(가로/세로)을
   한꺼번에 이용해 가장 안정적으로 각도를 잡는다(2-pass: 대략검출→정합→재검출).
 · EDGE 구분 강화 : 각 die 에 두 가지 edge 플래그를 모두 부여(둘 다 선택 가능).
     - is_edge_partial : die 사각형이 wafer 원 밖으로 일부라도 나간 '부분 die'.
     - is_edge_ring    : die 격자에서 8방향 이웃이 다 차 있지 않은 '최외곽 줄'.
   build_die_map(edge_mode="circle"|"ring"|"both") 로 is_edge 가 무엇을 가리킬지 선택.
   locate_die 결과에 is_edge(+ is_edge_partial / is_edge_ring)가 함께 반환된다.

[기능] (기존 wafer_die_map.py 대비)
 1. Notch 중심점 반환 : wafer '아래쪽'의 파인 곳(notch)만 탐색하고, 그 파임의
    중심 픽셀점을 기준으로 사용. dm.notch_center_px 로 반환.
 2. Angle 이중 검증   : notch 로 구한 각도를, die 격자(sawline) 방향으로 독립
    측정해 교차검증. dm.angle_verified / dm.die_grid_angle_resid.
 3. 4분면 맵 검증     : center corner 에서 die 를 확장해 만든 맵이 4분면(TL/TR/
    BL/BR) 가장자리까지 균형 있게 채워졌는지 확인. dm.quadrant_report.
 4. Wafer 원판 정리   : 시작 시 wafer 원판(가장 큰 연결성분) 밖을 모두 검정으로
    채워 외부 노이즈 제거 (clean_wafer). dm.aligned_image 가 정리된 이미지.
 5. 항상 aligned 반환 : 회전각이 0.0 이어도 dm.aligned_image 를 항상 반환
    (정리/보정 후 실제 사용 이미지).

[V5.1 각도 고도화 + 회전 깨짐 수정]
 · 각도 고도화 : die_render(투영 주기성)에 'FFT 스펙트럼' 교차검증을 추가했다.
   두 독립 단서가 일치하면 신뢰↑(dm.angle_agree/angle_confidence). 불일치/대각이면
   넓게(±44°) 재탐색해 '투영 peak 가 가장 큰' 각을 채택 → 큰 기울기(±10° 이상)도
   잡고, 격자 검출이 실패해도 '이미지 픽셀' 만으로 측정해 '조용히 0' 으로 실패하지 않는다.
 · 회전 깨짐 수정 : 회전 보간을 INTER_NEAREST → INTER_CUBIC 로 변경. 예전엔 die 격자
   같은 미세 주기 패턴이 NEAREST 회전 시 계단/모아레로 '깨져' 보였다. CUBIC 으로 매끈.

[회전(angle) 보정 방식 — build_die_map(angle_align_method=...) 로 선택]
 · "die_render"   (V5 기본) : die 격자(굵기 3 구조)의 '열/행 투영 주기성' + 'FFT 스펙트럼'
    교차검증으로 기울기 산출. 모든 die·양축 사용, 위상/극성 무관, 큰 기울기까지 견고.
 · "notch"        (옵션) : 아래쪽 notch 위치로 보정. 작은 각도까지 정밀(notch 필요).
 · "vertical_line"(옵션) : 이진화(Otsu)로 두꺼운 die 선을 잡고 '세선화'로 1px 중심선화한 뒤,
    '가장 긴 세로선'의 정확한 수직(90°) 대비 기울기로 보정(HoughLinesP). 이어서 가로축
    잔차까지 '순차(V->H)' 재보정. die 무늬가 뚜렷할 때 robust(notch 불필요).
    - 세선화 이유: 두꺼운 선은 양 가장자리가 다른 각도로 잡혀 각도가 흔들림 → 중심선 1개로 안정.
================================================================================

이 파일 "하나"만 다른 코드에 통째로 복붙하면 됩니다. (별도 import 불필요)
필요 외부 패키지는 numpy, opencv-python 뿐이며, 로컬 모듈 의존성은 없습니다.
공개 함수 2개만 쓰면 됩니다:

  1) build_die_map(image, ...)        -> WaferDieMap
        wafer 이미지 한 장을 넣으면 wafer/격자를 검출하고 EDGE 포함 전체 die map 생성.

  2) locate_die(die_map, point|bbox)  -> dict
        픽셀 좌표 또는 BBox(YOLO 등)를 넣으면 그 위치의
          - die_index (ix, iy)
          - die rect 픽셀 좌표 (x1, y1, x2, y2)
          - 실측 좌표/거리 (BBox 면 중심 기준)
        를 반환.

검출(웨이퍼 영역 + die 격자) 로직은 wafer_die_index_extract_39.py 의 것을
**그대로(verbatim)** 가져왔습니다. (detect_wafer / detect_grid / clip_die + 헬퍼)
미사용 함수(_norm01, _detect_line_hue, _wafer_color_mask, _refine_corner)만 제외.

인덱스 규칙 (원본과 동일)
--------------------------
- wafer 중앙 근처 die 4개가 만나는 격자 코너 = grid origin (x0, y0)
- 그 코너의 우측 상단 die 가 (ix=0, iy=0)
- ix +1 -> 한 칸 오른쪽(이미지 x 증가),  iy +1 -> 한 칸 위쪽(이미지 y 감소)
- die (ix, iy) 중심 픽셀:
      cx = x0 + ix*pitch_x + pitch_x/2
      cy = y0 - iy*pitch_y - pitch_y/2

실측 좌표 (real_coord)
-----------------------
- pixel_per_unit (default 32 px = 1 unit) 로 wafer 중심 기준 상대 좌표 환산
      rx = (px - wafer_cx) / pixel_per_unit
      ry = (wafer_cy - py) / pixel_per_unit       # 화면 위쪽이 +y

의존성: numpy, opencv-python
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["WaferDieMap", "build_die_map", "locate_die", "crop_die",
           "detect_notch_angle", "detect_notch", "align_wafer_by_notch",
           "align_wafer_by_vertical_line", "clean_wafer",
           "measure_die_grid_angle", "measure_vertical_line_angle",
           "measure_horizontal_line_angle", "measure_axis_line_angle",
           "validate_quadrant_edges",
           "render_die_grid_mask", "measure_die_render_angle",
           "align_wafer_by_die_render", "measure_wafer_angle_robust"]


# #############################################################################
# #                                                                           #
# #   CORE DETECTION  (원본 로직 그대로 — 수정 금지 영역)                      #
# #   wafer_die_index_extract_39.py 에서 복사. 동작/결과 동일.                #
# #                                                                           #
# #############################################################################

# =============================================================================
# 1) Wafer 영역(원) 검출
# =============================================================================
def detect_wafer(image_bgr: np.ndarray,
                 bg_threshold: int = 20) -> Tuple[int, int, int]:
    """검정 배경을 제외한 가장 큰 contour 를 wafer 로 간주. -> (cx, cy, radius)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Wafer region not found "
                           "(전부 배경이거나 bg_threshold 가 너무 높음).")
    wafer_cnt = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(wafer_cnt)
    return int(round(cx)), int(round(cy)), int(round(radius))


# =============================================================================
# 2) Die 격자 자동 검출 (+ 내부 헬퍼)
# =============================================================================
def _autocorr_period(profile: np.ndarray,
                     min_lag: int = 50,
                     max_lag: Optional[int] = None,
                     harmonic_threshold: float = 0.7) -> int:
    """1D 신호의 fundamental period 를 autocorrelation 으로 추정."""
    p = profile.astype(np.float64)
    p -= p.mean()
    n = len(p)
    if n < min_lag * 4:
        raise RuntimeError("Profile too short for autocorrelation.")

    corr = np.correlate(p, p, mode='full')[n - 1:]
    if max_lag is None:
        max_lag = n // 3

    peaks: List[Tuple[int, float]] = []
    for lag in range(min_lag, min(max_lag, len(corr) - 1)):
        if corr[lag] > corr[lag - 1] and corr[lag] > corr[lag + 1] and corr[lag] > 0:
            peaks.append((lag, float(corr[lag])))
    if not peaks:
        raise RuntimeError("Failed to estimate period from autocorrelation.")

    best_lag, best_val = max(peaks, key=lambda x: x[1])

    candidates = [best_lag]
    threshold = best_val * harmonic_threshold
    for k in (2, 3, 4, 5):
        sub = best_lag // k
        if sub < min_lag:
            break
        tol = max(2, sub // 20)
        for lag, val in peaks:
            if abs(lag - sub) <= tol and val >= threshold \
                    and abs(lag * k - best_lag) <= max(3, best_lag // 30):
                candidates.append(lag)
                break

    return min(candidates)


def _best_phase(profile: np.ndarray, pitch: int) -> int:
    """fallback : profile 을 pitch 로 슬라이딩해 grid-line 평균 최대 phase 반환."""
    best_ph, best_val = 0, -np.inf
    for ph in range(pitch):
        v = float(profile[ph::pitch].mean())
        if v > best_val:
            best_val = v
            best_ph = ph
    return best_ph


def _find_periodic_peaks(profile: np.ndarray,
                          approx_pitch: float,
                          min_score_ratio: float = 0.3
                          ) -> List[int]:
    """profile 에서 approx_pitch 간격으로 분포한 local maxima 위치 반환."""
    n = len(profile)
    if n < 3:
        return []
    max_val = float(profile.max())
    thr = max_val * min_score_ratio
    min_spacing = max(2, int(approx_pitch * 0.6))

    peaks: List[int] = []
    for i in range(1, n - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > thr:
            if not peaks or i - peaks[-1] >= min_spacing:
                peaks.append(i)
            elif profile[i] > profile[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _refine_origin_with_template(image_bgr: np.ndarray,
                                  die_template_bgr: np.ndarray,
                                  pitch_x: float, pitch_y: float,
                                  approx_x0: int, approx_y0: int,
                                  wafer_cx: int, wafer_cy: int, wafer_r: int
                                  ) -> Tuple[float, float, int, int, float]:
    """(옵션) die_sample 이미지로 matchTemplate -> sub-pixel pitch + phase 재추정."""
    tw, th = int(round(pitch_x)), int(round(pitch_y))
    template = cv2.resize(die_template_bgr, (tw, th), interpolation=cv2.INTER_AREA)

    def _edge(g: np.ndarray) -> np.ndarray:
        gf = g.astype(np.float32)
        sx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(sx * sx + sy * sy).astype(np.float32)

    e_w = _edge(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    e_t = _edge(cv2.cvtColor(template,  cv2.COLOR_BGR2GRAY))

    half = int(wafer_r * 0.5)
    rx1 = max(wafer_cx - half, 0)
    ry1 = max(wafer_cy - half, 0)
    rx2 = min(wafer_cx + half, e_w.shape[1])
    ry2 = min(wafer_cy + half, e_w.shape[0])
    if rx2 - rx1 <= tw + 2 or ry2 - ry1 <= th + 2:
        return pitch_x, pitch_y, approx_x0, approx_y0, 0.0
    roi = e_w[ry1:ry2, rx1:rx2]

    res = cv2.matchTemplate(roi, e_t, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    col_proj = res.max(axis=0)
    row_proj = res.max(axis=1)

    x_peaks = _find_periodic_peaks(col_proj, pitch_x)
    y_peaks = _find_periodic_peaks(row_proj, pitch_y)

    if len(x_peaks) >= 3:
        slope_x, intercept_x = np.polyfit(np.arange(len(x_peaks)), x_peaks, 1)
        pitch_x_ref = float(slope_x)
        phase_x = float(intercept_x)
    else:
        pitch_x_ref = float(pitch_x)
        phase_x = float(x_peaks[0]) if x_peaks else 0.0
    if len(y_peaks) >= 3:
        slope_y, intercept_y = np.polyfit(np.arange(len(y_peaks)), y_peaks, 1)
        pitch_y_ref = float(slope_y)
        phase_y = float(intercept_y)
    else:
        pitch_y_ref = float(pitch_y)
        phase_y = float(y_peaks[0]) if y_peaks else 0.0

    kx = round((wafer_cx - rx1 - phase_x) / pitch_x_ref)
    ky = round((wafer_cy - ry1 - phase_y) / pitch_y_ref)
    x0 = int(round(rx1 + phase_x + kx * pitch_x_ref))
    y0 = int(round(ry1 + phase_y + ky * pitch_y_ref))

    return pitch_x_ref, pitch_y_ref, x0, y0, float(max_val)


def _robust_phase(peaks: List[int], pitch: float,
                  profile: np.ndarray) -> float:
    """여러 periodic peak 위치로부터 노이즈에 강인한 phase(0~pitch) 추정."""
    if not peaks:
        return float(_best_phase(profile, int(round(pitch))))
    mods = np.array([p % pitch for p in peaks], dtype=np.float64)
    ref = float(mods[0])
    adj = ((mods - ref + pitch / 2.0) % pitch) - pitch / 2.0
    return float((ref + np.median(adj)) % pitch)


def _grid_profiles_std(gray_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[method='std'] column/row STD 의 최소점(=균일 boundary 라인) 신호."""
    col_std = gray_roi.std(axis=0)
    row_std = gray_roi.std(axis=1)
    col_profile = float(col_std.max()) - col_std
    row_profile = float(row_std.max()) - row_std
    return col_profile.astype(np.float64), row_profile.astype(np.float64)


def _grid_profiles_color(image_bgr: np.ndarray, gray: np.ndarray,
                         x1: int, x2: int, y1: int, y2: int,
                         wafer_cx: int, wafer_cy: int, wafer_r: int,
                         line_hue: Optional[int], hue_delta: int,
                         sat_min: int, val_min: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """[method='color'/'hybrid'] die 중앙 컬러 stripe 다발(채도) projection."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2]

    valid = (val >= val_min).astype(np.float32)
    sat_roi = sat[y1:y2, x1:x2]
    valid_roi = valid[y1:y2, x1:x2]

    col_num = (sat_roi * valid_roi).sum(axis=0)
    col_den = valid_roi.sum(axis=0) + 1e-6
    row_num = (sat_roi * valid_roi).sum(axis=1)
    row_den = valid_roi.sum(axis=1) + 1e-6
    col_profile = (col_num / col_den).astype(np.float64)
    row_profile = (row_num / row_den).astype(np.float64)
    return col_profile, row_profile


# --- corner-grid detection (codex "corner" 방식: street 선 자체로 코너 직접 검출) ---
def _street_color_mask(image_bgr: np.ndarray,
                       x1: int, x2: int, y1: int, y2: int,
                       min_brightness: float,
                       min_channel: int,
                       min_color_delta: int,
                       max_color_delta: int) -> np.ndarray:
    """밝은 wafer street/grid 색만 mask 로 분리한다.

    검은 노이즈와 회색 die 면은 제외하고, 내부 rainbow stripe 처럼 색 변화가
    너무 큰 패턴도 제외한다. Noise 샘플의 코너 교차점 검출에 쓰는 전용 mask.
    """
    roi = image_bgr[y1:y2, x1:x2]
    maxc = roi.max(axis=2).astype(np.int16)
    minc = roi.min(axis=2).astype(np.int16)
    brightness = roi.mean(axis=2)
    color_delta = maxc - minc

    mask = (
        (brightness > min_brightness)
        & (maxc > min_channel)
        & (color_delta >= min_color_delta)
        & (color_delta <= max_color_delta)
    )
    mask = mask.astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )


def _smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    """1D projection 을 이동 평균으로 완만하게 만든다."""
    if window <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _find_projection_bands(profile: np.ndarray,
                           offset: int,
                           min_width: int,
                           threshold_ratio: float,
                           sigma: float,
                           min_projection: float
                           ) -> List[Tuple[float, int, int, float]]:
    """projection 에서 연속된 강한 line band 들을 찾는다.

    Returns
    -------
    [(center, start, end, score), ...]
    """
    if profile.size == 0 or float(profile.max()) <= 0.0:
        return []

    threshold = max(
        float(profile.mean() + sigma * profile.std()),
        float(profile.max() * threshold_ratio),
        float(min_projection),
    )
    above = profile >= threshold

    bands: List[Tuple[float, int, int, float]] = []
    i = 0
    n = len(profile)
    while i < n:
        if not above[i]:
            i += 1
            continue

        j = i
        while j < n and above[j]:
            j += 1

        if j - i >= min_width:
            segment = profile[i:j]
            weights = np.maximum(segment, 1e-6)
            center = float((np.arange(i, j) * weights).sum() / weights.sum())
            bands.append((offset + center, offset + i, offset + j, float(segment.max())))
        i = j

    return bands


def _choose_previous_band(bands: List[Tuple[float, int, int, float]],
                          center: float) -> Tuple[float, int, int, float]:
    """wafer 중심보다 작거나 같은 가장 가까운 band 를 선택한다."""
    if not bands:
        raise RuntimeError("No wafer street/grid line was found near wafer center.")

    previous = [band for band in bands if band[0] <= center]
    if previous:
        return max(previous, key=lambda band: band[0])
    return min(bands, key=lambda band: abs(band[0] - center))


def _choose_nearest_band(bands: List[Tuple[float, int, int, float]],
                         center: float) -> Tuple[float, int, int, float]:
    """wafer 중심에 가장 가까운 band 를 선택한다."""
    if not bands:
        raise RuntimeError("No wafer street/grid line was found near wafer center.")
    return min(bands, key=lambda band: abs(band[0] - center))


def _median_band_spacing(bands: List[Tuple[float, int, int, float]],
                         fallback_pitch: float,
                         min_pitch: int,
                         max_pitch: Optional[int]) -> float:
    """검출된 street band 사이 간격의 median 으로 pitch 를 보정한다."""
    if len(bands) < 3:
        return float(fallback_pitch)

    centers = np.array(sorted(band[0] for band in bands), dtype=np.float64)
    diffs = np.diff(centers)
    upper = float(max_pitch) if max_pitch is not None else float(fallback_pitch * 1.8)
    lower = float(min_pitch)
    diffs = diffs[(diffs >= lower) & (diffs <= upper)]
    if diffs.size == 0:
        return float(fallback_pitch)
    return float(np.median(diffs))


def detect_corner_grid(image_bgr: np.ndarray,
                       wafer_cx: int, wafer_cy: int, wafer_r: int,
                       roi_half: Optional[int] = None,
                       min_pitch: int = 50,
                       max_pitch: Optional[int] = None,
                       min_brightness: float = 115.0,
                       min_channel: int = 130,
                       min_color_delta: int = 35,
                       max_color_delta: int = 130,
                       open_length: int = 60,
                       smooth_window: int = 9,
                       min_width: int = 3,
                       threshold_ratio: float = 0.35,
                       sigma: float = 1.8,
                       min_projection: float = 5.0
                       ) -> Tuple[float, float, int, int]:
    """Noise wafer 의 실제 4-way 코너 교차점을 직접 찾는 Grid 검출.

    기존 `hybrid` 방식은 die 내부 stripe peak 에 실측 offset 을 더해 코너를
    계산한다. 이 함수는 코너를 이루는 밝은 wafer street 선 자체를 mask 로 잡고,
    wafer 중심 근처의 세로/가로 line band 교차점을 `grid_origin` 으로 반환한다.

    Returns
    -------
    (pitch_x, pitch_y, x0, y0)
        x0, y0 는 코너 한 점이며, overlay 의 박스는 이 점을 확인하는 시각화 용도다.
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if roi_half is None:
        roi_half = min(900, max(300, int(wafer_r * 0.25)))
    x1 = max(wafer_cx - roi_half, 0)
    x2 = min(wafer_cx + roi_half, W)
    y1 = max(wafer_cy - roi_half, 0)
    y2 = min(wafer_cy + roi_half, H)

    blurred_low = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    sx_low = np.abs(cv2.Sobel(blurred_low, cv2.CV_32F, 1, 0, ksize=3))
    sy_low = np.abs(cv2.Sobel(blurred_low, cv2.CV_32F, 0, 1, ksize=3))
    pitch_x_rough = float(_autocorr_period(sx_low[y1:y2, x1:x2].mean(axis=0),
                                           min_lag=min_pitch, max_lag=max_pitch))
    pitch_y_rough = float(_autocorr_period(sy_low[y1:y2, x1:x2].mean(axis=1),
                                           min_lag=min_pitch, max_lag=max_pitch))

    street_mask = _street_color_mask(
        image_bgr, x1, x2, y1, y2,
        min_brightness=min_brightness,
        min_channel=min_channel,
        min_color_delta=min_color_delta,
        max_color_delta=max_color_delta)

    open_length = max(15, int(open_length))
    vertical_mask = cv2.morphologyEx(
        street_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_length)),
    )
    horizontal_mask = cv2.morphologyEx(
        street_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (open_length, 1)),
    )

    x_profile = _smooth_projection(vertical_mask.mean(axis=0), smooth_window)
    y_profile = _smooth_projection(horizontal_mask.mean(axis=1), smooth_window)
    x_bands = _find_projection_bands(
        x_profile, x1, min_width, threshold_ratio, sigma, min_projection)
    y_bands = _find_projection_bands(
        y_profile, y1, min_width, threshold_ratio, sigma, min_projection)

    # x 축은 wafer 원 검출 중심이 1 px 정도 흔들릴 수 있어 가장 가까운 세로선을 선택한다.
    # y 축은 중심 바로 위의 가로 street 가 grid origin 이므로 previous band 를 선택한다.
    x_band = _choose_nearest_band(x_bands, float(wafer_cx))
    y_band = _choose_previous_band(y_bands, float(wafer_cy))
    pitch_x = _median_band_spacing(x_bands, pitch_x_rough, min_pitch, max_pitch)
    pitch_y = _median_band_spacing(y_bands, pitch_y_rough, min_pitch, max_pitch)
    return pitch_x, pitch_y, int(round(x_band[0])), int(round(y_band[0]))


def detect_grid(image_bgr: np.ndarray,
                wafer_cx: int, wafer_cy: int, wafer_r: int,
                method: str = "corner",
                roi_ratio: float = 0.6,
                min_pitch: int = 50,
                max_pitch: Optional[int] = None,
                die_template_bgr: Optional[np.ndarray] = None,
                line_hue: Optional[int] = None,
                hue_delta: int = 20,
                sat_min: int = 50,
                val_min: int = 50
                ) -> Tuple[float, float, int, int]:
    """Die 격자 (pitch + origin) 자동 검출. -> (pitch_x, pitch_y, x0, y0).

    method: "corner"(기본, street 선으로 코너 직접 검출) | "std" | "color" | "hybrid"
    """
    # "corner" : 밝은 wafer street 선 자체를 mask 로 잡아 코너 교차점을 직접 검출.
    if method in ("corner", "corner_grid", "street"):
        return detect_corner_grid(
            image_bgr, wafer_cx, wafer_cy, wafer_r,
            min_pitch=min_pitch, max_pitch=max_pitch)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    half = int(wafer_r * roi_ratio)
    x1 = max(wafer_cx - half, 0)
    x2 = min(wafer_cx + half, image_bgr.shape[1])
    y1 = max(wafer_cy - half, 0)
    y2 = min(wafer_cy + half, image_bgr.shape[0])

    blurred_low = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    sx_low = np.abs(cv2.Sobel(blurred_low, cv2.CV_32F, 1, 0, ksize=3))
    sy_low = np.abs(cv2.Sobel(blurred_low, cv2.CV_32F, 0, 1, ksize=3))
    pitch_x_rough = _autocorr_period(sx_low[y1:y2, x1:x2].mean(axis=0),
                                     min_lag=min_pitch, max_lag=max_pitch)
    pitch_y_rough = _autocorr_period(sy_low[y1:y2, x1:x2].mean(axis=1),
                                     min_lag=min_pitch, max_lag=max_pitch)

    col_std, row_std = _grid_profiles_std(gray[y1:y2, x1:x2])
    col_sat, row_sat = _grid_profiles_color(
        image_bgr, gray, x1, x2, y1, y2,
        wafer_cx, wafer_cy, wafer_r,
        line_hue, hue_delta, sat_min, val_min)

    if method == "std":
        col_profile, row_profile = col_std, row_std
        off_x, off_y = 0.0, 0.0
    elif method == "color":
        col_profile, row_profile = col_sat, row_sat
        off_x, off_y = 0.5, 0.5
    elif method == "hybrid":
        col_profile, row_profile = col_sat, row_sat
        off_x, off_y = 0.63, 0.493
    else:
        raise ValueError(
            f"Unknown method: {method!r} (use 'corner' | 'std' | 'color' | 'hybrid')")

    x_peaks = _find_periodic_peaks(col_profile, pitch_x_rough)
    y_peaks = _find_periodic_peaks(row_profile, pitch_y_rough)
    pitch_x = float(pitch_x_rough)
    pitch_y = float(pitch_y_rough)
    phase_x = _robust_phase(x_peaks, pitch_x, col_profile)
    phase_y = _robust_phase(y_peaks, pitch_y, row_profile)

    phase_x += off_x * pitch_x
    phase_y += off_y * pitch_y

    bias_x = pitch_x * 0.05
    bias_y = pitch_y * 0.05
    kx = int(math.floor((wafer_cx - x1 - phase_x - bias_x) / pitch_x + 0.5))
    ky = int(math.floor((wafer_cy - y1 - phase_y - bias_y) / pitch_y + 0.5))
    x0 = int(round(x1 + phase_x + kx * pitch_x))
    y0 = int(round(y1 + phase_y + ky * pitch_y))

    if die_template_bgr is not None:
        pitch_x, pitch_y, x0, y0, _ = _refine_origin_with_template(
            image_bgr, die_template_bgr, pitch_x, pitch_y,
            x0, y0, wafer_cx, wafer_cy, wafer_r)

    return float(pitch_x), float(pitch_y), x0, y0


# =============================================================================
# 3) Die clip (한 die crop)
# =============================================================================
def clip_die(image: np.ndarray, center_x: int, center_y: int,
             die_w: int, die_h: int,
             border_mode: str = "pad") -> Optional[np.ndarray]:
    """(center_x, center_y) 기준 die_w x die_h die crop 반환 (pad / crop)."""
    H, W = image.shape[:2]
    x1 = center_x - die_w // 2
    y1 = center_y - die_h // 2
    x2 = x1 + die_w
    y2 = y1 + die_h

    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None

    ix1, iy1 = max(x1, 0), max(y1, 0)
    ix2, iy2 = min(x2, W), min(y2, H)
    crop = image[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None

    if border_mode == "crop":
        return crop.copy()

    if border_mode == "pad":
        if image.ndim == 3:
            canvas = np.zeros((die_h, die_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((die_h, die_w), dtype=image.dtype)
        ox, oy = ix1 - x1, iy1 - y1
        canvas[oy:oy + (iy2 - iy1), ox:ox + (ix2 - ix1)] = crop
        return canvas

    raise ValueError(f"Unknown border_mode: {border_mode!r}")


# #############################################################################
# #                                                                           #
# #   PUBLIC API  (요청한 2개 함수 + 재사용 데이터 구조)                       #
# #                                                                           #
# #############################################################################

# --- 사용자 조정 기본값 -----------------------------------------------------
DEFAULT_GRID_METHOD = "corner"   # "corner"(권장, street 선으로 코너 직접 검출) | "hybrid" | "std" | "color"
DEFAULT_PIXEL_PER_UNIT = 32      # 실측 좌표 환산 (px / unit)
DEFAULT_EDGE_MARGIN = 1.0        # die 포함 기준: 중심거리 <= r * 이값.
                                 #   1.0=원 안의 die 전부(EDGE 포함), 0.98=가장자리 제외

# --- EDGE die 판정 방식 (둘 다 계산되어 entry 에 저장; is_edge 가 무엇을 가리킬지 선택) ---
#   "circle" : is_edge = is_edge_partial (die 사각형이 wafer 원 밖으로 일부라도 나감)
#   "ring"   : is_edge = is_edge_ring    (die 격자에서 8방향 이웃이 다 차 있지 않은 최외곽)
#   "both"   : is_edge = is_edge_partial OR is_edge_ring
DEFAULT_EDGE_MODE = "circle"

# crop 영역 보정/확장 (die 사이 street 포함, 미세 정렬 오차 보정용)
DEFAULT_OFFSET_X = 0   # crop 중심 X 위치 보정 (px). +면 오른쪽, -면 왼쪽으로 이동
DEFAULT_OFFSET_Y = 0   # crop 중심 Y 위치 보정 (px). +면 아래쪽, -면 위쪽으로 이동
DEFAULT_MARGIN_X = 0   # 좌/우로 각각 더 포함할 영역 (px). die 폭이 +2*margin_x 만큼 커짐
DEFAULT_MARGIN_Y = 0   # 상/하로 각각 더 포함할 영역 (px). die 높이가 +2*margin_y 만큼 커짐

# --- Notch 회전(angle) 보정 ---------------------------------------------------
DEFAULT_NOTCH_ALIGN = True       # build_die_map 시작 시 notch 로 회전 보정 (notch 없으면 자동 skip)
DEFAULT_NOTCH_REF_DEG = 90.0     # notch 의 정상 위치 (이미지 좌표 각도. 90 = 아래쪽/6시 방향)
DEFAULT_NOTCH_MIN_ANGLE = 0.05   # 이보다 작은 오차(deg)는 보정 생략 (불필요한 워핑 방지)
DEFAULT_NOTCH_MIN_DEPTH = 4.0    # notch 인정 절대 최소 파임 깊이 (px). 엣지 노이즈 하한
DEFAULT_NOTCH_NOISE_K = 3.0      # ★ 홈 크기 강인성: 실효임계 = max(MIN_DEPTH, 림노이즈*K)
                                 #   거친 엣지 웨이퍼에서 자동으로 임계가 올라가 오검출 방지
DEFAULT_NOTCH_OPEN_KSIZE = 3     # ★ 림 컬러 노이즈 강인성: 경계를 '가장 큰 연결성분'으로
                                 #   잡고, 이 크기 open 으로 얇은 노이즈 다리를 끊음(0=open끔)

# --- Angle alignment method --------------------------------------------------
DEFAULT_ANGLE_ALIGN_METHOD = "die_render"  # "die_render"(V5 기본) | "notch" | "vertical_line" | "none"

# --- die_render 얼라인 (검출 die 를 굵기 3 사각형으로 렌더한 격자에 sawline 정합) ---
DEFAULT_DIE_RENDER_THICKNESS = 3   # ★ dm.dies 를 cv2.rectangle 이 굵기로 렌더(=sawline 모양)
DEFAULT_DIE_RENDER_SEARCH_DEG = 6.0  # ★ 회전 탐색 범위 ±(deg). 넓혀서 큰 기울기도 잡음(예전 3.5)
DEFAULT_DIE_RENDER_COARSE_STEP = 0.15 # 1차 탐색 간격(deg) — 범위 넓힘에 맞춰 약간 키움(속도)
DEFAULT_DIE_RENDER_FINE_STEP = 0.02   # 2차 정밀 탐색 간격(deg)
DEFAULT_DIE_RENDER_ROI_RATIO = 0.55   # 정합에 쓰는 중앙 ROI 반경 비율(wafer_r 대비)
DEFAULT_DIE_RENDER_MAX_DIM = 1400     # 정합 ROI 다운스케일 한계(px) — 클수록 각도 정밀↑(속도↓)
DEFAULT_DIE_RENDER_MAX_ITER = 3       # 2-pass 반복(잔차 수렴까지)
# ★ 고도화: FFT 교차검증 + 합의(agreement). 두 독립 단서가 일치하면 신뢰↑, 어긋나면 재탐색.
DEFAULT_ANGLE_FFT_MAX_DIM = 1024   # FFT ROI 다운스케일 한계(px)
DEFAULT_ANGLE_AGREE_TOL_DEG = 0.40 # projection vs FFT 가 이 안에서 일치하면 '합의'로 본다
DEFAULT_ANGLE_FULL_SCAN_DEG = 44.0 # 합의 실패 시 ±이 범위까지 넓게 재탐색(거의 모든 기울기)
DEFAULT_VERTICAL_LINE_MAX_DEG = 6.0        # accept near-vertical lines within +/-deg
DEFAULT_VERTICAL_LINE_ROI_RATIO = 0.70     # inner wafer ROI used for longest-line scan
DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO = 0.25 # min Hough line length as wafer_r ratio
DEFAULT_VERTICAL_LINE_MAX_ROI_SIZE = 1200  # downsample ROI limit for Hough speed
DEFAULT_AXIS_SEGMENT_MIN_LEN_RATIO = 0.035 # allow broken pieces; cluster decides total
DEFAULT_AXIS_CLUSTER_POS_TOL_RATIO = 0.010 # same-axis grouping tolerance
DEFAULT_AXIS_AGREE_TOL_DEG = 0.75          # vertical/horizontal agreement guard
# vertical_line 보정의 선(線) 소스 방식:
#   True  = 이진화(Otsu)로 두꺼운 die 선을 통째로 잡은 뒤 1px 중심선으로 '세선화'해 비교
#           (사용자 요청: "두껍게 잡은 선을 얇게 해서 비교"). 각도 비교가 두께에 흔들리지 않음.
#   False = 기존 Canny 엣지(이미 가늘다) 사용.
DEFAULT_AXIS_BINARIZE_LINES = True

# --- V2 전용 파라미터 ---------------------------------------------------------
DEFAULT_NOTCH_SECTOR_DEG = 70.0  # [기능1] notch 를 wafer '아래쪽'(ref±이 각도)에서만 탐색
DEFAULT_NOTCH_NOISE_MARGIN = 3.0 # [기능1] notch 인정 임계 = max(MIN_DEPTH, 림노이즈floor + 이값)
                                 #   (가산형 — 림 노이즈 floor 가 커도 진짜 notch 를 안 놓침)
DEFAULT_NOTCH_SMOOTH_DEG = 0.25  # [기능1] notch 검출 전 둘레 깊이를 이 각도폭으로 스무딩.
                                 #   notch(넓고 매끄러움)는 보존, 가장자리 거칠기(좁은 bite)는
                                 #   눌러서 -> 오염이 notch 깊이만큼 심해도 notch 만 골라냄.
DEFAULT_CLEAN_WAFER = True       # [기능4] 시작 시 wafer 원판 밖을 검정으로 (외부 노이즈 제거)
DEFAULT_VERIFY_TOL_DEG = 0.5     # [기능2] notch 각도 vs die 격자 각도 허용 오차(deg)
DEFAULT_GRID_ANGLE_RANGE = 4.0   # [기능2] die 격자 각도 탐색 범위 ±(deg)
DEFAULT_QUAD_BALANCE_TOL = 0.08  # [기능3] 4분면 coverage 허용 편차 (이내면 balanced)


@dataclass
class WaferDieMap:
    """build_die_map() 결과. locate_die() 등에서 재사용하는 격자/웨이퍼 정보 묶음.

    필드
    ----
    wafer_cx, wafer_cy, wafer_r : 웨이퍼 중심/반지름 (px)
    pitch_x, pitch_y            : die 가로/세로 pitch (px, sub-pixel float)
    x0, y0                      : grid origin(중심 코너) (px)
    die_w, die_h                : die crop 크기 (= round(pitch)) (px)
    pixel_per_unit              : 실측 좌표 환산 단위 (px/unit)
    dies                        : die entry 리스트 (아래 형식)
    dies_by_index               : {(ix,iy): entry} 빠른 조회용
    image_shape                 : (H, W) 원본 이미지 크기

    die entry(dict)
    ---------------
    {
      "index":       (ix, iy),
      "center_px":   (cx, cy),
      "rect_px":     (x1, y1, x2, y2),   # 순수 die 영역
      "crop_rect_px":(x1, y1, x2, y2),   # offset/margin 적용된 crop 영역
      "real_coord":  (rx, ry),           # die 중심 기준 실측 좌표
      "is_edge_partial": bool,           # 정의① die 사각형이 wafer 원 밖으로 일부라도 나감
      "is_edge_ring":    bool,           # 정의② 격자에서 8방향 이웃이 다 차 있지 않은 최외곽
      "is_edge":     bool,               # edge_mode 가 가리키는 값(circle→partial / ring→ring / both→OR)
      "image":       np.ndarray,         # with_crops=True 일 때만 (crop_rect_px 영역)
    }
    """
    wafer_cx: int
    wafer_cy: int
    wafer_r: int
    pitch_x: float
    pitch_y: float
    x0: int
    y0: int
    die_w: int
    die_h: int
    pixel_per_unit: int
    dies: List[Dict[str, Any]] = field(default_factory=list)
    dies_by_index: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)
    image_shape: Tuple[int, int] = (0, 0)
    rotation_deg: float = 0.0     # notch 보정으로 적용된 회전 각도 (0 = 보정 없음)
    aligned_image: Optional[np.ndarray] = field(default=None, repr=False)
                                  # ★ [기능5] 항상 채워짐 = clean+align 후 실제 사용 이미지.
                                  #   모든 좌표(rect/center)는 이 이미지 기준이므로
                                  #   crop/시각화/YOLO 도 이 이미지를 사용해야 좌표가 맞는다.
    # --- V2 추가 필드 ---
    notch_center_px: Optional[Tuple[int, int]] = None   # [기능1] notch 파임의 중심 픽셀점
    die_grid_angle_resid: float = 0.0   # [기능2] 보정 후 die 격자 잔여 기울기(deg)
    angle_verified: bool = False        # [기능2] notch 각도와 die 격자 각도 일치 여부
    quadrant_report: Dict[str, Any] = field(default_factory=dict)  # [기능3] 4분면 검증 결과
    edge_mode: str = DEFAULT_EDGE_MODE  # [V5] is_edge 가 가리키는 기준(circle|ring|both)
    angle_confidence: float = 1.0  # [V5 고도화] 각도 신뢰도 0~1 (projection·FFT 합의 기반)
    angle_agree: bool = True        # [V5 고도화] projection 과 FFT 가 합의했는지

    def get_die(self, ix: int, iy: int) -> Optional[Dict[str, Any]]:
        """(ix, iy) die entry 반환 (없으면 None)."""
        return self.dies_by_index.get((ix, iy))

    @property
    def num_dies(self) -> int:
        return len(self.dies)


def _rect_crosses_circle(x1: int, y1: int, x2: int, y2: int,
                         cx: int, cy: int, r: int) -> bool:
    """die rect 의 한 모서리라도 웨이퍼 원 밖이면 True (=정의①: 부분 die)."""
    r2 = r * r
    for (px, py) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        if (px - cx) ** 2 + (py - cy) ** 2 > r2:
            return True
    return False


def _normalize_edge_mode(edge_mode: str) -> str:
    """edge_mode 문자열 정규화 -> "circle" | "ring" | "both"."""
    m = str(edge_mode).lower().strip()
    if m in ("circle", "partial", "disc", "crop", "1"):
        return "circle"
    if m in ("ring", "neighbor", "outer", "outermost", "grid", "2"):
        return "ring"
    if m in ("both", "or", "all", "union"):
        return "both"
    raise ValueError("edge_mode must be 'circle', 'ring', or 'both'.")


def _resolve_edge_flag(is_partial: bool, is_ring: bool, edge_mode: str) -> bool:
    """edge_mode 에 따라 is_edge 가 가리킬 값 결정 (mode 는 정규화된 값)."""
    if edge_mode == "circle":
        return bool(is_partial)
    if edge_mode == "ring":
        return bool(is_ring)
    return bool(is_partial or is_ring)   # "both"


def _load_bgr(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """경로(str/Path) 또는 BGR ndarray 를 받아 BGR 이미지로 반환."""
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(image))
    return img


def _rotate_wafer_keep_size(image_bgr: np.ndarray,
                            wafer_cx: int, wafer_cy: int,
                            angle_deg: float,
                            interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Rotate around wafer center without changing output image size.

    ★ 보간법 = INTER_CUBIC (기본). 예전 INTER_NEAREST 는 die 격자처럼 '주기적인
    미세 패턴' 을 회전할 때 계단/모아레(aliasing) 가 심해 결과가 '깨져' 보였다.
    CUBIC(컬러는 LINEAR 대비 더 매끈)으로 바꿔 회전 후에도 격자가 또렷하다.
    각도 측정은 다운스케일·이진화 기반이라 약한 보간 블러에 영향받지 않는다.
    회전각 0 이면 보간 자체를 건너뛰어 원본을 그대로 보존(불필요한 워핑 방지).
    """
    if abs(float(angle_deg)) < 1e-9:
        return image_bgr.copy()
    H, W = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((float(wafer_cx), float(wafer_cy)), angle_deg, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (W, H),
                             flags=interp, borderValue=(0, 0, 0))
    assert rotated.shape[:2] == (H, W)
    return rotated


def _crop_rect(cx: float, cy: float, die_w: int, die_h: int,
               offset_x: int, offset_y: int,
               margin_x: int, margin_y: int) -> Tuple[int, int, int, int]:
    """die 중심에 offset(위치 보정) + margin(영역 확장)을 적용한 crop 사각 좌표.

    crop 중심 = (cx + offset_x, cy + offset_y)
    crop 크기 = (die_w + 2*margin_x, die_h + 2*margin_y)
    반환: (x1, y1, x2, y2)  ← die 사이 street 를 포함하려면 margin 을 키운다.
    """
    ccx = cx + offset_x
    ccy = cy + offset_y
    half_w = die_w / 2.0 + margin_x
    half_h = die_h / 2.0 + margin_y
    return (int(round(ccx - half_w)), int(round(ccy - half_h)),
            int(round(ccx + half_w)), int(round(ccy + half_h)))


def crop_die(image: np.ndarray, center_x: float, center_y: float,
             die_w: int, die_h: int, *,
             offset_x: int = DEFAULT_OFFSET_X, offset_y: int = DEFAULT_OFFSET_Y,
             margin_x: int = DEFAULT_MARGIN_X, margin_y: int = DEFAULT_MARGIN_Y,
             border_mode: str = "pad") -> Optional[np.ndarray]:
    """die 중심 기준으로 offset/margin 을 적용해 crop 한 이미지를 반환.

    - offset_x/y : crop 위치를 (px) 만큼 이동해 미세 정렬 오차를 보정.
    - margin_x/y : 각 변으로 (px) 만큼 영역을 더 포함 (die 사이 street/이웃 일부 포함).
    실제 crop = 중심 (center_x+offset_x, center_y+offset_y),
                크기 (die_w+2*margin_x, die_h+2*margin_y).
    원본 clip_die 를 그대로 재사용한다 (border_mode: "pad" 고정크기 | "crop" 가변).
    """
    return clip_die(image,
                    int(round(center_x + offset_x)),
                    int(round(center_y + offset_y)),
                    int(round(die_w + 2 * margin_x)),
                    int(round(die_h + 2 * margin_y)),
                    border_mode=border_mode)


# =============================================================================
# (0) Notch 기반 회전(angle) 보정 — build_die_map 의 모든 연산 전에 적용
# =============================================================================
def _wafer_silhouette(gray: np.ndarray, black_thr: int, open_ksize: int) -> np.ndarray:
    """wafer 실루엣 마스크(가장 큰 연결성분) — 림 주변 컬러 노이즈에 강인.

    notch 검출의 경계 측정이 '림 밖 컬러 노이즈(wafer 색과 다른 선/얼룩)'를 경계로
    오인하지 않도록, 단순 임계 마스크에서 (옵션) 얇은 노이즈 다리를 open 으로 끊고
    '가장 큰 연결성분' 만 남긴다. notch 같은 오목부(concavity)는 보존된다.
    """
    _, mask = cv2.threshold(gray, black_thr, 255, cv2.THRESH_BINARY)
    if open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if ncomp <= 1:
        return mask
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # 배경(0) 제외 최대 성분
    return (labels == big).astype(np.uint8)


def detect_notch_angle(image_bgr: np.ndarray,
                       wafer_cx: int, wafer_cy: int, wafer_r: int,
                       notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                       n_angles: int = 14400,
                       min_depth: float = DEFAULT_NOTCH_MIN_DEPTH,
                       noise_k: float = DEFAULT_NOTCH_NOISE_K,
                       min_span_deg: float = 0.06,
                       black_thr: int = 20,
                       open_ksize: int = DEFAULT_NOTCH_OPEN_KSIZE) -> Optional[float]:
    """Notch(웨이퍼 림의 작은 홈) 위치로 회전 오차(deg)를 측정 — 홈 크기에 강인.

    원리: 웨이퍼 둘레를 각도별로 방사형 스캔해 경계 반지름을 구하면,
    notch 위치에서만 경계가 안쪽으로 파인다(indentation). 그 파임 구간의
    깊이-가중 원형 centroid 각도와 기준 각도(notch_ref_deg)의 차이 = 회전 오차.

    홈 크기 강인성 (★ 적응형 임계)
    -----------------------------
    1) 후보 = (깊이 > min_depth) & (연속 각도폭 >= min_span_deg) 인 가장 깊은 구간.
    2) 림 노이즈(stair-step edge) floor 를 후보 '밖' 둘레에서 추정하고,
       실효 임계 = max(min_depth, 노이즈floor * noise_k) 로 자동 조정.
       => 엣지가 거친 웨이퍼에서도 오검출 없이, 작은 홈(≈8px↑)~큰 홈까지 동일 처리.

    Parameters
    ----------
    notch_ref_deg : notch 의 정상 위치 (이미지 좌표 각도. 90 = 아래쪽/6시 방향)
    n_angles      : 둘레 각도 샘플 수 (14400 = 0.025도 간격)
    min_depth     : notch 인정 절대 최소 파임 깊이 (px). 엣지 노이즈 하한.
    noise_k       : 적응형 임계 배율. 실효임계 = max(min_depth, 림노이즈*noise_k).
    min_span_deg  : notch 최소 각도 폭 (deg) — 단일-샘플 점노이즈 제외용.
    black_thr     : 배경(검정) 판정 임계.

    Returns
    -------
    float : 회전 오차 (deg). cv2.getRotationMatrix2D 의 angle 로 그대로 사용.
    None  : notch 를 찾지 못함 (notch 없는 이미지 -> 보정 불필요)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ★ 림 주변 '컬러 노이즈'(wafer 색과 다른 선/얼룩)에 강인하도록, 단순 임계 대신
    #   '가장 큰 연결성분(wafer 실루엣)'을 경계로 사용 (림 밖 노이즈를 경계로 오인 X).
    sil = _wafer_silhouette(gray, black_thr, open_ksize)

    # 둘레 방사형 스캔 (vectorized): 각도별 경계(실루엣 최외곽) 반지름.
    #   rs 범위를 넓게(0.93~1.015r) 잡아 깊은 홈도 끝까지 측정.
    angs = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    rs = np.linspace(wafer_r * 0.93, wafer_r * 1.015, 200)
    xs = (wafer_cx + rs[None, :] * np.cos(angs)[:, None]).astype(np.int32)
    ys = (wafer_cy + rs[None, :] * np.sin(angs)[:, None]).astype(np.int32)
    np.clip(xs, 0, W - 1, out=xs)
    np.clip(ys, 0, H - 1, out=ys)
    on_wafer = sil[ys, xs] > 0
    idx = np.where(on_wafer.any(axis=1),
                   on_wafer.shape[1] - 1 - np.argmax(on_wafer[:, ::-1], axis=1), 0)
    radii = rs[idx]
    depth = np.median(radii) - radii          # 양수 = 안쪽으로 파임

    # 1) 후보 구간: 절대 하한(min_depth) 이상 + 연속(span) 인 구간들
    above = np.where(depth > min_depth)[0]
    if len(above) == 0:
        return None
    clusters: List[List[int]] = [[above[0]]]
    for v in above[1:]:
        if v - clusters[-1][-1] <= 2:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # 0도/360도 경계에 걸친 cluster 병합
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == n_angles - 1:
        clusters[0] = clusters[-1] + [c + n_angles for c in clusters[0]]
        clusters.pop()
    clusters = [c for c in clusters
                if (c[-1] - c[0]) * 360.0 / n_angles >= min_span_deg]
    if not clusters:
        return None

    # 2) 가장 깊은(깊이합 최대) 구간 = notch 후보
    cand = max(clusters,
               key=lambda c: float(depth[[i % n_angles for i in c]].sum()))

    # 3) ★ 적응형 임계: 후보 '밖' 둘레의 노이즈 floor 로 실효 임계 결정
    keep = np.ones(n_angles, dtype=bool)
    keep[[i % n_angles for i in cand]] = False
    outside = depth[keep]
    noise_floor = float(np.percentile(outside, 99.5)) if outside.size else 0.0
    eff_thr = max(min_depth, noise_floor * noise_k)
    d = depth[[i % n_angles for i in cand]]
    if float(d.max()) < eff_thr:               # 노이즈 대비 충분히 깊지 않음 -> notch 아님
        return None

    # 4) 깊이-가중 원형 centroid -> notch 각도 -> 기준 대비 오차 (-180~+180 wrap)
    a = np.array([2.0 * np.pi * (i % n_angles) / n_angles for i in cand])
    notch_deg = math.degrees(math.atan2(float((np.sin(a) * d).sum()),
                                        float((np.cos(a) * d).sum()))) % 360.0
    return ((notch_deg - notch_ref_deg + 180.0) % 360.0) - 180.0


def align_wafer_by_notch(image_bgr: np.ndarray,
                         notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                         min_angle_deg: float = DEFAULT_NOTCH_MIN_ANGLE,
                         max_iter: int = 2) -> Tuple[np.ndarray, float]:
    """Notch 로 회전 오차를 측정/보정한 이미지 반환 -> (보정 이미지, 적용 각도 deg).

    - notch 가 없거나 오차가 min_angle_deg 미만이면 원본 그대로 반환 (각도 0.0).
    - 회전은 웨이퍼 중심 기준, INTER_NEAREST 사용
      (보간 블러가 생기면 die 내부 보조선이 격자 검출에 끼어들 수 있어 픽셀 보존).
    - max_iter=2 : 1차 보정 후 잔차가 남으면 누적 각도로 "원본에서" 다시 1회 회전
      (항상 원본 -> 1회 워핑이므로 이중 보간 없음).
    - ★ 출력 이미지 크기는 입력과 '동일하게 유지'된다 (10000x10000 -> 10000x10000).
      캔버스를 키우지 않으므로 회전 후에도 해상도/좌표계가 변하지 않는다.
    """
    def _angle(im, cx_, cy_, r_):   # V2: 아래쪽 sector notch 의 각도만 추출
        res = detect_notch(im, cx_, cy_, r_, notch_ref_deg=notch_ref_deg)
        return None if res is None else res[0]

    wafer_cx, wafer_cy, wafer_r = detect_wafer(image_bgr)
    err = _angle(image_bgr, wafer_cx, wafer_cy, wafer_r)
    if err is None or abs(err) < min_angle_deg:
        return image_bgr, 0.0

    H, W = image_bgr.shape[:2]

    def _rotate(total_deg: float) -> np.ndarray:
        # dsize=(W, H) 로 출력 캔버스를 입력과 '동일 크기'로 고정한다.
        #   (PIL 의 rotate(expand=True) 처럼 캔버스가 커지지 않음.
        #    wafer 는 중앙에 있어 작은 각도 회전으로 잘리지 않는다.)
        M = cv2.getRotationMatrix2D((float(wafer_cx), float(wafer_cy)), total_deg, 1.0)
        rotated = cv2.warpAffine(image_bgr, M, (W, H),
                                 flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
        assert rotated.shape[:2] == (H, W)   # 크기 불변 보장
        return rotated

    total = float(err)
    aligned = _rotate(total)
    for _ in range(max(0, max_iter - 1)):        # 잔차 정밀 보정
        cx2, cy2, r2 = detect_wafer(aligned)
        res = _angle(aligned, cx2, cy2, r2)
        if res is None or abs(res) < min_angle_deg:
            break
        total += float(res)
        aligned = _rotate(total)
    return aligned, total


def _axis_deviation_deg(x1: int, y1: int, x2: int, y2: int,
                        axis: str) -> Optional[float]:
    """Return cv2 rotation angle needed to make a segment vertical/horizontal."""
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if dx == 0.0 and dy == 0.0:
        return None
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    if axis == "vertical":
        return angle - 90.0
    if axis == "horizontal":
        return angle if angle <= 90.0 else angle - 180.0
    raise ValueError("axis must be 'vertical' or 'horizontal'.")


def _vertical_deviation_deg(x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
    """Return cv2 rotation angle needed to make an undirected segment vertical."""
    return _axis_deviation_deg(x1, y1, x2, y2, "vertical")


def _horizontal_deviation_deg(x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
    """Return cv2 rotation angle needed to make an undirected segment horizontal."""
    return _axis_deviation_deg(x1, y1, x2, y2, "horizontal")


def _thin_binary(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to one-pixel strokes for more stable Hough angles."""
    binary = ((mask > 0).astype(np.uint8) * 255)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(binary)

    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    for _ in range(512):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel


def _axis_position_at_center(x1: int, y1: int, x2: int, y2: int,
                             axis: str, center_x: float,
                             center_y: float) -> Optional[float]:
    """Project a segment to wafer center and return its axis position."""
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    mx = (float(x1) + float(x2)) * 0.5
    my = (float(y1) + float(y2)) * 0.5
    if axis == "vertical":
        if abs(dy) < 1e-6:
            return None
        return mx + (dx / dy) * (center_y - my)
    if axis == "horizontal":
        if abs(dx) < 1e-6:
            return None
        return my + (dy / dx) * (center_x - mx)
    raise ValueError("axis must be 'vertical' or 'horizontal'.")


def _cluster_axis_segments(segments: List[Dict[str, Any]],
                           axis: str,
                           pos_tol: float,
                           min_total_len: float,
                           center_x: float,
                           center_y: float,
                           roi_origin_x: int,
                           roi_origin_y: int,
                           scale: float
                           ) -> Optional[Tuple[float, Tuple[int, int, int, int], float]]:
    """Merge broken collinear Hough segments and return the strongest axis."""
    if not segments:
        return None

    clusters: List[List[Dict[str, Any]]] = []
    for seg in sorted(segments, key=lambda s: float(s["pos"])):
        if not clusters:
            clusters.append([seg])
            continue
        prev = clusters[-1]
        prev_len = sum(float(s["length"]) for s in prev)
        prev_pos = sum(float(s["pos"]) * float(s["length"]) for s in prev) / prev_len
        if abs(float(seg["pos"]) - prev_pos) <= pos_tol:
            prev.append(seg)
        else:
            clusters.append([seg])

    best: Optional[Tuple[float, Tuple[int, int, int, int], float]] = None
    for cluster in clusters:
        total_len = sum(float(s["length"]) for s in cluster)
        if total_len < min_total_len:
            continue
        dev = sum(float(s["dev"]) * float(s["length"]) for s in cluster) / total_len
        pos = sum(float(s["pos"]) * float(s["length"]) for s in cluster) / total_len

        pts = []
        for s in cluster:
            x1, y1, x2, y2 = s["line"]
            pts.extend([(float(x1), float(y1)), (float(x2), float(y2))])

        if axis == "vertical":
            ys = [p[1] for p in pts]
            y_a, y_b = min(ys), max(ys)
            a = math.radians(90.0 + dev)
            slope = math.cos(a) / max(math.sin(a), 1e-6)
            x_a = pos + slope * (y_a - center_y)
            x_b = pos + slope * (y_b - center_y)
        else:
            xs = [p[0] for p in pts]
            x_a, x_b = min(xs), max(xs)
            slope = math.tan(math.radians(dev))
            y_a = pos + slope * (x_a - center_x)
            y_b = pos + slope * (x_b - center_x)

        line = (
            int(round(x_a / scale)) + roi_origin_x,
            int(round(y_a / scale)) + roi_origin_y,
            int(round(x_b / scale)) + roi_origin_x,
            int(round(y_b / scale)) + roi_origin_y,
        )
        length = total_len / scale
        if best is None or length > best[2]:
            best = (float(dev), line, float(length))
    return best


def _detect_long_axis_lines(image_bgr: np.ndarray,
                            wafer_cx: int, wafer_cy: int, wafer_r: int,
                            axes: Tuple[str, ...] = ("vertical", "horizontal"),
                            max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                            roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                            min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO,
                            segment_min_length_ratio: float = DEFAULT_AXIS_SEGMENT_MIN_LEN_RATIO,
                            cluster_pos_tol_ratio: float = DEFAULT_AXIS_CLUSTER_POS_TOL_RATIO,
                            canny_low: int = 50,
                            canny_high: int = 150,
                            hough_threshold: int = 80,
                            max_line_gap_ratio: float = 0.035,
                            max_roi_size: int = DEFAULT_VERTICAL_LINE_MAX_ROI_SIZE,
                            thin_edges: bool = True,
                            binarize_lines: bool = DEFAULT_AXIS_BINARIZE_LINES
                            ) -> Dict[str, Tuple[float, Tuple[int, int, int, int], float]]:
    """Detect strongest vertical/horizontal axes, merging broken line pieces.

    선(線) 소스는 두 가지:
      - binarize_lines=True : ROI를 Otsu 이진화해 '두꺼운' die 선을 통째로 잡은 뒤
        thin_edges면 1px 중심선으로 세선화한다. 두께가 각도 측정을 흔들지 않게 됨
        (사용자 요청: "두껍게 잡은 선을 얇게 해서 비교"). 빈 결과면 Canny로 폴백.
      - binarize_lines=False: 기존 Canny 엣지(이미 가늘다)를 그대로 사용.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    roi_r = max(8, int(round(wafer_r * roi_ratio)))
    x0 = max(0, wafer_cx - roi_r)
    x1 = min(W, wafer_cx + roi_r + 1)
    y0 = max(0, wafer_cy - roi_r)
    y1 = min(H, wafer_cy + roi_r + 1)
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return {}

    roi = gray[y0:y1, x0:x1]
    local_cx = wafer_cx - x0
    local_cy = wafer_cy - y0
    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    mask = ((xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= roi_r ** 2).astype(np.uint8) * 255

    scale = 1.0
    max_dim = max(roi.shape[:2])
    if max_dim > max_roi_size:
        scale = float(max_roi_size) / float(max_dim)
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    center_x = float(local_cx) * scale
    center_y = float(local_cy) * scale
    blur = cv2.GaussianBlur(roi, (3, 3), 0)

    edges = None
    if binarize_lines:
        # (1) 이진화: 두꺼운 die 선/무늬를 통째로 잡는다 (Otsu 자동 임계).
        _, binr = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binr = cv2.bitwise_and(binr, binr, mask=mask)
        # (2) 세선화: 두꺼운 선을 1px 중심선으로 → 각도 비교가 선 두께에 흔들리지 않음.
        line_mask = _thin_binary(binr) if thin_edges else binr
        if int(cv2.countNonZero(line_mask)) >= 1:
            edges = line_mask
    if edges is None:
        # 폴백(또는 binarize_lines=False): 기존 Canny 엣지(이미 가늘다).
        edges = cv2.Canny(blur, canny_low, canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        if thin_edges:
            edges = _thin_binary(edges)

    segment_min_len = max(12, int(round(wafer_r * segment_min_length_ratio * scale)))
    max_gap = max(6, int(round(wafer_r * max_line_gap_ratio * scale)))
    threshold = max(25, int(round(hough_threshold * min(1.0, scale))))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800.0,
                            threshold=threshold,
                            minLineLength=segment_min_len,
                            maxLineGap=max_gap)
    if lines is None:
        return {}

    candidates: Dict[str, List[Dict[str, Any]]] = {axis: [] for axis in axes}
    for raw in lines.reshape(-1, 4):
        lx1, ly1, lx2, ly2 = (int(v) for v in raw)
        length = math.hypot(float(lx2 - lx1), float(ly2 - ly1))
        if length < segment_min_len:
            continue
        for axis in axes:
            dev = _axis_deviation_deg(lx1, ly1, lx2, ly2, axis)
            if dev is None or abs(dev) > max_deviation_deg:
                continue
            pos = _axis_position_at_center(lx1, ly1, lx2, ly2, axis, center_x, center_y)
            if pos is None:
                continue
            candidates[axis].append({
                "dev": float(dev),
                "pos": float(pos),
                "length": float(length),
                "line": (lx1, ly1, lx2, ly2),
            })

    min_total_len = max(float(segment_min_len) * 1.5,
                        float(wafer_r) * min_length_ratio * scale)
    pos_tol = max(4.0, float(wafer_r) * cluster_pos_tol_ratio * scale)
    out: Dict[str, Tuple[float, Tuple[int, int, int, int], float]] = {}
    for axis, segs in candidates.items():
        best = _cluster_axis_segments(segs, axis, pos_tol, min_total_len,
                                      center_x, center_y, x0, y0, scale)
        if best is not None:
            out[axis] = best
    return out


def _detect_longest_vertical_line(image_bgr: np.ndarray,
                                  wafer_cx: int, wafer_cy: int, wafer_r: int,
                                  max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                                  roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                                  min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO,
                                  **kwargs: Any
                                  ) -> Optional[Tuple[float, Tuple[int, int, int, int], float]]:
    """Find the strongest near-vertical axis, merging broken pieces."""
    axes = _detect_long_axis_lines(
        image_bgr, wafer_cx, wafer_cy, wafer_r,
        axes=("vertical",),
        max_deviation_deg=max_deviation_deg,
        roi_ratio=roi_ratio,
        min_length_ratio=min_length_ratio,
        **kwargs)
    return axes.get("vertical")


def _detect_longest_horizontal_line(image_bgr: np.ndarray,
                                    wafer_cx: int, wafer_cy: int, wafer_r: int,
                                    max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                                    roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                                    min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO,
                                    **kwargs: Any
                                    ) -> Optional[Tuple[float, Tuple[int, int, int, int], float]]:
    """Find the strongest near-horizontal axis, merging broken pieces."""
    axes = _detect_long_axis_lines(
        image_bgr, wafer_cx, wafer_cy, wafer_r,
        axes=("horizontal",),
        max_deviation_deg=max_deviation_deg,
        roi_ratio=roi_ratio,
        min_length_ratio=min_length_ratio,
        **kwargs)
    return axes.get("horizontal")


def measure_vertical_line_angle(image_bgr: np.ndarray,
                                wafer_cx: int, wafer_cy: int, wafer_r: int,
                                max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                                roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                                min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO
                                ) -> Optional[float]:
    """Measure rotation from the strongest vertical wafer axis."""
    res = _detect_longest_vertical_line(
        image_bgr, wafer_cx, wafer_cy, wafer_r,
        max_deviation_deg=max_deviation_deg,
        roi_ratio=roi_ratio,
        min_length_ratio=min_length_ratio)
    return None if res is None else res[0]


def measure_horizontal_line_angle(image_bgr: np.ndarray,
                                  wafer_cx: int, wafer_cy: int, wafer_r: int,
                                  max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                                  roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                                  min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO
                                  ) -> Optional[float]:
    """Measure rotation from the strongest horizontal wafer axis."""
    res = _detect_longest_horizontal_line(
        image_bgr, wafer_cx, wafer_cy, wafer_r,
        max_deviation_deg=max_deviation_deg,
        roi_ratio=roi_ratio,
        min_length_ratio=min_length_ratio)
    return None if res is None else res[0]


def measure_axis_line_angle(image_bgr: np.ndarray,
                            wafer_cx: int, wafer_cy: int, wafer_r: int,
                            max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                            roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                            min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO,
                            agree_tol_deg: float = DEFAULT_AXIS_AGREE_TOL_DEG
                            ) -> Optional[float]:
    """Measure rotation from merged vertical and horizontal wafer axes."""
    axes = _detect_long_axis_lines(
        image_bgr, wafer_cx, wafer_cy, wafer_r,
        axes=("vertical", "horizontal"),
        max_deviation_deg=max_deviation_deg,
        roi_ratio=roi_ratio,
        min_length_ratio=min_length_ratio)
    if not axes:
        return None

    vertical = axes.get("vertical")
    horizontal = axes.get("horizontal")
    if vertical is not None and horizontal is not None:
        v_angle, _, v_len = vertical
        h_angle, _, h_len = horizontal
        if abs(v_angle - h_angle) <= agree_tol_deg:
            return float((v_angle * v_len + h_angle * h_len) / (v_len + h_len))
        return float(v_angle if v_len >= h_len else h_angle)
    only = vertical if vertical is not None else horizontal
    return None if only is None else float(only[0])


def align_wafer_by_vertical_line(image_bgr: np.ndarray,
                                 min_angle_deg: float = DEFAULT_NOTCH_MIN_ANGLE,
                                 max_deviation_deg: float = DEFAULT_VERTICAL_LINE_MAX_DEG,
                                 roi_ratio: float = DEFAULT_VERTICAL_LINE_ROI_RATIO,
                                 min_length_ratio: float = DEFAULT_VERTICAL_LINE_MIN_LEN_RATIO,
                                 max_iter: int = 2) -> Tuple[np.ndarray, float]:
    """Align wafer sequentially: vertical axis first, then horizontal residual."""
    wafer_cx, wafer_cy, _ = detect_wafer(image_bgr)
    total = 0.0
    aligned = image_bgr

    for _ in range(max(1, max_iter)):
        changed = False
        cx, cy, r = detect_wafer(aligned)
        v_err = measure_vertical_line_angle(
            aligned, cx, cy, r,
            max_deviation_deg=max_deviation_deg,
            roi_ratio=roi_ratio,
            min_length_ratio=min_length_ratio)
        if v_err is not None and abs(v_err) >= min_angle_deg:
            total += float(v_err)
            aligned = _rotate_wafer_keep_size(image_bgr, wafer_cx, wafer_cy, total)
            changed = True

        cx, cy, r = detect_wafer(aligned)
        h_err = measure_horizontal_line_angle(
            aligned, cx, cy, r,
            max_deviation_deg=max_deviation_deg,
            roi_ratio=roi_ratio,
            min_length_ratio=min_length_ratio)
        if h_err is not None and abs(h_err) >= min_angle_deg:
            total += float(h_err)
            aligned = _rotate_wafer_keep_size(image_bgr, wafer_cx, wafer_cy, total)
            changed = True

        if not changed:
            break
    return aligned, total


# =============================================================================
# (V5) die_render 얼라인 : dm.dies 를 굵기 3 사각형으로 렌더한 격자에 sawline 정합
# =============================================================================
def render_die_grid_mask(image_shape: Tuple[int, ...],
                         dies: List[Any],
                         thickness: int = DEFAULT_DIE_RENDER_THICKNESS,
                         key: str = "rect_px") -> np.ndarray:
    """모든 die 를 cv2.rectangle(굵기 thickness)로 그린 격자 마스크(uint8) 반환.

    dies 항목은 dict(rect_px 키) 또는 (x1,y1,x2,y2) 튜플 둘 다 허용.
    굵기 3 사각 테두리들의 합집합 = wafer 의 sawline 격자 모양 = 정합/시각화 기준 템플릿.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), np.uint8)
    t = max(1, int(thickness))
    for d in dies:
        if isinstance(d, dict):
            x1, y1, x2, y2 = d[key]
        else:
            x1, y1, x2, y2 = d
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, t)
    return mask


def _prelim_die_rects(image_bgr: np.ndarray,
                      wafer_cx: int, wafer_cy: int, wafer_r: int,
                      grid_method: str = DEFAULT_GRID_METHOD,
                      edge_margin: float = 1.0
                      ) -> Tuple[List[Tuple[int, int, int, int]],
                                 Tuple[float, float, int, int, int, int]]:
    """현재 이미지에서 die 격자를 검출해 wafer 원 안의 die 사각형 목록을 만든다.

    build_die_map 의 die 순회와 동일한 중심/rect 공식(축정렬). die_render 정합의
    '이상적 격자 템플릿' 재료(렌더할 사각형들)로 쓰인다.

    주의: 이 단계는 '아직 보정 전'(기울어진) 이미지에서 호출될 수 있다. "corner"
    격자검출은 정렬된 이미지를 전제로 해 기울면 실패하므로, 실패 시 기울기에
    강인한 autocorrelation 기반("std"->"hybrid")으로 자동 폴백한다. 템플릿은
    주기적 격자라 원점(phase)이 약간 어긋나도 '각도' 정합 결과는 동일하다.
    """
    pitch_x = pitch_y = None
    x0 = y0 = 0
    for m in (grid_method, "std", "hybrid"):
        try:
            pitch_x, pitch_y, x0, y0 = detect_grid(
                image_bgr, wafer_cx, wafer_cy, wafer_r, method=m)
            break
        except Exception:
            pitch_x = None
            continue
    if pitch_x is None or pitch_y is None or pitch_x <= 1 or pitch_y <= 1:
        return [], (0.0, 0.0, 0, 0, 0, 0)
    die_w = int(round(pitch_x))
    die_h = int(round(pitch_y))
    max_ix = int(np.ceil(wafer_r / pitch_x)) + 2
    max_iy = int(np.ceil(wafer_r / pitch_y)) + 2
    r_lim_sq = (wafer_r * edge_margin) ** 2
    rects: List[Tuple[int, int, int, int]] = []
    for iy in range(-max_iy, max_iy + 1):
        for ix in range(-max_ix, max_ix + 1):
            cx_d = int(round(x0 + ix * pitch_x + pitch_x / 2))
            cy_d = int(round(y0 - iy * pitch_y - pitch_y / 2))
            dx = cx_d - wafer_cx
            dy = cy_d - wafer_cy
            if dx * dx + dy * dy > r_lim_sq:
                continue
            x_a = cx_d - die_w // 2
            y_a = cy_d - die_h // 2
            rects.append((x_a, y_a, x_a + die_w, y_a + die_h))
    return rects, (float(pitch_x), float(pitch_y), x0, y0, die_w, die_h)


def _grid_projection_score(image_bgr: np.ndarray,
                           wafer_cx: int, wafer_cy: int, wafer_r: int,
                           roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
                           max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM):
    """중앙 ROI 를 Otsu 이진화해 '열/행 투영 분산' 점수함수를 만든다.

    반환: score(a) -> float  (없으면 None). score 는 격자를 a 만큼 회전 후
    열 투영·행 투영의 분산 합. 격자가 축에 맞으면 최대. die 격자 검출(grid 함수)에
    의존하지 않고 '이미지 픽셀' 만으로 동작하므로, 격자 검출이 실패해도 각도를 잰다.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    half = max(16, int(round(wafer_r * roi_ratio)))
    x0r, x1r = max(0, wafer_cx - half), min(W, wafer_cx + half)
    y0r, y1r = max(0, wafer_cy - half), min(H, wafer_cy + half)
    if x1r <= x0r + 8 or y1r <= y0r + 8:
        return None

    roi_w, roi_h = x1r - x0r, y1r - y0r
    scale = min(1.0, float(max_dim) / float(max(roi_w, roi_h)))
    sw = max(8, int(round(roi_w * scale)))
    sh = max(8, int(round(roi_h * scale)))
    roi = gray[y0r:y1r, x0r:x1r]
    roi_s = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else roi

    lcx = (wafer_cx - x0r) * scale
    lcy = (wafer_cy - y0r) * scale
    yy, xx = np.ogrid[:sh, :sw]
    cmask = ((xx - lcx) ** 2 + (yy - lcy) ** 2 <= (half * scale) ** 2)

    blur = cv2.GaussianBlur(roi_s, (3, 3), 0)
    _, binr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grid = binr.astype(np.float32)
    grid[~cmask] = 0.0
    if float(grid.sum()) < 1.0:
        return None

    rot_center = (sw / 2.0, sh / 2.0)
    inner = (((xx - lcx) ** 2 + (yy - lcy) ** 2 <= (half * scale * 0.92) ** 2)
             .astype(np.float32))

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(rot_center, float(a), 1.0)
        rot = cv2.warpAffine(grid, M, (sw, sh), flags=cv2.INTER_LINEAR)
        rot *= inner
        return float(rot.sum(axis=0).var() + rot.sum(axis=1).var())

    return score


def _search_peak(score, center: float, search_deg: float,
                 coarse_step: float, fine_step: float) -> Tuple[float, float]:
    """score(a) 를 center±search_deg 에서 coarse->fine->포물선보간으로 최대화.

    반환: (best_angle, best_score).
    """
    coarse = np.arange(center - search_deg, center + search_deg + 1e-9, coarse_step)
    cs = [score(a) for a in coarse]
    ci = int(np.argmax(cs))
    best = float(coarse[ci])

    fine = np.arange(best - coarse_step, best + coarse_step + 1e-9, fine_step)
    fs = [score(a) for a in fine]
    fi = int(np.argmax(fs))
    ang = float(fine[fi])
    peak = float(fs[fi])
    if 0 < fi < len(fine) - 1:
        ym, y0v, yp = fs[fi - 1], fs[fi], fs[fi + 1]
        denom = (ym - 2.0 * y0v + yp)
        if abs(denom) > 1e-9:
            ang += 0.5 * (ym - yp) / denom * fine_step
    return ang, peak


def measure_die_render_angle(image_bgr: np.ndarray,
                             wafer_cx: int, wafer_cy: int, wafer_r: int,
                             *,
                             die_rects: Optional[List[Tuple[int, int, int, int]]] = None,
                             dies: Optional[List[Dict[str, Any]]] = None,
                             grid_method: str = DEFAULT_GRID_METHOD,
                             thickness: int = DEFAULT_DIE_RENDER_THICKNESS,
                             search_deg: float = DEFAULT_DIE_RENDER_SEARCH_DEG,
                             coarse_step: float = DEFAULT_DIE_RENDER_COARSE_STEP,
                             fine_step: float = DEFAULT_DIE_RENDER_FINE_STEP,
                             center: float = 0.0,
                             roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
                             max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM
                             ) -> Optional[float]:
    """die 격자(= 모든 die 를 굵기 3 사각형으로 그린 구조)를 후보 각도로 회전하며
    '열/행 투영 주기성(분산)' 이 최대가 되는 각도 = wafer 기울기(deg) 를 잰다.

    ★ V5 고도화: 격자 검출(_prelim_die_rects)에 의존하지 않고 '이미지 픽셀' 만으로
    측정한다(검출 실패해도 각도를 잼). center±search_deg 범위를 탐색.
    die_rects/dies 인자는 하위호환용이며 측정 자체엔 쓰지 않는다.

    반환: 정렬에 적용할 회전각(deg). 신호가 없으면 None.
    """
    score = _grid_projection_score(image_bgr, wafer_cx, wafer_cy, wafer_r,
                                   roi_ratio, max_dim)
    if score is None:
        return None
    ang, _ = _search_peak(score, center, search_deg, coarse_step, fine_step)
    return float(ang)


def _measure_tilt_fft(image_bgr: np.ndarray,
                      wafer_cx: int, wafer_cy: int, wafer_r: int,
                      roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
                      max_dim: int = DEFAULT_ANGLE_FFT_MAX_DIM) -> Optional[float]:
    """2D FFT 스펙트럼으로 격자 기울기(=정렬 적용각, [-45,45)) 를 독립 측정.

    die 격자는 2D 주기 패턴이라 진폭 스펙트럼에 격자 축 방향으로 강한 peak 가 생긴다.
    세로/가로 두 축 peak 는 90° 차 → '4배각(4φ)' 합벡터로 모아 위상/4 = 기울기.
    이미지 전체를 반영하므로 projection 과 '독립적인' 교차검증 단서가 된다.
    반환은 projection 과 같은 '적용 회전각' 부호 규약(이미지 y축이 아래라 +1).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    half = max(16, int(round(wafer_r * roi_ratio)))
    x0r, x1r = max(0, wafer_cx - half), min(W, wafer_cx + half)
    y0r, y1r = max(0, wafer_cy - half), min(H, wafer_cy + half)
    roi = gray[y0r:y1r, x0r:x1r]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return None
    scale = min(1.0, float(max_dim) / float(max(roi.shape[:2])))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    n = min(roi.shape[:2])
    oy = (roi.shape[0] - n) // 2
    ox = (roi.shape[1] - n) // 2
    sq = roi[oy:oy + n, ox:ox + n].astype(np.float32)

    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((sq - float(sq.mean())) * win))
    mag2 = (F.real ** 2 + F.imag ** 2)

    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    dx = (xx - c).astype(np.float64)
    dy = (yy - c).astype(np.float64)
    rho = np.sqrt(dx * dx + dy * dy)

    rmin = max(4.0, n * 0.012)
    rmax = n * 0.45
    band = (rho >= rmin) & (rho <= rmax)
    if int(band.sum()) < 16:
        return None
    radial = np.bincount(rho.astype(np.int32)[band].ravel(),
                         weights=mag2[band].ravel(),
                         minlength=int(rmax) + 2)
    if radial.size == 0 or radial.max() <= 0:
        return None
    r_star = float(np.argmax(radial))
    if r_star < rmin:
        r_star = float(rmin + 1)

    ann = (np.abs(rho - r_star) <= max(2.0, r_star * 0.45)) & (rho >= rmin)
    if int(ann.sum()) < 16:
        ann = band
    phi = np.arctan2(dy[ann], dx[ann])
    w = mag2[ann].astype(np.float64)
    vec = np.sum(w * np.exp(1j * 4.0 * phi))
    if float(np.sum(w)) <= 0:
        return None
    tilt = float(np.degrees(np.angle(vec)) / 4.0)
    return float(((tilt + 45.0) % 90.0) - 45.0)   # wrap to [-45,45)


def measure_wafer_angle_robust(image_bgr: np.ndarray,
                               wafer_cx: int, wafer_cy: int, wafer_r: int,
                               *,
                               roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
                               max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM,
                               search_deg: float = DEFAULT_DIE_RENDER_SEARCH_DEG,
                               coarse_step: float = DEFAULT_DIE_RENDER_COARSE_STEP,
                               fine_step: float = DEFAULT_DIE_RENDER_FINE_STEP,
                               agree_tol: float = DEFAULT_ANGLE_AGREE_TOL_DEG,
                               full_scan_deg: float = DEFAULT_ANGLE_FULL_SCAN_DEG
                               ) -> Dict[str, Any]:
    """★ 고도화된 각도 측정 : projection(정밀) + FFT(독립) 교차검증으로 견고하게.

    절차
    ----
    1) projection 으로 ±search_deg 정밀 탐색(기본 단서).
    2) FFT 로 기울기를 '독립' 추정([-45,45), 큰 기울기도 잡음).
    3) 두 값이 agree_tol 안에서 일치 → 합의(신뢰↑), projection 채택(정밀).
    4) 불일치/한쪽 실패 → FFT 중심으로 projection 재탐색 + 0° 중심 광역(full_scan)
       재탐색까지 후보로 모아, '투영 peak 점수가 가장 큰' 각을 채택(진짜 격자 정렬).
       => 탐색범위를 벗어난 큰 기울기/이상치에도 조용히 0 으로 실패하지 않는다.

    반환 dict: {angle, confidence, agree, projection, fft, candidates}
    """
    score = _grid_projection_score(image_bgr, wafer_cx, wafer_cy, wafer_r,
                                   roi_ratio, max_dim)
    fft_a = _measure_tilt_fft(image_bgr, wafer_cx, wafer_cy, wafer_r,
                              roi_ratio=roi_ratio)
    if score is None:
        # projection 신호 없음 → FFT 라도 사용
        if fft_a is None:
            return {"angle": 0.0, "confidence": 0.0, "agree": False,
                    "projection": None, "fft": None, "candidates": []}
        return {"angle": float(fft_a), "confidence": 0.45, "agree": False,
                "projection": None, "fft": float(fft_a), "candidates": [fft_a]}

    proj_a, proj_s = _search_peak(score, 0.0, search_deg, coarse_step, fine_step)

    if fft_a is not None and abs(proj_a - fft_a) <= agree_tol:
        # 두 독립 단서 합의 → 가장 신뢰. projection(정밀) 채택.
        return {"angle": float(proj_a), "confidence": 0.97, "agree": True,
                "projection": float(proj_a), "fft": float(fft_a),
                "candidates": [proj_a, fft_a]}

    # 불일치/대각: 여러 후보를 모아 '투영 peak 점수' 가 가장 큰 각을 채택
    cand: List[Tuple[float, float]] = [(proj_a, proj_s)]
    if fft_a is not None:
        rng = max(coarse_step * 3, 1.0)
        cand.append(_search_peak(score, fft_a, rng, coarse_step, fine_step))
    # 0° 중심 광역 스캔(거의 모든 기울기 포함) — 탐색범위 밖 큰 기울기 구제
    cand.append(_search_peak(score, 0.0, full_scan_deg,
                             max(coarse_step * 2, 0.3), fine_step))
    best_a, best_s = max(cand, key=lambda t: t[1])
    agree = bool(fft_a is not None and abs(best_a - fft_a) <= agree_tol)
    conf = 0.9 if agree else 0.6
    return {"angle": float(best_a), "confidence": conf, "agree": agree,
            "projection": float(proj_a),
            "fft": (None if fft_a is None else float(fft_a)),
            "candidates": [c[0] for c in cand]}


def align_wafer_by_die_render(image_bgr: np.ndarray,
                              *,
                              grid_method: str = DEFAULT_GRID_METHOD,
                              thickness: int = DEFAULT_DIE_RENDER_THICKNESS,
                              search_deg: float = DEFAULT_DIE_RENDER_SEARCH_DEG,
                              coarse_step: float = DEFAULT_DIE_RENDER_COARSE_STEP,
                              fine_step: float = DEFAULT_DIE_RENDER_FINE_STEP,
                              roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
                              max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM,
                              min_angle_deg: float = DEFAULT_NOTCH_MIN_ANGLE,
                              max_iter: int = DEFAULT_DIE_RENDER_MAX_ITER,
                              return_info: bool = False):
    """die 격자(굵기 3) 투영 주기성 + FFT 교차검증으로 wafer 를 정렬 (반복 수렴).

    매 반복: 현재 이미지에서 robust 측정(projection+FFT) → 누적각으로 '원본'을
    CUBIC 회전(깨짐 없음) → 잔차가 작아지면 종료. 첫 반복 측정값을 신뢰도로 기록.

    return_info=False(기본): (aligned, total)
    return_info=True       : (aligned, total, info)  info={confidence, agree, ...}
    """
    base_cx, base_cy, _ = detect_wafer(image_bgr)
    total = 0.0
    aligned = image_bgr
    info: Dict[str, Any] = {"confidence": 0.0, "agree": False,
                            "projection": None, "fft": None}
    for it in range(max(1, max_iter)):
        cx, cy, r = detect_wafer(aligned)
        res = measure_wafer_angle_robust(
            aligned, cx, cy, r, roi_ratio=roi_ratio, max_dim=max_dim,
            search_deg=search_deg, coarse_step=coarse_step, fine_step=fine_step)
        if it == 0:
            info = res
        delta = float(res.get("angle") or 0.0)
        if abs(delta) < min_angle_deg:
            break
        total += delta
        aligned = _rotate_wafer_keep_size(image_bgr, base_cx, base_cy, total)
    if return_info:
        return aligned, total, info
    return aligned, total


# =============================================================================
# (V2-1) clean_wafer : wafer 원판 밖을 검정으로 (외부 노이즈 제거)  [기능4]
# =============================================================================
def clean_wafer(image_bgr: np.ndarray,
                black_thr: int = 20,
                open_ksize: int = DEFAULT_NOTCH_OPEN_KSIZE) -> np.ndarray:
    """wafer 원판 밖의 모든 픽셀을 검정으로 채워 반환 (외부 노이즈 제거).

    "wafer = 가운데 큰 원, 나머지는 검정" 을 구현. 유지 영역 =
       (가장 큰 연결성분 = wafer 실루엣)  AND  (검출 원판 disc: 중심거리<=r)
    - 실루엣 조건 : 떨어진(detached) 외부 노이즈 제거.
    - disc 조건  : 림에 '연결'되어 원 밖으로 삐져나온 노이즈까지 잘라냄.
    notch(원판의 오목부)는 wafer 소재가 아니므로 자연히 검정 유지(=보존).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sil = _wafer_silhouette(gray, black_thr, open_ksize)     # 1=wafer, 0=그외
    cx, cy, r = detect_wafer(image_bgr)
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    keep = (sil > 0) & disc
    out = image_bgr.copy()
    out[~keep] = 0
    return out


# =============================================================================
# (V2-2) detect_notch : wafer '아래쪽' notch + 중심 픽셀점 반환  [기능1]
# =============================================================================
def detect_notch(image_bgr: np.ndarray,
                 wafer_cx: int, wafer_cy: int, wafer_r: int,
                 notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                 sector_deg: float = DEFAULT_NOTCH_SECTOR_DEG,
                 n_angles: int = 14400,
                 min_depth: float = DEFAULT_NOTCH_MIN_DEPTH,
                 noise_margin: float = DEFAULT_NOTCH_NOISE_MARGIN,
                 min_span_deg: float = 0.06,
                 black_thr: int = 20,
                 open_ksize: int = DEFAULT_NOTCH_OPEN_KSIZE,
                 smooth_deg: float = DEFAULT_NOTCH_SMOOTH_DEG
                 ) -> Optional[Tuple[float, Tuple[int, int]]]:
    """wafer 아래쪽(ref±sector_deg) 의 파인 곳(notch) 검출.

    가장자리 오염이 notch 깊이만큼 심한 경우에도, 둘레 깊이를 각도폭 smooth_deg
    로 스무딩(넓은 notch 보존 / 좁은 bite 억제)한 뒤 후보를 찾으므로 robust.

    Returns
    -------
    (angle_err_deg, notch_center_px) 또는 None
      angle_err_deg : 기준 위치(notch_ref_deg, 90=6시) 대비 회전 오차(deg).
      notch_center_px : 파임 구간의 '깊이-가중 중심' 픽셀점 (cx, cy).
                        -> 이 점을 notch 기준점으로 사용.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    sil = _wafer_silhouette(gray, black_thr, open_ksize)

    angs = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    rs = np.linspace(wafer_r * 0.93, wafer_r * 1.015, 200)
    xs = (wafer_cx + rs[None, :] * np.cos(angs)[:, None]).astype(np.int32)
    ys = (wafer_cy + rs[None, :] * np.sin(angs)[:, None]).astype(np.int32)
    np.clip(xs, 0, W - 1, out=xs)
    np.clip(ys, 0, H - 1, out=ys)
    on = sil[ys, xs] > 0
    idx = np.where(on.any(axis=1),
                   on.shape[1] - 1 - np.argmax(on[:, ::-1], axis=1), 0)
    radii = rs[idx]
    depth = np.median(radii) - radii

    # ★ 둘레 깊이 각도 스무딩 (circular moving average):
    #   넓고 매끄러운 notch 는 보존되고, 좁은 가장자리 bite(거칠기)는 평균되어 눌린다.
    #   -> 가장자리 오염이 notch 깊이만큼 심해도 notch 가 가장 깊은 '지속' dip 으로 남음.
    win = max(3, int(round(smooth_deg / 360.0 * n_angles)))
    if win >= 3:
        ker = np.ones(win, dtype=np.float64) / win
        padded = np.concatenate([depth[-win:], depth, depth[:win]])
        depth = np.convolve(padded, ker, mode="same")[win:win + n_angles]

    # 아래쪽 sector(ref±sector_deg) 안에서만 notch 후보를 찾는다.
    degs = np.degrees(angs)
    in_sector = np.abs(((degs - notch_ref_deg + 180.0) % 360.0) - 180.0) <= sector_deg
    above = np.where((depth > min_depth) & in_sector)[0]
    if len(above) == 0:
        return None
    clusters: List[List[int]] = [[above[0]]]
    for v in above[1:]:
        if v - clusters[-1][-1] <= 2:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    clusters = [c for c in clusters
                if (c[-1] - c[0]) * 360.0 / n_angles >= min_span_deg]
    if not clusters:
        return None
    cand = max(clusters, key=lambda c: float(depth[c].sum()))

    # 적응형 임계(가산형) : 노이즈 floor 는 sector '밖' 둘레에서 추정.
    #   곱셈형(floor*k)은 floor 가 커지면 임계가 과도하게 올라가 진짜 notch 를
    #   놓치므로, floor + margin 으로 안정화한다.
    noise_floor = float(np.percentile(depth[~in_sector], 99.5)) if (~in_sector).any() else 0.0
    eff_thr = max(min_depth, noise_floor + noise_margin)
    d = depth[cand]
    if float(d.max()) < eff_thr:
        return None

    a = angs[cand]
    notch_deg = math.degrees(math.atan2(float((np.sin(a) * d).sum()),
                                        float((np.cos(a) * d).sum()))) % 360.0
    err = ((notch_deg - notch_ref_deg + 180.0) % 360.0) - 180.0
    # 중심 픽셀점 : 파임 경계점(cx+radii*cos, ...)을 깊이로 가중 평균
    bx = wafer_cx + radii[cand] * np.cos(a)
    by = wafer_cy + radii[cand] * np.sin(a)
    cxp = int(round(float((bx * d).sum() / d.sum())))
    cyp = int(round(float((by * d).sum() / d.sum())))
    return err, (cxp, cyp)


# =============================================================================
# (V2-3) measure_die_grid_angle : die 격자 기울기 독립 측정  [기능2]
# =============================================================================
def measure_die_grid_angle(image_bgr: np.ndarray,
                           wafer_cx: int, wafer_cy: int, wafer_r: int,
                           search_deg: float = DEFAULT_GRID_ANGLE_RANGE,
                           coarse_step: float = 0.2,
                           fine_step: float = 0.02,
                           roi_ratio: float = 0.45) -> float:
    """die 격자(세로 sawline)를 '진짜 수직'으로 만드는 회전각 = 격자 기울기(deg).

    notch 와 무관하게 die 무늬만으로 각도를 독립 측정 -> notch 각도 교차검증용.
    중앙 ROI 를 후보 각도로 역회전 후, 세로 edge(|Sobel_x|) 열-프로파일의 분산이
    최대가 되는 각도를 찾는다(격자가 수직이면 열 프로파일이 가장 뾰족 = 분산 최대).
    coarse -> fine 2단계 탐색.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    half = int(wafer_r * roi_ratio)
    y0, y1 = max(0, wafer_cy - half), min(gray.shape[0], wafer_cy + half)
    x0, x1 = max(0, wafer_cx - half), min(gray.shape[1], wafer_cx + half)
    roi = gray[y0:y1, x0:x1]
    ch, cw = roi.shape[0] / 2.0, roi.shape[1] / 2.0

    def sharpness(a: float) -> float:
        M = cv2.getRotationMatrix2D((cw, ch), a, 1.0)
        rot = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_NEAREST)
        sx = np.abs(cv2.Sobel(rot, cv2.CV_32F, 1, 0, ksize=3))
        return float(sx.mean(axis=0).var())     # 세로 edge 열-프로파일 분산

    coarse = np.arange(-search_deg, search_deg + 1e-9, coarse_step)
    best = max(coarse, key=sharpness)
    fine = np.arange(best - coarse_step, best + coarse_step + 1e-9, fine_step)
    return float(max(fine, key=sharpness))


# =============================================================================
# (V2-4) validate_quadrant_edges : 4분면 가장자리 맵 검증  [기능3]
# =============================================================================
def validate_quadrant_edges(dies: List[Dict[str, Any]],
                            wafer_cx: int, wafer_cy: int, wafer_r: int,
                            balance_tol: float = DEFAULT_QUAD_BALANCE_TOL) -> Dict[str, Any]:
    """center corner 에서 확장한 die 맵이 4분면(TL/TR/BL/BR) 가장자리까지
    균형 있게 채워졌는지 검증.

    각 분면별로 die 중심의 최대 도달거리/wafer_r = coverage 를 계산하고,
    4분면 coverage 편차가 balance_tol 이내면 balanced=True (정상적으로 채워짐).
    한 분면만 짧으면 그쪽 맵 생성이 잘못된 것.
    """
    quad: Dict[str, List[float]] = {"TR": [], "TL": [], "BL": [], "BR": []}
    edge: Dict[str, int] = {"TR": 0, "TL": 0, "BL": 0, "BR": 0}
    for d in dies:
        dx = d["center_px"][0] - wafer_cx
        dy = d["center_px"][1] - wafer_cy
        key = ("T" if dy < 0 else "B") + ("R" if dx > 0 else "L")
        quad[key].append(math.hypot(dx, dy))
        if d.get("is_edge"):
            edge[key] += 1

    per: Dict[str, Dict[str, Any]] = {}
    covs = []
    for k, arr in quad.items():
        if arr:
            cov = max(arr) / float(wafer_r)
            per[k] = {"n_dies": len(arr), "max_dist": round(max(arr), 1),
                      "coverage": round(cov, 4), "edge_dies": edge[k]}
            covs.append(cov)
        else:
            per[k] = {"n_dies": 0, "max_dist": 0.0, "coverage": 0.0, "edge_dies": 0}
            covs.append(0.0)
    spread = (max(covs) - min(covs)) if covs else 1.0
    balanced = bool(spread <= balance_tol and min(covs) > 0.8)
    return {"per_quadrant": per,
            "coverage_spread": round(spread, 4),
            "balanced": balanced,
            "min_coverage": round(min(covs), 4) if covs else 0.0}


# =============================================================================
# (1) wafer 이미지 -> Die Map (EDGE 포함)
# =============================================================================
def build_die_map(image: Union[str, Path, np.ndarray],
                  *,
                  grid_method: str = DEFAULT_GRID_METHOD,
                  pixel_per_unit: int = DEFAULT_PIXEL_PER_UNIT,
                  include_edge: bool = True,
                  edge_margin: float = DEFAULT_EDGE_MARGIN,
                  die_template_path: Optional[str] = None,
                  with_crops: bool = False,
                  border_mode: str = "pad",
                  offset_x: int = DEFAULT_OFFSET_X,
                  offset_y: int = DEFAULT_OFFSET_Y,
                  margin_x: int = DEFAULT_MARGIN_X,
                  margin_y: int = DEFAULT_MARGIN_Y,
                  notch_align: bool = DEFAULT_NOTCH_ALIGN,
                  notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                  angle_align_method: str = DEFAULT_ANGLE_ALIGN_METHOD,
                  edge_mode: str = DEFAULT_EDGE_MODE,
                  clean: bool = DEFAULT_CLEAN_WAFER,
                  notch_sector_deg: float = DEFAULT_NOTCH_SECTOR_DEG,
                  verify_angle: bool = True,
                  verify_tol_deg: float = DEFAULT_VERIFY_TOL_DEG) -> WaferDieMap:
    """wafer 이미지 한 장 -> 전체 die map (EDGE die 포함). [V5]

    처리 순서: clean(외부노이즈 제거) -> 회전 보정(기본 die_render) -> notch 중심점 ->
               die 격자 -> die map(+edge 플래그) -> angle 검증 -> 4분면 검증.

    Parameters
    ----------
    image            : wafer 이미지 경로(str/Path) 또는 BGR ndarray
    grid_method      : 격자 검출 방식 "corner"(기본) | "hybrid" | "std" | "color"
    pixel_per_unit   : 실측 좌표 환산 (px/unit)
    include_edge     : True 면 웨이퍼 원 안 die 전부 포함(가장자리 잘린 die 포함).
    edge_margin      : die 포함 기준 = (중심거리 <= r * edge_margin).
    die_template_path: (옵션) die_sample 이미지로 격자 보정.
    with_crops       : True 면 각 die entry 에 "image"(crop) 포함.
    border_mode      : with_crops 시 clip 방식 "pad" | "crop".
    offset_x/offset_y, margin_x/margin_y : crop 위치보정 / 영역확장 (px).
    notch_align      : True(기본) 면 angle_align_method 로 회전(angle) 보정.
    notch_ref_deg    : notch 의 정상 위치 (90 = 아래쪽/6시 방향).
    angle_align_method: "die_render"(V5 기본) | "notch" | "vertical_line" | "none".
    edge_mode        : ★ EDGE 판정 기준. "circle"(기본,부분 die) | "ring"(격자 최외곽) | "both".
                       각 die 에는 is_edge_partial / is_edge_ring 가 모두 저장되고,
                       is_edge 는 edge_mode 가 가리키는 값이 된다.
    clean            : ★[기능4] True 면 시작 시 wafer 원판 밖을 검정으로(외부노이즈 제거).
    notch_sector_deg : ★[기능1] notch 를 아래쪽 ref±이 각도에서만 탐색.
    verify_angle     : ★[기능2] True 면 die 격자 각도로 notch 각도를 교차검증.
    verify_tol_deg   : ★[기능2] 두 각도 차이가 이 값 이내면 angle_verified=True.

    Returns
    -------
    WaferDieMap (V2 필드: notch_center_px, angle_verified, die_grid_angle_resid,
                 quadrant_report. aligned_image 는 항상 채워짐[기능5]. edge_mode 저장.)
    """
    img = _load_bgr(image)

    # 0a) ★[기능4] wafer 원판 밖을 검정으로 정리 (외부 노이즈 제거) — 가장 먼저
    if clean:
        img = clean_wafer(img)

    # 0b) ★ 회전(angle) 보정
    rotation_deg = 0.0
    angle_confidence = 1.0
    angle_agree = True
    align_method = angle_align_method.lower().replace("-", "_").strip()
    if notch_align:
        if align_method in ("die_render", "die", "render", "grid_render"):
            img, rotation_deg, _ainfo = align_wafer_by_die_render(
                img, grid_method=grid_method, return_info=True)
            angle_confidence = float(_ainfo.get("confidence", 1.0))
            angle_agree = bool(_ainfo.get("agree", False))
        elif align_method == "notch":
            img, rotation_deg = align_wafer_by_notch(
                img, notch_ref_deg=notch_ref_deg)
        elif align_method in ("vertical_line", "longest_vertical_line", "line"):
            img, rotation_deg = align_wafer_by_vertical_line(img)
        elif align_method in ("none", "off", "false"):
            pass
        else:
            raise ValueError(
                "angle_align_method must be 'die_render', 'notch', "
                "'vertical_line', or 'none'.")

    H, W = img.shape[:2]

    # 1) 웨이퍼 영역
    wafer_cx, wafer_cy, wafer_r = detect_wafer(img)

    # 1b) ★[기능1] 보정된 이미지에서 아래쪽 notch + 중심점 측정 (잔여 오차 포함)
    notch_center_px = None
    notch_resid = 0.0
    nres = detect_notch(img, wafer_cx, wafer_cy, wafer_r,
                        notch_ref_deg=notch_ref_deg, sector_deg=notch_sector_deg)
    if nres is not None:
        notch_resid, notch_center_px = nres

    # 1c) ★[기능2] die 격자 각도로 notch 각도 교차검증
    die_grid_angle_resid = 0.0
    angle_verified = False
    if verify_angle:
        die_grid_angle_resid = measure_die_grid_angle(img, wafer_cx, wafer_cy, wafer_r)
        if align_method == "notch":
            # 두 독립 측정(notch 잔여 / 격자 잔여)이 모두 작고 서로 가까우면 검증 성공
            angle_verified = bool(abs(die_grid_angle_resid - notch_resid) <= verify_tol_deg
                                  and abs(die_grid_angle_resid) <= verify_tol_deg)
        else:
            angle_verified = bool(abs(die_grid_angle_resid) <= verify_tol_deg)

    # 2) die 격자 (원본 로직 그대로) -- 옵션 template 보정
    die_template_bgr = None
    if die_template_path is not None:
        die_template_bgr = cv2.imread(str(die_template_path), cv2.IMREAD_COLOR)
        if die_template_bgr is None:
            raise FileNotFoundError(str(die_template_path))
    pitch_x, pitch_y, x0, y0 = detect_grid(
        img, wafer_cx, wafer_cy, wafer_r,
        method=grid_method, die_template_bgr=die_template_bgr)

    die_w = int(round(pitch_x))
    die_h = int(round(pitch_y))

    # 3) 전체 격자 위치 순회 (원본 inspect_wafer 와 동일한 중심/실측 공식)
    max_ix = int(np.ceil(wafer_r / pitch_x)) + 2
    max_iy = int(np.ceil(wafer_r / pitch_y)) + 2
    margin = edge_margin if include_edge else 0.98
    r_lim_sq = (wafer_r * margin) ** 2

    dies: List[Dict[str, Any]] = []
    dies_by_index: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for iy in range(-max_iy, max_iy + 1):
        for ix in range(-max_ix, max_ix + 1):
            cx_d = int(round(x0 + ix * pitch_x + pitch_x / 2))
            cy_d = int(round(y0 - iy * pitch_y - pitch_y / 2))

            dx = cx_d - wafer_cx
            dy = cy_d - wafer_cy
            if dx * dx + dy * dy > r_lim_sq:     # 웨이퍼 원 밖 격자 위치 -> die 없음
                continue

            x_a = cx_d - die_w // 2
            y_a = cy_d - die_h // 2
            x_b = x_a + die_w
            y_b = y_a + die_h

            # offset/margin 적용된 실제 crop 영역 (margin=offset=0 이면 rect_px 와 동일)
            crop_rect = _crop_rect(cx_d, cy_d, die_w, die_h,
                                   offset_x, offset_y, margin_x, margin_y)

            rx = (cx_d - wafer_cx) / pixel_per_unit
            ry = (wafer_cy - cy_d) / pixel_per_unit

            entry: Dict[str, Any] = {
                "index":       (ix, iy),
                "center_px":   (cx_d, cy_d),
                "rect_px":     (x_a, y_a, x_b, y_b),
                "crop_rect_px": crop_rect,
                "real_coord":  (rx, ry),
                # ★ 두 가지 edge 플래그를 모두 저장 (is_edge 는 아래서 edge_mode 로 결정)
                "is_edge_partial": _rect_crosses_circle(x_a, y_a, x_b, y_b,
                                                        wafer_cx, wafer_cy, wafer_r),
                "is_edge_ring": False,   # 8방향 이웃 확정 후 채움
                "is_edge":      False,   # edge_mode 적용 후 채움
            }

            if with_crops:
                crop = crop_die(img, cx_d, cy_d, die_w, die_h,
                                offset_x=offset_x, offset_y=offset_y,
                                margin_x=margin_x, margin_y=margin_y,
                                border_mode=border_mode)
                if crop is None:
                    continue
                entry["image"] = crop

            dies.append(entry)
            dies_by_index[(ix, iy)] = entry

    # 3b) ★ EDGE 플래그 확정 : is_edge_ring(8방향 이웃 결손) 계산 후 edge_mode 로 is_edge 선택
    emode = _normalize_edge_mode(edge_mode)
    present = set(dies_by_index.keys())
    for d in dies:
        ix, iy = d["index"]
        ring = any((ix + dxn, iy + dyn) not in present
                   for dxn in (-1, 0, 1) for dyn in (-1, 0, 1)
                   if not (dxn == 0 and dyn == 0))
        d["is_edge_ring"] = bool(ring)
        d["is_edge"] = _resolve_edge_flag(d["is_edge_partial"], d["is_edge_ring"], emode)

    # 4) ★[기능3] 4분면 가장자리 맵 검증
    quadrant_report = validate_quadrant_edges(dies, wafer_cx, wafer_cy, wafer_r)

    return WaferDieMap(
        wafer_cx=wafer_cx, wafer_cy=wafer_cy, wafer_r=wafer_r,
        pitch_x=pitch_x, pitch_y=pitch_y, x0=x0, y0=y0,
        die_w=die_w, die_h=die_h, pixel_per_unit=pixel_per_unit,
        dies=dies, dies_by_index=dies_by_index, image_shape=(H, W),
        rotation_deg=rotation_deg,
        aligned_image=img,                       # ★[기능5] 항상 반환 (clean+align 결과)
        notch_center_px=notch_center_px,         # ★[기능1]
        die_grid_angle_resid=die_grid_angle_resid,  # ★[기능2]
        angle_verified=angle_verified,           # ★[기능2]
        angle_confidence=angle_confidence,       # ★[V5 고도화] 각도 신뢰도(0~1)
        angle_agree=angle_agree,                 # ★[V5 고도화] projection↔FFT 합의 여부
        edge_mode=emode,                         # ★[V5] is_edge 기준
        quadrant_report=quadrant_report,         # ★[기능3]
    )


# =============================================================================
# (2) 픽셀 좌표 / BBox -> die index + die rect + 실측 좌표
# =============================================================================
def locate_die(die_map: WaferDieMap,
               point: Optional[Tuple[float, float]] = None,
               bbox: Optional[Tuple[float, float, float, float]] = None,
               *,
               offset_x: int = DEFAULT_OFFSET_X, offset_y: int = DEFAULT_OFFSET_Y,
               margin_x: int = DEFAULT_MARGIN_X, margin_y: int = DEFAULT_MARGIN_Y
               ) -> Dict[str, Any]:
    """픽셀 좌표 또는 BBox(YOLO 등)의 위치에 해당하는 die 정보 반환.

    Parameters
    ----------
    die_map : build_die_map() 결과
    point   : (x, y) 픽셀 좌표  (point 또는 bbox 중 하나만)
    bbox    : (x1, y1, x2, y2) 픽셀 BBox. 내부적으로 중심점으로 변환해 사용.
    offset_x/offset_y : crop 중심 위치 보정 (px). crop_rect_px 에 반영.
    margin_x/margin_y : 각 변으로 더 포함할 영역 (px, die 사이 street 포함). crop_rect_px 에 반영.

    Returns
    -------
    dict
        {
          "input_type"   : "point" | "bbox",
          "query_px"     : (qx, qy),           # 사용한 픽셀 좌표 (bbox면 중심)
          "die_index"    : (ix, iy),
          "die_center_px": (cx, cy),
          "die_rect_px"  : (x1, y1, x2, y2),   # 해당 die 의 사각 영역(순수 die)
          "crop_rect_px" : (x1, y1, x2, y2),   # offset/margin 적용된 crop 영역
          "real_coord"   : (rx, ry),           # 실측 좌표 (query 기준, 위쪽 +y)
          "real_distance": float,              # 웨이퍼 중심으로부터 실측 거리(스칼라)
          "die_real_coord": (drx, dry),        # 참고: die 중심 기준 실측 좌표
          "wafer_center_px": (wcx, wcy),       # 웨이퍼 중심점 (검출값)
          "corner_px"    : (x0, y0),           # 격자 코너(원점) 점 (검출값)
          "is_edge"      : bool,               # edge_mode 가 가리키는 edge 여부
          "is_edge_partial": bool,             # 정의① die 가 wafer 원 밖으로 일부 나감
          "is_edge_ring" : bool,               # 정의② 격자 최외곽(8방향 이웃 결손)
          "edge_mode"    : str,                # 이 맵의 is_edge 기준(circle|ring|both)
          "in_wafer"     : bool,               # query 점이 웨이퍼 원 안인지
        }

    Notes
    -----
    - die_index 는 격자 공식으로 해석적으로 계산하므로, die_map 에 미포함된
      위치(웨이퍼 밖 등)도 인덱스/실측값을 반환합니다. 포함 여부는 in_wafer 로 판단.
    - crop_rect_px 로 실제 이미지에서 crop 하려면 crop_die() 또는 슬라이싱 사용.
    """
    if (point is None) == (bbox is None):
        raise ValueError("point 또는 bbox 중 정확히 하나를 지정하세요.")

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        qx = (float(x1) + float(x2)) / 2.0       # BBox 중심
        qy = (float(y1) + float(y2)) / 2.0
        input_type = "bbox"
    else:
        qx, qy = float(point[0]), float(point[1])
        input_type = "point"

    px = die_map.pitch_x
    py = die_map.pitch_y
    x0 = die_map.x0
    y0 = die_map.y0

    # --- 좌표 -> die index (build_die_map 의 중심 공식의 역변환, 규칙 동일) ---
    ix = int(math.floor((qx - x0) / px))
    iy = int(math.floor((y0 - qy) / py))         # iy +는 위쪽(y 감소)

    # --- die 중심 / rect (build_die_map 과 동일하게 계산) -------------------
    cx_d = int(round(x0 + ix * px + px / 2))
    cy_d = int(round(y0 - iy * py - py / 2))
    x_a = cx_d - die_map.die_w // 2
    y_a = cy_d - die_map.die_h // 2
    x_b = x_a + die_map.die_w
    y_b = y_a + die_map.die_h

    # offset/margin 적용된 실제 crop 영역 (margin=offset=0 이면 die_rect_px 와 동일)
    crop_rect = _crop_rect(cx_d, cy_d, die_map.die_w, die_map.die_h,
                           offset_x, offset_y, margin_x, margin_y)

    # --- 실측 좌표/거리 -----------------------------------------------------
    ppu = die_map.pixel_per_unit
    rx = (qx - die_map.wafer_cx) / ppu           # query(=bbox 중심) 기준 실측 좌표
    ry = (die_map.wafer_cy - qy) / ppu
    real_distance = math.hypot(rx, ry)
    drx = (cx_d - die_map.wafer_cx) / ppu        # die 중심 기준 실측 좌표(참고)
    dry = (die_map.wafer_cy - cy_d) / ppu

    # --- edge 여부(두 정의 모두) / wafer 내부 여부 --------------------------
    emode = _normalize_edge_mode(getattr(die_map, "edge_mode", DEFAULT_EDGE_MODE))
    entry = die_map.get_die(ix, iy)
    if entry is not None:
        # die map 에 있으면 build 때 확정한 플래그 그대로 사용
        is_edge_partial = bool(entry.get("is_edge_partial",
                                         entry.get("is_edge", False)))
        is_edge_ring = bool(entry.get("is_edge_ring", False))
    else:
        # 맵에 없는(웨이퍼 밖 등) 위치 -> 즉석 계산
        is_edge_partial = _rect_crosses_circle(
            x_a, y_a, x_b, y_b,
            die_map.wafer_cx, die_map.wafer_cy, die_map.wafer_r)
        is_edge_ring = any(
            (ix + dxn, iy + dyn) not in die_map.dies_by_index
            for dxn in (-1, 0, 1) for dyn in (-1, 0, 1)
            if not (dxn == 0 and dyn == 0))
    is_edge = _resolve_edge_flag(is_edge_partial, is_edge_ring, emode)
    in_wafer = ((qx - die_map.wafer_cx) ** 2 + (qy - die_map.wafer_cy) ** 2
                <= die_map.wafer_r ** 2)

    return {
        "input_type":    input_type,
        "query_px":      (qx, qy),
        "die_index":     (ix, iy),
        "die_center_px": (cx_d, cy_d),
        "die_rect_px":   (x_a, y_a, x_b, y_b),
        "crop_rect_px":  crop_rect,
        "real_coord":    (rx, ry),
        "real_distance": real_distance,
        "die_real_coord": (drx, dry),
        "wafer_center_px": (die_map.wafer_cx, die_map.wafer_cy),  # 웨이퍼 중심점
        "corner_px":     (die_map.x0, die_map.y0),                # 격자 코너(원점)
        "is_edge":       is_edge,             # edge_mode 가 가리키는 값
        "is_edge_partial": is_edge_partial,   # 정의① 부분 die(원 밖으로 나감)
        "is_edge_ring":  is_edge_ring,        # 정의② 격자 최외곽(이웃 결손)
        "edge_mode":     emode,               # 이 맵의 is_edge 기준
        "in_wafer":      bool(in_wafer),
    }


# =============================================================================
# 사용 예시  (복붙해서 쓰는 파일이라 자동 실행 안 되도록 주석 처리해 둠.
#            그대로 본인 코드에서 호출하면 됩니다.)
# =============================================================================
#
#   # 1) wafer 이미지 -> die map (EDGE 포함)
#   dm = build_die_map("wafer.jpg", grid_method="corner",
#                      pixel_per_unit=32, include_edge=True)
#   print(dm.num_dies, dm.pitch_x, dm.pitch_y, (dm.x0, dm.y0))
#
#   # 2-a) 픽셀 좌표로 조회
#   r1 = locate_die(dm, point=(5499, 4700))
#   print(r1["die_index"], r1["die_rect_px"], r1["real_coord"])
#
#   # 2-b) BBox(YOLO box, x1,y1,x2,y2) 로 조회 (중심 기준)
#   r2 = locate_die(dm, bbox=(4880, 5080, 4980, 5180))
#   print(r2["die_index"], r2["die_rect_px"], r2["real_coord"],
#         r2["wafer_center_px"], r2["corner_px"])
#
#   # 3) offset(위치 보정) + margin(die 사이 street 포함) 으로 더 넓게 crop
#   #    - offset_x/y : 미세 정렬 오차 보정 (px)
#   #    - margin_x/y : 각 변으로 더 포함할 영역 (px)  ← die 간격만큼 더 따고 싶을 때
#   r3 = locate_die(dm, bbox=(4880, 5080, 4980, 5180),
#                   offset_x=0, offset_y=0, margin_x=8, margin_y=8)
#   x1, y1, x2, y2 = r3["crop_rect_px"]          # 확장된 crop 영역
#   patch = crop_die(img_bgr, *r3["die_center_px"], dm.die_w, dm.die_h,
#                    margin_x=8, margin_y=8)      # 또는 직접 crop (clip_die 재사용)
#
#   # build_die_map 단계에서 일괄로 margin/offset crop 을 받으려면:
#   dm2 = build_die_map("wafer.jpg", with_crops=True, margin_x=8, margin_y=8)
#   #   -> 각 entry["image"] 가 margin 포함 crop, entry["crop_rect_px"] 가 그 영역
#
#   # 4) ★ 회전(angle) 보정 — 기본 ON (V5 기본 방식 = "die_render"). 연산 전에 자동 수행
#   dm = build_die_map("wafer_rotated.jpg")        # angle_align_method="die_render" 기본
#   print(dm.rotation_deg)                          # 적용된 보정 각도 (0 = 보정 없음)
#   img_use = dm.aligned_image                      # ★ 항상 채워짐 = 좌표 기준 이미지!
#   #   crop / YOLO / 시각화는 모두 img_use(=dm.aligned_image) 에서 해야 좌표가 맞음
#
#   # 4-a) 회전 보정 방식 선택 (die_render 기본 / notch / vertical_line / none)
#   dm = build_die_map("wafer.jpg", angle_align_method="notch")
#   dm = build_die_map("wafer.jpg", angle_align_method="vertical_line")
#
#   # 4-b) 각도만 따로 쓰고 싶을 때 (YOLO 전에 이미지부터 정렬)
#   aligned, deg = align_wafer_by_die_render(img_bgr)   # 또는 align_wafer_by_notch(...)
#   dm = build_die_map(aligned, angle_align_method="none")  # 이미 정렬 -> 보정 OFF
#
#   # 5) ★ EDGE die 구분 — is_edge 기준 선택 (둘 다 entry 에 저장됨)
#   dm = build_die_map("wafer.jpg", edge_mode="circle")  # 부분 die(원 밖) / 기본
#   dm = build_die_map("wafer.jpg", edge_mode="ring")    # 격자 최외곽 줄
#   dm = build_die_map("wafer.jpg", edge_mode="both")    # 둘 중 하나라도면 edge
#   r = locate_die(dm, point=(5499, 4700))
#   print(r["is_edge"], r["is_edge_partial"], r["is_edge_ring"], r["edge_mode"])

# =============================================================================
# 반환값 정리  (각 함수가 돌려주는 값 레퍼런스)
# =============================================================================
#
# [1] build_die_map(...) -> WaferDieMap  (필드)
#     wafer_cx        # int  : 웨이퍼 중심 X 픽셀 (검출값)
#     wafer_cy        # int  : 웨이퍼 중심 Y 픽셀 (검출값)
#     wafer_r         # int  : 웨이퍼 반지름 픽셀
#     pitch_x         # float: die 가로 간격(px, sub-pixel) ← 한 die 의 폭
#     pitch_y         # float: die 세로 간격(px, sub-pixel) ← 한 die 의 높이
#     x0              # int  : 격자 코너(원점) X 픽셀 ← die(0,0) 기준점
#     y0              # int  : 격자 코너(원점) Y 픽셀
#     die_w           # int  : crop 용 die 폭  = round(pitch_x)
#     die_h           # int  : crop 용 die 높이 = round(pitch_y)
#     pixel_per_unit  # int  : 실측 좌표 환산 단위 (px/unit). 32 → 32px = 1unit
#     image_shape     # (H,W): 원본 이미지 크기
#     num_dies        # int  : die 총 개수 (property)
#     dies            # list : die entry(dict) 리스트 (아래 [1-a] 참고)
#     dies_by_index   # dict : {(ix,iy): entry} 빠른 조회용
#     get_die(ix,iy)  # method: entry 또는 None 반환
#     rotation_deg    # float: ★ 회전 보정으로 적용된 각도 (0 = 보정 없음). 기본 die_render
#     aligned_image   # ndarray: ★ 항상 채워짐 = clean+align 후 실제 사용 이미지(CUBIC 회전).
#                     #   모든 좌표는 이 이미지 기준 -> crop/YOLO 에 반드시 이걸 사용
#     angle_confidence# float: ★[고도화] 각도 신뢰도 0~1 (projection·FFT 합의 기반)
#     angle_agree     # bool : ★[고도화] projection 과 FFT 가 합의했는지 (False면 의심 -> 검토)
#     edge_mode       # str  : ★ is_edge 가 가리키는 기준 ("circle"|"ring"|"both")
#
# [1-a] dies 안의 die entry(dict) 하나의 형식
#     "index"        # (ix,iy)        die 격자 인덱스 (오른쪽 +ix, 위쪽 +iy, 코너 위-오른쪽=(0,0))
#     "center_px"    # (cx,cy)        die 중심 픽셀 좌표
#     "rect_px"      # (x1,y1,x2,y2)  순수 die 사각 영역(좌상~우하) 픽셀 좌표
#     "crop_rect_px" # (x1,y1,x2,y2)  offset/margin 적용된 crop 영역 (margin=offset=0이면 rect_px와 동일)
#     "real_coord"   # (rx,ry)        die 중심의 실측 좌표
#                    #                = ((cx-wafer_cx)/ppu, (wafer_cy-cy)/ppu), 위쪽 +y
#     "is_edge_partial" # bool        ★정의① die 사각형이 wafer 원 밖으로 일부라도 나감
#     "is_edge_ring"    # bool        ★정의② 격자에서 8방향 이웃이 다 차 있지 않은 최외곽 줄
#     "is_edge"      # bool           ★ edge_mode 가 가리키는 값(circle→partial/ring→ring/both→OR)
#     "image"        # np.ndarray     crop_rect_px 영역 crop (with_crops=True 일 때만 존재)
#
# [2] locate_die(...) -> dict  (키)
#     "input_type"     # str          "point" | "bbox" (어떤 입력으로 조회했는지)
#     "query_px"       # (qx,qy)      실제 조회에 쓴 픽셀 좌표 (bbox면 그 중심)
#     "die_index"      # (ix,iy)      그 좌표가 속한 die 의 격자 인덱스
#     "die_center_px"  # (cx,cy)      그 die 의 중심 픽셀 좌표
#     "die_rect_px"    # (x1,y1,x2,y2) 그 die 의 순수 사각 영역(좌상~우하) 픽셀 좌표
#     "crop_rect_px"   # (x1,y1,x2,y2) offset/margin 적용된 crop 영역 (die 사이 street 포함용)
#     "real_coord"     # (rx,ry)      조회 좌표(=bbox면 중심)의 실측 좌표
#                      #              = ((qx-wafer_cx)/ppu, (wafer_cy-qy)/ppu), 위쪽 +y
#     "real_distance"  # float        웨이퍼 중심으로부터의 실측 거리(스칼라) = hypot(rx,ry)
#     "die_real_coord" # (drx,dry)    참고용 — die '중심' 기준 실측 좌표
#     "wafer_center_px"# (wcx,wcy)    웨이퍼 중심점 (검출값) = (dm.wafer_cx, dm.wafer_cy)
#     "corner_px"      # (x0,y0)      격자 코너(원점) 점 (검출값) = (dm.x0, dm.y0)
#     "is_edge"        # bool         ★ edge_mode 가 가리키는 edge 여부
#     "is_edge_partial"# bool         ★정의① die 가 wafer 원 밖으로 일부 나감(부분 die)
#     "is_edge_ring"   # bool         ★정의② 격자 최외곽(8방향 이웃 결손)
#     "edge_mode"      # str          ★ 이 맵의 is_edge 기준 ("circle"|"ring"|"both")
#     "in_wafer"       # bool         조회 좌표가 웨이퍼 원 '안'인지 (밖이어도 index 는 계산됨)

# =============================================================================
# V6 COLOUR-ROBUST EXTENSION (standalone: no local-module import required)
# =============================================================================
# This section is generated from wafer_die_map_color_robust.py.
# build_die_map() remains the original V5 entry point.
# build_die_map_robust() is the colour-invariant entry point.

__all__ = list(__all__) + [
    "ColorRobustConfig",
    "detect_grid_color_robust",
    "build_die_map_robust",
    "make_grid_diagnostic",
]

"""Color-invariant wafer / die-grid detection, kept separate from V5.

This module deliberately does not modify ``wafer_die_map_py``.  Its public
``build_die_map_robust`` function returns the same ``WaferDieMap`` object, so
existing calls to ``locate_die`` and ``crop_die`` continue to work.

The detector does *not* assume a particular street or die colour.  It combines
normalised gradients from L*, a*, b* and grayscale channels, then finds the
two repeating orthogonal signals.  Therefore a bright, dark, coloured, or
mixed/noisy street can be detected as long as it creates a repeatable boundary.
"""

# Standalone extension public API: ColorRobustConfig, detect_grid_color_robust,
# build_die_map_robust, make_grid_diagnostic.


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
    global_pitch_refine: bool = True
    global_pitch_refine_window_ratio: float = 0.30
    global_pitch_refine_min_lines: int = 8
    global_pitch_refine_max_delta_ratio: float = 0.035
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
    peaks = _find_periodic_peaks(profile, pitch, min_score_ratio=0.20)
    if not peaks:
        return float(_best_phase(profile, max(2, int(round(pitch)))))
    return float(_robust_phase(peaks, pitch, profile))


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


def _fit_global_lattice_axis(response: np.ndarray, initial_origin: float,
                             initial_pitch: float, min_lines: int,
                             window_ratio: float, max_delta_ratio: float
                             ) -> Tuple[float, float, Dict[str, float]]:
    """Fit one lattice to all visible street centres, including fractional pitch.

    Period estimation from autocorrelation intentionally starts at pixel
    precision.  Drawing a full-wafer grid with that rounded period can create
    a small but visible accumulated error at the rim.  Here, each predicted
    street is locally re-centred in the invariant street response, then a
    robust least-squares line fits ``position = origin + index * pitch``.
    This corrects systematic drift while retaining the original candidate if
    the observations are too sparse or disagree with it.
    """
    response = response.astype(np.float64)
    pitch = float(initial_pitch)
    if pitch <= 2.0 or response.size < 3:
        return float(initial_origin), pitch, {"used": 0.0, "lines": 0.0, "rms_px": 0.0}

    half_window = max(3, int(round(pitch * float(window_ratio))))
    line_ids = np.arange(
        int(np.ceil((-float(initial_origin) - half_window) / pitch)),
        int(np.floor((response.size - 1 - float(initial_origin) + half_window) / pitch)) + 1,
    )
    observed: List[float] = []
    observed_ids: List[float] = []
    for line_id in line_ids:
        predicted = float(initial_origin) + float(line_id) * pitch
        lo = max(0, int(np.floor(predicted - half_window)))
        hi = min(response.size, int(np.ceil(predicted + half_window + 1)))
        if hi - lo < 3:
            continue
        segment = response[lo:hi]
        peak = lo + int(np.argmax(segment))
        # The box-integrated profile is a broad street-centre hill.  A small
        # weighted centroid makes the fit subpixel without being drawn toward
        # die-internal circuit texture elsewhere in the search interval.
        radius = max(2, min(7, half_window // 3))
        c_lo, c_hi = max(lo, peak - radius), min(hi, peak + radius + 1)
        coords = np.arange(c_lo, c_hi, dtype=np.float64)
        weights = response[c_lo:c_hi] - float(np.min(segment))
        if float(weights.sum()) <= 1e-9:
            position = float(peak)
        else:
            position = float(np.dot(coords, weights) / weights.sum())
        observed.append(position)
        observed_ids.append(float(line_id))

    ids = np.asarray(observed_ids, dtype=np.float64)
    positions = np.asarray(observed, dtype=np.float64)
    if positions.size < max(3, int(min_lines)):
        return float(initial_origin), pitch, {"used": 0.0, "lines": float(positions.size), "rms_px": 0.0}

    keep = np.ones(positions.size, dtype=bool)
    origin, fitted_pitch = float(initial_origin), pitch
    for _ in range(4):
        if int(keep.sum()) < max(3, int(min_lines)):
            break
        fitted_pitch, origin = np.polyfit(ids[keep], positions[keep], 1)
        residual = positions - (origin + ids * fitted_pitch)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        tolerance = max(1.25, 3.5 * 1.4826 * mad)
        next_keep = np.abs(residual - median) <= tolerance
        if np.array_equal(next_keep, keep):
            keep = next_keep
            break
        keep = next_keep

    if int(keep.sum()) < max(3, int(min_lines)) or abs(fitted_pitch - pitch) > pitch * float(max_delta_ratio):
        return float(initial_origin), pitch, {"used": 0.0, "lines": float(keep.sum()), "rms_px": 0.0}
    residual = positions[keep] - (origin + ids[keep] * fitted_pitch)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return float(origin), float(fitted_pitch), {"used": 1.0, "lines": float(keep.sum()), "rms_px": rms}


def _gradient_grid_candidate(image_bgr: np.ndarray, wafer_cx: int, wafer_cy: int,
                             wafer_r: int, config: ColorRobustConfig
                             ) -> Tuple[float, float, int, int, float, Dict[str, np.ndarray]]:
    gx, gy = _colour_invariant_edges(image_bgr, config.blur_sigma)
    half = max(80, int(wafer_r * config.roi_ratio))
    x1, x2 = max(0, wafer_cx - half), min(image_bgr.shape[1], wafer_cx + half)
    y1, y2 = max(0, wafer_cy - half), min(image_bgr.shape[0], wafer_cy + half)
    col = _smooth(gx[y1:y2, x1:x2].mean(axis=0), config.projection_smooth)
    row = _smooth(gy[y1:y2, x1:x2].mean(axis=1), config.projection_smooth)
    px = float(_autocorr_period(col, min_lag=config.min_pitch, max_lag=config.max_pitch))
    py = float(_autocorr_period(row, min_lag=config.min_pitch, max_lag=config.max_pitch))
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
    px, py, x0, y0 = detect_grid(
        image_bgr, wafer_cx, wafer_cy, wafer_r, method="std",
        roi_ratio=config.roi_ratio, min_pitch=config.min_pitch, max_pitch=config.max_pitch)
    x0, y0 = _snap_origin_to_mode(x0, y0, px, py, wafer_cx, wafer_cy, config.origin_mode)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    half = max(80, int(wafer_r * config.roi_ratio))
    x1, x2 = max(0, wafer_cx - half), min(gray.shape[1], wafer_cx + half)
    y1, y2 = max(0, wafer_cy - half), min(gray.shape[0], wafer_cy + half)
    col, row = _grid_profiles_std(gray[y1:y2, x1:x2])
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
    global_fit: Dict[str, Dict[str, float]] = {
        "x": {"used": 0.0, "lines": 0.0, "rms_px": 0.0},
        "y": {"used": 0.0, "lines": 0.0, "rms_px": 0.0},
    }
    if config.global_pitch_refine and phase_diag is not None:
        # Only use a central wafer band: rim/edge energy is not a street and
        # otherwise biases a projection near the ends of the lattice.
        half = max(80, int(wafer_r * config.roi_ratio))
        y1, y2 = max(0, wafer_cy - half), min(phase_diag["gx_centre"].shape[0], wafer_cy + half)
        x1, x2 = max(0, wafer_cx - half), min(phase_diag["gy_centre"].shape[1], wafer_cx + half)
        x0, px, global_fit_x = _fit_global_lattice_axis(
            phase_diag["gx_centre"][y1:y2].mean(axis=0), x0, px,
            config.global_pitch_refine_min_lines,
            config.global_pitch_refine_window_ratio,
            config.global_pitch_refine_max_delta_ratio,
        )
        y0, py, global_fit_y = _fit_global_lattice_axis(
            phase_diag["gy_centre"][:, x1:x2].mean(axis=1), y0, py,
            config.global_pitch_refine_min_lines,
            config.global_pitch_refine_window_ratio,
            config.global_pitch_refine_max_delta_ratio,
        )
        global_fit = {"x": global_fit_x, "y": global_fit_y}
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
        "global_pitch_refinement": global_fit,
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
        return detect_wafer(image_bgr)
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
                         pixel_per_unit: int = DEFAULT_PIXEL_PER_UNIT,
                         include_edge: bool = True,
                         edge_margin: float = DEFAULT_EDGE_MARGIN,
                         edge_mode: str = DEFAULT_EDGE_MODE,
                         with_crops: bool = False,
                         border_mode: str = "pad",
                         offset_x: int = 0, offset_y: int = 0,
                         margin_x: int = 0, margin_y: int = 0,
                         return_info: bool = False,
                         ) -> Union[WaferDieMap, Tuple[WaferDieMap, Dict[str, Any]]]:
    """Build a V5-compatible die map with colour-invariant grid detection.

    The supplied image is never recoloured.  ``aligned_image`` contains only
    optional geometric rotation/wafer cleanup, so crops retain their original
    die colours.  When ``return_info=True`` diagnostics include the selected
    detector and score for logging/acceptance thresholds.
    """
    img = _load_bgr(image).copy()
    if config.clean:
        try:
            img = clean_wafer(img)
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
            img, rotation_deg, angle_info = align_wafer_by_die_render(
                img, grid_method="std", return_info=True)
            angle_confidence = float(angle_info.get("confidence", 0.0))
            angle_source = "projection_fft"
            wafer_cx, wafer_cy, wafer_r = _detect_wafer_robust(img)
        except Exception:
            rotation_deg, angle_confidence = _estimate_grid_angle(
                img, wafer_cx, wafer_cy, wafer_r, config)
            angle_source = "hough_fallback"
            if abs(rotation_deg) >= 0.05 and angle_confidence >= 0.15:
                img = _rotate_wafer_keep_size(img, wafer_cx, wafer_cy, rotation_deg)
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
                "crop_rect_px": _crop_rect(cx, cy, die_w, die_h, offset_x, offset_y, margin_x, margin_y),
                "real_coord": ((cx - wafer_cx) / pixel_per_unit, (wafer_cy - cy) / pixel_per_unit),
                "is_edge_partial": _rect_crosses_circle(*rect, wafer_cx, wafer_cy, wafer_r),
                "is_edge_ring": False, "is_edge": False,
            }
            if with_crops:
                crop = crop_die(img, cx, cy, die_w, die_h, offset_x=offset_x, offset_y=offset_y,
                                   margin_x=margin_x, margin_y=margin_y, border_mode=border_mode)
                if crop is None:
                    continue
                entry["image"] = crop
            dies.append(entry)
            by_index[(ix, iy)] = entry
    emode = _normalize_edge_mode(edge_mode)
    present = set(by_index)
    for entry in dies:
        ix, iy = entry["index"]
        entry["is_edge_ring"] = any((ix + dx, iy + dy) not in present
                                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                    if dx or dy)
        entry["is_edge"] = _resolve_edge_flag(entry["is_edge_partial"], entry["is_edge_ring"], emode)
    report = validate_quadrant_edges(dies, wafer_cx, wafer_cy, wafer_r)
    result = WaferDieMap(
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


def make_grid_diagnostic(image: Union[str, Path, np.ndarray], die_map: WaferDieMap,
                         thickness: int = 2) -> np.ndarray:
    """Return an overlay on the aligned original image for visual acceptance checks."""
    base = die_map.aligned_image if die_map.aligned_image is not None else _load_bgr(image)
    overlay = base.copy()
    for die in die_map.dies:
        x1, y1, x2, y2 = die["rect_px"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    cv2.circle(overlay, (die_map.wafer_cx, die_map.wafer_cy), 5, (0, 0, 255), -1)
    return overlay
