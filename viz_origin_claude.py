# -*- coding: utf-8 -*-
"""
격자 원점 (x0, y0) 검출 파이프라인 단계별 시각화
==================================================
작성: Claude (Anthropic)

``wafer_color_v6_claude.py`` 가 오버레이의 **자홍 X** 표시(= grid 원점,
``locate_die_v6()["corner_px"]``)를 어떻게 찾는지, 각 단계의 중간 산출물을
PNG 로 떨궈서 "어디를 고쳐야 하는지" 눈으로 짚을 수 있게 한다.

이 스크립트는 v6 의 내부 함수를 **그대로 호출**한다. 로직을 다시 구현하지
않으므로 여기 그려지는 그림은 실제 검출 경로와 100% 같다.

파이프라인 (자세한 행번호는 README_claude.md 참조)
--------------------------------------------------
    detect_wafer_adaptive()      wafer_cx, wafer_cy, wafer_r
      -> _grid_roi()             중앙 정사각 ROI
      -> _downscale()            profile_max_dim 로 축소
      -> _feature_bank()         7채널 (L,a,b,chroma,sat,maxmin,stdL)
      -> 축(x,y) 별로:
           _analyze_channel()      채널별 pitch / score / street 위치
             _spectral_pitch()       FFT 로 pitch
             _fourier_phase()        거친 위상
             _fold()                 1주기 템플릿
             _street_from_template() street 정밀 위치 + 극성
           _consensus_pitch()      채널간 pitch 합의
           _consensus_phase()      채널간 street 위상 합의 (원형평균)
           k = round((중심 - 위상) / pitch)      <-- 원점 확정
           origin = 위상 + k * pitch

사용법
------
    python viz_origin_claude.py Img/real_casio_top_p092.png --out viz_out
    python viz_origin_claude.py Img/*.png --out viz_out --axis x
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import wafer_color_v6_claude as v6

# --- 색 (BGR) ---------------------------------------------------------------
BG        = (24, 24, 24)
FG        = (235, 235, 235)
GRID      = (60, 60, 60)
C_ACCENT  = (0, 165, 255)     # 주황 - 격자/피크
C_ORIGIN  = (255, 0, 255)     # 자홍 - 원점
C_CENTER  = (0, 255, 255)     # 노랑 - 웨이퍼 중심
C_BAND    = (70, 100, 70)     # 탐색 밴드
C_USED    = (120, 255, 120)   # 채택 채널
C_DROP    = (110, 110, 110)   # 탈락 채널

CH_COLORS = {
    "L":      (255, 255, 255),
    "a":      (120, 120, 255),
    "b":      (255, 200, 120),
    "chroma": (120, 255, 255),
    "sat":    (255, 120, 255),
    "maxmin": (120, 255, 160),
    "stdL":   (180, 180, 255),
}


# =============================================================================
# 그리기 도우미 (matplotlib 없이 OpenCV 만으로)
# =============================================================================
def _canvas(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), BG, np.uint8)


def _text(img, s, org, color=FG, scale=0.42, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _frame(img, box, color=GRID):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, 1)


def _curve(img, vals, box, color, thick=1, lo=None, hi=None):
    """vals(1D) 를 box=(x1,y1,x2,y2) 안에 꽉 채워 그린다."""
    x1, y1, x2, y2 = box
    v = np.asarray(vals, np.float64)
    n = v.size
    if n < 2:
        return
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    rng = max(hi - lo, 1e-12)
    xs = x1 + (np.arange(n) * (x2 - x1 - 1) / (n - 1)).round().astype(np.int32)
    ys = (y2 - 1 - ((v - lo) / rng * (y2 - y1 - 1))).round().astype(np.int32)
    ys = np.clip(ys, y1, y2 - 1)
    cv2.polylines(img, [np.stack([xs, ys], 1)], False, color, thick, cv2.LINE_AA)


def _vline_at(img, box, frac, color, thick=1, dashed=False):
    """box 안에서 가로 비율 frac(0~1) 위치에 세로선."""
    x1, y1, x2, y2 = box
    px = int(round(x1 + frac * (x2 - x1 - 1)))
    if not (x1 <= px < x2):
        return
    if dashed:
        for yy in range(y1, y2, 6):
            cv2.line(img, (px, yy), (px, min(yy + 3, y2 - 1)), color, thick)
    else:
        cv2.line(img, (px, y1), (px, y2 - 1), color, thick)


# =============================================================================
# 파이프라인 재현 (v6 내부 함수를 그대로 호출)
# =============================================================================
def _recompute(img_bgr: np.ndarray, cfg: v6.ColorProfile,
               clean: bool = True, align_angle: bool = True) -> Dict:
    """build_die_map_v6 와 **같은 순서**로 돌리면서 중간값을 전부 모아 반환.

    주의: 격자 검출은 원본이 아니라 *정렬(회전)된* 이미지 위에서 돈다
    (wafer_color_v6_claude.py:2155-2186). 이 단계를 빼먹으면 원점이
    실제 결과와 어긋난다 — 그래서 여기서도 그대로 재현한다.
    """
    diag = v6.GridDiagnostics()

    # 1) 원본 기준 wafer 검출
    cx0, cy0, r0, sil0 = v6.detect_wafer_adaptive(img_bgr, cfg, diag)
    fill = tuple(cfg.wafer_fill_bgr) if cfg.wafer_fill_bgr is not None \
        else diag.background_bgr

    # 2) clean
    base = v6.clean_wafer_v6(img_bgr, cx0, cy0, r0, sil0, fill) if clean \
        else img_bgr.copy()

    # 3) 회전 정렬 (누적각으로 항상 원본에서 1회 워핑)
    rotation_deg = 0.0
    img_al = base
    if align_angle:
        total = 0.0
        for _ in range(max(1, cfg.angle_max_iter)):
            a_proj, _a_fft, _c, _ag = v6.estimate_grid_angle_adaptive(
                img_al, cx0, cy0, r0, cfg)
            if abs(a_proj) < 0.01:
                break
            total += a_proj
            img_al = v6._rotate_keep_size(base, cx0, cy0, total, fill)
        rotation_deg = total

    # 회전 후 wafer 재검출
    cx, cy, r, mask = v6.detect_wafer_adaptive(img_al, cfg, diag)
    img_bgr = img_al

    roi, rx1, ry1 = v6._grid_roi(img_bgr, cx, cy, r, cfg.roi_ratio)
    small, scale = v6._downscale(roi, cfg.profile_max_dim)
    names = tuple(cfg.feature_channels) if cfg.feature_channels else v6.FEATURE_NAMES
    feats = v6._feature_bank(small, names)

    axes: Dict[str, Dict] = {}
    for axis in ("x", "y"):
        n_prof = float(small.shape[1] if axis == "x" else small.shape[0])
        profs, chans = {}, []
        for nm in names:
            f = feats[nm]
            prof = f.mean(axis=0) if axis == "x" else f.mean(axis=1)
            profs[nm] = prof
            fixed = cfg.pitch_x if axis == "x" else cfg.pitch_y
            chans.append(v6._analyze_channel(nm, prof, cfg,
                                             (fixed * scale) if fixed else None))

        pitch_local, agree, cluster = v6._consensus_pitch(
            [c for c in chans if c.score >= cfg.min_channel_score],
            cfg.pitch_cluster_tol)
        if not cluster:
            pitch_local, agree, cluster = v6._consensus_pitch(
                chans, cfg.pitch_cluster_tol)
        for c in cluster:
            c.used = True
        pitch = pitch_local / max(scale, 1e-12)

        folded_map = {}
        for c in cluster:
            prof = profs[c.name]
            xd = v6._detrend(prof, v6._odd(int(round(pitch_local * 2)) + 1))
            phi = v6._fourier_phase(xd, pitch_local)
            fol = v6._fold(xd, pitch_local, phi)
            pos, pol, width, _ = v6._street_from_template(
                fol, pitch_local, phi, cfg.street_polarity)
            c.polarity, c.street_width_frac, c.street_pos = pol, float(width), pos
            folded_map[c.name] = (fol, phi, pos)

        base = float(rx1 if axis == "x" else ry1)
        street_ph, phase_conf = v6._consensus_phase(cluster, pitch, base, scale)
        anchor = cx if axis == "x" else cy
        k = round((anchor - street_ph) / pitch)
        origin = street_ph + k * pitch

        max_pitch_local = max(cfg.min_pitch_px * 3.0, n_prof * cfg.max_pitch_ratio)
        axes[axis] = dict(
            profs=profs, chans=chans, cluster=cluster, folded=folded_map,
            pitch_local=pitch_local, pitch=pitch, agree=agree,
            street_ph=street_ph, phase_conf=phase_conf, k=k, origin=origin,
            anchor=float(anchor), base=base, n_prof=n_prof,
            max_pitch_local=max_pitch_local, min_pitch_local=cfg.min_pitch_px,
        )

    return dict(cx=cx, cy=cy, r=r, mask=mask, roi=roi, rx1=rx1, ry1=ry1,
                small=small, scale=scale, names=names, axes=axes, diag=diag,
                aligned=img_bgr, rotation_deg=rotation_deg)


# =============================================================================
# STAGE 1 - 웨이퍼 검출 + ROI
# =============================================================================
def stage1(img_bgr, S, cfg) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    cx, cy, r = S["cx"], S["cy"], S["r"]

    # 마스크 경계 (초록)
    if S["mask"] is not None:
        cnts, _ = cv2.findContours(S["mask"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 0), max(1, w // 700))

    t = max(1, w // 500)
    cv2.circle(vis, (cx, cy), r, C_CENTER, t)
    cv2.drawMarker(vis, (cx, cy), C_CENTER, cv2.MARKER_CROSS,
                   max(20, w // 30), t + 1)

    roi_h, roi_w = S["roi"].shape[:2]
    cv2.rectangle(vis, (S["rx1"], S["ry1"]),
                  (S["rx1"] + roi_w, S["ry1"] + roi_h), C_ACCENT, t + 1)

    x0 = int(round(S["axes"]["x"]["origin"]))
    y0 = int(round(S["axes"]["y"]["origin"]))
    cv2.drawMarker(vis, (x0, y0), C_ORIGIN, cv2.MARKER_TILTED_CROSS,
                   max(24, w // 26), t + 2)

    bar = _canvas(96, vis.shape[1])
    _text(bar, "STAGE 1  clean -> rotate-align -> wafer detect -> grid ROI",
          (10, 24), FG, 0.6, 2)
    _text(bar, f"rotation={S['rotation_deg']:+.3f}deg   center=({cx},{cy})  "
               f"r={r}   ROI={roi_w}x{roi_h} (roi_ratio={cfg.roi_ratio})",
          (10, 50), C_CENTER)
    _text(bar, f"origin(x0,y0)=({x0},{y0})   green=wafer mask  "
               f"orange=ROI  yellow=center  magenta=origin", (10, 74), C_ORIGIN)
    return np.vstack([bar, vis])


# =============================================================================
# STAGE 2 - 채널 프로파일 + 점수
# =============================================================================
def stage2(S, axis: str, cfg) -> np.ndarray:
    A = S["axes"][axis]
    names = list(S["names"])
    W, PH, PAD, TOP = 1180, 96, 12, 78
    H = TOP + len(names) * (PH + PAD) + 20
    img = _canvas(H, W)
    _text(img, f"STAGE 2  [{axis}] per-channel 1D profile "
               f"(ROI mean along the other axis)", (12, 26), FG, 0.6, 2)
    _text(img, "green name = adopted by consensus, grey = dropped.  "
               "score = 0.6*corr(T) + 0.4*corr(2T)", (12, 52), FG, 0.42)

    used = {c.name for c in A["cluster"]}
    by = {c.name: c for c in A["chans"]}
    for i, nm in enumerate(names):
        c = by[nm]
        y1 = TOP + i * (PH + PAD)
        box = (210, y1, W - 14, y1 + PH)
        _frame(img, box)
        col = CH_COLORS.get(nm, FG)
        _curve(img, A["profs"][nm], box, col, 1)
        tag = C_USED if nm in used else C_DROP
        _text(img, nm, (14, y1 + 22), tag, 0.62, 2)
        _text(img, f"score {c.score:.3f}", (14, y1 + 44), tag, 0.40)
        _text(img, f"pitch {c.pitch / max(S['scale'],1e-9):7.2f}px",
              (14, y1 + 62), tag, 0.40)
        pol = "bright" if c.polarity > 0 else "dark"
        _text(img, f"{pol} w={c.street_width_frac:.2f}", (14, y1 + 80), tag, 0.40)
        if nm in used:
            # 채택 채널엔 street 격자선을 얹는다
            n = A["profs"][nm].size
            ph, p = c.street_pos, A["pitch_local"]
            j = ph
            while j < n:
                _vline_at(img, box, j / max(n - 1, 1), C_ACCENT, 1, dashed=True)
                j += p
    return img


# =============================================================================
# STAGE 3 - FFT 파워 스펙트럼 + 탐색 밴드
# =============================================================================
def stage3(S, axis: str, cfg) -> np.ndarray:
    A = S["axes"][axis]
    names = list(S["names"])
    W, PH, PAD, TOP = 1180, 92, 12, 100
    H = TOP + len(names) * (PH + PAD) + 20
    img = _canvas(H, W)
    _text(img, f"STAGE 3  [{axis}] _spectral_pitch(): FFT power vs pitch(px)",
          (12, 26), FG, 0.6, 2)
    _text(img, f"search band = [min_pitch_px={cfg.min_pitch_px}, "
               f"max_pitch={A['max_pitch_local']/max(S['scale'],1e-9):.0f}px]  "
               f"(max_pitch = n_prof * max_pitch_ratio={cfg.max_pitch_ratio})",
          (12, 52), C_BAND)
    _text(img, "orange = adopted pitch.  A true pitch ABOVE the band cannot be "
               "seen -> FFT silently locks onto pitch/2.", (12, 76), FG, 0.42)

    used = {c.name for c in A["cluster"]}
    by = {c.name: c for c in A["chans"]}
    lo_p = cfg.min_pitch_px
    hi_p = A["max_pitch_local"]
    grid_p = np.linspace(lo_p, hi_p, 600)

    for i, nm in enumerate(names):
        c = by[nm]
        y1 = TOP + i * (PH + PAD)
        box = (210, y1, W - 14, y1 + PH)
        _frame(img, box)

        p = A["profs"][nm].astype(np.float64)
        n = p.size
        x = (p - p.mean()) * np.hanning(n)
        nfft = 1 << int(math.ceil(math.log2(max(64, n * 8))))
        P = np.abs(np.fft.rfft(x, nfft)) ** 2
        # pitch 축으로 리샘플
        kk = nfft / np.clip(grid_p, 1e-9, None)
        ki = np.clip(kk.round().astype(np.int64), 0, P.size - 1)
        spec = np.log10(P[ki] + 1e-12)
        _curve(img, spec, box, CH_COLORS.get(nm, FG), 1)

        tag = C_USED if nm in used else C_DROP
        _text(img, nm, (14, y1 + 24), tag, 0.60, 2)
        _text(img, f"peak {c.pitch/max(S['scale'],1e-9):7.2f}px",
              (14, y1 + 48), tag, 0.40)
        if c.note:
            _text(img, c.note[:22], (14, y1 + 68), (140, 180, 255), 0.34)
        if c.pitch > 0 and lo_p <= c.pitch <= hi_p:
            _vline_at(img, box, (c.pitch - lo_p) / max(hi_p - lo_p, 1e-9),
                      C_ACCENT, 1)
    return img


# =============================================================================
# STAGE 4 - 1주기 접힘 템플릿 -> street 위치/극성
# =============================================================================
def stage4(S, axis: str, cfg) -> np.ndarray:
    A = S["axes"][axis]
    items = [(c.name, A["folded"][c.name], c) for c in A["cluster"]
             if c.name in A["folded"]]
    W, PH, PAD, TOP = 1180, 110, 14, 104
    H = TOP + max(1, len(items)) * (PH + PAD) + 20
    img = _canvas(H, W)
    _text(img, f"STAGE 4  [{axis}] _fold() + _street_from_template()",
          (12, 26), FG, 0.6, 2)
    _text(img, "profile folded into ONE period. street = bin farthest from the "
               "median (|t - median| max).", (12, 52), FG, 0.42)
    _text(img, "The winning side (max vs min) IS the polarity - no separate "
               "polarity fit.", (12, 74), FG, 0.42)
    _text(img, "orange = detected street center.  Shifting this shifts the "
               "origin by the same amount.", (12, 96), C_ACCENT, 0.42)

    for i, (nm, (fol, phi, pos), c) in enumerate(items):
        y1 = TOP + i * (PH + PAD)
        box = (210, y1, W - 14, y1 + PH)
        _frame(img, box)
        _curve(img, fol, box, CH_COLORS.get(nm, FG), 2)
        med = float(np.median(fol))
        lo, hi = float(fol.min()), float(fol.max())
        fr = (med - lo) / max(hi - lo, 1e-12)
        yy = int(round(box[3] - 1 - fr * (box[3] - box[1] - 1)))
        cv2.line(img, (box[0], yy), (box[2] - 1, yy), (90, 90, 90), 1)
        # street 위치를 접힘 좌표(0..pitch)로 환산
        frac = ((pos - phi) % A["pitch_local"]) / A["pitch_local"]
        _vline_at(img, box, frac, C_ACCENT, 2)
        _text(img, nm, (14, y1 + 24), C_USED, 0.60, 2)
        _text(img, f"street_pos {pos:6.2f}", (14, y1 + 48), FG, 0.40)
        _text(img, f"polarity {'bright' if c.polarity>0 else 'dark'}",
              (14, y1 + 68), FG, 0.40)
        _text(img, f"width {c.street_width_frac:.3f}", (14, y1 + 88), FG, 0.40)
    if not items:
        _text(img, "(no adopted channel)", (220, TOP + 30), C_DROP, 0.6)
    return img


# =============================================================================
# STAGE 5 - 위상 합의(원형평균) + 원점 확정
# =============================================================================
def stage5(S, axis: str, cfg) -> np.ndarray:
    A = S["axes"][axis]
    W, H = 1180, 470
    img = _canvas(H, W)
    _text(img, f"STAGE 5  [{axis}] _consensus_phase() -> origin",
          (12, 26), FG, 0.6, 2)

    # --- 5a) 원형 다이어그램 -------------------------------------------------
    ccx, ccy, R = 175, 210, 118
    cv2.circle(img, (ccx, ccy), R, GRID, 1)
    _text(img, "circular mean of street phase", (12, 350), FG, 0.40)
    _text(img, f"phase_conf = |R| = {A['phase_conf']:.3f}", (12, 372),
          C_USED if A["phase_conf"] >= 0.5 else (80, 80, 255), 0.44)
    _text(img, "(low conf => streets disagree;", (12, 394), FG, 0.36)
    _text(img, " origin can land on a wrong line)", (12, 412), FG, 0.36)

    acc = 0j
    wsum = 0.0
    for c in A["cluster"]:
        g = A["base"] + c.street_pos / max(S["scale"], 1e-12)
        th = 2.0 * np.pi * (g % A["pitch"]) / A["pitch"]
        acc += c.score * np.exp(1j * th)
        wsum += c.score
        ex = int(round(ccx + R * math.cos(th)))
        ey = int(round(ccy - R * math.sin(th)))
        cv2.arrowedLine(img, (ccx, ccy), (ex, ey),
                        CH_COLORS.get(c.name, FG), 2, cv2.LINE_AA, tipLength=0.08)
    if wsum > 1e-12:
        m = acc / wsum
        ex = int(round(ccx + R * m.real))
        ey = int(round(ccy - R * m.imag))
        cv2.arrowedLine(img, (ccx, ccy), (ex, ey), C_ACCENT, 3,
                        cv2.LINE_AA, tipLength=0.10)

    # --- 5b) 수식 -----------------------------------------------------------
    bx = 380
    lines = [
        ("inputs", FG),
        (f"  street_ph (global street phase) = {A['street_ph']:10.4f} px", FG),
        (f"  pitch                           = {A['pitch']:10.4f} px", FG),
        (f"  anchor (wafer_c{axis})            = {A['anchor']:10.4f} px", C_CENTER),
        ("", FG),
        ("origin rule   (wafer_color_v6_claude.py:1673-1675)", C_ACCENT),
        ("  k      = round((anchor - street_ph) / pitch)", FG),
        ("  origin = street_ph + k * pitch", FG),
        ("", FG),
        (f"  (anchor - street_ph)/pitch = "
         f"{(A['anchor']-A['street_ph'])/A['pitch']:10.4f}", FG),
        (f"  k                          = {A['k']:10d}", C_ACCENT),
        (f"  origin                     = {A['origin']:10.4f} px", C_ORIGIN),
        ("", FG),
        (f"  offset from center = {A['origin']-A['anchor']:+8.3f} px"
         f"  = {(A['origin']-A['anchor'])/A['pitch']:+.4f} pitch",
         C_ORIGIN),
    ]
    yy = 62
    for s, col in lines:
        if s:
            _text(img, s, (bx, yy), col, 0.46)
        yy += 24

    ok = abs((A["origin"] - A["anchor"]) / A["pitch"]) <= 0.5
    _text(img, "round() => |offset| <= 0.5 pitch by construction.  "
               + ("OK" if ok else "VIOLATED"),
          (bx, yy + 8), C_USED if ok else (80, 80, 255), 0.46, 2)
    _text(img, "If you swap round() for floor()/ceil() here, the origin jumps",
          (bx, yy + 34), (150, 150, 150), 0.40)
    _text(img, "a whole pitch - this was the old 'magenta X sits one row up' bug.",
          (bx, yy + 54), (150, 150, 150), 0.40)
    return img


# =============================================================================
# STAGE 6 - 최종 확대 검증
# =============================================================================
def stage6(img_bgr, S, cfg, zoom_pitches: float = 3.0) -> np.ndarray:
    cx, cy = S["cx"], S["cy"]
    px, py = S["axes"]["x"]["pitch"], S["axes"]["y"]["pitch"]
    x0 = S["axes"]["x"]["origin"]
    y0 = S["axes"]["y"]["origin"]
    half = int(round(max(px, py) * zoom_pitches))
    h, w = img_bgr.shape[:2]
    a, b = max(0, cx - half), max(0, cy - half)
    c, d = min(w, cx + half), min(h, cy + half)
    crop = img_bgr[b:d, a:c].copy()
    if crop.size == 0:
        return _canvas(80, 600)

    Z = max(1, int(round(760 / max(crop.shape[:2]))))
    crop = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)

    def gx(v): return int(round((v - a) * Z))
    def gy(v): return int(round((v - b) * Z))

    n = int(zoom_pitches) + 2
    for i in range(-n, n + 1):
        X, Y = gx(x0 + i * px), gy(y0 + i * py)
        if 0 <= X < crop.shape[1]:
            cv2.line(crop, (X, 0), (X, crop.shape[0] - 1), C_ACCENT, 1)
        if 0 <= Y < crop.shape[0]:
            cv2.line(crop, (0, Y), (crop.shape[1] - 1, Y), C_ACCENT, 1)

    cv2.drawMarker(crop, (gx(cx), gy(cy)), C_CENTER, cv2.MARKER_CROSS, 46, 2)
    cv2.drawMarker(crop, (gx(x0), gy(y0)), C_ORIGIN, cv2.MARKER_TILTED_CROSS,
                   40, 2)
    cv2.line(crop, (gx(cx), gy(cy)), (gx(x0), gy(y0)), (255, 255, 255), 1)

    dx = (x0 - cx) / px
    dy = (y0 - cy) / py
    bar = _canvas(104, crop.shape[1])
    _text(bar, "STAGE 6  final check (nearest-neighbour zoom)", (10, 24), FG, 0.6, 2)
    _text(bar, f"pitch=({px:.3f},{py:.3f})px   center=({cx},{cy})   "
               f"origin=({x0:.1f},{y0:.1f})", (10, 50), FG, 0.44)
    good = abs(dx) <= 0.5 and abs(dy) <= 0.5
    _text(bar, f"origin - center = ({dx:+.3f}, {dy:+.3f}) pitch   "
               + ("OK  (<= 0.5)" if good else "OFF!  (> 0.5 -> check pitch/phase)"),
          (10, 76), C_USED if good else (80, 80, 255), 0.48, 2)
    return np.vstack([bar, crop])


# =============================================================================
# 실행
# =============================================================================
def visualize(path: str, outdir: str, cfg: v6.ColorProfile,
              axes: Sequence[str] = ("x", "y")) -> List[str]:
    img = v6._load_bgr(path)
    S = _recompute(img, cfg)
    img = S["aligned"]          # 격자 검출이 실제로 돌아간 이미지
    stem = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(outdir, stem)
    os.makedirs(d, exist_ok=True)

    written: List[str] = []

    def w(name, im):
        p = os.path.join(d, name)
        v6._imwrite_unicode(p, im)
        written.append(p)

    w("01_wafer_roi.png", stage1(img, S, cfg))
    for ax in axes:
        w(f"02_{ax}_channels.png", stage2(S, ax, cfg))
        w(f"03_{ax}_fft.png", stage3(S, ax, cfg))
        w(f"04_{ax}_fold_street.png", stage4(S, ax, cfg))
        w(f"05_{ax}_origin.png", stage5(S, ax, cfg))
    w("06_zoom_verify.png", stage6(img, S, cfg))

    # --- 자기검증: 재현한 값이 build_die_map_v6 결과와 같은가 ----------------
    ref = v6.build_die_map_v6(path, profile=cfg)
    got = (int(round(S["axes"]["x"]["origin"])),
           int(round(S["axes"]["y"]["origin"])))
    match = (got == (ref.x0, ref.y0))

    # 텍스트 요약
    A, B = S["axes"]["x"], S["axes"]["y"]
    txt = [
        f"# {stem}",
        f"self-check vs build_die_map_v6(): "
        f"{'MATCH' if match else 'MISMATCH'}  "
        f"viz={got}  build={(ref.x0, ref.y0)}",
        f"rotation applied = {S['rotation_deg']:+.4f} deg",
        f"wafer center = ({S['cx']}, {S['cy']})   r = {S['r']}",
        f"ROI          = {S['roi'].shape[1]}x{S['roi'].shape[0]} "
        f"@({S['rx1']},{S['ry1']})   downscale={S['scale']:.4f}",
        "",
        f"{'axis':4} {'pitch':>9} {'agree':>6} {'street_ph':>10} "
        f"{'conf':>6} {'k':>5} {'origin':>10} {'off(pitch)':>11}",
    ]
    for nm, X in (("x", A), ("y", B)):
        txt.append(f"{nm:4} {X['pitch']:9.4f} {X['agree']:6.2f} "
                   f"{X['street_ph']:10.4f} {X['phase_conf']:6.2f} "
                   f"{X['k']:5d} {X['origin']:10.3f} "
                   f"{(X['origin']-X['anchor'])/X['pitch']:+11.4f}")
    txt.append("")
    txt.append("channels (score / pitch_px / used):")
    for nm, X in (("x", A), ("y", B)):
        used = {c.name for c in X["cluster"]}
        for c in X["chans"]:
            txt.append(f"  [{nm}] {c.name:7} score={c.score:.3f} "
                       f"pitch={c.pitch/max(S['scale'],1e-9):8.2f} "
                       f"{'USED' if c.name in used else '-':5} {c.note}")
    if S["diag"].warnings:
        txt.append("")
        txt.append("warnings:")
        txt.extend(f"  ! {s}" for s in S["diag"].warnings)
    p = os.path.join(d, "summary.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    written.append(p)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="격자 원점(x0,y0) 검출 파이프라인 단계별 시각화")
    ap.add_argument("images", nargs="+", help="이미지 경로 (glob 가능)")
    ap.add_argument("--out", default="viz_out", help="출력 폴더")
    ap.add_argument("--axis", choices=["x", "y", "both"], default="both")
    ap.add_argument("--roi-ratio", type=float, default=None)
    ap.add_argument("--min-pitch-px", type=float, default=None)
    ap.add_argument("--max-pitch-ratio", type=float, default=None)
    ap.add_argument("--profile-max-dim", type=int, default=None)
    a = ap.parse_args(argv)

    kw = {}
    if a.roi_ratio is not None:
        kw["roi_ratio"] = a.roi_ratio
    if a.min_pitch_px is not None:
        kw["min_pitch_px"] = a.min_pitch_px
    if a.max_pitch_ratio is not None:
        kw["max_pitch_ratio"] = a.max_pitch_ratio
    if a.profile_max_dim is not None:
        kw["profile_max_dim"] = a.profile_max_dim
    cfg = v6.ColorProfile(**kw)

    axes = ("x", "y") if a.axis == "both" else (a.axis,)
    paths: List[str] = []
    for pat in a.images:
        g = sorted(glob.glob(pat))
        paths.extend(g if g else [pat])

    rc = 0
    for p in paths:
        try:
            files = visualize(p, a.out, cfg, axes)
            print(f"[ok] {p}  -> {len(files)} files in "
                  f"{os.path.dirname(files[0])}")
        except Exception as e:                       # noqa: BLE001
            print(f"[FAIL] {p}: {type(e).__name__}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
