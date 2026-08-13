# -*- coding: utf-8 -*-
"""합성 웨이퍼 생성기 + 정답 대조 검증 하네스  (V6 / claude)

기존 `--robustness` 하네스는 *실제 이미지에 색 변환을 입히는* 방식이라
- 정답(ground truth)이 없고 (자기 자신의 original 을 기준으로 상대 비교만 함)
- 변환으로 도달할 수 없는 색 조합은 아예 시험하지 못한다.

이 스크립트는 반대로 **정답을 알고 있는 웨이퍼를 직접 합성**한다.
pitch / 중심 / 반지름 / 회전각 / street 위상을 우리가 지정하므로
검출 결과를 절대 기준과 대조할 수 있다.

====================================================================
                          사 용 법  (USAGE)
====================================================================

[1] 준비
--------
    pip install numpy opencv-python

    같은 폴더에 `wafer_color_v6_claude.py` 가 있어야 `--eval` 이 된다.
    `--v5` 까지 쓰려면 `wafer_die_map_v5.py` 도 있어야 한다.


[2] CLI -- 이것만 알면 된다
--------------------------
    # (a) 이미지만 생성 (29장)
    python make_color_wafers_claude.py --out synth/

    # (b) 생성 + V6 로 검출해서 정답과 대조
    python make_color_wafers_claude.py --out synth/ --eval

    # (c) V5 와 나란히 비교  ★ V6 의 존재 이유가 여기서 보인다
    python make_color_wafers_claude.py --out synth/ --eval --v5

    # (d) 실패한 케이스만 다시 파고들 때
    python make_color_wafers_claude.py --out synth/ --eval \
           --only 15_lowcontrast 22_large_pitch --overlay

    # (e) 팔레트 이름 목록
    python make_color_wafers_claude.py --list

옵션 전체:
    --out DIR      출력 디렉터리 (기본 synth/)
    --eval         검출까지 수행해 정답과 대조
    --v5           V5 도 같이 돌려 한 표에 비교
    --overlay      팔레트별 검출 오버레이 PNG 저장 (<name>__v6.png)
    --only A B ...  특정 팔레트만 (공백 구분, 이름 그대로)
    --size N       캔버스 한 변을 N 으로 덮어쓰기 (빠른 실험용)
    --json PATH    결과를 JSON 으로 저장
    --list         팔레트 목록 출력 후 종료


[3] 결과 표 읽는 법
------------------
    name              v6                      v5
    01_classic_dark   OK  px+0.00% py+0.00%   FAIL  RuntimeError
                      ^^                      ^^^^
                      정답과 대조한 판정       V5 는 대개 예외로 죽는다

    OK / FAIL 은 아래 허용오차(TOL) 를 모두 통과했는지로 정한다.

        pitch  : 1.0%   (상대오차)
        center : 15 px  (절대)
        radius : 3.0%   (상대)
        angle  : 0.50 deg
        phase  : pitch 의 25% 이내 (street 위치, 원형 거리로 계산)

    맨 아래 요약줄에 `v6 29/29 (100.0%)` 처럼 합계가 나온다.


[4] 팔레트 29종 -- 무엇을 노린 시험인가
--------------------------------------
    01_classic_dark      검은 배경 / 어두운 street  (V5 가 상정한 조건)
    02_classic_bright    밝은 die
    03_bright_street     street 가 die 보다 밝음 -> 극성 자동판별 시험
    04_white_street_brown 흰 street + 갈색 얼룩  ★사용자가 지목한 실패 케이스
    05_grayscale         완전 무채색 -> chroma/sat 채널이 죽는다
    06_isoluminant       밝기는 같고 색상만 다름 -> L 채널이 죽는다
    07_isochromatic      색상은 같고 밝기만 다름
    08_neon              고채도 원색
    09_pastel            저채도 파스텔
    10_sepia             세피아
    11_green_pcb         PCB 녹색
    12_copper            구리색
    13_purple            보라
    14_cyan_on_white     흰 바탕에 청록 die
    15_lowcontrast       극저대비  ★pitch 2배 오검출이 나던 케이스
    16_bright_bg         배경이 웨이퍼보다 밝음 -> 배경 자동추정 시험
    17_rim_ring          웨이퍼 테두리 링 -> 원 반지름 부풀림 시험
    18_rot_p3            +3.0도 회전
    19_rot_m2            -2.2도 회전
    20_aniso_pitch       가로/세로 pitch 가 다름
    21_small_pitch       아주 작은 pitch
    22_large_pitch       아주 큰 pitch + 넓은 lane  ★마스크 파편화 케이스
    23_notch_left        notch 가 왼쪽
    24_heavy_noise       강한 노이즈
    25_jpeg_low          JPEG 저품질 압축
    26_illum_strong      강한 조명 기울기
    27_blur_soft         블러
    28_speckle_die       die 위에 얼룩
    29_worst_case        위 악조건을 한꺼번에

    ※ 이름을 그대로 `--only` 에 넣으면 된다.


[5] 파이썬에서 직접 쓰기 -- 내 팔레트 추가하기
---------------------------------------------
`WaferSpec` 에 적은 값이 **그대로 정답** 이 된다.

    from make_color_wafers_claude import WaferSpec, synth_wafer
    import cv2

    spec = WaferSpec(
        name    = "my_case",
        size    = 1600,          # 정사각 캔버스 한 변
        radius  = 700,           # 웨이퍼 반지름 (px)
        cx=None, cy=None,        # None -> 캔버스 중앙
        pitch_x = 64.0,          # 가로 pitch (px)
        pitch_y = 64.0,
        phase_x = 0.0,           # street 중심의 x 오프셋 (0..pitch_x)
        phase_y = 0.0,
        street_w= 6.0,           # street 폭 (px)
        angle_deg = 0.0,         # 반시계 회전 (deg)
        notch_deg = 270.0,       # notch 방향 (화면좌표, 아래=270)
        notch_r   = 0.045,       # notch 반지름 / 웨이퍼 반지름

        # 색 (모두 B,G,R 순서)
        bg     = (255, 255, 255),  # 웨이퍼 바깥
        die    = (60, 90, 160),    # die 바탕
        street = (255, 255, 255),  # scribe lane
        ink    = (30, 50, 110),    # die 내부 회로 무늬
        rim    = None,             # 테두리 링 색 (None -> 없음)

        # 열화
        die_jitter  = 10.0,        # die 별 밝기 흔들림 (0 -> 완전 동일)
        noise_sigma = 3.0,         # 가우시안 노이즈 sigma
        speckle     = (40, 70, 110),  # 얼룩 색 (None -> 없음)
        speckle_amt = 0.35,        # 얼룩 밀도 0..1
        speckle_on  = "street",    # "street" | "die" | "all"
        illum       = 0.0,         # 조명 기울기 0..1
        blur        = 0.0,         # 가우시안 블러 sigma
        jpeg_q      = 0,           # >0 이면 JPEG 왕복 압축
        seed        = 1234,
    )

    img, truth = synth_wafer(spec)   # img: BGR ndarray, truth: 정답 dict
    cv2.imwrite("my_case.png", img)
    print(truth)                     # pitch/cx/cy/r/angle/phase ...

    # 바로 검증까지
    from make_color_wafers_claude import evaluate
    from pathlib import Path
    res = evaluate([spec], Path("synth/"), with_v5=False, save_overlay=True)

`spec.truth()` 만 따로 불러도 정답 dict 를 얻을 수 있다.


[6] 재현성 (중요)
-----------------
팔레트별 seed 는 `zlib.crc32(name)` 으로 만든다.
파이썬 내장 `hash()` 는 PYTHONHASHSEED 때문에 **프로세스마다 값이
달라져서** 같은 명령을 두 번 돌려도 다른 이미지가 나온다.
그러면 "고쳤는데 왜 또 실패하지" 같은 유령 디버깅을 하게 된다.
직접 팔레트를 추가할 때도 seed 는 반드시 고정값으로 준다.

    확인:  md5sum synth/15_lowcontrast.png   # 두 번 돌려도 같아야 한다
"""
from __future__ import annotations

import math
import os
import sys
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["WaferSpec", "PALETTES", "synth_wafer", "evaluate"]

BGR = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# 사양 / 정답
# ---------------------------------------------------------------------------
@dataclass
class WaferSpec:
    """합성 웨이퍼 1장의 사양. 여기 적힌 값이 곧 정답이다."""
    name: str
    size: int = 1600                    # 정사각 캔버스 한 변
    radius: int = 700
    cx: Optional[int] = None            # None -> 캔버스 중앙
    cy: Optional[int] = None
    pitch_x: float = 64.0
    pitch_y: float = 64.0
    phase_x: float = 0.0                # street 중심의 x 오프셋 (0..pitch_x)
    phase_y: float = 0.0
    street_w: float = 6.0               # street 폭 (px)
    angle_deg: float = 0.0              # 시계 반대방향 회전
    notch_deg: float = 270.0            # notch 각도 (화면 좌표계, 아래=270)
    notch_r: float = 0.045              # notch 반지름 / 웨이퍼 반지름

    # --- 색 ---
    bg: BGR = (0, 0, 0)                 # 웨이퍼 바깥
    die: BGR = (120, 110, 100)          # die 바탕
    street: BGR = (40, 40, 40)          # scribe lane
    ink: BGR = (170, 160, 150)          # die 내부 회로 무늬
    rim: Optional[BGR] = None           # 웨이퍼 가장자리 링 (None -> 없음)

    # --- 열화 ---
    die_jitter: float = 10.0            # die 별 밝기 흔들림 (0 -> 완전 동일)
    noise_sigma: float = 3.0            # 가우시안 노이즈
    speckle: Optional[BGR] = None       # 특정 색 얼룩 (예: 갈색 노이즈)
    speckle_amt: float = 0.0            # 얼룩 밀도 0..1
    speckle_on: str = "street"          # street | die | all
    illum: float = 0.0                  # 조명 기울기 세기 0..1
    blur: float = 0.0                   # 가우시안 블러 sigma
    jpeg_q: int = 0                     # >0 이면 JPEG 왕복 압축 열화

    seed: int = 0

    def resolved_center(self) -> Tuple[int, int]:
        c = self.size // 2
        return (c if self.cx is None else self.cx,
                c if self.cy is None else self.cy)

    def truth(self) -> Dict[str, Any]:
        cx, cy = self.resolved_center()
        return {
            "name": self.name, "size": self.size,
            "cx": cx, "cy": cy, "r": self.radius,
            "pitch_x": self.pitch_x, "pitch_y": self.pitch_y,
            "phase_x": self.phase_x % self.pitch_x,
            "phase_y": self.phase_y % self.pitch_y,
            "street_w": self.street_w, "angle_deg": self.angle_deg,
            "bg": list(self.bg), "die": list(self.die),
            "street": list(self.street),
        }


# ---------------------------------------------------------------------------
# 렌더링 재료
# ---------------------------------------------------------------------------
def _die_template(rng: np.random.Generator, tw: int = 96) -> np.ndarray:
    """die 내부 회로 무늬 (0..1 alpha). 모든 die 가 공유하는 하나의 패턴.

    실제 die 처럼 '반복되지만 내부는 복잡한' 구조를 만들어야
    검출기가 단순 계단파가 아닌 현실적인 신호를 보게 된다.
    """
    t = np.zeros((tw, tw), np.float32)
    # 큰 블록 몇 개
    for _ in range(rng.integers(3, 6)):
        x0, y0 = rng.integers(4, tw - 20, 2)
        w, h = rng.integers(10, 28, 2)
        t[y0:y0 + h, x0:x0 + w] = rng.uniform(0.35, 1.0)
    # 가는 배선
    for _ in range(rng.integers(6, 12)):
        if rng.random() < 0.5:
            y0 = int(rng.integers(2, tw - 2))
            t[y0:y0 + int(rng.integers(1, 3)), 2:tw - 2] = rng.uniform(0.5, 1.0)
        else:
            x0 = int(rng.integers(2, tw - 2))
            t[2:tw - 2, x0:x0 + int(rng.integers(1, 3))] = rng.uniform(0.5, 1.0)
    # 패드 배열
    for gy in range(3, tw - 6, 14):
        for gx in range(3, tw - 6, 14):
            if rng.random() < 0.35:
                t[gy:gy + 4, gx:gx + 4] = rng.uniform(0.6, 1.0)
    return cv2.GaussianBlur(t, (0, 0), 0.7)


def _rot_coords(size: int, cx: float, cy: float,
                angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """웨이퍼 중심 기준 회전 좌표계 (u, v) 를 해석적으로 만든다.

    이미지를 실제로 회전시키지 않으므로 보간 아티팩트가 없고,
    정답 각도가 정확히 보존된다.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    return dx * ca + dy * sa, -dx * sa + dy * ca


def _soft_band(frac: np.ndarray, pitch: float, width: float) -> np.ndarray:
    """셀 내 위치(0..pitch)에서 경계(=0 또는 pitch)에 걸친 street alpha."""
    d = np.minimum(frac, pitch - frac)          # 가장 가까운 경계까지 거리
    return np.clip(width * 0.5 - d + 0.5, 0.0, 1.0)


def _apply_speckle(img: np.ndarray, mask: np.ndarray, color: BGR,
                   amount: float, rng: np.random.Generator) -> None:
    """지정 영역에 특정 색 얼룩을 뿌린다 (제자리 수정).

    단순 픽셀 노이즈가 아니라 blob 형태여야 실제 오염과 비슷해진다.
    """
    if amount <= 0:
        return
    h, w = img.shape[:2]
    n = rng.random((h // 4, w // 4)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    n = cv2.GaussianBlur(n, (0, 0), 1.5)
    thr = float(np.quantile(n, 1.0 - np.clip(amount, 0.0, 1.0)))
    sel = (n > thr) & mask
    if not sel.any():
        return
    strength = np.clip((n[sel] - thr) / max(1e-6, n.max() - thr), 0, 1)
    strength = (0.45 + 0.55 * strength)[:, None]
    img[sel] = (img[sel] * (1.0 - strength) +
                np.array(color, np.float32) * strength)


def _apply_illum(img: np.ndarray, strength: float,
                 rng: np.random.Generator) -> np.ndarray:
    """비스듬한 조명 기울기."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    ang = float(rng.uniform(0, 2 * math.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = ((xx / w - 0.5) * math.cos(ang) + (yy / h - 0.5) * math.sin(ang))
    return img * (1.0 + strength * g)[:, :, None]


def synth_wafer(spec: WaferSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """사양 1개 -> (BGR uint8 이미지, 정답 dict)."""
    rng = np.random.default_rng(spec.seed)
    n = spec.size
    cx, cy = spec.resolved_center()
    u, v = _rot_coords(n, cx, cy, spec.angle_deg)

    px, py = float(spec.pitch_x), float(spec.pitch_y)
    su = (u - spec.phase_x) % px
    sv = (v - spec.phase_y) % py
    iu = np.floor((u - spec.phase_x) / px).astype(np.int32)
    iv = np.floor((v - spec.phase_y) / py).astype(np.int32)

    # --- die 바탕 + 개체별 밝기 흔들림 ---
    img = np.empty((n, n, 3), np.float32)
    img[:] = np.array(spec.die, np.float32)
    if spec.die_jitter > 0:
        span = 256
        jt = rng.normal(0.0, spec.die_jitter,
                        (span, span)).astype(np.float32)
        img += jt[iv % span, iu % span][:, :, None]

    # --- die 내부 회로 무늬 ---
    tmpl = _die_template(rng)
    tw = tmpl.shape[0]
    tx = np.clip((su / px * tw).astype(np.int32), 0, tw - 1)
    ty = np.clip((sv / py * tw).astype(np.int32), 0, tw - 1)
    ink_a = tmpl[ty, tx][:, :, None]
    img = img * (1.0 - ink_a) + np.array(spec.ink, np.float32) * ink_a

    # --- scribe lane ---
    st_a = np.maximum(_soft_band(su, px, spec.street_w),
                      _soft_band(sv, py, spec.street_w))[:, :, None]
    img = img * (1.0 - st_a) + np.array(spec.street, np.float32) * st_a
    street_mask = st_a[:, :, 0] > 0.5

    # --- 웨이퍼 실루엣 (원 - notch) ---
    dist = np.sqrt((np.arange(n)[None, :] - cx) ** 2.0 +
                   (np.arange(n)[:, None] - cy) ** 2.0)
    wafer = dist <= spec.radius
    if spec.notch_r > 0:
        na = math.radians(spec.notch_deg)
        nx = cx + spec.radius * math.cos(na)
        ny = cy + spec.radius * math.sin(na)
        nr = spec.notch_r * spec.radius
        wafer &= ((np.arange(n)[None, :] - nx) ** 2.0 +
                  (np.arange(n)[:, None] - ny) ** 2.0) > nr ** 2.0

    if spec.rim is not None:
        ring = wafer & (dist > spec.radius - max(3.0, spec.radius * 0.012))
        img[ring] = np.array(spec.rim, np.float32)
        street_mask &= ~ring

    # --- 열화 ---
    if spec.speckle is not None and spec.speckle_amt > 0:
        if spec.speckle_on == "street":
            m = street_mask & wafer
        elif spec.speckle_on == "die":
            m = (~street_mask) & wafer
        else:
            m = wafer.copy()
        _apply_speckle(img, m, spec.speckle, spec.speckle_amt, rng)

    img[~wafer] = np.array(spec.bg, np.float32)

    img = _apply_illum(img, spec.illum, rng)
    if spec.noise_sigma > 0:
        img = img + rng.normal(0.0, spec.noise_sigma,
                               img.shape).astype(np.float32)
    if spec.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), spec.blur)

    out = np.clip(img, 0, 255).astype(np.uint8)
    if spec.jpeg_q > 0:
        ok, buf = cv2.imencode(".jpg", out,
                               [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    truth = spec.truth()
    # 이미지 좌표계에서의 street 위상 (회전 0 일 때만 의미 있음)
    truth["street_x_mod"] = (cx + spec.phase_x) % px
    truth["street_y_mod"] = (cy + spec.phase_y) % py
    return out, truth


# ---------------------------------------------------------------------------
# 팔레트 (색상 + 난이도 조합)
# ---------------------------------------------------------------------------
def _S(name: str, **kw: Any) -> WaferSpec:
    # zlib.crc32 은 프로세스마다 값이 바뀌지 않는다. 내장 hash() 는
    # PYTHONHASHSEED 때문에 실행마다 달라져서 이미지가 재현되지 않는다.
    kw.setdefault("seed", zlib.crc32(name.encode("utf-8")) % 100000)
    return WaferSpec(name=name, **kw)


PALETTES: List[WaferSpec] = [
    # ---- 기본형 ----
    _S("01_classic_dark", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=64, pitch_y=64, street_w=6),
    _S("02_classic_bright", bg=(250, 250, 248), die=(150, 148, 142),
       street=(52, 50, 48), ink=(205, 200, 192), pitch_x=72, pitch_y=72,
       street_w=7),

    # ---- 극성 반전: street 가 die 보다 밝다 ----
    _S("03_bright_street", bg=(12, 12, 12), die=(48, 44, 40),
       street=(225, 228, 232), ink=(96, 90, 84), pitch_x=60, pitch_y=60,
       street_w=6),

    # ---- 사용자가 보고한 실패 케이스: 흰 street + 갈색 노이즈 ----
    _S("04_white_street_brown", bg=(8, 8, 8), die=(126, 108, 92),
       street=(246, 246, 244), ink=(168, 150, 130),
       speckle=(38, 78, 138), speckle_amt=0.55, speckle_on="street",
       pitch_x=68, pitch_y=68, street_w=9, noise_sigma=5),

    # ---- 무채색 ----
    _S("05_grayscale", bg=(0, 0, 0), die=(128, 128, 128), street=(46, 46, 46),
       ink=(186, 186, 186), pitch_x=56, pitch_y=56, street_w=5),

    # ---- 등휘도(isoluminant): L 채널로는 안 보이고 색으로만 구분된다 ----
    _S("06_isoluminant", bg=(20, 30, 25), die=(150, 90, 90),
       street=(55, 90, 125), ink=(110, 95, 100),
       pitch_x=64, pitch_y=64, street_w=8, noise_sigma=2),

    # ---- 등색상(isochromatic): 색은 같고 밝기만 다르다 ----
    _S("07_isochromatic", bg=(5, 5, 5), die=(100, 120, 140),
       street=(42, 50, 58), ink=(150, 180, 210),
       pitch_x=64, pitch_y=64, street_w=6),

    # ---- 색 계열 ----
    _S("08_neon", bg=(30, 0, 30), die=(200, 30, 180), street=(40, 220, 220),
       ink=(255, 120, 240), pitch_x=60, pitch_y=60, street_w=6),
    _S("09_pastel", bg=(245, 240, 250), die=(215, 225, 235),
       street=(180, 200, 190), ink=(235, 240, 245),
       pitch_x=66, pitch_y=66, street_w=7),
    _S("10_sepia", bg=(20, 26, 34), die=(96, 130, 170), street=(38, 54, 76),
       ink=(140, 178, 214), pitch_x=64, pitch_y=64, street_w=6),
    _S("11_green_pcb", bg=(10, 20, 10), die=(40, 110, 30), street=(20, 55, 15),
       ink=(120, 190, 90), pitch_x=70, pitch_y=70, street_w=7),
    _S("12_copper", bg=(6, 10, 16), die=(60, 110, 175), street=(28, 52, 86),
       ink=(110, 165, 225), pitch_x=62, pitch_y=62, street_w=6),
    _S("13_purple", bg=(24, 6, 24), die=(160, 70, 130), street=(70, 26, 58),
       ink=(210, 130, 190), pitch_x=58, pitch_y=58, street_w=6),
    _S("14_cyan_on_white", bg=(255, 255, 255), die=(190, 160, 60),
       street=(240, 230, 200), ink=(220, 200, 110),
       pitch_x=64, pitch_y=64, street_w=8),

    # ---- 저대비 / 배경 반전 ----
    _S("15_lowcontrast", bg=(96, 96, 96), die=(128, 126, 124),
       street=(118, 116, 114), ink=(136, 134, 132),
       pitch_x=64, pitch_y=64, street_w=7, noise_sigma=1.5),
    _S("16_bright_bg", bg=(252, 250, 248), die=(70, 66, 62),
       street=(150, 146, 142), ink=(40, 38, 36),
       pitch_x=64, pitch_y=64, street_w=6),

    # ---- 구조 변화 ----
    _S("17_rim_ring", bg=(0, 0, 0), die=(118, 112, 104), street=(40, 38, 36),
       ink=(175, 168, 158), rim=(210, 215, 220),
       pitch_x=64, pitch_y=64, street_w=6),
    _S("18_rot_p3", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=64, pitch_y=64, street_w=6,
       angle_deg=3.0),
    _S("19_rot_m2", bg=(250, 250, 250), die=(140, 138, 132),
       street=(48, 46, 44), ink=(200, 196, 188),
       pitch_x=68, pitch_y=68, street_w=7, angle_deg=-2.2),
    _S("20_aniso_pitch", bg=(0, 0, 0), die=(110, 120, 130),
       street=(36, 40, 44), ink=(170, 180, 190),
       pitch_x=76, pitch_y=48, street_w=6),
    _S("21_small_pitch", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=28, pitch_y=28, street_w=3),
    _S("22_large_pitch", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=150, pitch_y=150, street_w=13),
    _S("23_notch_left", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=64, pitch_y=64, street_w=6,
       notch_deg=180.0),

    # ---- 촬영 열화 ----
    _S("24_heavy_noise", bg=(0, 0, 0), die=(120, 110, 96), street=(44, 42, 40),
       ink=(172, 160, 146), pitch_x=64, pitch_y=64, street_w=6,
       noise_sigma=20),
    _S("25_jpeg_low", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=64, pitch_y=64, street_w=6, jpeg_q=35),
    _S("26_illum_strong", bg=(14, 14, 14), die=(120, 110, 96),
       street=(38, 36, 34), ink=(178, 165, 148),
       pitch_x=64, pitch_y=64, street_w=6, illum=0.45),
    _S("27_blur_soft", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), pitch_x=64, pitch_y=64, street_w=6, blur=2.2),
    _S("28_speckle_die", bg=(0, 0, 0), die=(120, 110, 96), street=(38, 36, 34),
       ink=(178, 165, 148), speckle=(200, 210, 220), speckle_amt=0.35,
       speckle_on="die", pitch_x=64, pitch_y=64, street_w=6),

    # ---- 복합 최악 조건 ----
    _S("29_worst_case", bg=(248, 246, 244), die=(150, 152, 148),
       street=(196, 198, 194), ink=(132, 134, 130),
       speckle=(60, 96, 150), speckle_amt=0.45, speckle_on="all",
       pitch_x=64, pitch_y=64, street_w=7,
       angle_deg=1.6, noise_sigma=9, illum=0.25, jpeg_q=60),
]

PALETTE_BY_NAME = {s.name: s for s in PALETTES}


# ---------------------------------------------------------------------------
# 정답 대조 검증
# ---------------------------------------------------------------------------
TOL = {
    "pitch": 0.010,      # 상대 1.0%
    "center": 15.0,      # px (회전 있으면 25px)
    "radius": 0.030,     # 상대 3%
    "angle": 0.50,       # deg
    "phase": 0.25,       # pitch 대비 (street 안에 떨어졌는가)
}


def _circ_err(a: float, b: float, period: float) -> float:
    """주기 period 안에서의 최소 거리."""
    d = abs(a - b) % period
    return min(d, period - d)


def _score(det: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, Any]:
    """검출 결과 vs 정답 -> 항목별 오차와 통과 여부."""
    rot = abs(truth["angle_deg"]) > 1e-6
    e: Dict[str, Any] = {}
    e["d_pitch_x"] = abs(det["pitch_x"] - truth["pitch_x"]) / truth["pitch_x"]
    e["d_pitch_y"] = abs(det["pitch_y"] - truth["pitch_y"]) / truth["pitch_y"]
    e["d_center"] = math.hypot(det["cx"] - truth["cx"], det["cy"] - truth["cy"])
    e["d_radius"] = abs(det["r"] - truth["r"]) / truth["r"]
    # 회전 부호 규약이 구현마다 달라 절대값 기준으로 비교한다
    e["d_angle"] = min(abs(det["angle"] - truth["angle_deg"]),
                       abs(det["angle"] + truth["angle_deg"]))
    if not rot and det.get("x0") is not None:
        e["d_phase_x"] = _circ_err(det["x0"], truth["street_x_mod"],
                                   truth["pitch_x"]) / truth["pitch_x"]
        e["d_phase_y"] = _circ_err(det["y0"], truth["street_y_mod"],
                                   truth["pitch_y"]) / truth["pitch_y"]
    else:
        e["d_phase_x"] = e["d_phase_y"] = float("nan")

    ok = (e["d_pitch_x"] <= TOL["pitch"] and e["d_pitch_y"] <= TOL["pitch"]
          and e["d_center"] <= (25.0 if rot else TOL["center"])
          and e["d_radius"] <= TOL["radius"]
          and e["d_angle"] <= TOL["angle"])
    if not math.isnan(e["d_phase_x"]):
        ok = ok and e["d_phase_x"] <= TOL["phase"] \
                and e["d_phase_y"] <= TOL["phase"]
    e["pass"] = bool(ok)
    return e


def _run_v6(img: np.ndarray) -> Dict[str, Any]:
    import wafer_color_v6_claude as v6
    dm = v6.build_die_map_v6(img)
    return {"pitch_x": float(dm.pitch_x), "pitch_y": float(dm.pitch_y),
            "cx": int(dm.wafer_cx), "cy": int(dm.wafer_cy),
            "r": int(dm.wafer_r), "angle": float(dm.rotation_deg),
            "x0": float(dm.x0), "y0": float(dm.y0),
            "n_dies": int(dm.num_dies),
            "ok": bool(dm.diagnostics.ok)}


def _run_v5(img: np.ndarray) -> Dict[str, Any]:
    import wafer_die_map_v5 as v5
    dm = v5.build_die_map(img)
    return {"pitch_x": float(dm.pitch_x), "pitch_y": float(dm.pitch_y),
            "cx": int(dm.wafer_cx), "cy": int(dm.wafer_cy),
            "r": int(dm.wafer_r),
            "angle": float(getattr(dm, "rotation_deg", 0.0)),
            "x0": float(dm.x0), "y0": float(dm.y0),
            "n_dies": int(len(dm.dies)), "ok": True}


def evaluate(specs: Sequence[WaferSpec], out_dir: Path,
             with_v5: bool = False, save_overlay: bool = False,
             verbose: bool = True) -> Dict[str, Any]:
    """합성 -> 검출 -> 정답 대조. 결과 dict 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    engines: List[Tuple[str, Any]] = [("v6", _run_v6)]
    if with_v5:
        engines.append(("v5", _run_v5))

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        img, truth = synth_wafer(spec)
        path = out_dir / f"{spec.name}.png"
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(str(path))

        row: Dict[str, Any] = {"name": spec.name, "truth": truth}
        for eng, fn in engines:
            try:
                det = fn(img)
                row[eng] = {"det": det, "err": _score(det, truth)}
            except Exception as ex:                       # noqa: BLE001
                row[eng] = {"error": f"{type(ex).__name__}: {ex}"}
        rows.append(row)
        if verbose:
            print(_row_line(row, [e for e, _ in engines]), flush=True)

        if save_overlay:
            try:
                import wafer_color_v6_claude as v6
                dm = v6.build_die_map_v6(img)
                v6.save_debug_overlay(
                    dm, str(out_dir / f"{spec.name}__v6.png"))
            except Exception:                             # noqa: BLE001
                pass

    return {"rows": rows, "engines": [e for e, _ in engines]}


# ---------------------------------------------------------------------------
# 표 출력
# ---------------------------------------------------------------------------
_W_NAME = 22
_CELL = 46


def _cell(entry: Dict[str, Any]) -> str:
    if "error" in entry:
        return f"ERR {entry['error']}"[:_CELL].ljust(_CELL)
    e, d = entry["err"], entry["det"]
    ph = "  -  " if math.isnan(e["d_phase_x"]) else \
         f"{max(e['d_phase_x'], e['d_phase_y']) * 100:4.0f}%"
    return (f"{'PASS' if e['pass'] else 'FAIL'} "
            f"p{max(e['d_pitch_x'], e['d_pitch_y']) * 100:5.2f}% "
            f"c{e['d_center']:5.1f} r{e['d_radius'] * 100:4.1f}% "
            f"a{e['d_angle']:4.2f} ph{ph} n{d['n_dies']:5d}").ljust(_CELL)


def _row_line(row: Dict[str, Any], engines: Sequence[str]) -> str:
    return (" " + row["name"][:_W_NAME].ljust(_W_NAME) + "| " +
            "| ".join(_cell(row[e]) for e in engines))


def _header(engines: Sequence[str]) -> str:
    w = _W_NAME + 3 + len(engines) * (_CELL + 2)
    top = "=" * w
    h1 = (" " + "spec".ljust(_W_NAME) + "| " +
          "| ".join(e.upper().ljust(_CELL) for e in engines))
    h2 = (" " + "".ljust(_W_NAME) + "| " +
          "| ".join("verdict pitch  center radius angle phase  dies"
                    .ljust(_CELL) for _ in engines))
    return f"{top}\n{h1}\n{h2}\n{'-' * w}"


def _summary(res: Dict[str, Any]) -> str:
    engines = res["engines"]
    w = _W_NAME + 3 + len(engines) * (_CELL + 2)
    parts = []
    for e in engines:
        tot = len(res["rows"])
        n = sum(1 for r in res["rows"]
                if "err" in r.get(e, {}) and r[e]["err"]["pass"])
        parts.append(f"{n}/{tot} pass ({n / max(1, tot) * 100:.0f}%)"
                     .ljust(_CELL))
    return ("-" * w + "\n " + "TOTAL".ljust(_W_NAME) + "| " +
            "| ".join(parts) + "\n" + "=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        prog="make_color_wafers_claude",
        description="정답을 아는 합성 웨이퍼로 색상 강인성을 검증한다")
    ap.add_argument("--out", default="synth", help="출력 디렉터리")
    ap.add_argument("--eval", action="store_true", help="검출까지 수행")
    ap.add_argument("--v5", action="store_true", help="V5 도 같이 비교")
    ap.add_argument("--overlay", action="store_true", help="오버레이 저장")
    ap.add_argument("--only", nargs="*", help="특정 팔레트만")
    ap.add_argument("--size", type=int, default=0, help="캔버스 크기 덮어쓰기")
    ap.add_argument("--json", help="결과 JSON 경로")
    ap.add_argument("--list", action="store_true", help="팔레트 목록")
    a = ap.parse_args(argv)

    if a.list:
        for s in PALETTES:
            print(f"  {s.name:22s} pitch=({s.pitch_x:g},{s.pitch_y:g}) "
                  f"die={s.die} street={s.street} angle={s.angle_deg:+g}")
        return 0

    specs = PALETTES
    if a.only:
        want = set(a.only)
        specs = [s for s in PALETTES if s.name in want]
        if not specs:
            print("[error] 일치하는 팔레트 없음", file=sys.stderr)
            return 2
    if a.size:
        import copy
        specs = [copy.replace(s, size=a.size) if hasattr(copy, "replace")
                 else s for s in specs]

    out = Path(a.out)
    if not a.eval:
        out.mkdir(parents=True, exist_ok=True)
        for s in specs:
            img, _ = synth_wafer(s)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                buf.tofile(str(out / f"{s.name}.png"))
            print(f"  {s.name}.png")
        print(f"\n{len(specs)} images -> {out}")
        return 0

    engines = ["v6"] + (["v5"] if a.v5 else [])
    print(f" SYNTHETIC GROUND-TRUTH HARNESS   {len(specs)} palettes")
    print(_header(engines))
    res = evaluate(specs, out, with_v5=a.v5, save_overlay=a.overlay)
    print(_summary(res))

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str),
                                encoding="utf-8")
        print(f"JSON -> {a.json}")
    n_fail = sum(1 for r in res["rows"]
                 if "err" not in r.get("v6", {}) or not r["v6"]["err"]["pass"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
