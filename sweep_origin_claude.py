# -*- coding: utf-8 -*-
"""sweep_origin_claude.py — 격자 원점 (x0, y0) 파라미터 민감도 스윕.

목적
----
"어느 파라미터를 건드리면 원점이 움직이는가"를 **측정으로** 답한다.
코드를 읽고 추측하는 대신, 파라미터를 하나씩 바꿔가며
`build_die_map_v6()` 를 실제로 돌리고 baseline 대비 원점 이동량을 표로 낸다.

읽는 법
-------
원점은 pitch 의 정수배만큼 애매하다 (격자선은 여러 개, 그중 하나를 고를 뿐).
그래서 이동량을 두 가지로 쪼개서 본다:

  dx        = x0(variant) - x0(baseline)            [px]
  lat       = round(dx / pitch)                     [격자 몇 칸 이동했나]
  sub       = dx - lat*pitch                        [칸 안에서 얼마나 밀렸나, px]

  * lat != 0  -> **다른 격자선을 골랐다**. die 인덱스가 통째로 밀린다. 치명적.
  * lat == 0, |sub| 작음 -> 같은 선, 미세한 수치 차이. 무해.

사용법
------
    python sweep_origin_claude.py Img/real_casio_top_p092.png
    python sweep_origin_claude.py Img/*.png --out sweep_out
    python sweep_origin_claude.py Img/real_casio_top_p092.png --md report.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import wafer_color_v6_claude as v6

# Windows 콘솔(cp949)에서 em-dash 등이 깨지지 않도록. 파일 저장은 항상 utf-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
    except Exception:                                        # noqa: BLE001
        pass


# =============================================================================
# 스윕 정의 — (그룹, 라벨, ColorProfile 오버라이드 dict, build 인자 dict)
# =============================================================================
def _variants() -> List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    """돌려볼 조합 목록.

    그룹은 '원점 파이프라인의 어느 단계를 건드리는가' 기준으로 묶었다.
    """
    V: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []

    def add(group: str, label: str, prof: Dict[str, Any] = None,
            build: Dict[str, Any] = None) -> None:
        V.append((group, label, prof or {}, build or {}))

    # -- A) 전처리 (원점 검출 *이전* 단계) --------------------------------
    add("A preprocess", "align_angle=False", {}, {"align_angle": False})
    add("A preprocess", "clean=False", {}, {"clean": False})
    add("A preprocess", "angle_max_iter=1", {"angle_max_iter": 1})
    add("A preprocess", "angle_max_iter=6", {"angle_max_iter": 6})
    add("A preprocess", "angle_fine_step=0.005", {"angle_fine_step": 0.005})
    add("A preprocess", "angle_fine_step=0.10", {"angle_fine_step": 0.10})

    # -- B) ROI / 해상도 (_grid_roi, _downscale) --------------------------
    for r in (0.45, 0.55, 0.70, 0.80):
        add("B roi/scale", f"roi_ratio={r}", {"roi_ratio": r})
    for d in (512, 1024, 3072):
        add("B roi/scale", f"profile_max_dim={d}", {"profile_max_dim": d})

    # -- C) pitch 탐색 밴드 (_spectral_pitch) -----------------------------
    for m in (4.0, 16.0, 32.0):
        add("C pitch band", f"min_pitch_px={m}", {"min_pitch_px": m})
    for m in (0.15, 0.25, 0.50):
        add("C pitch band", f"max_pitch_ratio={m}", {"max_pitch_ratio": m})

    # -- D) 채널 합의 (_consensus_pitch / min_channel_score) --------------
    for t in (0.01, 0.10, 0.20):
        add("D consensus", f"pitch_cluster_tol={t}", {"pitch_cluster_tol": t})
    for s in (0.0, 0.30, 0.60, 0.90):
        add("D consensus", f"min_channel_score={s}", {"min_channel_score": s})

    # -- E) 채널 구성 (_feature_bank) — 원점의 '재료' --------------------
    for nm in v6.FEATURE_NAMES:
        add("E channels", f"only [{nm}]", {"feature_channels": (nm,)})
    add("E channels", "no stdL",
        {"feature_channels": tuple(n for n in v6.FEATURE_NAMES if n != "stdL")})
    add("E channels", "L only-ish [L,stdL]", {"feature_channels": ("L", "stdL")})
    add("E channels", "color only [a,b,chroma,sat]",
        {"feature_channels": ("a", "b", "chroma", "sat")})

    # -- F) street 극성 (_street_from_template) --------------------------
    add("F polarity", "street_polarity='dark'", {"street_polarity": "dark"})
    add("F polarity", "street_polarity='bright'", {"street_polarity": "bright"})

    # -- G) pitch 강제 지정 ----------------------------------------------
    add("G fixed pitch", "pitch_x/y = baseline round()", {})  # 런타임에 채움

    return V


# =============================================================================
# 실행
# =============================================================================
def _run(path: str, prof_kw: Dict[str, Any],
         build_kw: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    """한 조합 실행. 성공하면 (DieMap, ""), 실패하면 (None, 사유)."""
    try:
        cfg = replace(v6.ColorProfile(), **prof_kw)
        return v6.build_die_map_v6(path, profile=cfg, **build_kw), ""
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _fmt_delta(d: float, pitch: float) -> Tuple[int, float]:
    """이동량을 (격자 칸 수, 칸 안 잔차 px) 로 분해."""
    if pitch <= 0:
        return 0, d
    lat = int(round(d / pitch))
    return lat, d - lat * pitch


def sweep_one(path: str, verbose: bool = True) -> Dict[str, Any]:
    """이미지 한 장에 대해 전체 스윕. 결과 dict 반환."""
    name = os.path.splitext(os.path.basename(path))[0]
    if verbose:
        print(f"\n=== {name} ===", flush=True)

    t0 = time.time()
    base, err = _run(path, {}, {})
    if base is None:
        print(f"  [FAIL] baseline: {err}")
        return {"name": name, "baseline": None, "rows": []}
    bx, by, px_, py_ = base.x0, base.y0, base.pitch_x, base.pitch_y
    if verbose:
        print(f"  baseline: origin=({bx},{by})  pitch=({px_:.3f},{py_:.3f})  "
              f"dies={len(base.dies)}  [{time.time()-t0:.1f}s]", flush=True)

    rows: List[Dict[str, Any]] = []
    for group, label, prof_kw, build_kw in _variants():
        if label.startswith("pitch_x/y = baseline"):
            prof_kw = {"pitch_x": round(px_), "pitch_y": round(py_)}
            label = f"pitch_x={round(px_)}, pitch_y={round(py_)}"

        dm, e = _run(path, prof_kw, build_kw)
        if dm is None:
            rows.append({"group": group, "label": label, "error": e})
            if verbose:
                print(f"  {label:34s} FAIL  {e[:60]}", flush=True)
            continue

        dx, dy = dm.x0 - bx, dm.y0 - by
        latx, subx = _fmt_delta(dx, px_)
        laty, suby = _fmt_delta(dy, py_)
        row = {
            "group": group, "label": label, "error": "",
            "x0": dm.x0, "y0": dm.y0, "dx": dx, "dy": dy,
            "lat_x": latx, "lat_y": laty, "sub_x": subx, "sub_y": suby,
            "pitch_x": dm.pitch_x, "pitch_y": dm.pitch_y,
            "d_pitch_x": dm.pitch_x - px_, "d_pitch_y": dm.pitch_y - py_,
            "dies": len(dm.dies), "d_dies": len(dm.dies) - len(base.dies),
            "shift": (latx != 0 or laty != 0),
            # 칸 안 잔차를 pitch 비율로. 0.10 넘으면 street 중심을 놓쳤다고 본다.
            "drift": max(abs(subx) / max(px_, 1e-9), abs(suby) / max(py_, 1e-9)),
            # pitch 자체가 바뀌면 origin 이 우연히 같아도 '다른 격자'다.
            # (실측: white_brown 에서 only[a] 가 pitch 를 절반으로 잡아
            #  die 가 865 -> 1739 가 됐는데 origin 은 1px 차이라
            #  origin 만 보면 '안정'으로 오판된다.)
            "pitch_err": max(abs(dm.pitch_x / max(px_, 1e-9) - 1.0),
                             abs(dm.pitch_y / max(py_, 1e-9) - 1.0)),
        }
        rows.append(row)
        if verbose:
            flag = (" <<< PITCH BROKEN" if row["pitch_err"] > 0.05
                    else (" <<< LATTICE SHIFT" if row["shift"]
                          else (" <<< DRIFT" if row["drift"] > 0.10 else "")))
            print(f"  {label:34s} origin=({dm.x0:5d},{dm.y0:5d}) "
                  f"lat=({latx:+d},{laty:+d}) sub=({subx:+6.2f},{suby:+6.2f}) "
                  f"dies={len(dm.dies):5d}{flag}", flush=True)

    return {"name": name, "baseline": {
        "x0": bx, "y0": by, "pitch_x": px_, "pitch_y": py_,
        "dies": len(base.dies)}, "rows": rows}


# =============================================================================
# 리포트
# =============================================================================
def to_markdown(results: List[Dict[str, Any]]) -> str:
    L: List[str] = []
    L.append("# 격자 원점 (x0, y0) 파라미터 민감도")
    L.append("")
    L.append("`sweep_origin_claude.py` 자동 생성. baseline = `ColorProfile()` 기본값.")
    L.append("")
    L.append("- `lat` = 원점이 **격자 몇 칸** 움직였나 (0 이 아니면 die 인덱스가 통째로 밀림)")
    L.append("- `sub` = 같은 칸 안에서의 잔차 px (작으면 무해)")
    L.append("- `Δpitch` = pitch 변화. 이게 깨지면 origin 이 같아도 **다른 격자**다.")
    L.append("- `Δdies` = 검출 die 개수 변화")
    L.append("")

    for res in results:
        b = res["baseline"]
        L.append(f"## {res['name']}")
        L.append("")
        if b is None:
            L.append("baseline 자체가 실패했습니다.")
            L.append("")
            continue
        L.append(f"baseline: origin=({b['x0']}, {b['y0']}) "
                 f"pitch=({b['pitch_x']:.3f}, {b['pitch_y']:.3f}) "
                 f"dies={b['dies']}")
        L.append("")
        cur = None
        for r in res["rows"]:
            if r["group"] != cur:
                cur = r["group"]
                L.append("")
                L.append(f"### {cur}")
                L.append("")
                L.append("| 파라미터 | origin | lat (x,y) | sub px (x,y) | Δpitch | Δdies | 판정 |")
                L.append("|---|---|---|---|---|---|---|")
            if r.get("error"):
                L.append(f"| `{r['label']}` | — | — | — | — | — | **FAIL** {r['error'][:50]} |")
                continue
            if r["pitch_err"] > 0.05:
                verdict = f"**pitch 붕괴 {r['pitch_err']*100:.0f}%**"
            elif r["shift"]:
                verdict = "**격자 이동 (치명)**"
            elif r["drift"] > 0.10:
                verdict = f"**드리프트 {r['drift']*100:.0f}% pitch**"
            elif abs(r["sub_x"]) < 1 and abs(r["sub_y"]) < 1:
                verdict = "무해"
            else:
                verdict = "미세 이동"
            L.append(
                f"| `{r['label']}` | ({r['x0']}, {r['y0']}) "
                f"| ({r['lat_x']:+d}, {r['lat_y']:+d}) "
                f"| ({r['sub_x']:+.2f}, {r['sub_y']:+.2f}) "
                f"| ({r['d_pitch_x']:+.3f}, {r['d_pitch_y']:+.3f}) "
                f"| {r['d_dies']:+d} | {verdict} |")
        L.append("")
    return "\n".join(L) + "\n"


def to_summary(results: List[Dict[str, Any]]) -> str:
    """이미지 전체를 가로질러 '이 파라미터는 몇 장에서 격자를 움직였나' 집계."""
    tally: Dict[str, Dict[str, int]] = {}
    order: List[str] = []
    for res in results:
        for r in res["rows"]:
            key = f"{r['group']} :: {r['label']}"
            if key not in tally:
                tally[key] = {"pitch": 0, "shift": 0, "drift": 0,
                              "fail": 0, "n": 0}
                order.append(key)
            t = tally[key]
            t["n"] += 1
            # 우선순위: fail > pitch 붕괴 > 격자 이동 > 드리프트.
            # pitch 가 깨지면 origin 이 우연히 같아도 격자 자체가 다르므로
            # origin 기준 판정보다 먼저 걸러야 한다.
            if r.get("error"):
                t["fail"] += 1
            elif r["pitch_err"] > 0.05:
                t["pitch"] += 1
            elif r["shift"]:
                t["shift"] += 1
            elif r["drift"] > 0.10:
                t["drift"] += 1

    n_img = len(results)
    L = ["", "=" * 84,
         f"ORIGIN SENSITIVITY  ({n_img} images)",
         "  pitch = pitch itself changed >5% (grid is a DIFFERENT grid; worst)",
         "  shift = origin jumped to a DIFFERENT grid line (die index shifts)",
         "  drift = same line but street center off by >10% of pitch",
         "=" * 84,
         f"{'parameter':<50s} {'pitch':>7s} {'shift':>7s} {'drift':>7s} {'fail':>5s}"]
    quiet: List[str] = []
    for key in order:
        t = tally[key]
        if t["pitch"] or t["shift"] or t["drift"] or t["fail"]:
            L.append(f"{key:<50s} {t['pitch']:>5d}/{t['n']:<1d} "
                     f"{t['shift']:>5d}/{t['n']:<1d} "
                     f"{t['drift']:>5d}/{t['n']:<1d} {t['fail']:>5d}")
        else:
            quiet.append(key)
    L.append("")
    L.append(f"stable on all {n_img} images ({len(quiet)} params):")
    for key in quiet:
        L.append(f"  - {key}")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="격자 원점 파라미터 민감도 스윕")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--md", default="", help="마크다운 리포트 저장 경로")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    results = []
    for p in a.images:
        if not os.path.exists(p):
            print(f"[skip] not found: {p}")
            continue
        try:
            results.append(sweep_one(p, verbose=not a.quiet))
        except Exception:                                    # noqa: BLE001
            traceback.print_exc()

    print(to_summary(results))
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(to_markdown(results))
        print(f"\n[ok] markdown -> {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
