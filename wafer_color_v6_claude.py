# -*- coding: utf-8 -*-
"""
wafer_color_v6_claude.py  --  Color-agnostic Wafer Die Map (V6)
================================================================

작성: Claude (Anthropic)  /  V5(`wafer_die_map_v5.py`) 를 참고해 새로 작성.
**V5 를 import 하지 않는 완전 독립(standalone) 모듈입니다.**
(필요한 V5 헬퍼는 통째로 복사해 넣었습니다.)

왜 V6 인가
----------
V5 는 "wafer = 밝음 / 배경 = 검정", "street(die 사이) = 특정 밝기·채도 범위"
라는 **절대 색 임계값**에 의존한다. 대표적으로

    detect_wafer()        : gray > 20                      (배경=검정 가정)
    _street_color_mask()  : brightness>115, max(BGR)>130,
                            35 <= max(BGR)-min(BGR) <= 130 (street=유채색 가정)
    clean_wafer/detect_notch : black_thr=20

따라서 die 사이가 **흰색(무채색)** 이 되면 max-min ≈ 0 이라 street 마스크가
통째로 비어버리고, 배경이 흰색이면 wafer 검출부터 실패한다.

V6 의 원리 : "색"이 아니라 "주기성(periodicity)"으로 격자를 찾는다
------------------------------------------------------------------
die 격자는 **어떤 팔레트에서도 공간적으로 주기적**이다. 그래서 V6 는

 1) 배경색을 이미지 코너에서 **자동 추정** -> Lab 거리 + Otsu 로 wafer 검출
    (배경이 검정/흰색/갈색/어떤 색이든 무관)
 2) 색 공간을 여러 **feature 채널** 로 펼친다
        L, a, b, chroma, saturation, max(BGR)-min(BGR), local-std(L)
    - 흰 street  <-> 유채색 die   : chroma / sat / maxmin 채널이 강하게 반응
    - 검은 street <-> 밝은 die    : L 채널이 반응
    - 같은 색 but 질감만 다름     : stdL 채널이 반응
 3) 각 채널의 1D 프로파일에서 **FFT 로 pitch**, **Fourier 위상으로 원점**을 구하고
    lag-autocorrelation 으로 **주기성 점수**를 매긴다.
 4) 점수가 높은 채널만 뽑아 **pitch 합의(clustering) + 위상 원형평균(circular mean)**
    -> 어떤 색이 street 인지 사람이 알려줄 필요가 없다.
 5) street 극성(밝은 street / 어두운 street)은 한 주기로 접은(folded) 템플릿의
    **폭 비율 + 왜도(skewness)** 로 자동 판별한다.
 6) 회전각은 Lab 3채널 **컬러 그래디언트 에너지** 로 추정(회색조 Sobel 아님)
    + 2D FFT 로 교차검증.

즉 V6 에는 "밝기 115 이상" 같은 절대 색 임계가 **하나도 없다**.

====================================================================
                          사 용 법  (USAGE)
====================================================================

[1] 설치
--------
    pip install numpy opencv-python

    이 파일 하나만 있으면 동작한다. (V5 를 import 하지 않음)
    단, `--robustness` 의 "V5 와 비교" 기능만은 같은 폴더에
    `wafer_die_map_v5.py` 가 있어야 한다. 없으면 비교를 건너뛰고
    V6 결과만 출력한다.


[2] 가장 간단한 사용 -- 완전 자동
--------------------------------
아무 파라미터도 주지 않으면 배경색·street 극성·feature 채널·pitch·
위상·회전각을 전부 스스로 추정한다.

    from wafer_color_v6_claude import build_die_map_v6

    dm = build_die_map_v6("wafer.png")        # 경로 or np.ndarray(BGR)

    print(dm.pitch_x, dm.pitch_y)             # die 한 칸의 픽셀 크기
    print(dm.wafer_cx, dm.wafer_cy, dm.wafer_r)   # 웨이퍼 원 (px)
    print(dm.rotation_deg)                    # 보정한 회전각 (deg)
    print(len(dm.dies))                       # 검출된 die 개수

`dm.dies` 는 **dict 의 리스트** 다. (dataclass 아님. dict 그대로 쓴다)

    for d in dm.dies:
        d["index"]           # (ix, iy)  격자 인덱스. 중심 근처가 (0,0) 부근
        d["center_px"]       # (cx, cy)  die 중심 픽셀 좌표
        d["rect_px"]         # (x0, y0, x1, y1)  die 경계 사각형
        d["crop_rect_px"]    # margin/offset 을 반영한 실제 잘라낼 사각형
        d["real_coord"]      # (X, Y)  pixel_per_unit 로 환산한 실좌표
        d["is_edge_partial"] # 웨이퍼 경계에 걸쳐 잘린 die 인가
        d["is_edge_ring"]    # 최외곽 링에 속하는가
        d["is_edge"]         # 위 둘 중 하나라도 참인가

    # 인덱스로 바로 찾기
    d = dm.dies_by_index[(0, 0)]

    # die 이미지 잘라내기 -> with_crops=True 로 만들면 d["image"] 에 들어온다
    dm = build_die_map_v6("wafer.png", with_crops=True)
    cv2.imwrite("die_0_0.png", dm.dies_by_index[(0, 0)]["image"])


[3] 파라미터를 직접 주고 싶을 때 -- ColorProfile
-----------------------------------------------
아는 값만 넣으면 되고, 넣지 않은(None 인) 값은 전부 자동 추정된다.
즉 "전부 자동" 과 "전부 수동" 사이의 어떤 조합도 가능하다.

    from wafer_color_v6_claude import build_die_map_v6, ColorProfile

    prof = ColorProfile(
        # ---- 색 관련 (None 이면 자동) -------------------------------
        background_bgr   = (255, 255, 255),  # 웨이퍼 바깥 배경색 (B,G,R)
                                             #   None -> 코너 4곳에서 자동 추정
        street_polarity  = "bright",         # "bright" | "dark" | None(=auto)
                                             #   die 사이 street 가 밝은가 어두운가
        feature_channels = ("chroma", "sat"),# 사용할 채널을 직접 지정
                                             #   None -> 주기성 점수로 자동 선택

        # ---- 격자 관련 (None 이면 자동) -----------------------------
        pitch_x = 204.5,                     # 가로 pitch(px). None -> FFT 자동
        pitch_y = 204.5,                     # 세로 pitch(px). None -> FFT 자동

        # ---- 탐색 범위 조절 (기본값으로 충분한 경우가 대부분) --------
        min_pitch_px     = 8.0,   # 이보다 작은 주기는 노이즈로 보고 무시
        max_pitch_ratio  = 0.34,  # 최대 pitch = 웨이퍼 지름 * 이 비율
        roi_ratio        = 0.62,  # 프로파일을 뽑을 중심 ROI 크기(지름 대비)
        min_channel_score= 0.08,  # 이 점수 미만인 채널은 합의에서 제외
        pitch_cluster_tol= 0.04,  # pitch 합의 시 같은 값으로 묶는 허용오차(4%)

        # ---- 회전각 탐색 --------------------------------------------
        angle_search_deg = 6.0,   # +-6도 범위를 훑는다
        angle_coarse_step= 0.15,
        angle_fine_step  = 0.02,
        angle_full_scan_deg = 20.0,  # 6도 안에서 못 찾으면 +-20도로 확대

        # ---- 웨이퍼 마스크 모폴로지 ---------------------------------
        wafer_open_ksize  = 9,    # 점 노이즈 제거
        wafer_close_ksize = 25,   # 구멍 메우기 시작 커널
                                  #   (부족하면 내부에서 자동으로 키운다)
        wafer_fill_bgr    = None, # 웨이퍼 안을 특정 색으로 칠할 때
    )
    dm = build_die_map_v6("wafer.png", profile=prof)

feature_channels 로 쓸 수 있는 이름과 반응하는 상황:

    "L"       밝기        : 검은 street <-> 밝은 die  (가장 흔함)
    "a"       녹-적 축    : 색상이 반대쪽으로 갈리는 경우
    "b"       청-황 축    : 파란 street <-> 노란 die 등
    "chroma"  Lab 채도    : 흰색(무채) street <-> 유채색 die  ★
    "sat"     HSV 채도    : chroma 와 비슷하나 밝기 영향이 다름
    "maxmin"  max-min BGR : 무채/유채 대비. V5 가 쓰던 값과 같은 축
    "stdL"    국소 표준편차: 색은 같은데 질감만 다른 경우 ★

    ※ 그래디언트(Sobel) 채널은 일부러 넣지 않았다. street 하나가
      "양쪽 두 개의 엣지" 로 보여서 위상이 모호해지고 pitch 가 절반으로
      잘못 잡히기 때문이다.

일부만 바꾸고 싶으면 `.merged()` 가 편하다.

    prof2 = prof.merged(pitch_x=100.0, street_polarity=None)


[4] build_die_map_v6() 의 나머지 인자
------------------------------------
    build_die_map_v6(
        image,                    # 파일 경로(str) 또는 BGR np.ndarray
        profile      = None,      # 위 ColorProfile. None -> 완전 자동
        pixel_per_unit = 32,      # real_coord 환산 계수 (1 unit = 32 px)
        include_edge = True,      # 경계에 걸친 die 도 포함할 것인가
        edge_margin  = 1.0,       # 경계 판정 여유(die 크기 배수)
        edge_mode    = "circle",  # is_edge 를 무엇으로 정할 것인가
                                  #   "circle": 웨이퍼 원에 걸쳐 잘린 die
                                  #             (= is_edge_partial)
                                  #   "ring"  : 격자의 최외곽 링에 있는 die
                                  #             (= is_edge_ring)
                                  #   "both"  : 둘 중 하나라도 참이면 edge
        with_crops   = False,     # True 면 각 die 이미지를 d["image"] 에 넣는다
        border_mode  = "pad",     # "pad" | "crop"  (자를 영역이 이미지 밖일 때)
                                  #   pad : 부족한 부분을 0 으로 채워 크기 유지
                                  #   crop: 이미지 안쪽만 잘라 크기가 작아짐
        offset_x = 0, offset_y = 0,   # 잘라낼 사각형을 통째로 이동
        margin_x = 0, margin_y = 0,   # 잘라낼 사각형을 상하좌우로 확장
        clean        = True,      # 웨이퍼 마스크 정리(구멍/돌기 제거)
        align_angle  = True,      # 회전 보정 수행 여부
        notch_ref_deg   = 90.0,   # notch 가 있어야 할 기준 방향(deg)
        notch_sector_deg= 70.0,   # notch 를 찾을 각도 범위
        verify_angle = True,      # 2D FFT 로 회전각 교차검증
        verify_tol_deg = 0.5,     # 교차검증 허용 오차
    ) -> WaferDieMapV6

반환 객체 `WaferDieMapV6` 의 주요 속성:

    wafer_cx, wafer_cy, wafer_r     웨이퍼 원 (회전 보정 후 좌표계)
    pitch_x, pitch_y                die pitch (px)
    x0, y0                          격자 원점 (die (0,0) 의 기준점)
    die_w, die_h                    die 한 칸 크기 (= pitch 반올림)
    dies, dies_by_index             die 목록 / 인덱스 조회용 dict
    rotation_deg                    적용한 회전각 (deg)
    aligned_image                   회전 보정된 이미지 (BGR)
    wafer_mask                      웨이퍼 마스크 (uint8 0/255)
    notch_center_px                 notch 중심 (없으면 None)
    image_shape                     (H, W)
    die_grid_angle_resid            격자 잔여 기울기 (deg)
    angle_verified, angle_agree     회전각 검증 통과 여부
    angle_confidence                회전각 신뢰도 0~1
    quadrant_report                 사분면별 die 수 (편향 점검용)
    diagnostics                     자가진단 객체 ([5] 참고)

좌표 규약은 V5 와 동일하다 (기존 코드와 호환).

    cx = x0 + ix * pitch_x + pitch_x / 2      # x 는 오른쪽이 +
    cy = y0 - iy * pitch_y - pitch_y / 2      # y 는 위쪽이 +


[5] 검증 3종
------------
(a) 자가진단 리포트 -- 결과를 믿어도 되는지 스스로 점검

    dm = build_die_map_v6("wafer.png")
    print(dm.diagnostics.report())   # 사람이 읽는 텍스트 리포트
    if not dm.diagnostics.ok:
        print(dm.diagnostics.warnings)   # 경고 문자열 리스트
    d = dm.diagnostics.to_dict()         # JSON 으로 저장할 때

    리포트에는 배경색 추정값, 웨이퍼 coverage, 채널별 pitch/위상/점수,
    합의에 참여한 채널 수, 회전각 교차검증 결과가 들어간다.

(b) 디버그 오버레이 이미지 저장 -- 눈으로 확인

    from wafer_color_v6_claude import save_debug_overlay

    save_debug_overlay(dm, "debug.png",
                       max_dim   = 2000,   # 긴 변을 이 크기로 축소
                       draw_grid = True,   # 격자선
                       draw_dies = True,   # die 사각형 (경계 die 는 다른 색)
                       draw_panel= True)   # 좌상단 정보 패널

    ※ 경로에 한글이 있으면 cv2.imwrite 가 실패하므로 내부에서
      cv2.imencode + 파일 쓰기로 우회한다. 그냥 쓰면 된다.

(c) 색상 변형 테스트 하네스 -- 같은 웨이퍼를 15가지로 물들여 재검출

    from wafer_color_v6_claude import run_color_robustness_test

    res = run_color_robustness_test(
        "wafer.png",
        variants    = None,   # None = 전부. 이름 리스트로 일부만 지정 가능
        compare_v5  = True,   # 같은 폴더에 v5 있으면 함께 돌려 비교
        pitch_tol   = 0.02,   # pitch 가 원본 대비 2% 안이면 통과
        dies_tol    = 0.03,   # die 개수가 3% 안이면 통과
        overlay_dir = "out",  # 변형별 오버레이도 저장
    )

    쓸 수 있는 variant 이름(15종, `COLOR_VARIANTS` 의 키):
        original            원본 그대로 (기준값을 만든다)
        invert              BGR 반전 -- street/die 명암이 뒤집힌다
        gray                완전 무채색화 -- chroma/sat 채널이 죽는다
        hue+60, hue+150     색상환 회전 -- a/b 축이 통째로 바뀐다
        desat0.25           채도를 1/4 로
        gamma0.5, gamma2.0  감마 -- 밝기 분포가 비선형으로 눌린다
        lowcontrast         대비 축소
        brightbg            배경만 밝게 -- 배경 자동추정을 흔든다
        sepia               세피아 톤
        cool_tint           푸른 톤
        noise12             가우시안 노이즈 sigma=12
        illum_grad          조명 기울기 -- detrend 가 버티는지 본다
        white_street_brown  die 사이를 흰색+갈색 노이즈로 (V5 가 실패하는 케이스)
    목록은 CLI 로도 볼 수 있다:  --list-variants

    ※ 이 하네스는 "원본 자기 자신" 을 기준으로 한 **상대 비교** 다.
      **절대 정답(ground truth)** 과 대조하려면 같은 폴더의
      `make_color_wafers_claude.py` 를 쓴다. 29가지 팔레트의 웨이퍼를
      pitch/중심/반지름/회전각/street 위상을 지정해 직접 합성하므로
      검출값을 절대 기준과 비교할 수 있다.

          python make_color_wafers_claude.py --out synth/ --eval --v5


[6] 특정 좌표가 어느 die 인지 찾기
---------------------------------
    from wafer_color_v6_claude import locate_die_v6

    info = locate_die_v6(dm, point=(1234, 567))            # 점으로 조회
    info = locate_die_v6(dm, bbox=(1200, 540, 1300, 620))  # 사각형으로 조회
                                                           # (bbox 는 중심점으로 환산)

반환은 **dict 하나** 다. (겹치는 die 목록이 아니라, 질의 지점이 속한
die 하나를 돌려준다)

    info["die_index"]        # (ix, iy)   격자 인덱스   ★ "index" 아님
    info["die_center_px"]    # (cx, cy)   die 중심 픽셀
    info["die_rect_px"]      # (x0,y0,x1,y1) die 경계 사각형
    info["crop_rect_px"]     # offset/margin 반영한 잘라낼 사각형
    info["die_real_coord"]   # die 중심의 실좌표
    info["real_coord"]       # 질의 지점의 실좌표
    info["real_distance"]    # 웨이퍼 중심에서의 실거리
    info["query_px"]         # 실제로 사용된 질의 픽셀 좌표
    info["corner_px"]        # 격자 원점 (x0, y0) — die 좌상단이 **아니다**
                             #   웨이퍼 중심에 가장 가까운 street 교차점이며
                             #   질의 지점과 무관하게 항상 같은 값. (v5 와 동일)
    info["in_wafer"]         # 웨이퍼 원 안인가
    info["is_edge_partial"] / ["is_edge_ring"] / ["is_edge"]
    info["edge_mode"]        # 판정에 쓰인 edge_mode
    info["input_type"]       # "point" | "bbox"
    info["wafer_center_px"]  # 웨이퍼 중심

    offset_x / offset_y / margin_x / margin_y 는 build_die_map_v6 와
    같은 의미이며, 여기서 다시 주면 그 값으로 crop_rect_px 를 계산한다.


[7] CLI
-------
    # 기본 -- 결과 요약 출력
    python wafer_color_v6_claude.py wafer.png

    # 자가진단 리포트 + JSON 저장 + 오버레이 저장
    python wafer_color_v6_claude.py wafer.png --report \
           --json result.json --overlay out_dir/

    # 색상 변형 테스트 (V5 와 비교)
    python wafer_color_v6_claude.py wafer.png --robustness
    python wafer_color_v6_claude.py wafer.png --robustness --no-v5
    python wafer_color_v6_claude.py wafer.png --robustness \
           --variants whitestreet,invert,gray
    python wafer_color_v6_claude.py --list-variants

    # 파라미터 직접 지정
    python wafer_color_v6_claude.py wafer.png \
           --bg 255,255,255 \
           --polarity bright \
           --channels chroma,sat \
           --pitch 204.5              # 또는 --pitch 204.5,198.0
    python wafer_color_v6_claude.py wafer.png --min-pitch 20 --roi 0.5

    # 기타
    --ppu 32            pixel_per_unit
    --edge-mode ring    edge 판정 기준 (circle|ring|both)
    --no-align          회전 보정 끄기
    --no-clean          웨이퍼 마스크 정리 끄기

    ※ --overlay 는 파일이 아니라 **디렉터리** 를 받는다.
      디렉터리를 positional 로 주면 그 안의 이미지를 전부 처리한다.

    python wafer_color_v6_claude.py Img/ --report --overlay out/


[8] 잘 안 될 때 -- 순서대로 해볼 것
----------------------------------
먼저 `--report` 를 찍어서 경고(warnings)를 확인한다. 실제로 나오는
문구와 그에 대한 대처는 다음과 같다.

  "wafer detection fallback: coverage~1 ..."
      웨이퍼 마스크가 이미지 거의 전체를 덮었다.
      = 배경색과 웨이퍼 색이 구분되지 않았다는 뜻.
      -> --bg 로 배경색을 직접 지정한다.  예: --bg 255,255,255

  "wafer detection fallback: coverage~0 (wafer not separable...)"
      마스크가 거의 비었거나 die 단위로 산산조각 났다.
      (내접원으로 대체됐으므로 wafer_r 을 믿으면 안 된다)
      -> scribe lane 이 넓은 경우다. wafer_close_ksize 를 키운다.
         ColorProfile(wafer_close_ksize=41)

  "minEnclosingCircle inflated by rim noise ..."
      웨이퍼 테두리 밖 노이즈가 원을 부풀렸다는 뜻이며,
      이미 무게중심+면적반지름으로 자동 보정했다. 보통 무시해도 된다.

  "[x] weak pitch agreement 0.xx" / "[x] weak phase consensus 0.xx"
      채널들끼리 pitch 나 위상이 서로 다른 값을 말하고 있다.
      = 자동 선택된 feature 채널이 이 팔레트에 안 맞는다.
      -> --channels 로 직접 지정한다.
           흰(무채) street  -> chroma,sat,maxmin
           검은 street      -> L
           색은 같고 질감만 -> stdL
           색상 대비        -> a,b

  "[x] all channels below min_channel_score=0.08"
      어떤 채널에서도 주기성이 안 잡혔다.
      -> --roi 0.8 로 ROI 를 넓히거나 --min-pitch 를 낮춘다.
      -> 그래도 안 되면 --pitch 로 직접 준다.

  pitch 가 정확히 2배 또는 절반으로 나온다
      리포트의 pitch 옆 note 에 "x2 (r=...)" 또는 "/2 (r=...)" 가
      붙는지 본다. 이는 자동 보정이 개입했다는 표시다.
      -> 정답을 알면 --pitch 로 못 박는 것이 가장 확실하다.
      -> --min-pitch 를 실제 pitch 의 70% 정도로 올리면 절반 오검출이
         대부분 사라진다.

  "angle cross-check disagreed ..."
      격자 기반 각도와 2D FFT 각도가 서로 다르다. 회전이 크거나
      격자가 약할 때 나온다.
      -> ColorProfile(angle_search_deg=15) 로 탐색 범위를 넓힌다.
      -> 회전 보정이 필요 없으면 --no-align.

  "residual grid tilt +0.xxx deg"
      회전 보정 후에도 격자가 살짝 기울어 있다. 값이 0.1도 미만이면
      실무상 무시 가능하다.

  "quadrant coverage unbalanced ..."
      사분면별 die 수가 심하게 치우쳤다. 웨이퍼 중심이 잘못 잡혔거나
      이미지가 잘려 있을 가능성.
      -> 오버레이(save_debug_overlay)로 원 위치를 눈으로 확인한다.

  "no dies produced — grid detection is almost certainly wrong"
      격자는 나왔는데 die 가 하나도 안 만들어졌다. 위 경고들 중
      선행 원인이 반드시 같이 찍혀 있다. 그쪽을 먼저 고친다.

예외(RuntimeError)로 죽는 경우:

  "Grid ROI too small — wafer detection likely failed."
      웨이퍼 검출이 실패해 ROI 가 너무 작아졌다.
      -> --bg 로 배경색을 지정한다.

  "Implausible die size: WxH"
      pitch 가 비상식적인 값으로 나왔다.
      -> --min-pitch / --pitch 로 범위를 좁혀 준다.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "ColorProfile",
    "GridDiagnostics",
    "WaferDieMapV6",
    "build_die_map_v6",
    "locate_die_v6",
    "save_debug_overlay",
    "run_color_robustness_test",
    "detect_wafer_adaptive",
    "clean_wafer_v6",
    "detect_grid_adaptive",
    "estimate_grid_angle_adaptive",
    "detect_notch_v6",
    "clip_die",
    "crop_die",
]


# =============================================================================
# 0) 기본값 (V5 와 의미/이름을 최대한 맞춤)
# =============================================================================
DEFAULT_PIXEL_PER_UNIT = 32
DEFAULT_EDGE_MARGIN = 1.0
DEFAULT_EDGE_MODE = "circle"           # "circle" | "ring" | "both"
DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_MARGIN_X = 0
DEFAULT_MARGIN_Y = 0

DEFAULT_NOTCH_REF_DEG = 90.0           # 90 = 아래쪽(6시)
DEFAULT_NOTCH_SECTOR_DEG = 70.0
DEFAULT_NOTCH_MIN_DEPTH = 4.0
DEFAULT_NOTCH_NOISE_MARGIN = 3.0
DEFAULT_NOTCH_SMOOTH_DEG = 0.25
DEFAULT_VERIFY_TOL_DEG = 0.5
DEFAULT_QUAD_BALANCE_TOL = 0.08

# feature bank 이름 (auto 선택 후보). gradient 계열은 street 당 edge 가 2개라
# 위상(원점)이 모호하고 pitch 가 절반으로 잡힐 위험이 있어 제외한다.
FEATURE_NAMES: Tuple[str, ...] = (
    "L",        # Lab 밝기          : 검은/흰 street 대비
    "a",        # Lab green-red     : 갈색/적색 계열 street
    "b",        # Lab blue-yellow   : 청/황 계열 street
    "chroma",   # sqrt(a^2+b^2)     : "무채색 street <-> 유채색 die" (흰 street!)
    "sat",      # HSV S             : chroma 와 상보적
    "maxmin",   # max(BGR)-min(BGR) : V5 의 color_delta 를 '임계 없이' 프로파일로
    "stdL",     # local std of L    : 색이 같고 질감만 다른 경우
)

_EPS = 1e-12


# =============================================================================
# 1) 파라미터 / 진단 / 결과 자료구조
# =============================================================================
@dataclass
class ColorProfile:
    """색 관련 파라미터 묶음. **전부 옵션** — None 이면 이미지에서 자동 추정한다.

    "내가 파라미터를 주거나, 알아서 하거나" 두 가지를 모두 지원하기 위한 객체.
    아는 값만 채우면 나머지는 자동으로 채워진다.
    """

    # --- 사용자가 알려줄 수 있는 값 (None = 자동) -------------------------
    background_bgr: Optional[Tuple[int, int, int]] = None
    """웨이퍼 밖 배경색 (B, G, R). None 이면 이미지 4코너에서 자동 추정."""

    street_polarity: Optional[str] = None
    """street(die 사이)가 주변보다 밝은지: "bright" | "dark" | None(자동)."""

    feature_channels: Optional[Sequence[str]] = None
    """사용할 feature 채널 이름들. None 이면 주기성 점수로 자동 선택.
    가능: L, a, b, chroma, sat, maxmin, stdL"""

    pitch_x: Optional[float] = None
    pitch_y: Optional[float] = None
    """die pitch (px). 주면 검출을 건너뛰고 그 값을 사용(위상만 추정)."""

    # --- 탐색 범위 / 성능 --------------------------------------------------
    min_pitch_px: float = 8.0
    """이보다 작은 pitch 는 후보에서 제외 (노이즈 주기 방지)."""

    max_pitch_ratio: float = 0.34
    """pitch 상한 = 프로파일 길이 * 이 값 (최소 ~3주기는 보여야 신뢰).

    **알려진 한계** — 참 pitch 가 이 상한을 넘으면 FFT 는 그 값을 후보로
    보지 못하고 배음(1/2, 1/3 ...)을 고른다. 시야에 die 가 5개 미만으로
    보이는 매우 확대된 영상이 여기 해당한다.
      * 1/2 배음 : 자동으로 경고가 뜬다 ("2배값을 검증하지 못했습니다").
      * 1/3 이하 : street 가 pitch 의 10% 이상으로 매우 두꺼우면
        경고 없이 잘못된 값이 나올 수 있다 (실측: 참 300px, street 30px
        -> 30px 로 검출). 현실 웨이퍼 맵(20~40 die 이상)에서는 발생하지
        않지만, 확대 영상을 다룬다면 ``pitch_x`` / ``pitch_y`` 를 직접
        지정하거나 ``max_pitch_ratio`` 를 0.45 정도로 올려라."""

    roi_ratio: float = 0.62
    """격자 분석에 쓰는 중앙 정사각 ROI 반경 비율 (wafer_r 대비).
    0.62 -> 대각이 0.88r 이라 항상 웨이퍼 안쪽."""

    profile_max_dim: int = 2048
    """프로파일 계산 ROI 다운스케일 한계 (px)."""

    wafer_otsu_max_dim: int = 1024
    """Otsu 임계 계산용 다운스케일 한계 (임계값만 계산, 적용은 원본 해상도)."""

    # --- 채널 채택 기준 ----------------------------------------------------
    min_channel_score: float = 0.08
    """주기성 점수가 이보다 낮은 채널은 버린다 (0~1)."""

    pitch_cluster_tol: float = 0.04
    """채널간 pitch 합의 허용 상대오차 (4%)."""

    # --- 회전각 ------------------------------------------------------------
    angle_search_deg: float = 6.0
    angle_coarse_step: float = 0.15
    angle_fine_step: float = 0.02
    angle_max_iter: int = 3
    angle_agree_tol_deg: float = 0.50
    angle_full_scan_deg: float = 20.0
    angle_max_dim: int = 1100

    # --- wafer mask --------------------------------------------------------
    wafer_open_ksize: int = 9
    wafer_close_ksize: int = 25
    wafer_fill_bgr: Optional[Tuple[int, int, int]] = None
    """clean 시 웨이퍼 밖을 채울 색. None 이면 추정된 배경색으로 채움."""

    # -----------------------------------------------------------------------
    # 입력 검증 — "조용히 틀린 답" 을 막는다
    # -----------------------------------------------------------------------
    # 검증이 없으면 잘못된 파라미터가 예외 없이 통과해서, 겉보기엔 정상이지만
    # 완전히 틀린 결과를 돌려준다. 실제로 관측된 사례:
    #   max_pitch_ratio=0   -> pitch=(22.0,14.7) n=19111  (정답 88/88 n=800, 경고 0건)
    #   roi_ratio=2.0       -> ROI 가 웨이퍼 밖 배경까지 먹어서 pitch 가 조용히 어긋남
    #   pitch_x=3.0         -> min_pitch_px=8 보다 작은데도 통과, die 684,887 개 생성
    #   background_bgr=(999,..) -> numpy 의 날것 OverflowError
    #   feature_channels=() -> 사용자의 의도와 정반대로 "전체 채널" 로 조용히 대체
    # 이런 값은 전부 사용자 실수이므로, 조용히 틀리기보다 즉시 명확하게 실패한다.
    def __post_init__(self) -> None:
        def _bad(field: str, got: Any, want: str) -> None:
            raise ValueError(
                f"ColorProfile.{field}={got!r} 은(는) 사용할 수 없습니다. {want}")

        # --- 배경색: uint8 범위 --------------------------------------------
        if self.background_bgr is not None:
            bg = tuple(self.background_bgr)
            if len(bg) != 3:
                _bad("background_bgr", self.background_bgr, "(B, G, R) 3개여야 합니다.")
            for v in bg:
                if not isinstance(v, (int, float)) or not (0 <= float(v) <= 255):
                    _bad("background_bgr", self.background_bgr,
                         "각 채널은 0~255 여야 합니다.")
            self.background_bgr = (int(bg[0]), int(bg[1]), int(bg[2]))

        # --- street 극성: 문자열 계약 유지 ----------------------------------
        if self.street_polarity is not None:
            s = str(self.street_polarity).lower().strip()
            if s in ("auto", "none", "0"):
                self.street_polarity = None
            elif s in ("bright", "light", "+1", "1"):
                self.street_polarity = "bright"
            elif s in ("dark", "-1"):
                self.street_polarity = "dark"
            else:
                _bad("street_polarity", self.street_polarity,
                     '"bright" | "dark" | None 중 하나여야 합니다.')

        # --- feature 채널 ---------------------------------------------------
        # 빈 시퀀스를 truthiness 로 판정하면 "전체 채널" 로 조용히 대체되므로
        # 여기서 명시적으로 막는다.
        if self.feature_channels is not None:
            chans = tuple(self.feature_channels)
            if not chans:
                _bad("feature_channels", self.feature_channels,
                     f"비어 있을 수 없습니다. 자동 선택을 원하면 None 을 주세요. "
                     f"사용 가능: {list(FEATURE_NAMES)}")
            bad = [c for c in chans if c not in FEATURE_NAMES]
            if bad:
                _bad("feature_channels", bad,
                     f"사용 가능: {list(FEATURE_NAMES)}")
            self.feature_channels = chans

        # --- 고정 pitch -----------------------------------------------------
        # 0 이나 음수는 die 열거 루프에서 0개 혹은 폭발적인 개수를 만든다.
        for name in ("pitch_x", "pitch_y"):
            v = getattr(self, name)
            if v is None:
                continue
            v = float(v)
            if not math.isfinite(v) or v <= 0:
                _bad(name, v, "0 보다 큰 유한한 값이어야 합니다. "
                              "자동 검출을 원하면 None 을 주세요.")
            if v < self.min_pitch_px:
                _bad(name, v,
                     f"min_pitch_px({self.min_pitch_px}) 보다 작습니다. "
                     f"의도한 값이면 min_pitch_px 도 함께 낮추세요.")
            setattr(self, name, v)

        # --- 탐색 범위 ------------------------------------------------------
        if not math.isfinite(self.min_pitch_px) or self.min_pitch_px < 2.0:
            _bad("min_pitch_px", self.min_pitch_px, "2.0 이상이어야 합니다.")
        if not (0.0 < self.max_pitch_ratio <= 0.5):
            _bad("max_pitch_ratio", self.max_pitch_ratio,
                 "(0, 0.5] 범위여야 합니다. 기본값 0.34 는 최소 ~3주기를 보장합니다.")
        if not (0.0 < self.roi_ratio <= 1.0):
            _bad("roi_ratio", self.roi_ratio,
                 "(0, 1.0] 범위여야 합니다. 1.0 을 넘으면 ROI 가 웨이퍼 밖 "
                 "배경까지 포함해서 주기 분석이 망가집니다.")
        if self.profile_max_dim < 64:
            _bad("profile_max_dim", self.profile_max_dim, "64 이상이어야 합니다.")
        if self.wafer_otsu_max_dim < 64:
            _bad("wafer_otsu_max_dim", self.wafer_otsu_max_dim, "64 이상이어야 합니다.")

        # --- 채널 채택 기준 --------------------------------------------------
        if not (0.0 <= self.min_channel_score < 1.0):
            _bad("min_channel_score", self.min_channel_score, "[0, 1) 범위여야 합니다.")
        if not (0.0 < self.pitch_cluster_tol < 1.0):
            _bad("pitch_cluster_tol", self.pitch_cluster_tol, "(0, 1) 범위여야 합니다.")

        # --- 회전각 ----------------------------------------------------------
        if self.angle_search_deg < 0 or self.angle_full_scan_deg < 0:
            _bad("angle_search_deg/angle_full_scan_deg",
                 (self.angle_search_deg, self.angle_full_scan_deg), "0 이상이어야 합니다.")
        if self.angle_coarse_step <= 0 or self.angle_fine_step <= 0:
            _bad("angle_coarse_step/angle_fine_step",
                 (self.angle_coarse_step, self.angle_fine_step), "0 보다 커야 합니다.")
        if self.angle_max_iter < 1:
            _bad("angle_max_iter", self.angle_max_iter, "1 이상이어야 합니다.")

        # --- wafer mask -------------------------------------------------------
        if self.wafer_open_ksize < 1 or self.wafer_close_ksize < 1:
            _bad("wafer_open_ksize/wafer_close_ksize",
                 (self.wafer_open_ksize, self.wafer_close_ksize), "1 이상이어야 합니다.")
        if self.wafer_fill_bgr is not None:
            fb = tuple(self.wafer_fill_bgr)
            if len(fb) != 3 or any(not (0 <= float(v) <= 255) for v in fb):
                _bad("wafer_fill_bgr", self.wafer_fill_bgr,
                     "(B, G, R) 각 0~255 여야 합니다.")
            self.wafer_fill_bgr = (int(fb[0]), int(fb[1]), int(fb[2]))

    def merged(self, **kw: Any) -> "ColorProfile":
        """일부 필드만 바꾼 사본. (사본도 동일하게 검증된다)"""
        from dataclasses import replace
        return replace(self, **kw)


@dataclass
class ChannelResult:
    """feature 채널 1개의 축(axis) 분석 결과."""
    name: str
    pitch: float = 0.0
    score: float = 0.0             # 0~1 주기성 점수
    polarity: int = 0              # +1 = street 가 밝음(=최대), -1 = 어두움
    street_pos: float = 0.0        # ROI-local street 중심 (0 <= pos < pitch)
    street_width_frac: float = 0.0  # 한 주기 중 street 가 차지하는 비율
    used: bool = False
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "pitch": round(self.pitch, 3),
            "score": round(self.score, 4), "polarity": self.polarity,
            "street_pos": round(self.street_pos, 2),
            "street_width_frac": round(self.street_width_frac, 3),
            "used": self.used, "note": self.note,
        }


@dataclass
class GridDiagnostics:
    """자가진단 리포트 — 무엇을 어떻게 판단했는지 전부 기록."""
    background_bgr: Tuple[int, int, int] = (0, 0, 0)
    background_source: str = "auto"
    wafer_coverage: float = 0.0
    wafer_fallback: str = ""
    wafer_circle: Tuple[int, int, int] = (0, 0, 0)

    channels_x: List[ChannelResult] = field(default_factory=list)
    channels_y: List[ChannelResult] = field(default_factory=list)
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    pitch_x_agreement: float = 0.0      # 합의 클러스터 점수합 / 전체 점수합
    pitch_y_agreement: float = 0.0
    phase_conf_x: float = 0.0           # 채널간 위상 원형평균 resultant (0~1)
    phase_conf_y: float = 0.0
    polarity_x: str = "auto"
    polarity_y: str = "auto"

    angle_projection: float = 0.0
    angle_fft: Optional[float] = None
    angle_agree: bool = False
    angle_confidence: float = 0.0
    angle_applied: float = 0.0

    notch_found: bool = False
    n_dies: int = 0
    elapsed_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # ----- 편의 ------------------------------------------------------------
    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        """대략적인 '믿을 만한가' 판정."""
        return (self.pitch_x > 0 and self.pitch_y > 0
                and self.pitch_x_agreement >= 0.5
                and self.pitch_y_agreement >= 0.5
                and self.phase_conf_x >= 0.5 and self.phase_conf_y >= 0.5
                and not self.wafer_fallback)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_bgr": list(self.background_bgr),
            "background_source": self.background_source,
            "wafer_circle": list(self.wafer_circle),
            "wafer_coverage": round(self.wafer_coverage, 4),
            "wafer_fallback": self.wafer_fallback,
            "pitch_x": round(self.pitch_x, 3),
            "pitch_y": round(self.pitch_y, 3),
            "pitch_x_agreement": round(self.pitch_x_agreement, 3),
            "pitch_y_agreement": round(self.pitch_y_agreement, 3),
            "phase_conf_x": round(self.phase_conf_x, 3),
            "phase_conf_y": round(self.phase_conf_y, 3),
            "polarity_x": self.polarity_x, "polarity_y": self.polarity_y,
            "channels_x": [c.as_dict() for c in self.channels_x],
            "channels_y": [c.as_dict() for c in self.channels_y],
            "angle_projection": round(self.angle_projection, 4),
            "angle_fft": (None if self.angle_fft is None
                          else round(self.angle_fft, 4)),
            "angle_agree": self.angle_agree,
            "angle_confidence": round(self.angle_confidence, 3),
            "angle_applied": round(self.angle_applied, 4),
            "notch_found": self.notch_found,
            "n_dies": self.n_dies,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "ok": self.ok,
            "warnings": list(self.warnings),
        }

    def report(self) -> str:
        """사람이 읽는 자가진단 리포트 문자열."""
        L: List[str] = []
        L.append("=" * 72)
        L.append(" WAFER COLOR-AGNOSTIC DIE MAP  (V6 / claude)  SELF-DIAGNOSIS")
        L.append("=" * 72)
        b = self.background_bgr
        L.append(f" background BGR      : ({b[0]:3d},{b[1]:3d},{b[2]:3d})"
                 f"   [{self.background_source}]")
        cx, cy, r = self.wafer_circle
        L.append(f" wafer circle        : center=({cx},{cy})  r={r}"
                 f"   coverage={self.wafer_coverage:.3f}"
                 + (f"  FALLBACK={self.wafer_fallback}" if self.wafer_fallback else ""))
        L.append("-" * 72)
        L.append(f" pitch_x = {self.pitch_x:9.3f} px"
                 f"   agreement={self.pitch_x_agreement:.2f}"
                 f"   phase_conf={self.phase_conf_x:.2f}"
                 f"   street={self.polarity_x}")
        L.append(f" pitch_y = {self.pitch_y:9.3f} px"
                 f"   agreement={self.pitch_y_agreement:.2f}"
                 f"   phase_conf={self.phase_conf_y:.2f}"
                 f"   street={self.polarity_y}")
        for axis, chans in (("X", self.channels_x), ("Y", self.channels_y)):
            L.append("-" * 72)
            L.append(f" [{axis}] channel        pitch     score  pol  width%  used")
            for c in sorted(chans, key=lambda z: -z.score):
                mark = "USE" if c.used else " - "
                pol = "+" if c.polarity > 0 else ("-" if c.polarity < 0 else "?")
                L.append(f"      {c.name:<10s} {c.pitch:9.3f} {c.score:8.4f}"
                         f"   {pol}  {c.street_width_frac * 100:5.1f}   {mark}"
                         + (f"   {c.note}" if c.note else ""))
        L.append("-" * 72)
        fft_s = "n/a" if self.angle_fft is None else f"{self.angle_fft:+.3f}"
        L.append(f" angle projection    : {self.angle_projection:+.3f} deg")
        L.append(f" angle 2D-FFT        : {fft_s} deg"
                 f"   agree={self.angle_agree}")
        L.append(f" angle applied       : {self.angle_applied:+.3f} deg"
                 f"   confidence={self.angle_confidence:.2f}")
        L.append(f" notch found         : {self.notch_found}")
        L.append(f" dies                : {self.n_dies}")
        L.append(f" elapsed             : {self.elapsed_sec:.2f} s")
        L.append("-" * 72)
        L.append(f" OVERALL             : {'OK' if self.ok else 'CHECK WARNINGS'}")
        if self.warnings:
            for w in self.warnings:
                L.append(f"   ! {w}")
        L.append("=" * 72)
        return "\n".join(L)


@dataclass
class WaferDieMapV6:
    """build_die_map_v6() 결과. **V5 WaferDieMap 과 같은 형식** + 부가 필드 2개.

    이 모듈은 v5 를 import 하지 않는다. v5 의 정의를 통째로 복붙해 온 것이라
    이름/타입/순서가 같을 뿐이고, 런타임 의존성은 전혀 없다.
    (``wafer_die_map_v5.py`` 를 지워도 v6 는 그대로 동작한다.)

    형식이 같으므로 기존에 v5 결과를 쓰던 코드는 **그대로 v6 결과를 받는다.**
    필드 접근이든 위치 인자 언패킹이든 동일하게 동작한다.

    앞 22개 필드 (v5 와 이름/타입/순서 동일)
    ---------------------------------------
    wafer_cx, wafer_cy, wafer_r   웨이퍼 중심/반지름 (px)
    pitch_x, pitch_y              die pitch (px, sub-pixel float)
    x0, y0                        grid origin = 중심에 가장 가까운 street 교차점
    die_w, die_h                  die crop 크기 (= round(pitch))
    pixel_per_unit                실측 좌표 환산 단위 (px/unit)
    dies                          die entry 리스트
    dies_by_index                 {(ix,iy): entry}
    image_shape                   (H, W)
    rotation_deg                  적용된 회전 보정 각도
    aligned_image                 정렬 후 실제 사용 이미지 (모든 좌표의 기준)
    notch_center_px               notch 중심 (없으면 None)
    die_grid_angle_resid          보정 후 잔여 기울기(deg)
    angle_verified                notch 각도와 격자 각도 일치 여부
    quadrant_report               4분면 검증 결과
    edge_mode                     is_edge 기준 (circle|ring|both)
    angle_confidence              각도 신뢰도 0~1
    angle_agree                   projection 과 FFT 합의 여부

    die entry(dict) 키도 v5 와 완전히 같다::

        index, center_px, rect_px, crop_rect_px, real_coord,
        is_edge_partial, is_edge_ring, is_edge, image

    v6 전용 추가 필드 (반드시 맨 뒤 — 중간에 끼우면 위치 인자가 밀린다)
    ----------------------------------------------------------------
    wafer_mask   : 색상 적응형으로 구한 웨이퍼 전경 마스크 (uint8 0/255).
                   v5 는 밝기 임계값으로 매번 다시 만들었지만 v6 는
                   이미 구한 것을 그대로 넘겨준다.
    diagnostics  : 자가진단 리포트 (채널별 점수/합의도/경고).
                   ``print(m.diagnostics.report())`` 로 사람이 읽는 형태 출력.

    둘 다 뒤에 붙어 있을 뿐이라 무시하면 v5 결과와 구분되지 않는다.
    die 조회는 이 모듈 안의 :func:`locate_die_v6` 를 쓰면 된다.
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
    rotation_deg: float = 0.0
    aligned_image: Optional[np.ndarray] = field(default=None, repr=False)
    notch_center_px: Optional[Tuple[int, int]] = None
    die_grid_angle_resid: float = 0.0
    angle_verified: bool = False
    quadrant_report: Dict[str, Any] = field(default_factory=dict)
    edge_mode: str = DEFAULT_EDGE_MODE
    angle_confidence: float = 1.0
    angle_agree: bool = True
    # --- 여기까지가 V5 WaferDieMap 과 1:1 (이름/타입/순서) -------------------
    # 아래 2개는 v6 전용. 반드시 맨 뒤에 둬라 — 중간에 끼우면 뒤 필드가
    # 한 칸씩 밀려서 위치 인자 언패킹이 조용히 어긋난다.
    wafer_mask: Optional[np.ndarray] = field(default=None, repr=False)
    diagnostics: GridDiagnostics = field(default_factory=GridDiagnostics)

    def get_die(self, ix: int, iy: int) -> Optional[Dict[str, Any]]:
        return self.dies_by_index.get((ix, iy))

    @property
    def num_dies(self) -> int:
        return len(self.dies)


# =============================================================================
# 2) 공용 소도구
# =============================================================================
def _load_bgr(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """경로(str/Path) 또는 ndarray -> **uint8 3채널 BGR** 이미지.

    모든 공개 함수의 단일 입력 관문이다. 여기서 정규화를 끝내야
    하위 함수들이 dtype/채널 수를 다시 걱정하지 않는다.

    처리하는 것:
      * 2채널(회색조)      -> BGR 로 확장
      * 4채널(BGRA/RGBA)  -> 알파 버리고 BGR (PNG 에 흔하다)
      * uint8 이 아닌 dtype -> uint8 로 정규화

    dtype 정규화가 필요한 이유:
      cv2.cvtColor(..., COLOR_BGR2LAB) 는 float32 입력을 **0~1 범위**로
      해석한다. 0~255 범위의 float32 를 그냥 넘기면 Lab 값이 포화돼서
      예외 없이 조용히 다른 pitch 가 나온다 (실측: 88.43 vs 정답 87.98).
    """
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]          # 알파 폐기
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"지원하지 않는 이미지 형태 {image.shape}. "
                "(H,W) 회색조 또는 (H,W,3) BGR 또는 (H,W,4) BGRA 여야 합니다.")
        return _to_uint8(img)
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(image))
    return img


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """어떤 dtype 이든 0~255 uint8 BGR 로 맞춘다 (이미 uint8 이면 그대로)."""
    if img.dtype == np.uint8:
        return img
    a = np.asarray(img)
    if a.dtype == np.uint16:
        return (a.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise ValueError("이미지에 유효한(finite) 픽셀이 없습니다.")
    hi = float(finite.max())
    # 0~1 로 정규화된 float 인지, 0~255 스케일인지 판별한다.
    scale = 255.0 if hi <= 1.0 + 1e-6 else 1.0
    return np.nan_to_num(a * scale, nan=0.0).clip(0, 255).astype(np.uint8)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _downscale(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """긴 변이 max_dim 이하가 되도록 축소. -> (축소본, scale)  (scale<=1)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim or max_dim <= 0:
        return img, 1.0
    s = float(max_dim) / float(m)
    out = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA)
    return out, s


def _moving_average(p: np.ndarray, win: int) -> np.ndarray:
    """reflect 패딩 이동평균 (cv2.GaussianBlur 의 ksize=(0,0) 함정 회피)."""
    n = p.size
    win = _odd(max(3, int(win)))
    if win >= n:
        return np.full(n, float(p.mean()))
    pad = win // 2
    ext = np.pad(p.astype(np.float64), pad, mode="reflect")
    ker = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(ext, ker, mode="valid")[:n]


def _detrend(p: np.ndarray, win: int) -> np.ndarray:
    """저주파(조명 기울기) 제거."""
    return p.astype(np.float64) - _moving_average(p, win)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if d < _EPS:
        return 0.0
    return float((a * b).sum() / d)


def _corr_at_lag(x: np.ndarray, lag: float) -> float:
    """lag 만큼 shift 한 자기상관 (Pearson)."""
    L = int(round(lag))
    if L < 2 or L >= x.size - 4:
        return 0.0
    return _pearson(x[:-L], x[L:])


# =============================================================================
# 3) 배경색 자동 추정 + 색-무관 wafer 검출
# =============================================================================
def _estimate_background(img_bgr: np.ndarray) -> Tuple[Tuple[int, int, int], str]:
    """이미지 4코너 블록에서 배경 BGR 을 robust 하게 추정.

    웨이퍼는 보통 화면 중앙의 원이므로 코너는 배경이다. 4코너 중 서로 가장
    비슷한 3개를 채택(한 코너에 라벨/노이즈가 있어도 견딤)해 median 을 쓴다.
    """
    h, w = img_bgr.shape[:2]
    bs = max(8, min(h, w) // 16)
    blocks = [
        img_bgr[0:bs, 0:bs],
        img_bgr[0:bs, w - bs:w],
        img_bgr[h - bs:h, 0:bs],
        img_bgr[h - bs:h, w - bs:w],
    ]
    meds = np.array([np.median(b.reshape(-1, 3), axis=0) for b in blocks],
                    dtype=np.float64)                      # (4,3)
    # 4개 중 나머지 3개와의 거리합이 가장 큰 1개를 이상치로 제거
    d = np.abs(meds[:, None, :] - meds[None, :, :]).sum(axis=2).sum(axis=1)
    keep = np.argsort(d)[:3]
    bg = np.median(meds[keep], axis=0)
    return (int(round(bg[0])), int(round(bg[1])), int(round(bg[2]))), "corners"


def _bgr_to_lab_f32(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> Lab float32 (OpenCV 8U Lab 스케일: L 0..255, a/b 0..255@128)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _bg_distance_map(img_bgr: np.ndarray,
                     bg_bgr: Tuple[int, int, int]) -> np.ndarray:
    """배경색으로부터의 Lab 거리 맵 (uint8 0..255 정규화).

    ★ 색-무관의 핵심: "검정보다 밝은가"가 아니라 "배경색과 다른가"를 본다.
    """
    lab = _bgr_to_lab_f32(img_bgr)
    swatch = np.zeros((1, 1, 3), np.uint8)
    swatch[0, 0] = np.array(bg_bgr, dtype=np.uint8)
    bg_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d = lab - bg_lab.reshape(1, 1, 3)
    dist = np.sqrt((d * d).sum(axis=2))
    hi = float(np.percentile(dist, 99.5))
    if hi < _EPS:
        return np.zeros(dist.shape, np.uint8)
    return np.clip(dist * (255.0 / hi), 0, 255).astype(np.uint8)


def detect_wafer_adaptive(img_bgr: np.ndarray,
                          profile: Optional[ColorProfile] = None,
                          diag: Optional[GridDiagnostics] = None
                          ) -> Tuple[int, int, int, np.ndarray]:
    """색-무관 wafer 검출 -> (cx, cy, r, silhouette_mask uint8 0/1).

    절차
    ----
    1) 배경색 자동 추정 (또는 profile.background_bgr).
    2) 배경색과의 Lab 거리 맵 -> Otsu (임계는 다운스케일본에서 계산).
    3) morphology open/close -> 가장 큰 연결성분.
    4) 외곽 컨투어를 채워(fill) 실루엣 확정.
       ★ RETR_EXTERNAL + drawContours(-1) 이므로 내부 구멍(어두운 die)은 메워지고
         notch 같은 '경계 오목부'는 그대로 보존된다.
    5) minEnclosingCircle. 면적환산 반지름과 크게 어긋나면(림 돌출 노이즈)
       면적환산 반지름 + 무게중심으로 대체.
    6) coverage 가 비정상(≈1 또는 ≈0)이면 내접원 fallback.
    """
    prof = profile or ColorProfile()
    # 공개 함수는 단독 호출될 수 있으므로 여기서도 입력을 정규화한다.
    # (BGRA/회색조/float 입력이 cvtColor 로 곧장 들어가 조용히 깨지는 것을 막는다)
    img_bgr = _load_bgr(img_bgr)
    H, W = img_bgr.shape[:2]

    if prof.background_bgr is not None:
        bg = (int(prof.background_bgr[0]), int(prof.background_bgr[1]),
              int(prof.background_bgr[2]))
        src = "user"
    else:
        bg, src = _estimate_background(img_bgr)
    if diag is not None:
        diag.background_bgr = bg
        diag.background_source = src

    dist = _bg_distance_map(img_bgr, bg)

    # Otsu 임계는 다운스케일본에서(속도), 적용은 원본 해상도에서(정밀도)
    small, _ = _downscale(dist, prof.wafer_otsu_max_dim)
    thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(dist, float(thr), 255, cv2.THRESH_BINARY)

    ko = _odd(max(3, int(round(prof.wafer_open_ksize * min(H, W) / 3000.0)) or 3))
    kc = _odd(max(5, int(round(prof.wafer_close_ksize * min(H, W) / 3000.0)) or 5))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    # --- 점진적 CLOSE -------------------------------------------------
    # 고정 커널은 scribe lane 이 넓으면(예: pitch 150 / lane 13px) lane 을
    # 못 메워서 마스크가 die 단위로 산산조각 난다. 그러면 RETR_EXTERNAL 의
    # '가장 큰 컨투어' 가 die 하나가 되어 coverage~0 오판이 난다.
    # -> 최대 컨투어가 전경 면적의 대부분을 차지할 때까지 커널을 키운다.
    fg = float(np.count_nonzero(mask))
    k_max = _odd(max(kc, int(round(0.04 * min(H, W)))))
    k = _odd(kc)
    closed = mask
    while True:
        closed = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs or fg < _EPS:
            break
        if cv2.contourArea(max(cs, key=cv2.contourArea)) >= 0.85 * fg:
            break
        if k >= k_max:
            break
        k = _odd(min(k_max, int(round(k * 1.8)) + 1))
    mask = closed

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback = ""
    if not cnts:
        fallback = "no-contour"
    else:
        big = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(big))
        cov = area / float(H * W)
        sil = np.zeros((H, W), np.uint8)
        cv2.drawContours(sil, [big], -1, 1, thickness=-1)
        (ecx, ecy), er = cv2.minEnclosingCircle(big)
        r_area = math.sqrt(max(area, 1.0) / math.pi)
        if er > 1.10 * r_area:
            # 림 밖으로 삐져나온 노이즈가 원을 부풀림 -> 면적환산 + 무게중심
            m = cv2.moments(big)
            if abs(m["m00"]) > _EPS:
                ecx = m["m10"] / m["m00"]
                ecy = m["m01"] / m["m00"]
            er = r_area
            if diag is not None:
                diag.warn("minEnclosingCircle inflated by rim noise "
                          "-> used area-equivalent radius")
        if cov > 0.985:
            fallback = "coverage~1 (background estimate failed?)"
        elif cov < 0.02:
            fallback = "coverage~0 (wafer not separable from background)"
        else:
            if diag is not None:
                diag.wafer_coverage = cov
                diag.wafer_circle = (int(round(ecx)), int(round(ecy)),
                                     int(round(er)))
            return (int(round(ecx)), int(round(ecy)), int(round(er)), sil)

    # ---- fallback : 이미지 내접원 -----------------------------------------
    cx, cy = W // 2, H // 2
    r = int(min(H, W) // 2) - 1
    sil = np.zeros((H, W), np.uint8)
    cv2.circle(sil, (cx, cy), r, 1, thickness=-1)
    if diag is not None:
        diag.wafer_fallback = fallback
        diag.wafer_coverage = math.pi * r * r / float(H * W)
        diag.wafer_circle = (cx, cy, r)
        diag.warn(f"wafer detection fallback: {fallback}")
    return cx, cy, r, sil


def clean_wafer_v6(img_bgr: np.ndarray,
                   wafer_cx: int, wafer_cy: int, wafer_r: int,
                   sil: np.ndarray,
                   fill_bgr: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """wafer 원판 밖(실루엣 밖 OR 원 밖)을 fill_bgr 로 채워 반환.

    V5 는 항상 검정으로 채웠지만, 배경이 흰색인 이미지에서 검정으로 채우면
    없던 강한 경계가 생겨 이후 처리를 방해한다. V6 는 **추정된 배경색**으로
    채워 색상 팔레트에 중립적으로 동작한다.
    """
    H, W = img_bgr.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - wafer_cx) ** 2 + (yy - wafer_cy) ** 2 <= wafer_r * wafer_r
    keep = (sil > 0) & disc
    out = img_bgr.copy()
    out[~keep] = np.array(fill_bgr, dtype=img_bgr.dtype)
    return out


# =============================================================================
# 4) Feature bank  --  색 공간을 "여러 관점"으로 펼친다
# =============================================================================
def _feature_bank(roi_bgr: np.ndarray,
                  names: Sequence[str] = FEATURE_NAMES) -> Dict[str, np.ndarray]:
    """ROI(BGR uint8) -> {채널이름: float32 2D}.

    각 채널은 '무엇이 street 인지'에 대한 서로 다른 가설이다.
    어떤 팔레트라도 최소 한 채널은 die/street 를 분리해낸다는 것이 V6 의 전제.
    """
    out: Dict[str, np.ndarray] = {}
    need = set(names)

    lab = _bgr_to_lab_f32(roi_bgr)
    Lc, Ac, Bc = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    if "L" in need:
        out["L"] = Lc
    if "a" in need:
        out["a"] = Ac
    if "b" in need:
        out["b"] = Bc
    if "chroma" in need:
        # 무채색(흰/회/검) street <-> 유채색 die 를 분리하는 결정적 채널.
        # V5 의 min_color_delta=35 가 '버렸던' 정보를 임계 없이 살린다.
        da, db = Ac - 128.0, Bc - 128.0
        out["chroma"] = np.sqrt(da * da + db * db)
    if "sat" in need:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        out["sat"] = hsv[:, :, 1].astype(np.float32)
    if "maxmin" in need:
        f = roi_bgr.astype(np.float32)
        out["maxmin"] = f.max(axis=2) - f.min(axis=2)
    if "stdL" in need:
        # 평균색이 같고 '질감'만 다른 경우(예: 흰 street + 갈색 노이즈)를 잡는다.
        k = 5
        m = cv2.boxFilter(Lc, cv2.CV_32F, (k, k))
        m2 = cv2.boxFilter(Lc * Lc, cv2.CV_32F, (k, k))
        out["stdL"] = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return out


def _grid_roi(img_bgr: np.ndarray, cx: int, cy: int, r: int,
              roi_ratio: float) -> Tuple[np.ndarray, int, int]:
    """웨이퍼 중앙의 정사각 ROI -> (roi, x1, y1).  대각이 항상 원 안쪽."""
    H, W = img_bgr.shape[:2]
    half = max(16, int(round(r * roi_ratio)))
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)
    return img_bgr[y1:y2, x1:x2], x1, y1


# =============================================================================
# 5) 주기 검출 : FFT pitch / Fourier phase / 자기상관 점수 / 극성
# =============================================================================
def _spectral_pitch(p: np.ndarray, min_pitch: float,
                    max_pitch: float) -> Optional[float]:
    """1D 프로파일에서 pitch(주기, px)를 FFT 로 추정.

    - Hann 창 + 8배 zero-padding 으로 주파수 분해능 확보.
    - [min_pitch, max_pitch] 밴드 안에서만 탐색  -> 조명 기울기(초저주파)와
      픽셀 노이즈(초고주파)는 자동 배제. 별도 detrend 불필요.
    - ★ 기본파 선호: 최대 피크가 2/3/4 배 고조파일 수 있으므로 k/2, k/3, k/4
      근방에 충분한 파워(>=25%)가 있으면 더 낮은 주파수(=더 큰 pitch)를 채택.
      (그렇지 않으면 die 를 반쪽으로 쪼개는 pitch 가 나온다.)
    - log-power 포물선 보간으로 sub-bin 정밀도.
    """
    n = int(p.size)
    if n < 32:
        return None
    x = (p.astype(np.float64) - float(p.mean())) * np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(max(64, n * 8))))
    P = np.abs(np.fft.rfft(x, nfft)) ** 2
    nb = P.size

    k_lo = int(math.ceil(nfft / max(max_pitch, min_pitch + 1.0)))
    k_hi = int(math.floor(nfft / max(min_pitch, 2.0)))
    k_lo = max(k_lo, 1)
    k_hi = min(k_hi, nb - 2)
    if k_hi <= k_lo:
        return None

    k0 = k_lo + int(np.argmax(P[k_lo:k_hi + 1]))
    pk0 = float(P[k0])
    if pk0 < _EPS:
        return None

    kbest = k0
    for m in (2, 3, 4):
        kc = k0 / float(m)
        if kc < k_lo:
            break
        wnd = max(1.0, kc * 0.06)
        lo = int(max(k_lo, math.floor(kc - wnd)))
        hi = int(min(k_hi, math.ceil(kc + wnd)))
        if hi < lo:
            continue
        kk = lo + int(np.argmax(P[lo:hi + 1]))
        if float(P[kk]) >= 0.25 * pk0:
            kbest = min(kbest, kk)          # 더 낮은 주파수 = 더 큰 pitch = 기본파

    kf = float(kbest)
    if 1 <= kbest < nb - 1:
        y0 = math.log(float(P[kbest - 1]) + _EPS)
        y1 = math.log(float(P[kbest]) + _EPS)
        y2 = math.log(float(P[kbest + 1]) + _EPS)
        den = y0 - 2.0 * y1 + y2
        if abs(den) > _EPS:
            kf = kbest + max(-0.5, min(0.5, 0.5 * (y0 - y2) / den))
    if kf < _EPS:
        return None
    pitch = nfft / kf
    if not (min_pitch <= pitch <= max_pitch):
        return None
    return float(pitch)


def _resolve_period_multiple(prof: np.ndarray, pitch: float,
                             max_pitch: float, min_pitch: float = 0.0,
                             margin: float = 0.06,
                             gap: float = 0.25, strong: float = 0.75
                             ) -> Tuple[float, str]:
    """반주기(half-period) 오검출 교정 -> (pitch, note).

    스펙트럼만 보면 die 안에 비슷한 구조가 2개 있을 때 진짜 주기의 1/2 을
    고르기 쉽다 (주파수 영역 '기본파 선호' 는 고조파 파워비가 애매하면 실패).
    자기상관은 "정말 그 lag 에서 반복되는가" 를 직접 답하므로 훨씬 확실하다.

    실측: portable_bw_sample 의 X 축에서 pitch 45 로 잡혔지만
    r(45) = -0.105 (역위상!) / r(90) = +0.997 — 진짜 주기는 90 이었다.

    규칙: 승격은 "T 에서 반복이 실제로 깨질 때" 만 한다.
      (1) r(T) 가 충분히 높으면(>= ``strong``) 무조건 T 를 유지한다.
      (2) 최대값과 r(T) 의 차이가 ``gap`` 미만이면 T 를 유지한다.
    두 관문을 모두 통과했을 때만 T, 2T, 3T 중 최대값에서 ``margin`` 이내인
    **가장 작은** 것으로 승격한다.

    이 관문이 필요한 이유(실측): 15_lowcontrast 는 r(64)=+0.67,
    r(128)=+0.78 로 128 이 근소하게 높지만 진짜 주기는 64 다. 단순히
    "더 큰 쪽" 을 고르면 저대비 영상에서 주기가 2배로 부풀어 오른다.

    반대 방향(강등)도 같은 논리로 처리한다. 스펙트럼 피크는 2고조파에
    얹힐 수 있어서 진짜 주기의 2배가 잡히기도 한다(실측 15_lowcontrast X:
    스펙트럼 129, r(64)=+0.78 / r(128)=+0.86). T/2 에서도 사실상 똑같이
    잘 반복되면 기본 주기는 T/2 다.
    """
    # ---- 강등: T/2 에서도 거의 같은 수준으로 반복되면 T 는 2고조파다 ----
    half = pitch * 0.5
    if half >= max(min_pitch, 2.0):
        xh = _detrend(prof, _odd(int(round(pitch * 2)) + 1))
        if float(xh.std()) > _EPS:
            r_full = _corr_at_lag(xh, pitch)
            r_half = _corr_at_lag(xh, half)
            if r_half >= 0.5 and r_half >= r_full - 0.12:
                ref = _spectral_pitch(prof, half * 0.90,
                                      min(half * 1.12, max_pitch))
                c = ref if (ref and abs(ref - half) / half <= 0.05) else half
                return float(c), f"/2 (r={r_full:+.2f}->{r_half:+.2f})"

    cands = [pitch * m for m in (1, 2, 3) if pitch * m <= max_pitch]
    if len(cands) < 2:
        return float(pitch), ""

    # 가장 큰 후보 기준으로 detrend 해야 lag 간 비교가 공정하다
    xd = _detrend(prof, _odd(int(round(max(cands) * 2)) + 1))
    if float(xd.std()) < _EPS:
        return float(pitch), ""
    rs = [_corr_at_lag(xd, c) for c in cands]
    best = max(rs)
    if best <= 0.0:
        return float(pitch), ""

    # 승격 관문: T 에서 이미 반복이 잘 되면(또는 근소한 차이면) 건드리지 않는다.
    r1 = rs[0]
    if r1 >= strong or (best - r1) < gap:
        return float(pitch), ""

    for c, r in zip(cands, rs):
        if r >= best - margin:
            if c <= pitch * 1.0001:
                return float(pitch), ""
            m = int(round(c / pitch))
            # 승격된 주기를 좁은 밴드에서 다시 스펙트럼 정밀화
            ref = _spectral_pitch(prof, c * 0.90, min(c * 1.12, max_pitch))
            if ref and abs(ref - c) / c <= 0.05:
                c = ref
            return float(c), f"x{m} (r={rs[0]:+.2f}->{best:+.2f})"
    return float(pitch), ""


def _fourier_phase(p: np.ndarray, pitch: float) -> float:
    """pitch 주파수의 Fourier 계수 위상 -> 프로파일 '최대'의 위치 phi (0<=phi<pitch).

    지역 임계/밴드 탐색(V5 방식)과 달리 전 구간을 쓰는 최소제곱 최적해라
    한두 개 street 가 오염돼도 흔들리지 않는다.
    """
    n = int(p.size)
    i = np.arange(n, dtype=np.float64)
    x = (p.astype(np.float64) - float(p.mean())) * np.hanning(n)
    z = complex(np.sum(x * np.exp(-2j * np.pi * i / float(pitch))))
    if abs(z) < _EPS:
        return 0.0
    phi = (-np.angle(z)) * float(pitch) / (2.0 * np.pi)
    return float(phi % float(pitch))


def _fold(p: np.ndarray, pitch: float, phi: float) -> np.ndarray:
    """프로파일을 한 주기로 접은 평균 템플릿. index 0 = phi(=최대) 위치."""
    n = int(p.size)
    nb = int(max(8, min(int(round(pitch)), 128)))
    i = np.arange(n, dtype=np.float64)
    pos = ((i - phi) % pitch) / pitch * nb
    b = np.clip(pos.astype(np.int64), 0, nb - 1)
    cnt = np.bincount(b, minlength=nb).astype(np.float64)
    s = np.bincount(b, weights=p.astype(np.float64), minlength=nb)
    cnt[cnt < 1.0] = 1.0
    t = s / cnt
    return t - t.mean()


def _street_from_template(folded: np.ndarray, pitch: float, phi: float,
                          forced: Optional[str] = None
                          ) -> Tuple[float, int, float, str]:
    """접힌 템플릿에서 street 위치/극성/폭을 한 번에 구한다.

    반환: ``(street_pos, polarity, width_frac, note)``  (0 <= street_pos < pitch)

    왜 Fourier 위상을 그대로 쓰면 안 되는가
    --------------------------------------
    ``_fourier_phase`` 는 기본 주파수 성분의 위상 = 한 주기 파형의 무게중심이라,
    실제 scribe lane 처럼 **비대칭** 한 단면에서는 실제 street 에서 밀린다.
    (측정: pitch 92 웨이퍼에서 8 px = 8.7% 오차)
    깊이 가중 중심도 마찬가지로 밀린다 — scribe lane 은 '넓고 완만한 홈 +
    그 안의 좁고 밝은 금속 중심선' 처럼 다봉(multi-modal) 이라, 중심을 잡으면
    금속선이 아니라 홈 전체의 무게중심으로 끌려간다.

    그래서 실측 기반으로 채택한 규칙은 **중앙값 대비 가장 극단적인 지점**:

      street 는 한 주기에서 통계적으로 가장 튀는 좁은 특징이다.
      die 본체는 넓어서 중앙값을 지배하므로, ``|t - median(t)|`` 가 최대인
      지점이 곧 scribe lane 이다. 최대/최소 중 어느 쪽이 이겼는지가
      그대로 극성이 되므로 **극성을 따로 맞출 필요가 없다.**

    midrange(=(max+min)/2) 기준의 '좁은 쪽이 street' 휴리스틱과 달리, die 본체가
    밝고 street 안에 밝은 선과 어두운 홈이 함께 있어도 흔들리지 않는다.
    (검증: casio p092 에서 X 61 / Y 63 — V5 의 64 / 62 와 각각 3px / 1px 차이,
     기존 방식은 8px / 3px 차이였다.)
    """
    nb = int(folded.size)
    if nb < 4 or pitch <= 0:
        return (float(phi % pitch) if pitch > 0 else 0.0), -1, 0.0, ""

    t = folded.astype(np.float64)
    # 원형 스무딩 (1~2 px 노이즈 스파이크 제거, 실제 lane 폭은 보존)
    k = max(1, nb // 48)
    if k > 1:
        ker = np.ones(2 * k + 1) / float(2 * k + 1)
        t = np.convolve(np.r_[t[-k:], t, t[:k]], ker, mode="valid")

    dev = t - float(np.median(t))
    hi, lo = int(np.argmax(dev)), int(np.argmin(dev))
    note = ""

    f = str(forced).lower().strip() if forced is not None else None
    if f in ("bright", "light", "white", "max", "+1", "1"):
        pol, b, note = +1, hi, "forced"
    elif f in ("dark", "black", "min", "-1"):
        pol, b, note = -1, lo, "forced"
    else:
        pol = +1 if float(dev[hi]) >= -float(dev[lo]) else -1
        b = hi if pol > 0 else lo

    s = dev * float(pol)                    # street 쪽이 양수인 곡선
    peak = float(s[b])
    if peak < _EPS:
        return float(phi % pitch), pol, 0.0, note

    # 서브빈 포물선 보간 (원형)
    y0, y1, y2 = float(s[(b - 1) % nb]), peak, float(s[(b + 1) % nb])
    den = y0 - 2.0 * y1 + y2
    delta = 0.0 if abs(den) < _EPS else float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))

    off = ((b + delta) / float(nb)) % 1.0
    width = float((s > 0.5 * peak).mean())   # 반치폭 기준 street 점유율
    return float((phi + off * pitch) % pitch), pol, width, note


def _analyze_channel(name: str, prof1d: np.ndarray,
                     cfg: ColorProfile,
                     fixed_pitch: Optional[float]) -> ChannelResult:
    """채널 1개의 pitch / 점수 / 극성 / street 위치를 계산."""
    res = ChannelResult(name=name)
    n = int(prof1d.size)
    if n < 64:
        res.note = "profile too short"
        return res

    max_pitch = max(cfg.min_pitch_px * 3.0, n * cfg.max_pitch_ratio)
    pitch = fixed_pitch if fixed_pitch else _spectral_pitch(
        prof1d, cfg.min_pitch_px, max_pitch)
    if not pitch or pitch <= 0:
        res.note = "no periodic peak"
        return res

    # 반주기 오검출 교정 (사용자가 pitch 를 직접 준 경우는 건드리지 않는다)
    mult_note = ""
    if not fixed_pitch:
        pitch, mult_note = _resolve_period_multiple(
            prof1d, pitch, max_pitch, cfg.min_pitch_px)
    res.pitch = float(pitch)

    # pitch 를 알았으니 그 2배 창으로 detrend -> 주기 성분만 남긴다
    xd = _detrend(prof1d, _odd(int(round(pitch * 2)) + 1))
    if float(xd.std()) < _EPS:
        res.note = "flat"
        return res

    r1 = _corr_at_lag(xd, pitch)
    r2 = _corr_at_lag(xd, pitch * 2.0)
    res.score = float(0.6 * max(0.0, r1) + 0.4 * max(0.0, r2))

    phi = _fourier_phase(xd, pitch)
    folded = _fold(xd, pitch, phi)
    pos, pol, width, note = _street_from_template(
        folded, pitch, phi, cfg.street_polarity)
    res.polarity = pol
    res.street_width_frac = float(width)
    res.note = " ".join(s for s in (mult_note, note) if s)
    res.street_pos = pos

    if cfg.street_polarity is None and not (0.02 <= width <= 0.55):
        res.score *= 0.5
        res.note = (res.note + " " if res.note else "") + \
                   f"suspicious width {width:.2f}"
    return res


def _consensus_pitch(chans: List[ChannelResult],
                     tol: float) -> Tuple[float, float, List[ChannelResult]]:
    """채널별 pitch 를 상대오차 tol 로 클러스터링 -> (pitch, agreement, 채택채널).

    점수합이 가장 큰 클러스터를 채택하고, 그 안에서 점수 가중평균을 취한다.
    agreement = 채택 클러스터 점수합 / 전체 점수합 (0~1).
    """
    live = [c for c in chans if c.score > 0 and c.pitch > 0]
    if not live:
        return 0.0, 0.0, []
    total = sum(c.score for c in live)
    best_cluster: List[ChannelResult] = []
    best_sum = -1.0
    for anchor in live:
        cl = [c for c in live
              if abs(c.pitch - anchor.pitch) / anchor.pitch <= tol]
        s = sum(c.score for c in cl)
        if s > best_sum:
            best_sum, best_cluster = s, cl
    w = sum(c.score for c in best_cluster)
    pitch = sum(c.pitch * c.score for c in best_cluster) / max(w, _EPS)
    agree = float(best_sum / max(total, _EPS))
    return float(pitch), agree, best_cluster


def _consensus_phase(chans: List[ChannelResult], pitch: float,
                     base_offset: float, scale: float) -> Tuple[float, float]:
    """채택 채널들의 street 위치를 원형평균 -> (전역 street 위상, 신뢰도 0~1).

    각 채널 위치를 전역좌표로 올린 뒤 pitch 로 모듈로 -> 각도로 바꿔 가중 원형평균.
    resultant 길이 R 이 곧 채널간 일치도(=신뢰도).
    """
    if not chans or pitch <= 0:
        return 0.0, 0.0
    acc = 0j
    wsum = 0.0
    for c in chans:
        gpos = base_offset + c.street_pos / max(scale, _EPS)
        th = 2.0 * np.pi * (gpos % pitch) / pitch
        acc += c.score * np.exp(1j * th)
        wsum += c.score
    if wsum < _EPS or abs(acc) < _EPS:
        return 0.0, 0.0
    conf = float(abs(acc) / wsum)
    ph = float((np.angle(acc) % (2.0 * np.pi)) / (2.0 * np.pi) * pitch)
    return ph, conf


# =============================================================================
# 6) 색-무관 격자 검출 (pitch + origin)
# =============================================================================
def detect_grid_adaptive(img_bgr: np.ndarray,
                         wafer_cx: int, wafer_cy: int, wafer_r: int,
                         profile: Optional[ColorProfile] = None,
                         diag: Optional[GridDiagnostics] = None
                         ) -> Tuple[float, float, int, int]:
    """색에 의존하지 않는 die 격자 검출 -> (pitch_x, pitch_y, x0, y0).

    x0, y0 는 V5 와 **동일한 규약** — 두 축 모두 "가장 가까운" 이다:
      x0 = wafer 중심에 가장 가까운 세로 street 중심
      y0 = wafer 중심에 가장 가까운 가로 street 중심
      die 중심 = (x0 + ix*px + px/2,  y0 - iy*py - py/2)

    "가장 가까운" 이므로 원점은 중심에서 최대 ±0.5 pitch 안에 들어온다.
    (예전 docstring 은 y 를 "중심보다 위" 라고 적었는데, 그건 제거된
     math.floor 구현을 설명한 것이었다. 근거는 아래 origin 결정부 주석 참조.)
    """
    cfg = profile or ColorProfile()
    img_bgr = _load_bgr(img_bgr)          # 단독 호출 대비 입력 정규화
    roi, rx1, ry1 = _grid_roi(img_bgr, wafer_cx, wafer_cy, wafer_r, cfg.roi_ratio)
    if roi.size == 0 or min(roi.shape[:2]) < 64:
        raise RuntimeError(
            f"Grid ROI too small ({roi.shape[:2] if roi.size else (0, 0)}) — "
            f"wafer_r={wafer_r}, roi_ratio={cfg.roi_ratio}. "
            "웨이퍼 검출이 실패했거나 이미지가 너무 작습니다.")
    small, scale = _downscale(roi, cfg.profile_max_dim)

    names = tuple(cfg.feature_channels) if cfg.feature_channels else FEATURE_NAMES
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Unknown feature channel(s): {unknown}. "
                         f"Available: {list(FEATURE_NAMES)}")
    feats = _feature_bank(small, names)

    out: List[float] = []
    for axis in ("x", "y"):
        # 이 축의 1D 프로파일 길이 (축소본 기준). 주기 개수 / 탐색 상한 계산에 쓴다.
        n_prof = float(small.shape[1] if axis == "x" else small.shape[0])
        chans: List[ChannelResult] = []
        for nm in names:
            f = feats[nm]
            prof = f.mean(axis=0) if axis == "x" else f.mean(axis=1)
            fixed = cfg.pitch_x if axis == "x" else cfg.pitch_y
            fixed_local = (fixed * scale) if fixed else None
            chans.append(_analyze_channel(nm, prof, cfg, fixed_local))

        pitch_local, agree, cluster = _consensus_pitch(
            [c for c in chans if c.score >= cfg.min_channel_score],
            cfg.pitch_cluster_tol)
        if not cluster:
            # 점수 문턱을 못 넘으면 문턱 없이 재시도 (약한 신호라도 쓴다)
            pitch_local, agree, cluster = _consensus_pitch(
                chans, cfg.pitch_cluster_tol)
            if diag is not None:
                diag.warn(f"[{axis}] all channels below min_channel_score="
                          f"{cfg.min_channel_score}; using weak signals")
        if not cluster or pitch_local <= 0:
            raise RuntimeError(
                f"No periodic die grid found along {axis}. "
                f"Try ColorProfile(feature_channels=..., pitch_{axis}=...).")

        for c in cluster:
            c.used = True
        pitch = pitch_local / max(scale, _EPS)

        # 채택 채널의 위상을 consensus pitch 로 재계산 (pitch 가 살짝 바뀌므로)
        for c in cluster:
            f = feats[c.name]
            prof = f.mean(axis=0) if axis == "x" else f.mean(axis=1)
            xd = _detrend(prof, _odd(int(round(pitch_local * 2)) + 1))
            phi = _fourier_phase(xd, pitch_local)
            folded = _fold(xd, pitch_local, phi)
            pos, pol, width, note = _street_from_template(
                folded, pitch_local, phi, cfg.street_polarity)
            c.polarity, c.street_width_frac = pol, float(width)
            c.street_pos = pos
            if note and note not in c.note:
                c.note = (c.note + " " if c.note else "") + note

        base = float(rx1 if axis == "x" else ry1)
        street_ph, phase_conf = _consensus_phase(cluster, pitch, base, scale)

        # grid 원점(x0,y0) = 웨이퍼 중심에 **가장 가까운** street 교차점.
        # x/y 모두 round 를 쓴다 (중심에서 최대 ±0.5 pitch).
        #
        # v5 도 동일하다 (wafer_die_map_v5.py:588-589):
        #     kx = int(math.floor((wafer_cx - x1 - phase_x - bias_x) / pitch_x + 0.5))
        #     ky = int(math.floor((wafer_cy - y1 - phase_y - bias_y) / pitch_y + 0.5))
        # `floor(z + 0.5)` 는 round-half-up 이므로 v5 는 두 축 모두 반올림이다.
        #
        # 예전에는 y 축만 math.floor 를 썼다. floor 는 -inf 쪽으로 버리는데
        # 이미지 y 는 아래로 증가하므로, 원점이 항상 중심과 같거나 그 "위"
        # 격자선으로만 잡혔다. 게다가 웨이퍼는 die 격자를 중심에 맞춰 찍기
        # 때문에 중심 바로 아래(1~3px)에 street 선이 있는데, floor 는 아래쪽을
        # 못 잡으니 그 선을 건너뛰고 한 pitch 를 통째로 올라갔다.
        # 즉 격자가 잘 맞을수록 오차가 1 pitch 에 가까워지는 구조였다
        # (실측 9장 중 7장이 -0.89 pitch 이상 어긋남. 나머지 2장은 가장 가까운
        #  선이 마침 중심 위쪽이라 floor 와 round 가 같은 답을 냈다).
        # 결과: 1) 오버레이의 자홍 원점 마커가 중심 십자에서 한 칸 떨어져 보이고
        #       2) dies_by_index 의 iy 가 v5 대비 1 만큼 밀렸다.
        # 격자선 집합 자체는 같아서 pitch/die 크기/개수에는 영향이 없다.
        anchor = wafer_cx if axis == "x" else wafer_cy
        k = round((anchor - street_ph) / pitch)
        origin = street_ph + k * pitch

        pol_votes = sum(c.score * c.polarity for c in cluster)
        pol_str = "bright" if pol_votes > 0 else "dark"

        if diag is not None:
            if axis == "x":
                diag.channels_x = chans
                diag.pitch_x = pitch
                diag.pitch_x_agreement = agree
                diag.phase_conf_x = phase_conf
                diag.polarity_x = pol_str
            else:
                diag.channels_y = chans
                diag.pitch_y = pitch
                diag.pitch_y_agreement = agree
                diag.phase_conf_y = phase_conf
                diag.polarity_y = pol_str
            if agree < 0.5:
                diag.warn(f"[{axis}] weak pitch agreement {agree:.2f} "
                          f"— channels disagree on die pitch")
            if phase_conf < 0.5:
                diag.warn(f"[{axis}] weak phase consensus {phase_conf:.2f} "
                          f"— street position uncertain")

            # ---------------------------------------------------------------
            # 배음(harmonic) 모호성 경고 — "조용히 절반으로 나온 pitch" 방지
            # ---------------------------------------------------------------
            # 탐색 상한은 max_pitch = n * max_pitch_ratio 다. 만약 참 pitch 가
            # 이 상한을 넘으면 FFT 는 그 값을 아예 후보로 볼 수 없고, 대신
            # 2배음(참값/2)에 피크가 잡힌다. 이때 모든 채널이 똑같이 절반값을
            # 보므로 agreement=1.00 이 되어 기존 경고는 하나도 울리지 않는다.
            #
            # 실측 사례: 참 pitch=400 인 3x3 die 이미지 -> 199.57 (오차 50%),
            #            agree=1.00, 경고 0건. 완전히 조용한 오답.
            #
            # 판정 기준은 "2T 가 탐색 밴드 안에 있었는가" 다.
            #   2*pitch <= max_pitch  -> 2T 도 후보였는데 밀렸다 = 신뢰 가능
            #   2*pitch >  max_pitch  -> 2T 는 검사조차 못했다 = 배제 불가
            #
            # 단, 사용자가 pitch 를 직접 지정했다면 이미 정답을 아는 것이므로
            # 경고하지 않는다 (불필요한 소음).
            n_periods = n_prof / max(pitch_local, _EPS)
            max_pitch = max(cfg.min_pitch_px * 3.0, n_prof * cfg.max_pitch_ratio)
            user_fixed = (cfg.pitch_x if axis == "x" else cfg.pitch_y) is not None
            if user_fixed:
                pass
            elif 2.0 * pitch_local > max_pitch:
                diag.warn(
                    f"[{axis}] pitch {pitch:.2f}px 는 탐색 상한 "
                    f"({max_pitch / max(scale, _EPS):.0f}px) 때문에 2배값을 "
                    f"검증하지 못했습니다 — 참 pitch 가 이 값의 2배일 수 있습니다 "
                    f"(프로파일에 약 {n_periods:.1f}주기만 보임). "
                    f"die 가 크거나 시야가 좁으면 max_pitch_ratio 를 올리거나 "
                    f"pitch_{axis} 를 직접 지정하세요.")
            elif n_periods < 4.0:
                diag.warn(
                    f"[{axis}] 프로파일에 약 {n_periods:.1f}주기만 보입니다 "
                    f"— 주기 추정이 불안정할 수 있습니다 (4주기 이상 권장).")

        out.extend([pitch, origin])

    pitch_x, x0f, pitch_y, y0f = out[0], out[1], out[2], out[3]
    return float(pitch_x), float(pitch_y), int(round(x0f)), int(round(y0f))


# =============================================================================
# 7) 색-무관 회전각(격자 기울기) 추정
# =============================================================================
def _color_edge_energy(roi_bgr: np.ndarray) -> np.ndarray:
    """Lab 3채널 그래디언트 에너지 맵 (float32).

    V5 는 회색조 |Sobel_x| 만 봤기 때문에, 밝기는 같고 색만 다른 street
    (예: 회색 die 옆의 같은 밝기 갈색 street)를 아예 못 본다.
    V6 는 L/a/b 모두의 그래디언트를 합쳐 **어떤 색 대비든** 에너지로 잡는다.
    """
    lab = _bgr_to_lab_f32(roi_bgr)
    acc = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        ch = lab[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        acc += gx * gx + gy * gy
    return np.sqrt(acc)


def _angle_projection_score(E: np.ndarray, angle_deg: float,
                            inner: int) -> float:
    """에너지 맵을 angle 만큼 회전 후, 행/열 프로파일의 '뾰족함'.

    격자가 축과 나란해지면 열/행 합 프로파일이 뾰족해진다(분산 최대).
    회전으로 생기는 검은 모서리는 항상 유효한 내접 정사각(inner)만 잘라 배제.
    평균 제곱으로 정규화해 회전에 따른 밝기 변화가 점수를 왜곡하지 않게 한다.
    """
    h, w = E.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(E, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    oy = (h - inner) // 2
    ox = (w - inner) // 2
    sub = rot[oy:oy + inner, ox:ox + inner]
    col = sub.mean(axis=0)
    row = sub.mean(axis=1)
    sc = 0.0
    for pr in (col, row):
        m = float(pr.mean())
        if m > _EPS:
            sc += float(pr.var()) / (m * m)
    return sc


def _angle_from_fft(E: np.ndarray) -> Optional[float]:
    """2D FFT 스펙트럼의 90도-주기 방향 원형평균 -> 격자 기울기 (deg).

    projection 탐색과 **독립적인** 두 번째 단서. 두 값이 일치하면 신뢰도 상승.
    격자는 0deg/90deg 두 방향에 에너지를 만들므로 alpha 를 4배각으로 접어
    (mod 90) 가중 원형평균을 취한다.
    """
    h, w = E.shape[:2]
    if min(h, w) < 64:
        return None
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2((E - E.mean()) * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    u = (xx - cx).astype(np.float64)
    v = (yy - cy).astype(np.float64)
    rad = np.sqrt(u * u + v * v)
    rmax = min(cx, cy)
    band = (rad > rmax * 0.03) & (rad < rmax * 0.90)
    if not band.any():
        return None
    wgt = F[band].astype(np.float64) ** 2
    alpha = np.arctan2(v[band], u[band])
    acc = complex(np.sum(wgt * np.exp(4j * alpha)))
    if abs(acc) < _EPS:
        return None
    ang = math.degrees(np.angle(acc)) / 4.0        # -45 .. +45
    return float(((ang + 45.0) % 90.0) - 45.0)


def estimate_grid_angle_adaptive(img_bgr: np.ndarray,
                                 wafer_cx: int, wafer_cy: int, wafer_r: int,
                                 profile: Optional[ColorProfile] = None
                                 ) -> Tuple[float, Optional[float], float, bool]:
    """격자를 축에 정렬시키는 회전각 -> (angle_proj, angle_fft, confidence, agree).

    반환 각도를 cv2.getRotationMatrix2D 의 angle 로 그대로 쓰면 정렬된다.
    projection 탐색(coarse -> fine -> 포물선)과 2D-FFT 를 교차검증한다.
    """
    cfg = profile or ColorProfile()
    img_bgr = _load_bgr(img_bgr)          # 단독 호출 대비 입력 정규화
    roi, _, _ = _grid_roi(img_bgr, wafer_cx, wafer_cy, wafer_r, 0.55)
    if roi.size == 0 or min(roi.shape[:2]) < 64:
        return 0.0, None, 0.0, False
    small, _ = _downscale(roi, cfg.angle_max_dim)
    E = _color_edge_energy(small)

    a_fft = _angle_from_fft(E)

    def search(span: float) -> float:
        amax = math.radians(min(44.0, span))
        inner = int(min(E.shape[:2]) / (math.cos(amax) + math.sin(amax)))
        inner = max(32, min(inner, min(E.shape[:2])))
        coarse = np.arange(-span, span + 1e-9, cfg.angle_coarse_step)
        sc = [_angle_projection_score(E, float(a), inner) for a in coarse]
        b = int(np.argmax(sc))
        lo = coarse[b] - cfg.angle_coarse_step
        hi = coarse[b] + cfg.angle_coarse_step
        fine = np.arange(lo, hi + 1e-9, cfg.angle_fine_step)
        sf = [_angle_projection_score(E, float(a), inner) for a in fine]
        j = int(np.argmax(sf))
        if 0 < j < len(sf) - 1:
            den = sf[j - 1] - 2.0 * sf[j] + sf[j + 1]
            if abs(den) > _EPS:
                d = 0.5 * (sf[j - 1] - sf[j + 1]) / den
                if -1.0 < d < 1.0:
                    return float(fine[j] + d * cfg.angle_fine_step)
        return float(fine[j])

    a_proj = search(cfg.angle_search_deg)

    agree = False
    if a_fft is not None:
        agree = abs(a_proj - a_fft) <= cfg.angle_agree_tol_deg
        if not agree and cfg.angle_full_scan_deg > cfg.angle_search_deg:
            # 두 단서가 어긋나면 더 넓게 재탐색 (큰 기울기를 놓친 경우)
            a2 = search(cfg.angle_full_scan_deg)
            if abs(a2 - a_fft) <= cfg.angle_agree_tol_deg:
                a_proj, agree = a2, True

    if a_fft is None:
        conf = 0.6
    elif agree:
        conf = 1.0
    else:
        d = abs(a_proj - a_fft)
        conf = float(max(0.25, 0.7 - min(d, 5.0) * 0.09))
    return float(a_proj), a_fft, float(conf), bool(agree)


# =============================================================================
# 8) Notch (V5 로직 이식 — 단, 임계 대신 '적응형 실루엣 마스크'를 입력받음)
# =============================================================================
def detect_notch_v6(sil: np.ndarray,
                    wafer_cx: int, wafer_cy: int, wafer_r: int,
                    notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                    sector_deg: float = DEFAULT_NOTCH_SECTOR_DEG,
                    n_angles: int = 14400,
                    min_depth: float = DEFAULT_NOTCH_MIN_DEPTH,
                    noise_margin: float = DEFAULT_NOTCH_NOISE_MARGIN,
                    min_span_deg: float = 0.06,
                    smooth_deg: float = DEFAULT_NOTCH_SMOOTH_DEG
                    ) -> Optional[Tuple[float, Tuple[int, int]]]:
    """wafer 아래쪽(ref +- sector_deg) 의 notch 검출 -> (angle_err_deg, center_px).

    V5 detect_notch 와 동일한 방사 스캔/원형 스무딩/적응형 임계 로직이지만,
    입력이 `gray > black_thr` 마스크가 아니라 **색-무관 실루엣 마스크**다.
    -> 배경이 흰색/갈색이어도 그대로 동작한다.
    """
    H, W = sil.shape[:2]
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
    depth = np.median(radii) - radii            # 양수 = 안쪽으로 파임

    # 둘레 원형 이동평균: 넓은 notch 는 보존, 좁은 가장자리 거칠기는 억제
    win = max(3, int(round(smooth_deg / 360.0 * n_angles)))
    if win >= 3:
        ker = np.ones(win, dtype=np.float64) / win
        padded = np.concatenate([depth[-win:], depth, depth[:win]])
        depth = np.convolve(padded, ker, mode="same")[win:win + n_angles]

    degs = np.degrees(angs)
    in_sector = np.abs(((degs - notch_ref_deg + 180.0) % 360.0) - 180.0) <= sector_deg
    above = np.where((depth > min_depth) & in_sector)[0]
    if len(above) == 0:
        return None
    clusters: List[List[int]] = [[int(above[0])]]
    for v in above[1:]:
        if int(v) - clusters[-1][-1] <= 2:
            clusters[-1].append(int(v))
        else:
            clusters.append([int(v)])
    clusters = [c for c in clusters
                if (c[-1] - c[0]) * 360.0 / n_angles >= min_span_deg]
    if not clusters:
        return None
    cand = max(clusters, key=lambda c: float(depth[c].sum()))

    noise_floor = (float(np.percentile(depth[~in_sector], 99.5))
                   if (~in_sector).any() else 0.0)
    eff_thr = max(min_depth, noise_floor + noise_margin)
    d = depth[cand]
    if float(d.max()) < eff_thr or float(d.sum()) < _EPS:
        return None

    a = angs[cand]
    notch_deg = math.degrees(math.atan2(float((np.sin(a) * d).sum()),
                                        float((np.cos(a) * d).sum()))) % 360.0
    err = ((notch_deg - notch_ref_deg + 180.0) % 360.0) - 180.0
    bx = wafer_cx + radii[cand] * np.cos(a)
    by = wafer_cy + radii[cand] * np.sin(a)
    cxp = int(round(float((bx * d).sum() / d.sum())))
    cyp = int(round(float((by * d).sum() / d.sum())))
    return float(err), (cxp, cyp)


# =============================================================================
# 9) V5 에서 그대로 가져온 기하 헬퍼 (import 대신 통째 복사)
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


def _crop_rect(cx: float, cy: float, die_w: int, die_h: int,
               offset_x: int, offset_y: int,
               margin_x: int, margin_y: int) -> Tuple[int, int, int, int]:
    """die 중심에 offset(위치 보정) + margin(영역 확장)을 적용한 crop 사각 좌표."""
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
    """die 중심 기준으로 offset/margin 을 적용해 crop 한 이미지를 반환."""
    return clip_die(image,
                    int(round(center_x + offset_x)),
                    int(round(center_y + offset_y)),
                    int(round(die_w + 2 * margin_x)),
                    int(round(die_h + 2 * margin_y)),
                    border_mode=border_mode)


def _rect_crosses_circle(x1: int, y1: int, x2: int, y2: int,
                         cx: int, cy: int, r: int) -> bool:
    """die rect 의 한 모서리라도 웨이퍼 원 밖이면 True (=부분 die)."""
    r2 = r * r
    for (px, py) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        if (px - cx) ** 2 + (py - cy) ** 2 > r2:
            return True
    return False


def _normalize_edge_mode(edge_mode: str) -> str:
    m = str(edge_mode).lower().strip()
    if m in ("circle", "partial", "disc", "crop", "1"):
        return "circle"
    if m in ("ring", "neighbor", "outer", "outermost", "grid", "2"):
        return "ring"
    if m in ("both", "or", "all", "union"):
        return "both"
    raise ValueError("edge_mode must be 'circle', 'ring', or 'both'.")


def _resolve_edge_flag(is_partial: bool, is_ring: bool, edge_mode: str) -> bool:
    if edge_mode == "circle":
        return bool(is_partial)
    if edge_mode == "ring":
        return bool(is_ring)
    return bool(is_partial or is_ring)


def _rotate_keep_size(image_bgr: np.ndarray, cx: float, cy: float,
                      angle_deg: float,
                      fill: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """웨이퍼 중심 기준 회전(출력 크기 유지). 각 0 이면 워핑 자체를 생략."""
    if abs(float(angle_deg)) < 1e-9:
        return image_bgr.copy()
    H, W = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
    return cv2.warpAffine(image_bgr, M, (W, H),
                          flags=cv2.INTER_CUBIC,
                          borderValue=(float(fill[0]), float(fill[1]),
                                       float(fill[2])))


def validate_quadrant_edges(dies: List[Dict[str, Any]],
                            wafer_cx: int, wafer_cy: int, wafer_r: int,
                            balance_tol: float = DEFAULT_QUAD_BALANCE_TOL
                            ) -> Dict[str, Any]:
    """4분면(TL/TR/BL/BR) 가장자리까지 균형 있게 채워졌는지 검증."""
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
    covs: List[float] = []
    for k, arr in quad.items():
        if arr:
            cov = max(arr) / float(wafer_r)
            per[k] = {"n_dies": len(arr), "max_dist": round(max(arr), 1),
                      "coverage": round(cov, 4), "edge_dies": edge[k]}
            covs.append(cov)
        else:
            per[k] = {"n_dies": 0, "max_dist": 0.0, "coverage": 0.0,
                      "edge_dies": 0}
            covs.append(0.0)
    spread = (max(covs) - min(covs)) if covs else 1.0
    balanced = bool(spread <= balance_tol and min(covs) > 0.8)
    return {"per_quadrant": per, "coverage_spread": round(spread, 4),
            "balanced": balanced,
            "min_coverage": round(min(covs), 4) if covs else 0.0}


# =============================================================================
# 10) ★ 메인 엔트리 : build_die_map_v6
# =============================================================================
def build_die_map_v6(image: Union[str, Path, np.ndarray],
                     *,
                     profile: Optional[ColorProfile] = None,
                     pixel_per_unit: int = DEFAULT_PIXEL_PER_UNIT,
                     include_edge: bool = True,
                     edge_margin: float = DEFAULT_EDGE_MARGIN,
                     edge_mode: str = DEFAULT_EDGE_MODE,
                     with_crops: bool = False,
                     border_mode: str = "pad",
                     offset_x: int = DEFAULT_OFFSET_X,
                     offset_y: int = DEFAULT_OFFSET_Y,
                     margin_x: int = DEFAULT_MARGIN_X,
                     margin_y: int = DEFAULT_MARGIN_Y,
                     clean: bool = True,
                     align_angle: bool = True,
                     notch_ref_deg: float = DEFAULT_NOTCH_REF_DEG,
                     notch_sector_deg: float = DEFAULT_NOTCH_SECTOR_DEG,
                     verify_angle: bool = True,
                     verify_tol_deg: float = DEFAULT_VERIFY_TOL_DEG,
                     ) -> WaferDieMapV6:
    """wafer 이미지 한 장 -> die map. **색상에 의존하지 않는 단일 진입점.**

    처리 순서
    ---------
      1. 배경색 자동 추정 -> 색-무관 wafer 검출(원 + 실루엣)
      2. clean : 웨이퍼 밖을 배경색으로 (외부 노이즈 제거)
      3. 컬러 그래디언트 기반 회전 정렬 (projection + 2D-FFT 교차검증, 반복)
      4. 다채널 주기성 격자 검출 (pitch + street 위상 -> origin)
      5. notch 검출 (적응형 실루엣 기반)
      6. die 순회 + edge 플래그 + 4분면 검증
      7. 자가진단 리포트(diagnostics) 채움

    Parameters
    ----------
    image          : 경로(str/Path) 또는 BGR ndarray
    profile        : ColorProfile. **None 이면 전부 자동.** 아는 값만 채워도 됨
                     (background_bgr / street_polarity / feature_channels /
                      pitch_x / pitch_y ...)
    pixel_per_unit : 실측 좌표 환산 (px/unit)
    include_edge   : True 면 원 안 die 전부 포함(가장자리 잘린 die 포함)
    edge_margin    : die 포함 기준 = (중심거리 <= r * edge_margin)
    edge_mode      : is_edge 기준 "circle"(기본) | "ring" | "both"
    with_crops     : True 면 각 die entry 에 "image"(crop) 포함
    border_mode    : with_crops 시 "pad" | "crop"
    offset_x/y, margin_x/y : crop 위치보정 / 영역확장 (px)
    clean          : 웨이퍼 밖을 배경색으로 채움
    align_angle    : 격자 회전 정렬 수행 여부
    verify_angle   : 정렬 후 잔여 기울기로 검증

    Returns
    -------
    WaferDieMapV6  (V5 WaferDieMap 필드 호환 + `.diagnostics`, `.wafer_mask`)
    """
    t_start = time.time()
    cfg = profile or ColorProfile()
    diag = GridDiagnostics()

    img0 = _load_bgr(image)
    if img0.ndim != 3 or img0.shape[2] != 3:
        raise ValueError("image must be a 3-channel BGR image")

    # --- 1) wafer 검출 (원본 기준) -----------------------------------------
    cx0, cy0, r0, sil0 = detect_wafer_adaptive(img0, cfg, diag)
    fill = tuple(cfg.wafer_fill_bgr) if cfg.wafer_fill_bgr is not None \
        else diag.background_bgr

    # --- 2) clean ----------------------------------------------------------
    base = clean_wafer_v6(img0, cx0, cy0, r0, sil0, fill) if clean else img0.copy()

    # --- 3) 회전 정렬 (누적각으로 항상 '원본에서 1회' 워핑 -> 이중 보간 없음) --
    rotation_deg = 0.0
    a_proj = a_fft = None
    conf, agree = 1.0, True
    img = base
    if align_angle:
        total = 0.0
        for it in range(max(1, cfg.angle_max_iter)):
            a_proj, a_fft, conf, agree = estimate_grid_angle_adaptive(
                img, cx0, cy0, r0, cfg)
            if abs(a_proj) < 0.01:
                break
            total += a_proj
            img = _rotate_keep_size(base, cx0, cy0, total, fill)
        rotation_deg = total
        diag.angle_projection = float(a_proj or 0.0)
        diag.angle_fft = a_fft
        diag.angle_agree = bool(agree)
        diag.angle_confidence = float(conf)
        diag.angle_applied = float(rotation_deg)
        if not agree and a_fft is not None:
            diag.warn(f"angle cross-check disagreed "
                      f"(projection={a_proj:+.3f}, fft={a_fft:+.3f})")

    H, W = img.shape[:2]

    # --- 회전 후 wafer 재검출 (원의 중심/반지름 미세 변화 반영) --------------
    wafer_cx, wafer_cy, wafer_r, sil = detect_wafer_adaptive(img, cfg, diag)

    # --- 4) 격자 검출 -------------------------------------------------------
    pitch_x, pitch_y, x0, y0 = detect_grid_adaptive(
        img, wafer_cx, wafer_cy, wafer_r, cfg, diag)
    die_w = int(round(pitch_x))
    die_h = int(round(pitch_y))
    if die_w < 2 or die_h < 2:
        raise RuntimeError(f"Implausible die size: {die_w}x{die_h}")

    # --- 5) notch -----------------------------------------------------------
    notch_center_px: Optional[Tuple[int, int]] = None
    nres = detect_notch_v6(sil, wafer_cx, wafer_cy, wafer_r,
                           notch_ref_deg=notch_ref_deg,
                           sector_deg=notch_sector_deg)
    if nres is not None:
        _, notch_center_px = nres
    diag.notch_found = notch_center_px is not None

    # --- 잔여 기울기 검증 ---------------------------------------------------
    die_grid_angle_resid = 0.0
    angle_verified = False
    if verify_angle:
        die_grid_angle_resid, _, _, _ = estimate_grid_angle_adaptive(
            img, wafer_cx, wafer_cy, wafer_r, cfg)
        angle_verified = bool(abs(die_grid_angle_resid) <= verify_tol_deg)
        if not angle_verified:
            diag.warn(f"residual grid tilt {die_grid_angle_resid:+.3f} deg "
                      f"> tol {verify_tol_deg}")

    # --- 6) die 순회 (V5 와 완전히 동일한 중심/실측 공식) --------------------
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
            if dx * dx + dy * dy > r_lim_sq:
                continue

            x_a = cx_d - die_w // 2
            y_a = cy_d - die_h // 2
            x_b = x_a + die_w
            y_b = y_a + die_h
            crop_rect = _crop_rect(cx_d, cy_d, die_w, die_h,
                                   offset_x, offset_y, margin_x, margin_y)
            entry: Dict[str, Any] = {
                "index": (ix, iy),
                "center_px": (cx_d, cy_d),
                "rect_px": (x_a, y_a, x_b, y_b),
                "crop_rect_px": crop_rect,
                "real_coord": ((cx_d - wafer_cx) / pixel_per_unit,
                               (wafer_cy - cy_d) / pixel_per_unit),
                "is_edge_partial": _rect_crosses_circle(
                    x_a, y_a, x_b, y_b, wafer_cx, wafer_cy, wafer_r),
                "is_edge_ring": False,
                "is_edge": False,
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

    emode = _normalize_edge_mode(edge_mode)
    present = set(dies_by_index.keys())
    for d in dies:
        ix, iy = d["index"]
        ring = any((ix + dxn, iy + dyn) not in present
                   for dxn in (-1, 0, 1) for dyn in (-1, 0, 1)
                   if not (dxn == 0 and dyn == 0))
        d["is_edge_ring"] = bool(ring)
        d["is_edge"] = _resolve_edge_flag(d["is_edge_partial"],
                                          d["is_edge_ring"], emode)

    quadrant_report = validate_quadrant_edges(dies, wafer_cx, wafer_cy, wafer_r)
    if not quadrant_report["balanced"]:
        diag.warn(f"quadrant coverage unbalanced "
                  f"(spread={quadrant_report['coverage_spread']}, "
                  f"min={quadrant_report['min_coverage']})")

    diag.n_dies = len(dies)
    diag.elapsed_sec = time.time() - t_start
    if not dies:
        diag.warn("no dies produced — grid detection is almost certainly wrong")

    return WaferDieMapV6(
        wafer_cx=wafer_cx, wafer_cy=wafer_cy, wafer_r=wafer_r,
        pitch_x=pitch_x, pitch_y=pitch_y, x0=x0, y0=y0,
        die_w=die_w, die_h=die_h, pixel_per_unit=pixel_per_unit,
        dies=dies, dies_by_index=dies_by_index, image_shape=(H, W),
        rotation_deg=rotation_deg, aligned_image=img, wafer_mask=sil,
        notch_center_px=notch_center_px,
        die_grid_angle_resid=die_grid_angle_resid,
        angle_verified=angle_verified,
        quadrant_report=quadrant_report,
        edge_mode=emode,
        angle_confidence=float(conf),
        angle_agree=bool(agree),
        diagnostics=diag,
    )


# =============================================================================
# 11) 좌표 -> die 조회 (V5 locate_die 와 동일 규약)
# =============================================================================
def locate_die_v6(die_map: WaferDieMapV6,
                  point: Optional[Tuple[float, float]] = None,
                  bbox: Optional[Tuple[float, float, float, float]] = None,
                  *,
                  offset_x: int = DEFAULT_OFFSET_X,
                  offset_y: int = DEFAULT_OFFSET_Y,
                  margin_x: int = DEFAULT_MARGIN_X,
                  margin_y: int = DEFAULT_MARGIN_Y) -> Dict[str, Any]:
    """픽셀 좌표 또는 BBox 위치의 die 정보 반환 (V5 locate_die 와 동일한 키)."""
    if (point is None) == (bbox is None):
        raise ValueError("point 또는 bbox 중 정확히 하나를 지정하세요.")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        qx = (float(x1) + float(x2)) / 2.0
        qy = (float(y1) + float(y2)) / 2.0
        input_type = "bbox"
    else:
        qx, qy = float(point[0]), float(point[1])
        input_type = "point"

    px, py = die_map.pitch_x, die_map.pitch_y
    x0, y0 = die_map.x0, die_map.y0
    ix = int(math.floor((qx - x0) / px))
    iy = int(math.floor((y0 - qy) / py))

    cx_d = int(round(x0 + ix * px + px / 2))
    cy_d = int(round(y0 - iy * py - py / 2))
    x_a = cx_d - die_map.die_w // 2
    y_a = cy_d - die_map.die_h // 2
    x_b = x_a + die_map.die_w
    y_b = y_a + die_map.die_h
    crop_rect = _crop_rect(cx_d, cy_d, die_map.die_w, die_map.die_h,
                           offset_x, offset_y, margin_x, margin_y)

    ppu = die_map.pixel_per_unit
    rx = (qx - die_map.wafer_cx) / ppu
    ry = (die_map.wafer_cy - qy) / ppu
    drx = (cx_d - die_map.wafer_cx) / ppu
    dry = (die_map.wafer_cy - cy_d) / ppu

    emode = _normalize_edge_mode(getattr(die_map, "edge_mode", DEFAULT_EDGE_MODE))
    entry = die_map.get_die(ix, iy)
    if entry is not None:
        is_edge_partial = bool(entry.get("is_edge_partial",
                                         entry.get("is_edge", False)))
        is_edge_ring = bool(entry.get("is_edge_ring", False))
    else:
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
        "input_type": input_type,
        "query_px": (qx, qy),
        "die_index": (ix, iy),
        "die_center_px": (cx_d, cy_d),
        "die_rect_px": (x_a, y_a, x_b, y_b),
        "crop_rect_px": crop_rect,
        "real_coord": (rx, ry),
        "real_distance": math.hypot(rx, ry),
        "die_real_coord": (drx, dry),
        "wafer_center_px": (die_map.wafer_cx, die_map.wafer_cy),
        "corner_px": (die_map.x0, die_map.y0),
        "is_edge": is_edge,
        "is_edge_partial": is_edge_partial,
        "is_edge_ring": is_edge_ring,
        "edge_mode": emode,
        "in_wafer": bool(in_wafer),
    }


# =============================================================================
# 12) 검증 1 - 디버그 오버레이 이미지 저장
# =============================================================================
def _imwrite_unicode(path: str, img: np.ndarray) -> str:
    """한글/유니코드 경로에서도 안전하게 저장 (cv2.imwrite 는 실패함)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    buf.tofile(path)
    return path


def _put_text(canvas: np.ndarray, text: str, org: Tuple[int, int],
              scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255),
              thick: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def save_debug_overlay(die_map: WaferDieMapV6,
                       path: str,
                       *,
                       max_dim: int = 2000,
                       draw_grid: bool = True,
                       draw_dies: bool = True,
                       draw_panel: bool = True) -> str:
    """정렬된 웨이퍼 이미지 위에 판정 결과를 그려서 저장.

    색상 규약
    --------
    노랑    웨이퍼 원 + 중심 십자 (wafer_cx, wafer_cy)
    주황선  grid street 격자 (x0/y0 lattice)
    자홍    grid 원점(x0,y0) / notch 위치
    초록    내부(full) die
    청록    edge die

    참고 — 주황 격자선은 die 사각형과 정확히 같은 자리에 놓이는데,
    die 를 나중에 그리므로 ``draw_dies=True`` 면 대부분 초록/청록에
    덮여 보이지 않는다. 격자선만 보려면 ``draw_dies=False`` 로 호출하라.

    자홍 X 표시(grid 원점)는 웨이퍼 중심에 **가장 가까운** street 교차점이라
    노랑 십자와 반 pitch 이내로 붙어 있어야 정상이다. 한 칸 이상 떨어져
    보인다면 pitch 나 phase 추정이 틀린 것이므로 리포트를 확인하라.
    """
    canvas = die_map.aligned_image
    if canvas is None:
        raise ValueError("die_map.aligned_image 가 없습니다.")
    canvas = canvas.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    H, W = canvas.shape[:2]
    cx, cy, r = die_map.wafer_cx, die_map.wafer_cy, die_map.wafer_r
    px, py = float(die_map.pitch_x), float(die_map.pitch_y)
    lw = max(1, int(round(min(H, W) / 1400.0)))

    # --- 웨이퍼 원 --------------------------------------------------------
    cv2.circle(canvas, (cx, cy), r, (0, 255, 255), lw * 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS,
                   max(8, int(r * 0.10)), lw * 2, cv2.LINE_AA)

    # --- grid street 격자 -------------------------------------------------
    if draw_grid and px > 1.0 and py > 1.0:
        x0, y0 = float(die_map.x0), float(die_map.y0)
        k0 = int(math.floor((0 - x0) / px)) - 1
        k1 = int(math.ceil((W - x0) / px)) + 1
        for k in range(k0, k1 + 1):
            xv = int(round(x0 + k * px))
            if -2 <= xv <= W + 2:
                cv2.line(canvas, (xv, 0), (xv, H - 1), (0, 165, 255), lw, cv2.LINE_AA)
        m0 = int(math.floor((0 - y0) / py)) - 1
        m1 = int(math.ceil((H - y0) / py)) + 1
        for m in range(m0, m1 + 1):
            yv = int(round(y0 + m * py))
            if -2 <= yv <= H + 2:
                cv2.line(canvas, (0, yv), (W - 1, yv), (0, 165, 255), lw, cv2.LINE_AA)
        cv2.drawMarker(canvas, (int(round(x0)), int(round(y0))), (255, 0, 255),
                       cv2.MARKER_TILTED_CROSS, max(10, int(max(px, py) * 0.8)),
                       lw * 3, cv2.LINE_AA)

    # --- die 사각형 -------------------------------------------------------
    if draw_dies:
        for d in die_map.dies:
            xa, ya, xb, yb = d["rect_px"]
            col = (255, 255, 0) if d["is_edge"] else (0, 220, 0)
            cv2.rectangle(canvas, (int(xa), int(ya)), (int(xb) - 1, int(yb) - 1),
                          col, lw, cv2.LINE_AA)

    # --- notch ------------------------------------------------------------
    if die_map.notch_center_px is not None:
        nx, ny = int(die_map.notch_center_px[0]), int(die_map.notch_center_px[1])
        cv2.arrowedLine(canvas, (cx, cy), (nx, ny), (255, 0, 255),
                        lw * 2, cv2.LINE_AA, tipLength=0.03)
        cv2.circle(canvas, (nx, ny), max(6, int(r * 0.03)), (255, 0, 255),
                   lw * 3, cv2.LINE_AA)

    # --- 정보 패널 --------------------------------------------------------
    if draw_panel:
        g = die_map.diagnostics
        b = g.background_bgr
        cxs = max(g.channels_x, key=lambda c: c.score).name if g.channels_x else "-"
        cys = max(g.channels_y, key=lambda c: c.score).name if g.channels_y else "-"
        fft_s = "n/a" if g.angle_fft is None else f"{g.angle_fft:+.3f}"
        lines = [
            "wafer_color_v6_claude   COLOR-AGNOSTIC DIE MAP",
            f"wafer    c=({cx},{cy})  r={r}   cov={g.wafer_coverage:.3f}"
            + (f"  FB:{g.wafer_fallback}" if g.wafer_fallback else ""),
            f"bg BGR   ({b[0]},{b[1]},{b[2]})  [{g.background_source}]",
            f"pitch    x={px:8.3f}  y={py:8.3f} px",
            f"origin   x0={die_map.x0}  y0={die_map.y0}",
            f"channel  x={cxs}  y={cys}",
            f"street   x={g.polarity_x}  y={g.polarity_y}",
            f"agree    pitch=({g.pitch_x_agreement:.2f},{g.pitch_y_agreement:.2f})"
            f"  phase=({g.phase_conf_x:.2f},{g.phase_conf_y:.2f})",
            f"angle    applied={g.angle_applied:+.3f}  fft={fft_s}"
            f"  conf={g.angle_confidence:.2f}",
            f"notch    {'found' if g.notch_found else 'not found'}",
            f"dies     {die_map.num_dies}"
            f"  (edge {sum(1 for d in die_map.dies if d['is_edge'])})",
            f"OVERALL  {'OK' if g.ok else 'CHECK WARNINGS'}"
            f"   elapsed={g.elapsed_sec:.2f}s",
        ]
        for w in g.warnings[:3]:
            lines.append(f"  ! {w[:60]}")

        fs = max(0.45, min(H, W) / 2400.0)
        th = max(1, int(round(fs * 2.2)))
        (_, tw_h), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        step = int(round(tw_h * 1.9)) + 4
        pad = int(round(10 * max(1.0, fs / 0.5)))
        box_w = min(W - 4, max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                               fs, th)[0][0]
                               for t in lines) + pad * 2)
        box_h = min(H - 4, step * len(lines) + pad * 2)
        sub = canvas[2:2 + box_h, 2:2 + box_w]
        cv2.addWeighted(sub, 0.28, np.zeros_like(sub), 0.72, 0, sub)
        for i, t in enumerate(lines):
            col = (255, 255, 255)
            if t.lstrip().startswith("!"):
                col = (0, 200, 255)
            elif t.startswith("OVERALL"):
                col = (0, 255, 0) if g.ok else (0, 165, 255)
            _put_text(canvas, t, (2 + pad, 2 + pad + step * (i + 1) - 4),
                      scale=fs, color=col, thick=th)

    # --- 축소 후 저장 -----------------------------------------------------
    m = max(canvas.shape[:2])
    if max_dim and m > max_dim:
        s = max_dim / float(m)
        canvas = cv2.resize(canvas,
                            (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                            interpolation=cv2.INTER_AREA)
    return _imwrite_unicode(path, canvas)


# =============================================================================
# 13) 검증 2 - 색상 변형 테스트 하네스 : 변형 생성기
# =============================================================================
def _apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(img, lut.astype(np.uint8))


def _v_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return _apply_lut(img, np.clip((x ** gamma) * 255.0, 0, 255))


def _v_invert(img: np.ndarray) -> np.ndarray:
    return (255 - img)


def _v_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _v_hue_shift(img: np.ndarray, deg: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16) + int(round(deg / 2.0))  # OpenCV H: 0..179
    hsv[:, :, 0] = np.mod(h, 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _v_desaturate(img: np.ndarray, keep: float) -> np.ndarray:
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, keep, g, 1.0 - keep, 0.0)


def _v_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _v_tint(img: np.ndarray, bgr_gain: Tuple[float, float, float]) -> np.ndarray:
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= bgr_gain[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def _v_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _v_illumination(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """대각선 밝기 그라디언트 — 조명 불균일 시뮬레이션."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = (xx / max(1, W - 1) + yy / max(1, H - 1)) * 0.5      # 0..1
    gain = (1.0 - strength) + 2.0 * strength * ramp
    out = img.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _v_white_streets_brown_noise(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """사용자가 지적한 실패 케이스 재현.

    'die 사이(street) 가 흰색 + 갈색 노이즈로 변하는' 상황.
    웨이퍼 내부만 반전시켜 어두운 street 를 밝게 만들고,
    밝아진 영역에만 갈색 톤 노이즈를 섞는다. 배경은 그대로 둬서
    '배경은 어두운데 street 는 밝다' 는 역전 상황을 만든다.
    """
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]

    # 웨이퍼 영역 추정 (변형 전 원본 기준)
    try:
        cx, cy, r, sil = detect_wafer_adaptive(img)
        mask = sil > 0
    except Exception:
        mask = np.ones((H, W), dtype=bool)

    inv = (255 - img)
    out = img.copy()
    out[mask] = inv[mask]

    # 반전 후 '밝은' 영역 = 원래 street
    lum = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    thr = float(np.percentile(lum[mask], 70)) if mask.any() else 128.0
    street = mask & (lum > thr)
    if street.any():
        f = out.astype(np.float32)
        # 흰색으로 밀어올리기
        f[street] = f[street] * 0.35 + 255.0 * 0.65
        # 갈색(BGR ~ (40,80,140)) 노이즈 주입
        brown = np.array([40.0, 80.0, 140.0], dtype=np.float32)
        n = rng.normal(0.0, 1.0, (H, W, 1)).astype(np.float32)
        n = np.clip(n, -2.5, 2.5)
        speck = (rng.random((H, W, 1)) < 0.30).astype(np.float32)
        blend = (0.30 + 0.20 * n) * speck
        blend = np.clip(blend, 0.0, 0.6)
        f[street] = (f[street] * (1.0 - blend[street])
                     + brown[None, :] * blend[street])
        out = np.clip(f, 0, 255).astype(np.uint8)
    return out


#: 이름 -> 변형 함수. baseline("original") 은 항상 첫 항목.
COLOR_VARIANTS: Dict[str, Any] = {
    "original":            lambda im: im.copy(),
    "invert":              _v_invert,
    "gray":                _v_gray,
    "hue+60":              lambda im: _v_hue_shift(im, 60.0),
    "hue+150":             lambda im: _v_hue_shift(im, 150.0),
    "desat0.25":           lambda im: _v_desaturate(im, 0.25),
    "gamma0.5":            lambda im: _v_gamma(im, 0.5),
    "gamma2.0":            lambda im: _v_gamma(im, 2.0),
    "lowcontrast":         lambda im: _v_contrast(im, 0.35, 96.0),
    "brightbg":            lambda im: _v_contrast(im, 0.60, 110.0),
    "sepia":               lambda im: _v_tint(_v_gray(im), (0.62, 0.86, 1.12)),
    "cool_tint":           lambda im: _v_tint(im, (1.25, 1.00, 0.72)),
    "noise12":             lambda im: _v_noise(im, 12.0, seed=1),
    "illum_grad":          _v_illumination,
    "white_street_brown":  _v_white_streets_brown_noise,
}


# =============================================================================
# 14) 검증 3 - 색상 변형 테스트 하네스 : 실행/집계
# =============================================================================
def _v5_build(img_bgr: np.ndarray, **kw) -> Dict[str, Any]:
    """V5 를 (있으면) 지연 임포트해서 같은 이미지에 돌린다.

    V6 는 V5 에 의존하지 않는다 — 오직 이 비교 하네스 안에서만 선택적으로
    불러오며, 없거나 실패하면 조용히 skip 한다.
    """
    import importlib
    import inspect
    m = importlib.import_module("wafer_die_map_v5")
    # V6 전용 kwarg 가 섞여 있으므로 V5 가 받는 것만 골라 넘긴다
    accepted = set(inspect.signature(m.build_die_map).parameters)
    dm = m.build_die_map(img_bgr, **{k: v for k, v in kw.items()
                                     if k in accepted})
    return {
        "pitch_x": float(dm.pitch_x), "pitch_y": float(dm.pitch_y),
        "n_dies": int(len(dm.dies)),
        "wafer_r": int(dm.wafer_r),
        "cx": int(dm.wafer_cx), "cy": int(dm.wafer_cy),
    }


def _v6_build(img_bgr: np.ndarray, profile: Optional[ColorProfile] = None,
              **kw) -> Dict[str, Any]:
    dm = build_die_map_v6(img_bgr, profile=profile, **kw)
    g = dm.diagnostics
    return {
        "pitch_x": float(dm.pitch_x), "pitch_y": float(dm.pitch_y),
        "n_dies": int(dm.num_dies),
        "wafer_r": int(dm.wafer_r),
        "cx": int(dm.wafer_cx), "cy": int(dm.wafer_cy),
        "ok": bool(g.ok),
        "conf": float(min(g.pitch_x_agreement, g.pitch_y_agreement,
                          g.phase_conf_x, g.phase_conf_y)),
        "chan_x": (max(g.channels_x, key=lambda c: c.score).name
                   if g.channels_x else "-"),
        "chan_y": (max(g.channels_y, key=lambda c: c.score).name
                   if g.channels_y else "-"),
    }


def _deviation(res: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, float]:
    def rel(k: str) -> float:
        b = float(base.get(k) or 0.0)
        v = float(res.get(k) or 0.0)
        return abs(v - b) / b if b > 1e-9 else (0.0 if v == 0 else 1.0)
    return {
        "d_pitch_x": rel("pitch_x"),
        "d_pitch_y": rel("pitch_y"),
        "d_dies": rel("n_dies"),
        "d_r": rel("wafer_r"),
        "d_center": math.hypot(res["cx"] - base["cx"], res["cy"] - base["cy"]),
    }


def run_color_robustness_test(image: Union[str, Path, np.ndarray],
                              *,
                              variants: Optional[Sequence[str]] = None,
                              compare_v5: bool = True,
                              pitch_tol: float = 0.02,
                              dies_tol: float = 0.03,
                              overlay_dir: Optional[str] = None,
                              profile: Optional[ColorProfile] = None,
                              verbose: bool = True,
                              **build_kw) -> Dict[str, Any]:
    """같은 웨이퍼를 여러 색상으로 변형해서 검출이 흔들리는지 측정.

    판정 기준
    --------
    baseline("original") 결과 대비
      * pitch_x / pitch_y 상대오차 < ``pitch_tol`` (기본 2%)
      * die 개수 상대오차 < ``dies_tol`` (기본 3%)
    둘 다 만족하면 PASS. 예외가 나면 FAIL(ERROR).

    Returns
    -------
    dict : {"v6": {...}, "v5": {...}, "summary": {...}}
    """
    img0 = _load_bgr(image)
    src_name = (os.path.splitext(os.path.basename(str(image)))[0]
                if isinstance(image, (str, Path)) else "array")
    names = list(variants) if variants else list(COLOR_VARIANTS.keys())
    if "original" in names:
        names.remove("original")
    names.insert(0, "original")

    engines: List[Tuple[str, Any]] = [("v6", lambda im: _v6_build(im, profile, **build_kw))]
    if compare_v5:
        try:
            import importlib
            importlib.import_module("wafer_die_map_v5")
            engines.append(("v5", lambda im: _v5_build(im, **build_kw)))
        except Exception as e:      # noqa: BLE001
            if verbose:
                print(f"[info] V5 비교 생략 ({type(e).__name__}: {e})")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k, _ in engines}
    variant_imgs: Dict[str, np.ndarray] = {}

    for name in names:
        fn = COLOR_VARIANTS.get(name)
        if fn is None:
            if verbose:
                print(f"[warn] 알 수 없는 변형: {name}")
            continue
        try:
            vim = fn(img0)
        except Exception as e:      # noqa: BLE001
            if verbose:
                print(f"[warn] 변형 생성 실패 {name}: {e}")
            continue
        variant_imgs[name] = vim

        for eng, runner in engines:
            t0 = time.time()
            try:
                r = runner(vim)
                r["error"] = None
            except Exception as e:  # noqa: BLE001
                r = {"pitch_x": 0.0, "pitch_y": 0.0, "n_dies": 0,
                     "wafer_r": 0, "cx": 0, "cy": 0,
                     "error": f"{type(e).__name__}: {e}"}
            r["sec"] = time.time() - t0
            results[eng][name] = r

    # ---- baseline 대비 편차 + PASS/FAIL ----------------------------------
    summary: Dict[str, Any] = {}
    for eng, _ in engines:
        base = results[eng].get("original")
        n_pass = n_fail = 0
        for name, r in results[eng].items():
            if r.get("error") or base is None or base.get("error"):
                r["pass"] = False
                r["dev"] = {}
            else:
                dev = _deviation(r, base)
                r["dev"] = dev
                r["pass"] = (dev["d_pitch_x"] < pitch_tol
                             and dev["d_pitch_y"] < pitch_tol
                             and dev["d_dies"] < dies_tol)
            n_pass += int(bool(r["pass"]))
            n_fail += int(not r["pass"])
        total = max(1, n_pass + n_fail)
        summary[eng] = {"pass": n_pass, "fail": n_fail, "total": total,
                        "rate": n_pass / total}

    # ---- 오버레이 저장 ---------------------------------------------------
    if overlay_dir:
        for name, vim in variant_imgs.items():
            r = results["v6"].get(name, {})
            if r.get("error"):
                continue
            try:
                dm = build_die_map_v6(vim, profile=profile, **build_kw)
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                               for ch in name)
                save_debug_overlay(
                    dm, os.path.join(overlay_dir, f"{src_name}__{safe}.png"))
            except Exception as e:      # noqa: BLE001
                if verbose:
                    print(f"[warn] 오버레이 저장 실패 {name}: {e}")

    if verbose:
        print(_format_robustness_table(src_name, names, results, summary,
                                       pitch_tol, dies_tol))

    out: Dict[str, Any] = {k: results[k] for k in results}
    out["summary"] = summary
    out["source"] = src_name
    return out


def _format_robustness_table(src: str, names: Sequence[str],
                             results: Dict[str, Dict[str, Dict[str, Any]]],
                             summary: Dict[str, Any],
                             pitch_tol: float, dies_tol: float) -> str:
    L: List[str] = []
    L.append("=" * 96)
    L.append(f" COLOR ROBUSTNESS HARNESS   source={src}"
             f"   pitch_tol={pitch_tol:.1%}  dies_tol={dies_tol:.1%}")
    L.append("=" * 96)
    engines = list(results.keys())
    head = f" {'variant':<20s}"
    for eng in engines:
        head += f"|{eng:^37s}"
    L.append(head)
    sub = f" {'':<20s}"
    for _ in engines:
        sub += f"| {'pitch_x':>8s} {'pitch_y':>8s} {'dies':>6s} {'':>3s} {'':>6s}"
    L.append(sub)
    L.append("-" * 96)
    for name in names:
        row = f" {name:<20s}"
        for eng in engines:
            r = results[eng].get(name)
            if r is None:
                row += f"|{'-':^37s}"
            elif r.get("error"):
                row += f"| {('ERR ' + r['error'])[:34]:<35s}  "
            else:
                mark = "OK " if r.get("pass") else "BAD"
                dev = r.get("dev") or {}
                worst = max([dev.get("d_pitch_x", 0.0),
                             dev.get("d_pitch_y", 0.0),
                             dev.get("d_dies", 0.0)] or [0.0])
                row += (f"| {r['pitch_x']:8.3f} {r['pitch_y']:8.3f}"
                        f" {r['n_dies']:6d} {mark:>3s} {worst:5.2%}")
        L.append(row)
    L.append("-" * 96)
    tail = f" {'TOTAL':<20s}"
    for eng in engines:
        s = summary[eng]
        tail += f"| {s['pass']:d}/{s['total']:d} pass  ({s['rate']:.0%}){'':<16s}"
    L.append(tail)
    L.append("=" * 96)
    return "\n".join(L)


# =============================================================================
# 15) CLI
# =============================================================================
def _build_profile_from_args(a) -> Optional[ColorProfile]:
    """--bg / --pitch / --polarity / --channels 등 수동 파라미터를 프로필로."""
    kw: Dict[str, Any] = {}
    if a.bg:
        parts = [int(v) for v in a.bg.split(",")]
        if len(parts) != 3:
            raise SystemExit("--bg 는 B,G,R 세 개여야 합니다. 예: --bg 255,255,255")
        kw["background_bgr"] = (parts[0], parts[1], parts[2])
    if a.pitch:
        parts = [float(v) for v in a.pitch.split(",")]
        if len(parts) == 1:
            kw["pitch_x"] = kw["pitch_y"] = parts[0]
        elif len(parts) == 2:
            kw["pitch_x"], kw["pitch_y"] = parts[0], parts[1]
        else:
            raise SystemExit("--pitch 는 P 또는 PX,PY 형식입니다.")
    if a.polarity:
        m = {"bright": 1, "dark": -1, "auto": 0}
        if a.polarity not in m:
            raise SystemExit("--polarity 는 bright|dark|auto 중 하나입니다.")
        kw["street_polarity"] = m[a.polarity]
    if a.channels:
        chans = tuple(c.strip() for c in a.channels.split(",") if c.strip())
        bad = [c for c in chans if c not in FEATURE_NAMES]
        if bad:
            raise SystemExit(f"알 수 없는 채널 {bad}. 사용 가능: {list(FEATURE_NAMES)}")
        kw["feature_channels"] = chans
    if a.min_pitch:
        kw["min_pitch_px"] = float(a.min_pitch)
    if a.roi:
        kw["roi_ratio"] = float(a.roi)
    return ColorProfile(**kw) if kw else None


def _iter_images(paths: Sequence[str]) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in exts:
                    out.append(os.path.join(p, f))
        else:
            out.append(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    # Windows cp949 콘솔에서 유니코드 리포트가 깨지지 않도록
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(
        prog="wafer_color_v6_claude",
        description="색상 비의존(color-agnostic) 웨이퍼 die map 검출 V6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python wafer_color_v6_claude.py Img/real_piper_top_p088.png --report\n"
            "  python wafer_color_v6_claude.py Img --report --overlay out/\n"
            "  python wafer_color_v6_claude.py Img/x.png --robustness --overlay out/\n"
            "  python wafer_color_v6_claude.py Img/x.png --pitch 128 --polarity bright\n"
        ))
    # nargs="*" 이어야 `--list-variants` 만 단독으로 쓸 수 있다.
    # (이미지 필수 여부는 parse 후에 직접 검사한다)
    ap.add_argument("images", nargs="*", help="이미지 파일 또는 디렉터리")
    ap.add_argument("--report", action="store_true", help="자가진단 리포트 출력")
    ap.add_argument("--json", metavar="PATH", help="진단 결과 JSON 저장")
    ap.add_argument("--overlay", metavar="DIR", help="디버그 오버레이 저장 디렉터리")
    ap.add_argument("--robustness", action="store_true",
                    help="색상 변형 테스트 하네스 실행")
    ap.add_argument("--variants", help="테스트할 변형 이름 (쉼표구분)")
    ap.add_argument("--no-v5", action="store_true", help="V5 비교 생략")
    ap.add_argument("--list-variants", action="store_true",
                    help="사용 가능한 색상 변형 이름 출력 후 종료")

    ap.add_argument("--ppu", type=int, default=DEFAULT_PIXEL_PER_UNIT,
                    help=f"pixel per unit (기본 {DEFAULT_PIXEL_PER_UNIT})")
    ap.add_argument("--edge-mode", default=DEFAULT_EDGE_MODE,
                    help="edge die 판정 모드 (circle|ring|both|any)")
    ap.add_argument("--no-align", action="store_true", help="회전 정렬 생략")
    ap.add_argument("--no-clean", action="store_true", help="웨이퍼 외부 정리 생략")

    ap.add_argument("--bg", help="배경색 수동 지정 B,G,R")
    ap.add_argument("--pitch", help="pitch 수동 지정 P 또는 PX,PY (px)")
    ap.add_argument("--polarity", help="street 밝기 극성 bright|dark|auto")
    ap.add_argument("--channels", help="사용할 특징 채널 (쉼표구분)")
    ap.add_argument("--min-pitch", type=float, help="최소 pitch (px)")
    ap.add_argument("--roi", type=float, help="grid 분석 ROI 비율 (0~1)")
    a = ap.parse_args(argv)

    if a.list_variants:
        print("사용 가능한 색상 변형:")
        for k in COLOR_VARIANTS:
            print(f"  {k}")
        return 0

    if not a.images:
        ap.error("이미지 파일 또는 디렉터리를 하나 이상 지정하세요.")

    profile = _build_profile_from_args(a)
    build_kw: Dict[str, Any] = dict(
        pixel_per_unit=a.ppu,
        edge_mode=a.edge_mode,
        align_angle=not a.no_align,
        clean=not a.no_clean,
    )

    files = _iter_images(a.images)
    if not files:
        print("처리할 이미지가 없습니다.")
        return 2

    all_json: Dict[str, Any] = {}
    rc = 0
    for path in files:
        print(f"\n### {path}")
        if a.robustness:
            try:
                res = run_color_robustness_test(
                    path,
                    variants=(a.variants.split(",") if a.variants else None),
                    compare_v5=not a.no_v5,
                    overlay_dir=a.overlay,
                    profile=profile,
                    verbose=True,
                    **build_kw)
                all_json[path] = {"robustness": res["summary"]}
                if res["summary"].get("v6", {}).get("fail", 0) > 0:
                    rc = max(rc, 1)
            except Exception as e:      # noqa: BLE001
                print(f"[ERROR] {type(e).__name__}: {e}")
                rc = 3
            continue

        try:
            dm = build_die_map_v6(path, profile=profile, **build_kw)
        except Exception as e:          # noqa: BLE001
            print(f"[ERROR] {type(e).__name__}: {e}")
            rc = 3
            continue

        if a.report:
            print(dm.diagnostics.report())
        else:
            g = dm.diagnostics
            print(f"  pitch=({dm.pitch_x:.3f},{dm.pitch_y:.3f})  dies={dm.num_dies}"
                  f"  r={dm.wafer_r}  angle={g.angle_applied:+.3f}"
                  f"  {'OK' if g.ok else 'CHECK'}")
        all_json[path] = dm.diagnostics.to_dict()

        if a.overlay:
            base = os.path.splitext(os.path.basename(path))[0]
            out = save_debug_overlay(dm, os.path.join(a.overlay, f"{base}__v6.png"))
            print(f"  overlay -> {out}")

    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        print(f"\nJSON -> {a.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
