# 격자 원점 `(x0, y0)` 검출 로직 — 전체 정리

> 대상: `wafer_color_v6_claude.py` 의 **자홍색 X 마커**(오버레이의 grid origin).
> 목적: 어느 줄을 고치면 원점이 어떻게 움직이는지 한눈에 파악.

---

## 0. 먼저: "코너"라는 말이 세 군데에서 다른 뜻으로 쓰인다

| 위치 | 이름 | 실제 뜻 | 이 문서 대상 |
|---|---|---|---|
| `:1008` `_estimate_background()` | 이미지 4귀퉁이 | 배경색 추정용 픽셀 샘플 | ✗ |
| `:1679-1681` `detect_grid_adaptive()` | **grid origin `(x0,y0)`** | **웨이퍼 중심에 가장 가까운 street 교차점** | **✓** |
| `:2376` `locate_die_v6()` `corner_px` | 반환 필드 | `(die_map.x0, die_map.y0)` — 위와 **같은 값** | (파생) |

`corner_px` 는 "die 의 좌상단"이 아니라 **격자 원점**이다. v5 도 동일
(`wafer_die_map_v5.py:2200, 2287, 2400` 이 "격자 코너(원점)"으로 문서화).

---

## 1. 한 줄 요약

```python
# wafer_color_v6_claude.py:1679-1681
anchor = wafer_cx if axis == "x" else wafer_cy
k      = round((anchor - street_ph) / pitch)
origin = street_ph + k * pitch
```

`street_ph` 는 "street 격자선이 어디서 시작하나"(0 ≤ street_ph < pitch),
`pitch` 는 격자 간격. 이 둘로 만들어지는 **무한한 격자선 집합** 중에서
웨이퍼 중심(`anchor`)에 **가장 가까운 하나**를 고르는 것이 전부다.

따라서 원점 오류는 반드시 아래 셋 중 하나다.

| 증상 | 원인 위치 |
|---|---|
| 원점이 한 칸(=1 pitch) 통째로 밀림 | `:1680` **선택 규칙** (`round`/`floor`/`anchor`) |
| 원점이 격자선에서 몇 px 어긋남 | `:1658` **street 위상** (`_consensus_phase` 이하) |
| 격자 간격 자체가 틀림 | `:1624` **pitch 합의** (`_consensus_pitch` 이하) |

---

## 2. 호출 흐름도 (행번호 포함)

```
build_die_map_v6()                                        :2092
│
├─ _load_bgr()                                            :900
├─ detect_wafer_adaptive()          -> cx0, cy0, r0, sil0  :1054
├─ clean_wafer_v6()                 -> base               :1170
│     웨이퍼 바깥을 배경색으로 칠함 (ROI 통계 오염 방지)
│
├─ ┌ 회전 정렬 루프 ┐                                      :2166-2174
│  │ estimate_grid_angle_adaptive()                       :1821
│  │ _rotate_keep_size()                                  :2042
│  └ -> img (정렬본), rotation_deg     ※ casio = +0.9505°
│
├─ detect_wafer_adaptive()  (재검출)                       :2188
│     회전 후 원의 중심/반지름이 미세하게 바뀌므로 다시 잡음
│
└─ detect_grid_adaptive(img, wafer_cx, wafer_cy, wafer_r)  :1579
   │        ※ 원본이 아니라 **정렬된 이미지** 위에서 돈다
   │
   ├─ _grid_roi()          half = r * roi_ratio           :1231  (호출 :1597)
   ├─ _downscale()         max_dim = profile_max_dim      :959   (호출 :1603)
   ├─ _feature_bank()      7채널 생성                      :1192  (호출 :1610)
   │      L, a, b, chroma, sat, maxmin, stdL              :462
   │
   ├─ for axis in ("x","y"):                              :1613
   │   │
   │   ├─ prof = feat.mean(axis=0 or 1)   1D 프로파일      :1619
   │   │
   │   ├─ for 채널 in 7개:  _analyze_channel()             :1478  (호출 :1622)
   │   │     ├─ _spectral_pitch()   FFT 로 pitch          :1246
   │   │     ├─ _resolve_period_multiple()  배음 정리      :1308
   │   │     ├─ _detrend() + _corr_at_lag() -> score      :983 / :997
   │   │     ├─ _fourier_phase()    거친 위상 φ            :1381
   │   │     ├─ _fold()             1주기로 접기           :1397
   │   │     └─ _street_from_template()  street 위치/극성  :1411
   │   │
   │   ├─ _consensus_pitch()   -> pitch, agree, cluster   :1528  (호출 :1624)
   │   │     점수합 최대 클러스터 채택, 점수 가중평균
   │   │
   │   ├─ cluster 채널만 consensus pitch 로 위상 재계산     :1644-1655
   │   │     (pitch 가 살짝 바뀌었으므로 fold/street 다시)
   │   │
   │   ├─ _consensus_phase()   -> street_ph, phase_conf   :1553  (호출 :1658)
   │   │     전역좌표 mod pitch -> 각도 -> 점수 가중 원형평균
   │   │     |resultant| = phase_conf
   │   │
   │   └─ ★ origin 결정                                    :1679-1681
   │
   └─ return pitch_x, pitch_y, int(round(x0)), int(round(y0))  :1744
```

---

## 3. 단계별 시각화 자료 대응표

`viz_origin_claude.py` 가 생성하는 PNG 와 위 흐름도의 대응:

| 파일 | 보여주는 것 | 대응 행 |
|---|---|---|
| `01_wafer_roi.png` | 정렬본 위의 웨이퍼 원 / ROI 사각형 / 원점 X | `:2166-2188`, `:1597` |
| `02_{x,y}_channels.png` | 7채널 1D 프로파일 + score + 검출된 street 선 | `:1610`, `:1619`, `:1478` |
| `03_{x,y}_fft.png` | 채널별 FFT 파워 vs pitch, 탐색 밴드 표시 | `:1246` |
| `04_{x,y}_fold_street.png` | 1주기로 접은 템플릿 + street 위치/극성/폭 | `:1397`, `:1411` |
| `05_{x,y}_origin.png` | 원형 위상 다이어그램 + `k`/`origin` 수식 | `:1553`, `:1679-1681` |
| `06_zoom_verify.png` | 원점 주변 확대 — 중심 십자 vs 원점 X | 최종 검증 |
| `summary.txt` | 수치 요약 + **`build_die_map_v6()` 와 자가대조** | — |

실행:

```bash
python viz_origin_claude.py Img/real_casio_top_p092.png --out viz_out
```

`summary.txt` 2행이 항상 `MATCH` 여야 한다. 이 자가대조는 시각화가
실제 파이프라인과 어긋나는 것을 막는 안전장치다 — 실제로 개발 중
회전 정렬 단계를 빠뜨려 `(1549,1535)` vs `(1533,1539)` 로 어긋난 적이 있고,
이 체크가 그걸 잡아냈다.

---

## 4. 수정 지점 카탈로그

### ★ 4-1. 격자선 선택 규칙 — `:1679-1681`

가장 직접적인 수정 지점. **여기를 바꾸면 원점이 pitch 단위로 점프한다.**

```python
anchor = wafer_cx if axis == "x" else wafer_cy   # :1679
k      = round((anchor - street_ph) / pitch)     # :1680
origin = street_ph + k * pitch                   # :1681
```

| 바꾸고 싶은 것 | 수정 |
|---|---|
| 중심이 아니라 **좌상단 기준** 원점 | `anchor` 를 ROI/웨이퍼 좌상단으로, `k = ceil(...)` |
| 항상 중심 **위/왼쪽** 선을 잡기 | `k = math.floor(...)` ← **과거 버그**. 아래 주의 참조 |
| 항상 중심 **아래/오른쪽** 선 | `k = math.ceil(...)` |
| notch 기준 정렬 | `anchor` 를 notch 좌표로 |

> **주의 — `floor` 로 되돌리지 말 것.**
> 이미지 y 는 아래로 증가한다. `floor` 는 −∞ 쪽으로 버리므로 원점이
> 중심과 같거나 그 **위** 선으로만 잡힌다. 그런데 웨이퍼는 die 격자를
> 중심에 맞춰 찍기 때문에 중심 바로 **아래** 1~3 px 에 street 가 있고,
> `floor` 는 그 선을 못 잡아 한 pitch 를 통째로 건너뛴다.
> 실측 9장 중 7장이 −0.89 pitch 이상 어긋났다. 근거 주석: `:1660-1678`.

### 4-2. street 위상 — `:1553-1577`, 호출 `:1658`

원점을 **pitch 안에서** 미세하게 움직인다.

- `:1565` `gpos = base_offset + c.street_pos / scale` — 축소본→전역 좌표 환산.
  `_grid_roi` 오프셋(`base`)을 안 더하면 ROI 만큼 통째로 밀린다.
- `:1567` `acc += c.score * exp(iθ)` — **가중치가 `score`**.
  특정 채널을 더 믿게 하려면 여기서 가중치를 바꾼다.
- `:1571` `conf = |acc| / Σw` — `phase_conf`. 낮으면 채널들이 street 위치에
  동의하지 않는다는 뜻 (`:1702` 에서 0.5 미만이면 경고).

### 4-3. 채널별 street 검출 — `:1411-1475`

원점의 **원재료**. 여기가 틀리면 위 전부가 틀린다.

- `:1445-1448` 원형 스무딩 커널 `k = nb // 48`. lane 이 아주 좁으면 뭉갤 수 있다.
- `:1450` `dev = t - median(t)` — **판정 기준**. 중앙값 대신 평균/midrange 로
  바꾸면 die 본체가 밝을 때 흔들린다 (그래서 median 을 씀).
- `:1460` `pol = +1 if dev[hi] >= -dev[lo] else -1` — 극성 자동 결정.
  `street_polarity` 로 강제하면 `:1455-1458` 으로 우회.
- `:1469-1473` 서브빈 포물선 보간 — 소수점 정밀도. 빼면 ±0.5 bin 양자화 오차.
- `:1474` `width = (s > 0.5*peak).mean()` — 반치폭 기준 street 점유율.
  `_analyze_channel:1478` 에서 이 값이 `[0.02, 0.55]` 밖이면 score 를 반감시킨다.

### 4-4. pitch 합의 — `:1528-1550`, 호출 `:1624`

pitch 가 틀리면 원점은 중심 근처에선 맞아 보여도 가장자리에서 크게 벌어진다.

- `:1542-1543` 상대오차 `tol`(=`pitch_cluster_tol`) 로 클러스터링.
- `:1545` 점수합 최대 클러스터 채택 — **다수결이 아니라 점수합**.
- `:1548` 채택 클러스터 안에서 점수 가중평균.
- `:1625` `min_channel_score` 문턱. 전멸하면 `:1629` 에서 문턱 없이 재시도.

### 4-5. 입력 준비 — `:1597-1619`

- `:1597` `_grid_roi(..., cfg.roi_ratio)` — ROI 가 작으면 주기 수가 줄어 FFT 가 불안정.
- `:1603` `_downscale(roi, cfg.profile_max_dim)` — 축소하면 빨라지지만
  **원점 정밀도가 `1/scale` px 단위로 양자화**된다.
- `:1610` `_feature_bank(small, names)` — `feature_channels` 로 채널 선택.
- `:1619` `prof = f.mean(axis=...)` — **평균 투영**.
  결함/파티클에 강하게 하려면 `np.median` 으로 바꿀 수 있다 (미검증).

### 4-6. 원점 검출 *이전* 단계 — `:2161-2192`

여기서 이미지가 바뀌므로 원점도 따라 바뀐다. **재구현 시 가장 빠뜨리기 쉬운 곳.**

- `:2166-2174` 회전 정렬 루프. `align_angle=False` 면 통째로 생략.
- `:2188` 회전 후 `detect_wafer_adaptive()` 재호출 — `anchor` 가 여기서 정해진다.

---

## 5. 실측 값 (casio p092)

```
self-check vs build_die_map_v6(): MATCH  viz=(1533, 1539)  build=(1533, 1539)
rotation applied = +0.9505 deg
wafer center = (1536, 1536)   r = 1401
ROI          = 1738x1738 @(667,667)   downscale=1.0000

axis     pitch  agree  street_ph   conf     k     origin  off(pitch)
x      92.0045   1.00    61.2939   0.99    16   1533.366     -0.0286
y      92.0027   1.00    67.2538   0.91    16   1539.297     +0.0358
```

`off(pitch)` = `(origin - anchor) / pitch`. `round` 를 쓰는 한 구조적으로
|off| ≤ 0.5 가 보장된다. 실측은 0.03 수준 — 격자가 중심에 잘 맞아 있다는 뜻.

`y` 축 `conf=0.91` 이 `x` 의 0.99 보다 낮은데, `04_y_fold_street.png` 를 보면
`stdL` 채널만 street 위치가 다른 채널(37~41)과 크게 다르다(56). 점수 가중
원형평균이라 소수 의견이 눌렸다. `feature_channels` 에서 `stdL` 을 빼면
실제로 합의도가 올라간다 (실측):

```
baseline(7ch)  origin=(1533,1539)  phase_conf=(x 0.995, y 0.914)  dies=732
no stdL(6ch)   origin=(1533,1541)  phase_conf=(x 0.994, y 0.933)  dies=730
```

다만 **conf 가 올랐다고 원점이 더 맞는 것은 아니다.** y 원점이 2px 움직였고
어느 쪽이 참인지는 conf 가 답해주지 않는다. `conf` 는 "채널들이 서로
동의하는 정도"일 뿐, "정답에 가까운 정도"가 아니다. 소수 의견 채널을
빼면 당연히 동의도는 올라간다.

---

## 5-2. 파라미터 민감도 — 실측 요약

`sweep_origin_claude.py` 로 실제 이미지 9장(real 5 + NaturalColorSeries 4) ×
41개 조합을 돌린 결과. 판정 우선순위는 **pitch > shift > drift**:

| 판정 | 뜻 | 왜 이 순서인가 |
|---|---|---|
| `pitch` | pitch 자체가 5% 넘게 바뀜 | 격자 간격이 다르면 원점이 같아도 **다른 격자**다. 가장 나쁨 |
| `shift` | 원점이 **다른 격자선**으로 점프 | die 인덱스가 통째로 밀림 |
| `drift` | 같은 선인데 street 중심에서 pitch 10% 이상 벗어남 | 경계에 가까워짐 |

**건드리면 위험한 것 (9장 중 몇 장에서 깨졌나):**

| 파라미터 | pitch | shift | drift | fail |
|---|---|---|---|---|
| `min_pitch_px=32.0` | **3/9** | 0/9 | 0/9 | 0 |
| `only [a]` | **2/9** | 0/9 | 2/9 | 1 |
| `only [b]` | **2/9** | 0/9 | 0/9 | 1 |
| `only [sat]` | **2/9** | 0/9 | 2/9 | 1 |
| `only [chroma]` | **1/9** | 0/9 | 2/9 | 1 |
| `only [maxmin]` | **1/9** | 0/9 | 2/9 | 1 |
| `street_polarity='dark'` | 0/9 | 1/9 | **4/9** | 0 |
| `street_polarity='bright'` | 0/9 | 1/9 | **4/9** | 0 |
| `align_angle=False` | 0/9 | 1/9 | 2/9 | 0 |
| `only [stdL]` | 0/9 | 1/9 | 2/9 | 0 |
| `only [L]` | 0/9 | 0/9 | 3/9 | 0 |
| `profile_max_dim=512` | 0/9 | 0/9 | 3/9 | 0 |

**9장 전부에서 안전했던 것 (27개):** `roi_ratio` 전 범위(0.45~0.80),
`max_pitch_ratio` 전 범위, `pitch_cluster_tol` 전 범위,
`min_channel_score` ≤ 0.3, `angle_max_iter`/`angle_fine_step` 전 범위,
`clean=False`, `min_pitch_px` ≤ 16, pitch 직접 지정.

### 여기서 읽어야 할 것

1. **ROI·해상도·합의 임계값 튜닝은 원점을 거의 못 건드린다.** 이쪽을 만져서
   결과를 바꾸려는 시도는 대체로 헛수고다.
2. **진짜 위험은 세 곳에 몰려 있다** — `min_pitch_px`(탐색 밴드에서 참 pitch 를
   잘라냄), `street_polarity` 강제, 그리고 **단일 채널 운용**.
3. `min_pitch_px=32` 가 깬 3장은 **정확히** y 축 참 pitch 가 27.6px 인
   NCS 3장(amber / rose / teal)이다. 하한 32 가 참값 27.6 을 밴드에서
   잘라내자 FFT 는 남아 있는 **정수배**를 골랐다:

   | 이미지 | 참 pitch_y | 검출 pitch_y | 배수 |
   |---|---|---|---|
   | `natural_amber_olive_bronze` | 27.61 | 193.8 | 7× |
   | `natural_teal_bluegray` | 27.57 | 193.3 | 7× |
   | `natural_rose_violet_iceblue` | 27.59 | 84.45 | 3× |

   y pitch 가 35.7 인 `white_brown` 은 32 보다 커서 멀쩡했다. 즉 이건
   튜닝 실패가 아니라 **하한이 참 pitch 를 넘으면 반드시 터지는 구조**다.
   `min_pitch_px` 는 항상 참 pitch 보다 작게 잡아야 한다.
4. `only [a]` / `only [sat]` 가 `white_brown` 에서 pitch 를 **절반**으로 잡아
   die 가 865 → 1739 가 됐다. 이때 origin 은 1px 밖에 안 움직였다.
   **원점만 보면 "안정"으로 오판된다** — 그래서 판정에 pitch 를 1순위로 넣었다.
5. 7채널 합의 구조 자체가 방어선이다. 단일 채널은 1~2장에서 깨지지만
   7채널 합의는 9장 전부 통과했다. 채널을 줄이는 방향의 수정은 권하지 않는다.

전체 표(이미지별 41행)는 [`ORIGIN_SENSITIVITY_claude.md`](ORIGIN_SENSITIVITY_claude.md).

---

## 6. 자가대조가 붙은 이유

`viz_origin_claude.py` 는 `v6` 내부 함수를 직접 호출해 그림을 그린다.
따라서 **호출 순서를 하나라도 빠뜨리면 그림이 거짓말을 한다.**
개발 중 실제로 회전 정렬(`:2166-2174`)을 빠뜨려 원점이 `(1549,1535)` 로
나왔고, `build_die_map_v6()` 의 `(1533,1539)` 와 대조하는 체크가 이를 잡았다.
그래서 이 대조는 임시 디버그가 아니라 **영구 기능**으로 남겼다.
