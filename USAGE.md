# 사용 가이드: 색상 강인 Die Map

이 문서는 `wafer_die_map_color_robust.py`의 사용법입니다. 기존
`wafer_die_map_v5.py`는 수정하지 않으며, 새 함수가 반환하는 `dm`은 기존 V5의
`WaferDieMap` 형식과 호환됩니다.

다른 코드에 통째로 복사/붙여넣기할 때는 `wafer_die_map_v6_single.py` 하나만
사용하십시오. V5 본문과 이 문서의 colour-robust 확장이 이미 합쳐져 있으므로
`wafer_die_map_v5.py`나 `wafer_die_map_color_robust.py`를 함께 복사할 필요가
없습니다.

## 1. 설치

Python 3.9 이상과 아래 패키지가 필요합니다.

```powershell
pip install numpy opencv-python
```

## 2. 가장 간단한 실행

```python
from wafer_die_map_color_robust import build_die_map_robust

dm = build_die_map_robust(r"C:\data\wafer.png")

print("wafer center:", dm.wafer_cx, dm.wafer_cy)
print("pitch (px):", dm.pitch_x, dm.pitch_y)
print("origin:", dm.x0, dm.y0)
print("detected dies:", dm.num_dies)
```

기본 `origin_mode="nearest_center"`은 wafer 중심에 유클리드 거리로 가장 가까운
grid 교차점을 `(x0, y0)`으로 사용합니다. 기존 V5의 인덱스 기준(중심 바로 위
street)을 유지해야 하면 `ColorRobustConfig(origin_mode="upper_right")`를
명시하십시오.

기본 설정은 색상과 명암 변화에 모두 대응하는 `auto` 모드이며, 작은 회전은
자동 보정합니다.

## 3. 결과 품질과 선택된 방식 확인

```python
from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust

dm, info = build_die_map_robust(
    r"C:\data\wafer.png",
    config=ColorRobustConfig(),
    return_info=True,
)

print(info["grid"]["selected_method"])   # gradient 또는 std
print(info["grid"]["quality"])           # 반복 grid 품질 점수
print(info["grid"]["phase_refinement"])  # grid 미세 위치 보정값
print(info["rotation_deg"], info["angle_source"])
```

`quality`는 같은 촬영 조건의 기준 이미지와 비교할 때 유용합니다. 새 이미지의
값이 평소보다 크게 낮거나 `info["grid"]["errors"]`가 비어 있지 않으면 overlay를
확인하는 것을 권장합니다.

## 4. pitch 범위를 아는 경우: 가장 중요한 수동 설정

Die 내부 회로의 반복 무늬를 die pitch로 잘못 선택할 가능성을 줄이려면 실제
pitch 범위를 지정합니다. 예를 들어 실제 die pitch가 약 84 px이면 다음과 같이
사용합니다.

```python
cfg = ColorRobustConfig(
    min_pitch=75,
    max_pitch=95,
)
dm = build_die_map_robust(r"C:\data\wafer.png", config=cfg)
```

권장 순서는 다음과 같습니다.

1. 대표 이미지 1장으로 `pitch_x`, `pitch_y`를 확인합니다.
2. 허용 가능한 공정/배율 변화를 포함해 `min_pitch`, `max_pitch`를 설정합니다.
3. 이후 양산 이미지에는 같은 범위를 사용합니다.

## 5. 색상·노이즈가 심한 경우

기본 `auto`는 Lab/gray gradient 후보와 구조(`std`) 후보를 비교해 선택합니다.
흰색+갈색 noise, 색 반전, chroma 변화가 특히 심하면 gradient 방식만 강제할 수
있습니다.

```python
cfg = ColorRobustConfig(
    mode="gradient",
    min_pitch=70,
    max_pitch=100,
)
dm = build_die_map_robust(r"C:\data\wafer.png", config=cfg)
```

반대로 안정적인 단색 장비 이미지에서 구조 방식만 비교하려면
`mode="std"`를 사용합니다.

## 6. 회전 보정

기본 `angle_align="robust"`는 projection + FFT 방식으로 grid를 수평/수직에
가깝게 보정합니다.

```python
# 기본: 회전 보정 사용
dm = build_die_map_robust(r"C:\data\wafer.png")

# 입력이 이미 보정됐거나 원본 좌표계를 반드시 유지해야 할 때
cfg = ColorRobustConfig(angle_align="none")
dm = build_die_map_robust(r"C:\data\wafer.png", config=cfg)
```

중요: 회전 보정을 켠 경우 `dm`의 좌표, `locate_die` 결과, crop은
`dm.aligned_image` 기준입니다. YOLO 등 후속 검출도 반드시 같은 `aligned_image`에
실행해야 좌표가 맞습니다. 원본 이미지 좌표를 유지해야 하면
`angle_align="none"`을 사용하십시오.

## 7. 점 또는 BBox로 Die 찾기

```python
from wafer_die_map_v5 import locate_die

# 점 좌표
result = locate_die(dm, point=(1500, 1500))

# 검출 BBox (x1, y1, x2, y2)
result = locate_die(dm, bbox=(1450, 1470, 1520, 1540))

print(result["die_index"])
print(result["die_rect_px"])
print(result["die_center_px"])
print(result["is_edge"])
```

`die_index`는 오른쪽으로 `+ix`, 위쪽으로 `+iy`입니다. `is_edge`의 기준은
`edge_mode="circle"`(기본), `"ring"`, `"both"` 중에서 선택할 수 있습니다.

## 8. Die crop

```python
from wafer_die_map_v5 import crop_die

# 회전 보정이 켜져 있으면 반드시 aligned_image를 사용
die_image = crop_die(
    dm.aligned_image,
    *result["die_center_px"],
    dm.die_w,
    dm.die_h,
    margin_x=4,
    margin_y=4,
)
```

처음부터 모든 die crop을 저장하려면 `with_crops=True`를 사용합니다.

```python
dm = build_die_map_robust(
    r"C:\data\wafer.png",
    with_crops=True,
    margin_x=4,
    margin_y=4,
)
first_crop = dm.dies[0]["image"]
```

## 9. Overlay로 육안 확인

새 장비, 새 배율, 새 wafer 색상에서는 첫 이미지마다 overlay를 확인하십시오.

```python
import cv2
from wafer_die_map_color_robust import make_grid_diagnostic

overlay = make_grid_diagnostic(r"C:\data\wafer.png", dm, thickness=1)
cv2.imwrite(r"C:\data\wafer_grid_overlay.png", overlay)
```

초록색 grid 선이 실제 saw street 중앙과 평행하게 맞는지 확인합니다. 일정한
오프셋이 보이면 `phase_refine=False` 결과도 함께 비교하고, pitch 범위를 먼저
좁혀 내부 회로 패턴 선택을 배제하십시오.

## 10. 테스트

저장소 루트에서 실행합니다.

```powershell
# 제공된 실제 wafer 6장
python test_color_robust.py

# 실제 이미지 기반의 흰색/갈색, chroma, gamma/noise, 회전 적대적 변형
python test_adversarial_color_robust.py

# 실제 geometry를 유지한 색상 fixture 7종
python test_color_variants.py

# 자연스러운 street를 가진 AI 색상 시리즈 3종
python test_natural_color_series.py

# 자연스러운 pearl-white + 갈색 잔류물 street 이미지
python test_white_brown_natural_streets.py

# 중심에 가장 가까운 grid 코너: 실제 wafer 6장 + 좌/우/상/하 위상 fixture 4장
python test_nearest_center_corner.py

# 모든 원본 fixture의 grid + wafer center + 최근접 기준 코너 결과 생성
python make_all_diagnostics.py
# 위 19개 결과의 한 장짜리 미리보기 생성
python make_all_diagnostics_contact_sheet.py

# 실제/자연스러운 색상 wafer를 +/-0.5, 1, 2, 3도로 회전한 angle 보정 시험
python test_angle_alignment.py

# V5 + colour-robust이 합쳐진 단일 파일 검증
python test_single_file.py
```

테스트 원본 위치:

| 경로 | 내용 |
| --- | --- |
| `Img/` | 제공된 실제 wafer 이미지 6장 |
| `TestAssets/ColorVariants/` | 실제 84 px wafer geometry 기반 색상 fixture 7종 |
| `TestAssets/NaturalColorSeries/` | 자연스러운 좁은 dark street AI 이미지 3종 |
| `TestAssets/generated_multicolor_natural_streets_v2.png` | 자연스러운 street AI smoke-test 이미지 |

## 11. 문제 해결

| 증상 | 먼저 할 일 |
| --- | --- |
| pitch가 die 내부 패턴으로 잡힘 | `min_pitch`, `max_pitch`를 실제 pitch 근처로 좁힙니다. |
| overlay가 일정하게 어긋남 | `angle_align="none"`과 기본값을 모두 비교하고, `phase_refine` 정보와 실제 street 중심을 확인합니다. |
| 회전 보정 후 YOLO 좌표가 어긋남 | YOLO 입력을 원본이 아닌 `dm.aligned_image`로 통일합니다. |
| wafer 외부 배경이 복잡함 | 기본 `clean=True`를 유지하고, wafer가 프레임에서 충분히 크게 보이도록 촬영합니다. |
| grid가 거의 보이지 않음 | 노출/초점부터 확인하고, 색이 아니라 반복되는 경계 자체가 존재하는지 overlay로 검증합니다. |

## 12. 다른 구현과의 구분

`wafer_color_v6_claude.py`와 `README_claude.md`는 다른 기여자가 별도로 만든
독립 구현입니다. 이 문서의 API, 테스트, 설정값은 그 파일이 아니라
`wafer_die_map_color_robust.py`에만 해당합니다.

파라미터가 어느 단계에 영향을 주는지와 전체 처리 흐름은
[HOW_IT_WORKS_VISUAL.md](HOW_IT_WORKS_VISUAL.md)를 참고하십시오.
