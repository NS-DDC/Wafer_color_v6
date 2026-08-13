# 색상 강인 Die Grid 검출

기존 `wafer_die_map_v5.py`는 수정하지 않았습니다. 새 경로는
`wafer_die_map_color_robust.py`의 `build_die_map_robust()`입니다.

## 핵심 방식

- 고정 HSV/밝기/특정 street 색상을 쓰지 않습니다.
- Gray와 Lab의 L/a/b 채널 각각에서 경계 에너지를 추출하여 결합합니다.
  따라서 street가 흰색, 갈색, 어두운 색, 여러 색의 혼합이어도 반복되는
  Die 경계가 있으면 동작합니다.
- `auto` 모드에서는 색상 불변 gradient 방식과 기존의 `std` 구조 방식
  두 후보를 만들고, 반복 주기 품질 점수가 더 높은 결과를 고릅니다.
- 미세한 컬러 노이즈는 median blur, Gaussian blur, 긴 방향 투영, 주기성
  검증으로 억제합니다. 원본 Die 색상은 바꾸지 않습니다.

## 기본 사용

```python
from wafer_die_map_color_robust import build_die_map_robust, ColorRobustConfig
from wafer_die_map_v5 import locate_die

dm, info = build_die_map_robust(
    "Img/real_mips_top_p084.png",
    config=ColorRobustConfig(),       # auto + 색상 불변 회전 보정
    return_info=True,
)

print(info["grid"])                 # 선택 방식, 후보 품질 점수
print(locate_die(dm, point=(1500, 1500)))
```

`dm`은 기존 `WaferDieMap`이므로 `locate_die`, `crop_die`, `dies` 등 기존 후속
코드를 그대로 사용할 수 있습니다. 좌표와 crop은 `dm.aligned_image` 기준입니다.

## 조정이 필요한 경우

```python
# Die pitch가 84 px 근처임을 안다면 내부 회로 패턴을 pitch로 오인하지 않도록 제한
cfg = ColorRobustConfig(min_pitch=75, max_pitch=95)

# 매우 강한 색상 변화/갈색-흰색 노이즈일 때 gradient 방식만 강제
cfg = ColorRobustConfig(mode="gradient", min_pitch=70, max_pitch=100)

# 이미지가 이미 수평/수직 정렬되어 있어 회전 warp를 원하지 않을 때
cfg = ColorRobustConfig(angle_align="none")

dm = build_die_map_robust("wafer.png", config=cfg)
```

`min_pitch`와 `max_pitch`가 가장 중요한 수동 파라미터입니다. 실제 Die pitch의
대략적인 범위를 넣으면 Die 내부의 반복 패턴이나 심한 노이즈로 인한 오검출을
더 잘 막을 수 있습니다.

## 시각 검증 및 회귀 테스트

```powershell
python test_color_robust.py
```

실행하면 `_color_robust_diagnostics`에 초록색 grid overlay가 생성됩니다.
검증 코드에는 `Img`의 모든 PNG에 대해 비어 있지 않은 map과 최소 pitch를
확인하는 smoke test가 포함돼 있습니다.
