# 색상·노이즈 강인성 검증

실제 제공 이미지와 별도 생성 이미지를 모두 사용한 검증 기록입니다.

## 1. 실제 제공 wafer 이미지

`test_color_robust.py`는 `Img/`의 6개 실제 이미지에서 자동 검출을 실행합니다.
검출된 pitch는 각각 78, 84, 88, 90, 92 px였으며, 모든 샘플에서 비어 있지 않은
die map을 만들었습니다.

## 2. 실제 이미지 기반 적대적 변형

`test_adversarial_color_robust.py`는 실제 `real_mips_top_p084.png`의 die 형상은
바꾸지 않고 street만 변형합니다. 이는 색상 때문에 pitch가 틀어지는지 확인하기
위한 재현 가능한 시험입니다.

| 조건 | 검증 결과 |
| --- | --- |
| 흰색 + 갈색 24 px 타일 street, 강한 노이즈 | 84 x 84 px |
| 비슷한 명도, 다른 chroma의 street | 84 x 84 px |
| 전체 hue 이동, gamma 변화, 센서 노이즈 | 84 x 84 px |
| 갈색/흰색 noisy street + 5도 회전 | 84 x 84 px, 보정 -4.76도 |

## 3. AI 생성 다색 wafer smoke test

`TestAssets/generated_multicolor_noisy_wafer.png`은 검출기의 기존 fixture가 아닌
새로운 다색/불균일 saw-street wafer 이미지입니다. `image_gen`으로 만든 뒤 육안
확인하고, `TestAssets/generated_multicolor_noisy_wafer_overlay.png`에 검출 overlay를
저장했습니다. 이 입력에서도 die map 326개와 pitch 55 x 66 px를 검출했습니다.

생성 프롬프트의 핵심 조건은 다음과 같습니다: 검정 배경의 상단 촬영 원형 wafer,
반복되는 직교 die grid, 흰색·tan·갈색·푸른 회색 street, salt-and-pepper noise,
불균일 조명, 약 2도 회전, 텍스트/워터마크 없음.

## 4. 실제 geometry 기반 색상 fixture 7종

`generate_color_variants.py`는 실제 84 px wafer의 die geometry를 보존한 상태에서
street만 흰색/갈색, cyan/magenta, gold/blue, purple/green, 저대비 어두운 색,
다색 speckle로 각각 바꿉니다. 마지막 fixture는 흰색/갈색 noisy street에 5도
회전도 적용합니다. 결과 파일은 `TestAssets/ColorVariants/`에 있고,
`test_color_variants.py`가 모두 84 x 84 px pitch를 유지하는지 검증합니다.

## 실행

```powershell
python test_color_robust.py
python test_adversarial_color_robust.py
python test_color_variants.py
```

두 스크립트는 외부 데이터 다운로드 없이 이 저장소의 `Img/`와 `TestAssets/`만
사용합니다.
