# wafer_color_v6_claude.py

Color-agnostic wafer die-map detection — Claude's implementation.

Standalone (~2,360 lines). No import of `wafer_die_map_v5`; every helper it needs was
copied in. One entry point:

```python
from wafer_color_v6_claude import build_die_map_v6

dm = build_die_map_v6("wafer.png")          # fully automatic
print(dm.pitch_x, dm.pitch_y, dm.num_dies)
print(dm.diagnostics.report())              # self-diagnosis
```

### 📖 사용법은 소스 파일 맨 위 주석에 전부 들어 있습니다

두 파일 모두 모듈 docstring 에 상세 사용법을 한국어로 달아두었습니다.
에디터에서 파일을 열거나 `help()` 로 바로 볼 수 있습니다.

```bash
python -c "import wafer_color_v6_claude as m; print(m.__doc__)"
python -c "import make_color_wafers_claude as g; print(g.__doc__)"
```

`wafer_color_v6_claude.py` 의 목차:

| 절 | 내용 |
|---|---|
| [1] | 설치 |
| [2] | 완전 자동 사용 — `dm.dies` dict 의 모든 키 설명 |
| [3] | 파라미터 직접 지정 — `ColorProfile` 전 필드 + feature 채널 7종이 각각 어떤 상황에 반응하는지 |
| [4] | `build_die_map_v6()` 인자 18개 + 반환 속성 전체 + V5 호환 좌표 규약 |
| [5] | 검증 3종 — 자가진단 리포트 / 디버그 오버레이 / 색상 변형 하네스(variant 15종 각각의 의도) |
| [6] | `locate_die_v6()` — 반환 dict 의 키 16개 |
| [7] | CLI 전체 옵션과 예시 |
| [8] | **잘 안 될 때** — 실제로 출력되는 경고 문구별 대처법 |

`make_color_wafers_claude.py` 의 목차:

| 절 | 내용 |
|---|---|
| [1] | 준비 |
| [2] | CLI 전체 옵션 |
| [3] | 결과 표 읽는 법 + 허용오차(TOL) 기준 |
| [4] | 팔레트 29종이 각각 무엇을 노린 시험인지 |
| [5] | `WaferSpec` 로 내 팔레트 추가하기 (전 필드 설명) |
| [6] | 재현성 — seed 를 `zlib.crc32` 로 잡아야 하는 이유 |

---

## Headline result

Both engines run on the same image put through 15 color transforms. PASS = pitch within
2% and die count within 3% of that engine's own baseline.

| source | v6 | v5 |
|---|---|---|
| `real_casio_top_p092` | **15/15 (100%)** | 8/15 (53%) |
| `real_piper_top_p088` | **15/15 (100%)** | 0/15 (0%) |

Plus 29 synthetic wafers rendered with *known* ground truth across 29 color palettes
(isoluminant, isochromatic, neon, pastel, sepia, copper, bright background, ±3° rotation,
anisotropic pitch, heavy noise, JPEG q35, and more):

| | v6 | v5 |
|---|---|---|
| synthetic palettes | **29/29 (100%)** | 0/29 (0%) |

On the four images whose filenames encode ground-truth pitch, v6 recovers it exactly:

| image | truth | v6 pitch x / y | dies |
|---|---|---|---|
| `real_piper_top_p088` | 88 | 87.977 / 88.000 | 800 |
| `real_casio_top_p092` | 92 | 92.005 / 92.003 | 732 |
| `real_exposed_top_p078` | 78 | 78.002 / 78.002 | 1019 |
| `real_mips_top_p084` | 84 | 83.999 / 84.002 | 874 |

v5 raises `RuntimeError: No wafer street/grid line was found near wafer center.` on
**four of these five images** before any color transform is applied. On casio it agrees
with v6 (91.999/91.989, 732 dies, rotation 0.949 vs 0.950).

Full tables: [`results_claude/`](results_claude/).

---

## Why v5 breaks

| v5 mechanism | assumption | fails when |
|---|---|---|
| `gray > 20` wafer mask | wafer brighter than background | background is white / wafer is dark |
| `min_brightness=115`, `min_channel=130` | absolute BGR levels | gamma or exposure shift |
| `35 <= color_delta <= 130` in `_street_color_mask` | street is **chromatic** | grayscale or desaturated image → `max(BGR)-min(BGR) ≈ 0` → empty mask → RuntimeError |
| HSV-saturation projection | saturation carries the grid | inverted or achromatic imagery |

The dominant failure is the third: the color-delta gate empties the street mask, so
there is nothing left to find a grid in.

## What v6 does instead

Nothing keys off an absolute color. Every stage is relative or periodicity-driven.

**1 — Background estimated, not assumed.** Sample four corner blocks, reject outliers,
take the consensus BGR. Segment by *Lab distance from that background* + Otsu, so it
works whether the wafer is lighter or darker.

**2 — Wafer mask hardened.** Fill interior holes with `findContours(RETR_EXTERNAL)` +
`drawContours(..., -1)` — this closes dark dies inside the wafer while preserving the
notch concavity. Guards: if `minEnclosingCircle` radius exceeds 1.10× the
area-equivalent radius, rim speckle is inflating it, so fall back to centroid + area
radius; if coverage lands outside `[0.02, 0.985]`, fall back to the inscribed circle.

**3 — Multi-channel feature bank.** Build seven 1-D projections —
`L, a, b, chroma, sat, maxmin, stdL` — and score each by how *periodic* it is:

```
score = 0.6·max(0, r(T)) + 0.4·max(0, r(2T))     # Pearson lag-autocorrelation
```

The winning channel is whichever one actually shows the grid, so the code never needs to
know if streets are bright, dark, colored, or textured. Gradient channels are
deliberately excluded — a street has two edges, which halves the apparent pitch and makes
phase ambiguous.

**4 — Pitch from the spectrum, verified in the lag domain.** Hann-windowed, 8×
zero-padded rFFT restricted to `[min_pitch, max_pitch]`, with parabolic log-power
interpolation. Then `_resolve_period_multiple` re-tests T, 2T, 3T by autocorrelation and
promotes if a multiple is genuinely better. This caught a real bug: on
`portable_bw_sample` the FFT returned pitch 45, but

```
r(45) = -0.105     # negative — antiphase, alternating rows are NOT alike
r(90) = +0.997
```

so the true pitch is 90. The annotated companion image `portable_bw_overlay.png`
independently locks every chroma channel at exactly 90.000, confirming it.

**5 — Street localization without a shape assumption.** Fourier phase gives the location
of a *maximum*; the earlier code then assumed the street sat half a period away, which is
only true for a sinusoid. Real scribe lanes are asymmetric — wide die plateau, deep narrow
trough, bright metal centerline. `_street_from_template` instead takes the **largest
deviation from the median** of the folded template (with circular parabolic sub-bin
refinement) and derives polarity from which side wins.

**6 — Consensus.** Pitch clustered across channels at 4% tolerance with score-weighted
averaging; phase merged by circular mean. Both agreements are reported so a low-confidence
result is visible rather than silent.

**7 — Rotation.** Lab-Sobel scalar energy map, projection-variance sweep, cross-validated
against a 2-D FFT angle (4× angle mod 90°, weighted circular mean over an annulus).
Disagreement lowers the reported confidence instead of being hidden.

v5's grid conventions are preserved exactly, so downstream indexing is unchanged:
`cx = x0 + ix·pitch_x + pitch_x/2`, `cy = y0 − iy·pitch_y − pitch_y/2`.

---

## Verification

**Self-diagnosis** — `--report` prints per-channel pitch/score/polarity/width, which
channels were used, agreement and phase confidence per axis, both angle estimates, and an
`OK` / `CHECK WARNINGS` verdict.

```
 pitch_x =    92.005 px   agreement=1.00   phase_conf=0.99   street=dark
 [X] channel        pitch     score  pol  width%  used
      sat           92.005   0.9981   -    8.7   USE
      L             89.767   0.9971   -   14.4   USE   x2 (r=-0.11->+1.00)
```

**Debug overlay** — `--overlay DIR` writes the wafer circle, grid streets, origin, notch
vector, full vs edge dies, and the diagnostic panel. Unicode paths handled via
`cv2.imencode` + `tofile` (plain `cv2.imwrite` fails on them).

**Robustness harness** — `--robustness` runs 15 transforms through v6 and v5 side by side:
`invert, gray, hue+60, hue+150, desat0.25, gamma0.5, gamma2.0, lowcontrast, brightbg,
sepia, cool_tint, noise12, illum_grad, white_street_brown`.

`white_street_brown` reproduces the originally reported failure directly: invert inside
the wafer so dark streets become bright, push the brightest 30% toward white, then speckle
brown over it. v6 holds at 92.001/92.011 with 724 dies (1.09% deviation).

**Synthetic ground-truth harness** — `make_color_wafers_claude.py` renders wafers from
scratch across 29 color palettes, so pitch, center, radius, rotation and street phase are
*known exactly* rather than inferred. Transform-based harnesses can only perturb an
existing image; they cannot reach color combinations the source never had.

```bash
python make_color_wafers_claude.py --out synth --eval --v5   # generate + score
python make_color_wafers_claude.py --list                    # show palettes
```

| | v6 | v5 |
|---|---|---|
| synthetic palettes passed | **29 / 29 (100%)** | 0 / 29 (0%) |

Tolerances: pitch ≤1%, center ≤15px, radius ≤3%, angle ≤0.5°, street phase ≤25% of pitch.
Palettes cover isoluminant (die and street share luminance, differ only in hue),
isochromatic (identical hue, differ only in luminance), neon, pastel, sepia, green PCB,
copper, cyan-on-white, bright background, rim ring, ±3° rotation, anisotropic pitch
(76×48), pitch 28 and 150, left notch, σ=20 noise, JPEG q35, strong illumination gradient,
blur, and a composite worst case.

Rendering uses analytic rotated coordinates (`u = dx·cos a + dy·sin a`) rather than
rotating a rendered image, so the ground-truth angle carries no interpolation error.
Seeds come from `zlib.crc32(name)`, not the built-in `hash()`, which is randomized per
process and would make results irreproducible.

Two v6 defects were found and fixed by this harness:

- **Harmonic mis-pick.** `_spectral_pitch` can land on the 2nd harmonic. Measured on a
  low-contrast palette: spectral peak 129 px, but r(64)=+0.78 vs r(128)=+0.86. Since T/2
  repeats essentially as well, the fundamental is 64. `_resolve_period_multiple` now
  demotes (`r_half ≥ 0.5` and `r_half ≥ r_full − 0.12`) as well as promotes, and promotion
  is gated so a strong fundamental is never overridden by a marginally higher multiple.
- **Wafer mask fragmentation.** A fixed morphological close kernel cannot bridge wide
  scribe lanes. At pitch 150 / lane 13 px the mask split into 376 separate dies, so
  `findContours(RETR_EXTERNAL)` returned *one die* as the largest contour, coverage read
  0.007, and detection fell back to the image inscribed circle (r=799 vs true 700).
  Closing is now progressive: the kernel grows until the largest contour holds ≥85% of the
  foreground area, capped at 4% of `min(H, W)`.

Both fixes were regression-checked: the 6 real images and both 15-transform robustness
runs are byte-for-byte unchanged.

---

## Independent target set: `NaturalColorSeries`

A separate set of AI-generated wafers with thin, natural saw streets — not produced by
this project's generator, so it is an independent check. Reference die size and die count
come from that set's own README.

| image | v6 die | ref die | v6 dies | ref dies | Δ | warn | v5 |
|---|---|---|---|---|---|---|---|
| `natural_teal_bluegray` | 38×28 | 38×28 | 1133 | 1129 | 0.35% | 0 | **fails** |
| `natural_amber_olive_bronze` | 38×28 | 38×28 | 1131 | 1129 | 0.18% | 0 | **fails** |
| `natural_rose_violet_iceblue` | 38×28 | 38×28 | 1132 | 1129 | 0.27% | 0 | **fails** |
| `white_brown_natural_streets_ai` | 35×36 | 35×36 | 865 | 867 | 0.23% | 0 | **silently wrong** |

v6: **4/4**, die size exact on every image, count within 0.35% (the residual is edge-die
inclusion policy), no warnings, ~1 s each.

v5: the three `natural_*` images raise `RuntimeError: No wafer street/grid line was found
near wafer center`. `white_brown_natural_streets_ai` is the worse case — v5 returns
**no error at all** while reporting pitch `(70.00, 62.50)` against a true `(35.11, 35.74)`,
i.e. exactly double, and 248 dies instead of 865. That silent doubling is the failure mode
this project was created to remove.

---

## Input hardening

Static review plus dynamic probing turned up five ways to get a *silently wrong* answer.
Each was reproduced first, then fixed, then re-checked against the full regression.

| # | Problem | Observed before fix | Now |
|---|---|---|---|
| 1 | `ColorProfile` had no validation | `max_pitch_ratio=0` → pitch `(22.0, 14.7)`, 19,111 dies vs true 88/88 and 800 dies — **zero warnings**. `pitch_x=3.0` (below `min_pitch_px=8`) → 684,887 dies. `background_bgr=(999,…)` → raw `OverflowError` from NumPy | `__post_init__` validates every field and raises `ValueError` with a message naming the field, the bad value, and the accepted range |
| 2 | Pitch silently halved when the true pitch exceeds the FFT search band | true 400 px → 199.57 (49.8% error) with `agreement=1.00` and **zero warnings**, because every channel agreed on the same wrong half | Warns when `2·pitch > max_pitch`, i.e. when 2T was never inside the band and therefore could not be ruled out |
| 3 | Non-`uint8` input silently shifted the result | `float32` in 0–255 → 88.43 vs 87.98, no error. `cv2.cvtColor(…, BGR2LAB)` reads `float32` as 0–1, so Lab saturates | `_load_bgr` normalizes `uint16`/`float32`/`float64`/`bool`, auto-detecting 0–1 vs 0–255 scale |
| 4 | 4-channel BGRA accepted by the public helpers | `detect_wafer_adaptive(bgra)` returned a result instead of rejecting | Alpha dropped at the single input gate; all public entry points normalize |
| 5 | CLI `--polarity` violated its own type contract | mapped to `int` (`1`/`-1`/`0`) but the field is `Optional[str]`; worked only because `str()` happened to be called downstream | Normalized in `__post_init__`, so `"bright"`, `"dark"`, `"auto"`, `1`, `-1`, `0` and `None` are all accepted and the documented `street_polarity=-1` form now genuinely works |

Note on #2: the obvious check — "warn if fewer than 4 periods are visible" — **would not
have caught it**. The failing case had 4.35 visible periods. The correct test is whether
the 2× candidate was ever inside the search band at all.

After normalization, `uint8`, BGRA, `float32` (both 0–1 and 0–255), `float64` and `uint16`
all return byte-identical results; `(H,W,5)` and `(H,W,2)` now fail loudly.

Regression after all five fixes: **29/29 synthetic, 4/4 real, 4/4 NaturalColorSeries, 0
failures**, and no new warnings on any previously-clean image.

---

## Grid origin: `floor` vs `round` on the y axis

Reviewing the debug overlays turned up a sixth issue. The magenta origin marker
`(x0, y0)` should sit on the street intersection *nearest* the yellow wafer-centre
cross, but on 7 of 9 images it sat one row above it.

Cause — the two axes disagreed:

```python
if axis == "x":
    k = round((wafer_cx - street_ph) / pitch)     # nearest line, ±0.5 pitch
else:
    k = math.floor((wafer_cy - street_ph) / pitch)  # always toward smaller y
```

Measured origin-to-centre offset over all 9 real + NCS images:

| | before | after |
|---|---|---|
| `dx / pitch_x` | −0.13 … +0.45 | −0.13 … +0.45 (unchanged) |
| `dy / pitch_y` | −0.05 … −0.99, **7 of 9 beyond −0.89** | −0.47 … +0.09 |
| origins outside ±0.5 pitch | 7 / 9 | **0 / 9** |

Two things combine here, and the second is the interesting one.

**Direction** is forced by `math.floor`, which truncates toward −∞. Image `y`
grows downward, so smaller `y` is *higher* on screen, and picking the largest
lattice line `≤ wafer_cy` can only ever land on or above the centre.

**Magnitude** should then be anything in 0…1 pitch, but it clustered just under a
full pitch. That is because a wafer's die grid is laid out centred on the wafer,
so there is a street line very close to `wafer_cy` — the post-fix `round` offsets
show exactly how close (mips +0.012, casio +0.033, teal +0.036, amber +0.072).
They are all *positive*, i.e. the nearest line sits marginally **below** centre,
which is the one place `floor` may not go. So it stepped over a line 1–3 px away
and took the next one a whole pitch up:

```
    ─────────────   <- floor picked this (0.967 pitch up)
          ^
       92 px skipped
          v
    ──────+──────   <- wafer centre; correct line is 3 px below
    ─────────────   <- round picks this
```

The perverse consequence is that **the better the grid is centred, the closer the
`floor` error gets to a full die.**

The two images that did not move — `portable_bw_sample` (−0.466) and
`real_piper_top_p088` (−0.045) — happened to have their nearest line already
above centre, so `floor` and `round` agreed. `piper` has the weakest phase
estimate of the set (`phase_conf=0.41`), and that misalignment is precisely why
it escaped.

The `floor` was worth checking rather than assuming, since v5 counts `iy`
*upward* from the origin (`cy_d = y0 - iy*pitch_y - pitch_y/2`) while `ix` counts
rightward, so it could plausibly have been deliberate v5-index compatibility.
It was not — v5 rounds on **both** axes (`wafer_die_map_v5.py:588`):

```python
kx = int(math.floor((wafer_cx - x1 - phase_x - bias_x) / pitch_x + 0.5))
ky = int(math.floor((wafer_cy - y1 - phase_y - bias_y) / pitch_y + 0.5))
```

`floor(z + 0.5)` is round-half-up. Running both on `real_casio_top_p092.png`
confirmed it directly — same centre `(1536, 1536)`, same 92 px pitch:

| | v5 | v6 before | v6 after |
|---|---|---|---|
| origin | (1536, 1534) | (1533, **1447**) | (1533, **1539**) |
| offset from centre | (0.000, −0.022) pitch | (−0.033, **−0.967**) pitch | (−0.033, **+0.033**) pitch |

So the asymmetry was v6's own, and it put `dies_by_index` one row out of step
with v5. Fixed by using `round` on both axes.

Impact is limited to labelling and display: the lattice `y = y0 + m·pitch_y` is
the same set of lines whichever `k` anchors it, so pitch, die size and die count
are untouched — die counts are byte-identical before and after (casio 732,
piper 800, exposed 1019, mips 874, NCS 1131 / 1131 / 1132 / 865), and the
regression stays at **29/29 synthetic, 4/4 real, 4/4 NCS**.

Also corrected in the same pass: the overlay docstring advertised orange grid
lines, but die rectangles are drawn *after* the lattice and tile exactly on it,
so the orange is always overpainted green. Use `draw_dies=False` to actually see
the grid lines.

---

## Return format: same as v5

`build_die_map_v6()` returns `WaferDieMapV6`, which is **v5's `WaferDieMap`
format**, not a new one. It is a separate class only because v6 does not import
v5 — being standalone was a requirement, so the dataclass had to be redeclared.
The contents were copied field for field:

| | |
|---|---|
| v5 fields present in v6 | **22 / 22**, same names and types |
| positional order matches v5 | **yes** (`fields(v6)[:22] == fields(v5)`) |
| die entry dict keys | **identical**, zero difference |
| extra in v6 | `wafer_mask`, `diagnostics` — appended at the end |

So v5's downstream helpers accept a v6 object directly, no conversion:

```python
import wafer_die_map_v5 as v5
from wafer_color_v6_claude import build_die_map_v6

m = build_die_map_v6("wafer.png")
v5.locate_die(m, point=(1600, 1600))
# -> same 16 keys as v5, same die_index (0, -1)
```

If you need a real `v5.WaferDieMap` instance (for `isinstance` checks or
pickling), either conversion works:

```python
v5_obj = v5.WaferDieMap(**m.to_v5_kwargs())      # keyword
v5_obj = v5.WaferDieMap(*dc.astuple(m)[:22])     # positional
```

The two extra fields are additions, not replacements:

- **`wafer_mask`** — the colour-adaptive wafer foreground mask (`uint8` 0/255).
  v5 rebuilt an equivalent from a brightness threshold every time it needed one;
  v6 hands over the one it already computed.
- **`diagnostics`** — the self-diagnosis report (per-channel scores, agreement,
  warnings). `print(m.diagnostics.report())` for the human-readable form.

Both were moved to the **end** of the field list in this pass. `wafer_mask` had
been sitting in the middle, which broke positional conversion while leaving
attribute access fine — the kind of mismatch that shows up only once someone
tries tuple unpacking.

---

## CLI

```bash
python wafer_color_v6_claude.py Img/wafer.png --report
python wafer_color_v6_claude.py Img --report --overlay out/ --json out/r.json
python wafer_color_v6_claude.py Img/wafer.png --robustness
python wafer_color_v6_claude.py Img/wafer.png --pitch 92 --polarity dark   # manual
```

Accepts files or directories. Manual overrides available for background color, pitch,
polarity, channel subset, min pitch, and ROI ratio — anything left unset is solved
automatically.

```python
from wafer_color_v6_claude import build_die_map_v6, ColorProfile

dm = build_die_map_v6("wafer.png", profile=ColorProfile(
    pitch_x=92.0, pitch_y=92.0,      # optional
    street_polarity=-1,              # -1 dark, +1 bright, None auto
    feature_channels=("L", "sat"),   # None = auto-select
))
```

## Known limitations

- Angle projection vs FFT disagree on `portable_bw_sample` and `portable_bw_overlay`
  (projection ≈ 0.00°, FFT ≈ 1.05°). Flagged as `CHECK WARNINGS`, not hidden.
- `real_exposed_top_p078` reports low y-phase confidence (0.21) — pitch is exact, but the
  street position within the period is less certain.
- **Very large pitch relative to the field of view.** The FFT search band tops out at
  `profile_length × max_pitch_ratio` (0.34), so a true pitch above that cannot be a
  candidate and a sub-harmonic is picked instead. Fewer than ~5 dies visible across the
  wafer triggers this. The ½ case now warns; a ⅓-or-lower mis-pick can still be silent
  when streets are unusually thick (measured: true 300 px with 30 px streets → 30 px).
  Real wafer maps show 20–40+ dies across and are unaffected. If you work with zoomed
  crops, set `pitch_x`/`pitch_y` explicitly or raise `max_pitch_ratio` to ~0.45.
- Grid *phase* within a period is partly conventional: a rigid shift of a periodic tiling
  is self-consistent, so die-to-die similarity cannot pin it down. What is validated here
  is pitch exactness, phase stability across color variants, and landing inside a scribe
  lane rather than mid-die.

Requires Python 3.9+, NumPy, OpenCV. Tested on NumPy 2.4.6 / OpenCV 5.0.0.
