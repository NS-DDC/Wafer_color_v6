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
- Grid *phase* within a period is partly conventional: a rigid shift of a periodic tiling
  is self-consistent, so die-to-die similarity cannot pin it down. What is validated here
  is pitch exactness, phase stability across color variants, and landing inside a scribe
  lane rather than mid-die.

Requires Python 3.9+, NumPy, OpenCV. Tested on NumPy 2.4.6 / OpenCV 5.0.0.
