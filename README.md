# Wafer_color_v6

Colour-independent wafer die-grid detection and test fixtures.

This repository intentionally keeps multiple contributors' implementations
separate:

| Implementation | Main entry point | Notes |
| --- | --- | --- |
| `wafer_die_map_color_robust.py` | `build_die_map_robust()` | V5-compatible colour-robust path, with real-image and natural-colour fixture tests. |
| `wafer_color_v6_claude.py` | `build_die_map_v6()` | Independently contributed standalone implementation; see `README_claude.md`. |
| `wafer_die_map_v5.py` | `build_die_map()` | Original implementation, retained unchanged. |

For the recommended colour-robust V5-compatible workflow, see
[USAGE.md](USAGE.md) and the visual explanation in
[HOW_IT_WORKS_VISUAL.md](HOW_IT_WORKS_VISUAL.md). For the independently contributed standalone V6 path,
see [README_claude.md](README_claude.md).

## One-file copy/paste version

Use [wafer_die_map_v6_single.py](wafer_die_map_v6_single.py) when the target
application needs one file only. It includes the complete original V5 body and
the colour-robust extension, with no `import wafer_die_map_v5` dependency.

```python
from wafer_die_map_v6_single import ColorRobustConfig, build_die_map_robust

dm = build_die_map_robust(
    r"E:\data\wafer.png",
    config=ColorRobustConfig(min_pitch=75, max_pitch=95),
)
```

The original `build_die_map`, `locate_die`, and `crop_die` APIs remain in the
same file. Run `python test_single_file.py` to verify the standalone file.

## Long-range grid-fit correction

`build_die_map_robust()` first obtains a repeat period, then re-fits that
lattice against every visible street in the central wafer region. This retains
the fractional pitch (for example `92.18 px`) instead of drawing every die at
a rounded `92 px`, preventing the overlay and die rectangles from drifting at
the wafer rim. It is enabled by default with
`ColorRobustConfig(global_pitch_refine=True)`. Set it to `False` only when a
non-uniform/perspective image intentionally cannot be represented by one
global rectangular lattice.

```powershell
python test_global_pitch_refinement.py
```

## Real-camera X/Y reference profile

For the supplied raw camera images only, use
`wafer_die_map_real_axis.py`. It preserves the observed reference convention:
X is the nearest vertical street to wafer centre, while Y is the horizontal
street above wafer centre. This is intentionally separate from synthetic-test
behaviour.

```python
from wafer_die_map_real_axis import build_die_map_real_axis

dm = build_die_map_real_axis(r"E:\data\real_wafer.png")
```

Run `python test_real_axis_reference.py` for the raw `Img/` sources only, and
`python make_real_axis_diagnostics.py` to write their labelled overlays into
`TestAssets/AllDiagnostics/`.

## Quick start

```powershell
pip install numpy opencv-python
python test_color_robust.py

# Your image
python -c "from wafer_die_map_color_robust import build_die_map_robust; dm = build_die_map_robust(r'wafer.png'); print(dm.pitch_x, dm.pitch_y, dm.num_dies)"
```

The supplied source images are in `Img/`. Generated natural-colour examples
are in `TestAssets/NaturalColorSeries/`.

Angle verification is documented in [ANGLE_TESTING.md](ANGLE_TESTING.md).
