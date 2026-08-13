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
[USAGE.md](USAGE.md). For the independently contributed standalone V6 path,
see [README_claude.md](README_claude.md).

## Quick start

```powershell
pip install numpy opencv-python
python test_color_robust.py

# Your image
python -c "from wafer_die_map_color_robust import build_die_map_robust; dm = build_die_map_robust(r'wafer.png'); print(dm.pitch_x, dm.pitch_y, dm.num_dies)"
```

The supplied source images are in `Img/`. Generated natural-colour examples
are in `TestAssets/NaturalColorSeries/`.
