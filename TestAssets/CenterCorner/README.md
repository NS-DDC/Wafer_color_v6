# Nearest-centre corner fixtures

`01`~`04` are deterministic wafer fixtures with known grid phase.  They cover
all four positions of the nearest grid intersection relative to wafer centre:

| File | Nearest corner relative to centre |
| --- | --- |
| `01_left_top.png` | left / above |
| `02_right_top.png` | right / above |
| `03_left_bottom.png` | left / below |
| `04_right_bottom.png` | right / below |

Each matching `_diagnostic.png` shows yellow detected grid lines, a red wafer
centre marker, and the blue nearest grid-corner marker. `truth.json` stores
the known expected coordinates. Regenerate sources with
`python make_center_corner_fixtures.py` and validate both these fixtures and
the six supplied real images with `python test_nearest_center_corner.py`.
