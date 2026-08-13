# Natural-street AI colour series

These three AI-generated wafer photographs use the same natural thin, dark
saw-street style as `../generated_multicolor_natural_streets_v2.png`, while
changing only plausible low-saturation wafer reflection colours.

| Image | Reflection palette | Validation |
| --- | --- | --- |
| `natural_teal_bluegray.png` | teal, blue-gray, cool silver | 38 x 28 px, 1,129 dies |
| `natural_amber_olive_bronze.png` | amber, olive, bronze | 38 x 28 px, 1,129 dies |
| `natural_rose_violet_iceblue.png` | rose-gray, violet-gray, icy blue | 38 x 28 px, 1,129 dies |
| `white_brown_natural_streets_ai.png` | pearl-white streets with subtle amber/brown residue | 35 x 36 px, 867 dies |

Each `_overlay.png` is the corresponding detected grid. The grid and street
geometry was visually inspected, and all three images were tested by
`test_natural_color_series.py`.

`white_brown_natural_streets_ai.png` is a separate, natural-looking version
of the white/brown street condition. It deliberately has no artificial grid
outside the wafer and no square-tile intersections. Its regression test is
`test_white_brown_natural_streets.py`.
