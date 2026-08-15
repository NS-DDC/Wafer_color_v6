# Deterministic colour variants

`generate_color_variants.py` creates these images from the supplied real
`Img/real_mips_top_p084.png`.  The die geometry is preserved; only the street
appearance changes.  This makes visual overlay comparison meaningful.

| File prefix | Street condition |
| --- | --- |
| `01` | white, brown and noisy tiles |
| `02` | cyan/magenta high-chroma tiles |
| `03` | gold and blue tiles |
| `04` | purple and green tiles |
| `05` | dark, low-contrast tiles |
| `06` | mixed multicolour, speckled tiles |
| `07` | white/brown noisy image, additionally rotated 5 degrees |

They are JPEG test fixtures (quality 96), generated with fixed random seed
`20260813`. Run `python generate_color_variants.py` to recreate them exactly.
