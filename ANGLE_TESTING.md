# Angle alignment verification

`test_angle_alignment.py` rotates two source images synthetically by
`-3, -2, -1, -0.5, +0.5, +1, +2, +3` degrees and checks:

1. estimated correction matches the expected correction;
2. die pitch remains stable;
3. a second alignment pass reports residual tilt within 0.05 degrees.

The real MIPS source contains an original 1.250-degree tilt. Therefore its
expected correction is `1.250 - injected_angle`. The natural teal image starts
at 0.000 degrees.

## Recorded result

| Source | Angles tested | Maximum correction error | Maximum residual | Pitch |
| --- | ---: | ---: | ---: | --- |
| `Img/real_mips_top_p084.png` | 8 | 0.002 deg | 0.000 deg | 84 x 84 px |
| `TestAssets/NaturalColorSeries/natural_teal_bluegray.png` | 8 | 0.025 deg | 0.000 deg | 38 x 28 px |

Run the test from the repository root:

```powershell
python test_angle_alignment.py
```

The angle test uses `cv2.INTER_CUBIC` for the controlled input rotations, the
same interpolation family used by the detector's geometric correction.
