"""Regression test: fractional die pitch must not accumulate rim drift."""

from __future__ import annotations

from make_color_wafers_claude import WaferSpec, synth_wafer
from wafer_die_map_color_robust import ColorRobustConfig, build_die_map_robust


def _closest_line_error(origin: float, pitch: float, truth_origin: float,
                        truth_pitch: float, span: int) -> float:
    """Maximum difference between corresponding lines across a wafer span."""
    # Align the two line sequences at the centre so this measures accumulated
    # pitch drift rather than an arbitrary choice of representative period.
    index = round((origin - truth_origin) / truth_pitch)
    errors = []
    for offset in range(-span, span + 1):
        detected = origin + offset * pitch
        expected = truth_origin + (index + offset) * truth_pitch
        errors.append(abs(detected - expected))
    return max(errors)


def main() -> None:
    spec = WaferSpec(
        name="fractional_pitch", size=1800, radius=760, cx=893, cy=911,
        pitch_x=82.35, pitch_y=77.65, phase_x=21.4, phase_y=58.2,
        street_w=8.0, bg=(4, 4, 4), die=(126, 108, 88), street=(240, 239, 236),
        ink=(160, 142, 125), speckle=(80, 55, 145), speckle_amt=0.14,
        speckle_on="street", die_jitter=7.0, noise_sigma=2.5, illum=0.10, seed=1201,
    )
    image, truth = synth_wafer(spec)
    base = dict(min_pitch=60, max_pitch=100, angle_align="none", phase_refine=False)
    coarse = build_die_map_robust(image, config=ColorRobustConfig(**base, global_pitch_refine=False))
    refined, info = build_die_map_robust(image, config=ColorRobustConfig(**base, global_pitch_refine=True),
                                         return_info=True)

    # synth_wafer defines phase from (0, 0), while this public API stores the
    # nearest representative grid corner.  Compare whole lattices, not x0/y0.
    truth_x0 = truth["cx"] + truth["phase_x"]
    truth_y0 = truth["cy"] + truth["phase_y"]
    err_x_before = _closest_line_error(coarse.x0, coarse.pitch_x, truth_x0, truth["pitch_x"], 8)
    err_y_before = _closest_line_error(coarse.y0, coarse.pitch_y, truth_y0, truth["pitch_y"], 8)
    err_x_after = _closest_line_error(refined.x0, refined.pitch_x, truth_x0, truth["pitch_x"], 8)
    err_y_after = _closest_line_error(refined.y0, refined.pitch_y, truth_y0, truth["pitch_y"], 8)

    assert info["grid"]["global_pitch_refinement"]["x"]["used"] == 1.0
    assert info["grid"]["global_pitch_refinement"]["y"]["used"] == 1.0
    assert err_x_after < err_x_before * 0.45, (err_x_before, err_x_after)
    assert err_y_after < err_y_before * 0.45, (err_y_before, err_y_after)
    assert err_x_after <= 3.0, err_x_after
    assert err_y_after <= 3.0, err_y_after
    print("fractional-pitch drift (px): "
          f"X {err_x_before:.2f} -> {err_x_after:.2f}, "
          f"Y {err_y_before:.2f} -> {err_y_after:.2f}")


if __name__ == "__main__":
    main()
