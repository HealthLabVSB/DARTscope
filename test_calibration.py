"""
Tests for calibration - per-channel calibration from a corner reflector (roadmap 1.6).

The synthetic array is the IWR6843AOP one from test_capon_rae with a known gain and
phase error on every virtual channel. A calibration estimated with the reflector at
one angle has to fix the angle of a reflector somewhere else.

Run with:  python -m pytest test_calibration.py -q
"""
import os

import numpy as np
import pytest

import calibration
from AOA import compute_range_azimuth_elevation_map
from test_capon_rae import RANGE_STEP_M, make_params, make_synthetic_cube

R, RX, NTX, LOOPS, R_BIN = 64, 4, 3, 32, 30
rng = np.random.default_rng(7)
ERR_PHASE = np.deg2rad(rng.uniform(-60, 60, NTX * RX))
ERR_GAIN = rng.uniform(0.7, 1.3, NTX * RX)
G_TRUE = ERR_GAIN * np.exp(1j * ERR_PHASE)


def reflector(az, el, r_bin=R_BIN, g=G_TRUE, clutter=None, frames=20):
    """A static reflector seen through channel errors g; optional other static targets."""
    params = make_params(R)
    tg = [{"r_bin": r_bin, "az_deg": az, "el_deg": el, "amp": 5.0, "static": True}]
    cube = make_synthetic_cube(tg + (clutter or []), params, F=frames, loops=LOOPS, R=R,
                               noise=0.02)
    for t in range(NTX):
        for r in range(RX):
            cube[:, t::NTX, r, :] *= g[t * RX + r]
    return cube, params


def _angle_error(c, params, az, el, cal=None):
    p = dict(params)
    if cal is not None:
        p.update(cal.to_params())
    rae, meta = compute_range_azimuth_elevation_map(
        c, p, remove_static=False, method="bartlett",
        angles_az_deg=np.arange(-60, 60.5, 1.0), angles_el_deg=np.arange(-40, 40.5, 1.0))
    frame = rae[0, R_BIN - meta.get("r_start", 0)]
    e, a = np.unravel_index(np.argmax(frame), frame.shape)
    return float(np.hypot(meta["angles_az_deg"][a] - az, meta["angles_el_deg"][e] - el))


def test_the_channel_errors_are_recovered():
    cube, params = reflector(0.0, 0.0)
    cal = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M)
    assert cal.range_bin == R_BIN
    # c should undo g up to one common complex factor
    prod = cal.coefficients * G_TRUE
    prod = prod / prod[0]
    assert np.max(np.abs(np.angle(prod))) < np.deg2rad(2.0)
    assert np.max(np.abs(np.abs(prod) - 1.0)) < 0.03
    assert cal.coherence_before < 0.9 < 0.999 < cal.coherence_after


def test_a_calibration_from_boresight_fixes_another_angle():
    """The real test: estimated at (0, 0), applied to a reflector at (25, -12)."""
    cube0, params = reflector(0.0, 0.0)
    cal = calibration.estimate(cube0, params, range_m=R_BIN * RANGE_STEP_M)
    cube1, _ = reflector(25.0, -12.0)
    before = _angle_error(cube1, params, 25.0, -12.0)
    after = _angle_error(cube1, params, 25.0, -12.0, cal)      # through the RAE view
    assert before >= 2.0 and after <= 0.5
    v = calibration.verify(cube1, params, cal, R_BIN * RANGE_STEP_M, 25.0, -12.0)
    assert v["error_after_deg"] < 0.5 and v["error_before_deg"] > 1.5
    assert v["coherence_before"] < 0.7 and v["coherence_after"] > 0.99


def test_phase_only_leaves_the_gains_alone():
    cube, params = reflector(0.0, 0.0)
    cal = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M, amplitude=False)
    assert np.allclose(np.abs(cal.coefficients), 1.0)


def test_a_second_scatterer_in_the_cell_shows_up_and_background_removes_it():
    """
    Rank-1 over time cannot see it (all static); a second position shows it, and the
    empty-scene subtraction removes it.
    """
    other = [{"r_bin": R_BIN, "az_deg": -35.0, "el_deg": 10.0, "amp": 3.0, "static": True}]
    cube, params = reflector(0.0, 0.0, clutter=other)
    empty_only = cube - reflector(0.0, 0.0)[0]            # the scene without the reflector
    bad = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M)
    good = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M,
                                background=empty_only)
    assert bad.coherence_after > 0.95                       # it fits its own data anyway
    cube1, _ = reflector(25.0, -12.0)
    vb = calibration.verify(cube1, params, bad, R_BIN * RANGE_STEP_M, 25, -12)
    vg = calibration.verify(cube1, params, good, R_BIN * RANGE_STEP_M, 25, -12)
    assert vb["coherence_after"] < 0.8 and vb["error_after_deg"] > 1.0
    assert vg["coherence_after"] > 0.99 and vg["error_after_deg"] < 0.5


def test_reflector_margin_and_range_bias():
    cube, params = reflector(0.0, 0.0)
    cal = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M,
                               true_range_m=R_BIN * RANGE_STEP_M - 0.004)
    assert cal.reflector_margin_db > 30
    assert cal.range_bias_m == pytest.approx(0.004, abs=0.002)


def test_save_and_load_round_trip(tmp_path):
    cube, params = reflector(0.0, 0.0)
    cal = calibration.estimate(cube, params, range_m=R_BIN * RANGE_STEP_M)
    path = os.path.join(tmp_path, "cal.json")
    calibration.save(path, cal)
    back = calibration.load(path)
    assert np.allclose(back.coefficients, cal.coefficients)
    assert (back.num_tx, back.num_rx) == (NTX, RX)
    assert back.to_params() == cal.to_params()
    with open(path, "w") as f:
        f.write('{"format": "something-else"}')
    with pytest.raises(ValueError):
        calibration.load(path)


def test_a_calibration_for_another_array_is_not_applied():
    import AOA
    data = np.ones((2, 5, 12), complex)
    out = AOA._apply_calibration(data, {"channel_calibration": [[0.0, 1.0]] * 8}, 12)
    assert np.allclose(out, data)
    out = AOA._apply_calibration(data, {"channel_calibration": [[0.0, 1.0]] * 12}, 12)
    assert np.allclose(out, 1j)


def test_principal_is_the_snapshot_itself_not_its_conjugate():
    u = np.exp(1j * np.array([0.0, 0.5, 1.3, -0.7]))
    X = np.outer(np.exp(1j * np.linspace(0, 3, 10)), u)          # common phase per snapshot
    p = calibration.principal(X)
    assert abs(np.vdot(p, u)) / np.linalg.norm(u) == pytest.approx(1.0, abs=1e-9)


def test_the_command_line_tool_starts():
    from tools import calibrate
    with pytest.raises(SystemExit) as e:
        calibrate.main(["--help"])
    assert e.value.code == 0
