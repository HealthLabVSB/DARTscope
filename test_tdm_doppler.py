"""
Tests for TDM-MIMO Doppler compensation (AOA.compensate_tdm_doppler) in every AoA view.

A moving target shifts its phase between the TX slots of a loop; uncorrected, the
beamformer puts it at the wrong angle. The synthetic target here is a static one from
test_capon_rae with a Doppler ramp over the chirps, so the geometry and the element
phases are the ones the AoA code already agrees with.

Run with:  python -m pytest test_tdm_doppler.py -q
"""
import numpy as np
import pytest

from AOA import (compensate_tdm_doppler, compute_aoa_spectrum,
                 compute_azimuth_elevation_map, compute_range_azimuth_elevation_map)
from test_capon_rae import make_params, make_synthetic_cube

L, S, RX, R, R_BIN = 64, 3, 4, 64, 30


def moving_cube(k_loop_bins, az=20.0, el=-10.0, F=2):
    """Target at R_BIN moving at k_loop_bins Doppler bins over L loops (may exceed L/2)."""
    params = make_params(R)
    cube = make_synthetic_cube([{"r_bin": R_BIN, "az_deg": az, "el_deg": el, "amp": 5.0,
                                 "static": True}], params, F=F, loops=L, R=R, noise=0.01)
    n = np.arange(L * S)                                   # chirp slot index
    cube[:, :, :, R_BIN] *= np.exp(2j * np.pi * k_loop_bins * n / (L * S))[None, :, None]
    return cube, params


def _virt(cube):
    F = cube.shape[0]
    return cube.reshape(F, L, S, RX, R).reshape(F, L, S * RX, R)


def _tx_phase_error(virt, ref):
    """Largest phase error of a TX block relative to the static reference, radians."""
    x, y = virt[0, 0, :, R_BIN], ref[0, 0, :, R_BIN]
    d = np.angle(x * np.conj(y)).reshape(S, RX)
    d = d - d[0:1, :]                                   # common loop phase removed
    return float(np.max(np.abs(np.angle(np.exp(1j * d)))))


def test_compensation_restores_the_static_phases_between_tx():
    static, _ = moving_cube(0.0)
    moving, _ = moving_cube(20.4)
    before = _tx_phase_error(_virt(moving), _virt(static))
    comp, k = compensate_tdm_doppler(_virt(moving), S, RX)
    after = _tx_phase_error(comp, _virt(static))
    assert before > 1.0                                   # 2 * 2 pi 20.4 / 192 = 1.34 rad
    assert after < 0.02
    assert k[0, R_BIN] == pytest.approx(20.4, abs=0.05)


def test_a_static_target_is_left_alone():
    static, _ = moving_cube(0.0)
    comp, k = compensate_tdm_doppler(_virt(static), S, RX)
    assert np.allclose(k[:, R_BIN], 0.0, atol=1e-3)       # noise moves it by ~1e-5
    assert _tx_phase_error(comp, _virt(static)) < 1e-3


def test_negative_velocity():
    static, _ = moving_cube(0.0)
    moving, _ = moving_cube(-13.7)
    comp, k = compensate_tdm_doppler(_virt(moving), S, RX)
    assert k[0, R_BIN] == pytest.approx(-13.7, abs=0.05)
    assert _tx_phase_error(comp, _virt(static)) < 0.02


def test_folded_velocity_needs_the_right_hypothesis():
    """Beyond v_max the loop Doppler folds; the TX phase does not (roadmap 1.4)."""
    static, _ = moving_cube(0.0)
    moving, _ = moving_cube(20.4 + L)                     # one fold: reads as 20.4
    wrong, k = compensate_tdm_doppler(_virt(moving), S, RX, hypothesis=0)
    right, _ = compensate_tdm_doppler(_virt(moving), S, RX, hypothesis=1)
    assert k[0, R_BIN] == pytest.approx(20.4, abs=0.05)
    assert _tx_phase_error(wrong, _virt(static)) > 1.0
    assert _tx_phase_error(right, _virt(static)) < 0.02


def _peak_rae(cube, params, comp):
    rae, meta = compute_range_azimuth_elevation_map(
        cube, {**params, "tdm_doppler_compensation": comp}, remove_static=False,
        method="bartlett")
    _, el_i, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
    return meta["angles_az_deg"][az_i], meta["angles_el_deg"][el_i]


def test_rae_places_a_moving_target_correctly_only_with_compensation():
    cube, params = moving_cube(20.4, az=20.0, el=-10.0)
    az0, el0 = _peak_rae(cube, params, False)
    az1, el1 = _peak_rae(cube, params, True)
    assert max(abs(az0 - 20.0), abs(el0 + 10.0)) > 4.0
    assert abs(az1 - 20.0) <= 2.5 and abs(el1 + 10.0) <= 2.5


def test_azimuth_elevation_view_uses_it_too():
    cube, params = moving_cube(20.4, az=20.0, el=-10.0)

    def peak(comp):
        spec, meta = compute_azimuth_elevation_map(
            cube, {**params, "tdm_doppler_compensation": comp}, range_bin=R_BIN,
            remove_static=False)
        el_i, az_i = np.unravel_index(np.argmax(spec[0]), spec[0].shape)
        return meta["angles_az_deg"][az_i], meta["angles_el_deg"][el_i]

    az0, el0 = peak(False)
    az1, el1 = peak(True)
    assert max(abs(az0 - 20.0), abs(el0 + 10.0)) > 4.0
    assert abs(az1 - 20.0) <= 2.5 and abs(el1 + 10.0) <= 2.5


def test_one_dimensional_aoa_uses_it_too():
    cube, params = moving_cube(20.4, az=25.0, el=0.0)

    def peak(comp):
        spec, meta = compute_aoa_spectrum(
            cube, {**params, "tdm_doppler_compensation": comp}, range_bin=R_BIN,
            remove_static=False)
        return meta["angles_deg"][int(np.argmax(np.asarray(spec)[0]))]

    assert abs(peak(True) - 25.0) <= 2.0
    assert abs(peak(True) - 25.0) < abs(peak(False) - 25.0)


@pytest.mark.parametrize("az, el", [(20.0, -10.0), (-30.0, 15.0)])
def test_the_azimuth_elevation_view_and_the_rae_view_agree(az, el):
    """The Az-El Bartlett beamformer used to be unconjugated and drew the mirror image."""
    cube, params = moving_cube(0.0, az=az, el=el)
    spec, meta = compute_azimuth_elevation_map(cube, params, range_bin=R_BIN,
                                               remove_static=False)
    el_i, az_i = np.unravel_index(np.argmax(spec[0]), spec[0].shape)
    assert abs(meta["angles_az_deg"][az_i] - az) <= 2.5
    assert abs(meta["angles_el_deg"][el_i] - el) <= 2.5
    assert _peak_rae(cube, params, False) == pytest.approx((az, el), abs=2.5)
