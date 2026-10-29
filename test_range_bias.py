"""
Tests for the calibrated range bias on every range axis (roadmap 9.5).

Run with:  python -m pytest test_range_bias.py -q
"""
import numpy as np
import pytest

import calibration
import spectral
from radar_processing import (C0, compute_range_doppler_map, compute_range_fft_cube,
                              range_bias_m)
from range_profile import range_profile_dbfs, range_profile_rx

FS_MHZ, SLOPE, N = 10.0, 60.0, 256
PARAMS = {"digOutSampleRate_MHz": FS_MHZ, "freqSlopeMHz_us": SLOPE, "dataFmt_adcFmt": "Complex1X"}


def chirps_at(range_m, frames=2, chirps=4, rx=2):
    """ADC samples of a point target whose beat frequency puts it at range_m (raw)."""
    f_beat = 2 * SLOPE * 1e12 * range_m / C0
    t = np.arange(N) / (FS_MHZ * 1e6)
    sig = np.exp(2j * np.pi * f_beat * t)
    return np.broadcast_to(sig, (frames, chirps, rx, N)).astype(np.complex64).copy()


def test_no_bias_unless_it_is_set():
    assert range_bias_m({}) == 0.0 and range_bias_m(None) == 0.0
    assert range_bias_m({"range_bias_m": "0.012"}) == pytest.approx(0.012)
    assert range_bias_m({"range_bias_m": "junk"}) == 0.0


def test_every_axis_moves_by_the_bias():
    cube = chirps_at(1.5)
    b = 0.03
    _, m0 = compute_range_fft_cube(cube, PARAMS, return_complex=True)
    _, m1 = compute_range_fft_cube(cube, {**PARAMS, "range_bias_m": b}, return_complex=True)
    assert np.allclose(np.asarray(m0["range_m"]) - b, m1["range_m"])
    _, rd0 = compute_range_doppler_map(cube, PARAMS)
    _, rd1 = compute_range_doppler_map(cube, {**PARAMS, "range_bias_m": b})
    assert np.allclose(np.asarray(rd0["range_axis"]) - b, rd1["range_axis"])
    r0, _ = range_profile_dbfs(cube[0, :, 0, :], FS_MHZ * 1e6, SLOPE * 1e12, N, 16)
    r1, _ = range_profile_rx(cube[0], FS_MHZ * 1e6, SLOPE * 1e12, N, 16, range_bias_m=b)
    assert np.allclose(r0 - b, r1)


def test_the_zoom_band_is_asked_for_in_corrected_range():
    true_r, b = 1.2345, 0.02
    cube = chirps_at(true_r + b)                       # what the board reports
    r, z = spectral.zoom_range_profile(cube[0, 0, 0], {**PARAMS, "range_bias_m": b},
                                       1.0, 1.5, n=2001)
    assert r[0] == pytest.approx(1.0) and r[-1] < 1.5
    assert r[int(np.argmax(np.abs(z)))] == pytest.approx(true_r, abs=0.001)


def test_calibration_measures_the_bias_and_the_axis_then_shows_the_true_distance():
    true_r, b = 1.50, 0.025
    cube = chirps_at(true_r + b, frames=10)
    rc, meta = compute_range_fft_cube(cube, PARAMS, n_fft_range=1024, return_complex=True)
    cal = calibration.estimate(rc, {**PARAMS, "tx_order_masks": [1]}, true_r + b,
                               true_range_m=true_r, range_axis=meta["range_m"])
    assert cal.range_bias_m == pytest.approx(b, abs=0.002)
    _, m = compute_range_fft_cube(cube, {**PARAMS, "range_bias_m": cal.range_bias_m},
                                  n_fft_range=1024)
    p = np.abs(compute_range_fft_cube(cube, PARAMS, n_fft_range=1024, return_complex=True)[0])
    prof = p.mean(axis=(0, 1, 2))
    k = int(np.argmax(prof))
    r_peak, _ = spectral.refine_peak(m["range_m"], prof, k)   # the grid is 24 mm coarse
    assert r_peak == pytest.approx(true_r, abs=0.002)


def test_a_bias_already_applied_is_added_back():
    """Re-estimating with a calibration active must report the whole offset, not the rest."""
    true_r, b = 1.50, 0.025
    cube = chirps_at(true_r + b, frames=10)
    applied = {**PARAMS, "tx_order_masks": [1], "range_bias_m": 0.010}
    rc, meta = compute_range_fft_cube(cube, applied, n_fft_range=1024, return_complex=True)
    cal = calibration.estimate(rc, applied, true_r, true_range_m=true_r,
                               range_axis=meta["range_m"])
    assert cal.range_bias_m == pytest.approx(b, abs=0.002)
