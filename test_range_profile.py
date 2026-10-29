"""
Tests for range_profile - one frame's range profile over several RX (roadmap 9.3).

Run with:  python -m pytest test_range_profile.py -q
"""
import numpy as np
import pytest

from range_profile import range_profile_dbfs, range_profile_rx

FS, SLOPE, N = 10e6, 60e12, 256


def frame(phases_deg, gains=None, beat=40.0, chirps=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    gains = gains or [1.0] * len(phases_deg)
    n = np.arange(N)
    x = np.zeros((chirps, len(phases_deg), N), complex)
    for r, (p, g) in enumerate(zip(phases_deg, gains)):
        x[:, r, :] = 0.1 * g * np.exp(1j * (2 * np.pi * beat * n / N + np.deg2rad(p)))
    return x + noise * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))


def _peak_db(mode, x):
    r, db = range_profile_rx(x, FS, SLOPE, N, 16, mode=mode)
    return db.max()


def test_a_single_rx_is_that_channel():
    x = frame([0, 0, 0, 0], gains=[1, 2, 1, 1])
    r0, d0 = range_profile_dbfs(x[:, 1, :], FS, SLOPE, N, 16)
    r1, d1 = range_profile_rx(x, FS, SLOPE, N, 16, mode="RX1")
    assert np.allclose(d0, d1) and np.allclose(r0, r1)
    assert _peak_db("RX1", x) == pytest.approx(_peak_db("RX0", x) + 20 * np.log10(2), abs=0.01)


def test_mrc_does_not_care_about_phase_and_sum_does():
    in_phase = frame([0, 0, 0, 0])
    spread = frame([0, 90, 180, 270])                  # a target far off boresight, say
    assert _peak_db("MRC", in_phase) == pytest.approx(_peak_db("MRC", spread), abs=0.01)
    assert _peak_db("SUM", in_phase) == pytest.approx(_peak_db("RX0", in_phase), abs=0.01)
    assert _peak_db("SUM", spread) < _peak_db("SUM", in_phase) - 40


def test_mrc_is_the_rms_of_the_channels():
    x = frame([0, 30, 60, 90], gains=[1, 1, 2, 2])
    assert 10 ** (_peak_db("MRC", x) / 20) == pytest.approx(
        np.sqrt(np.mean(np.array([1, 1, 4, 4]))) * 10 ** (_peak_db("RX0", x) / 20), rel=1e-3)


def test_complex_samples_cover_the_whole_range_as_the_rti_does():
    """
    Complex 1x spans 0..fs: a target at 3/4 of the maximum range is in the profile at 3/4
    of it. The positive half alone ended the profile at half the range; real samples
    still give the positive half.
    """
    x = frame([0, 0, 0, 0], beat=192.0)
    r_max = 299_792_458.0 * FS / (2 * SLOPE)
    r, db = range_profile_rx(x, FS, SLOPE, N, 16)
    assert r.size == N and r[-1] == pytest.approx(r_max * (N - 1) / N)
    assert r[np.argmax(db)] == pytest.approx(0.75 * r_max, abs=r_max / N)
    r1, _ = range_profile_dbfs(x[:, 0, :], FS, SLOPE, N, 16)
    assert np.allclose(r1, r)
    r_real, _ = range_profile_dbfs(np.real(x[:, 0, :]), FS, SLOPE, N, 16)
    assert r_real.size == N // 2 + 1


def test_a_missing_channel_index_is_clamped_not_fatal():
    x = frame([0, 0])
    assert _peak_db("RX7", x) == pytest.approx(_peak_db("RX1", x))
