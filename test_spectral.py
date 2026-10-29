"""
Tests for spectral - sub-bin peak interpolation and zoom FFT.

Run with:  python -m pytest test_spectral.py -q
"""
import numpy as np
import pytest

import spectral
from detection import detect_targets_rae, detections_to_range_velocity, extract_peaks


@pytest.fixture(autouse=True)
def _restore_default():
    before = spectral.default_method()
    yield
    spectral.set_default_method(before)


def _hann_tone_error(method, pad=1, n=64):
    """Worst-case error in bins over offsets -0.5..0.5, Hann window as the pipeline uses."""
    worst = 0.0
    nfft = n * pad
    for off in np.linspace(-0.5, 0.5, 41):
        f0 = (20 + off) / n                         # in cycles/sample, bin 20 unpadded
        x = np.exp(2j * np.pi * f0 * np.arange(n)) * np.hanning(n)
        mag = np.abs(np.fft.fft(x, nfft))
        k = int(np.argmax(mag))
        est = (k + spectral.peak_offset(mag, k, method)) / pad
        worst = max(worst, abs(est - (20 + off)))
    return worst


def test_documented_accuracy_holds():
    """The numbers in the module docstring, which is how the default was chosen."""
    assert _hann_tone_error("gaussian") < 0.02
    assert _hann_tone_error("parabolic") < 0.06
    assert _hann_tone_error("gaussian", pad=4) < 0.001
    assert _hann_tone_error("parabolic", pad=4) < 0.004
    assert 0.45 < _hann_tone_error("none") <= 0.5


def test_power_and_db_give_the_same_vertex_as_magnitude():
    n = 64
    x = np.exp(2j * np.pi * (20.3 / n) * np.arange(n)) * np.hanning(n)
    mag = np.abs(np.fft.fft(x))
    k = int(np.argmax(mag))
    d = spectral.peak_offset(mag, k, "gaussian")
    assert spectral.peak_offset(mag ** 2, k, "gaussian") == pytest.approx(d)
    assert spectral.peak_offset(20 * np.log10(mag), k, "gaussian", scale="db") == pytest.approx(d)


def test_edges_and_none_stay_on_the_bin():
    axis = np.arange(10) * 0.5
    v = np.linspace(1, 2, 10)                   # maximum at the last bin
    assert spectral.refine_peak(axis, v, 9) == (4.5, 2.0)
    v2 = np.array([1, 3, 2, 1, 1.0])
    assert spectral.refine_peak(axis[:5], v2, 1, method="none")[0] == 0.5


def test_default_method_is_switchable_and_checked():
    v = np.array([1, 3, 2.0])
    spectral.set_default_method("none")
    assert spectral.peak_offset(v, 1) == 0.0
    spectral.set_default_method("parabolic")
    assert spectral.peak_offset(v, 1) > 0
    with pytest.raises(ValueError):
        spectral.set_default_method("jacobsen")


def test_refined_height_is_closer_to_the_true_peak():
    n = 64
    x = np.exp(2j * np.pi * (20.5 / n) * np.arange(n)) * np.hanning(n)
    mag = np.abs(np.fft.fft(x))
    true_peak = np.abs(np.sum(np.hanning(n)))   # window gain at zero offset
    k = int(np.argmax(mag))
    _, h = spectral.refine_peak(np.arange(n), mag, k)
    assert abs(h - true_peak) < 0.5 * abs(mag[k] - true_peak)


def test_zoom_spectrum_equals_zero_padded_fft():
    fs, n = 1000.0, 256
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    pad = 16
    ref = np.fft.fft(x * _periodic_hann(n), n * pad)
    k0, k1 = 40 * pad, 60 * pad
    f, z = spectral.zoom_spectrum(x, fs, k0 * fs / (n * pad), k1 * fs / (n * pad), n=k1 - k0)
    assert np.allclose(z, ref[k0:k1], atol=1e-9)
    assert f[0] == pytest.approx(k0 * fs / (n * pad))


def _periodic_hann(n):
    from scipy.signal import get_window
    return get_window("hann", n, fftbins=True)


def test_zoom_range_profile_places_a_target():
    params = {"digOutSampleRate_MHz": 10.0, "freqSlopeMHz_us": 60.0}
    fs, slope, c = 10e6, 60e12, 299_792_458.0
    r_true = 1.2345
    t = np.arange(256) / fs
    chirp = np.exp(2j * np.pi * (2 * slope * r_true / c) * t)
    r, spec = spectral.zoom_range_profile(chirp, params, 1.0, 1.5, n=1001)
    assert abs(r[np.argmax(np.abs(spec))] - r_true) < 0.001


def test_rd_detection_reports_fractional_range_and_velocity():
    n_d, n_r = 32, 64
    d = np.arange(n_d)[:, None]
    r = np.arange(n_r)[None, :]
    rd = 1e-3 + np.exp(-0.5 * ((d - 10.3) / 1.2) ** 2 - 0.5 * ((r - 20.6) / 1.2) ** 2)
    det = rd > 0.5
    peaks = extract_peaks(det, rd)
    assert len(peaks) == 1
    p = peaks[0]
    assert (p["doppler_idx"], p["range_idx"]) == (10, 21)
    assert p["doppler_pos"] == pytest.approx(10.3, abs=0.02)
    assert p["range_pos"] == pytest.approx(20.6, abs=0.02)
    out = detections_to_range_velocity(peaks, np.arange(n_r) * 0.05, np.arange(n_d) * 0.1 - 1.6)
    assert out[0]["range_m"] == pytest.approx(20.6 * 0.05, abs=0.002)
    assert out[0]["velocity_ms"] == pytest.approx(10.3 * 0.1 - 1.6, abs=0.003)


def test_rd_detection_without_interpolation_is_unchanged():
    spectral.set_default_method("none")
    rd = np.full((8, 8), 1e-3)
    rd[3, 5] = 1.0
    rd[3, 6] = 0.6
    p = extract_peaks(rd > 0.5, rd)[0]
    assert (p["doppler_pos"], p["range_pos"]) == (3.0, 5.0)


def test_rae_detection_is_interpolated_in_all_three_axes():
    R, EL, AZ = 64, 9, 61
    rr, ee, aa = np.meshgrid(np.arange(R), np.arange(EL), np.arange(AZ), indexing="ij")
    lin = 1e-4 + np.exp(-0.5 * (((rr - 30.4) / 1.0) ** 2 + ((ee - 4.3) / 1.0) ** 2
                              + ((aa - 35.7) / 1.5) ** 2))
    rae_db = 10 * np.log10(lin)
    rng_axis = np.arange(R) * 0.04
    az = np.linspace(-60, 60, AZ)
    el = np.linspace(-20, 20, EL)
    t = detect_targets_rae(rae_db, rng_axis, az, el, threshold_factor=4.0)
    assert len(t) == 1
    assert t[0]["range_m"] == pytest.approx(30.4 * 0.04, abs=0.002)
    assert t[0]["azimuth_deg"] == pytest.approx(-60 + 35.7 * 2.0, abs=0.05)
    assert t[0]["elevation_deg"] == pytest.approx(-20 + 4.3 * 5.0, abs=0.1)
