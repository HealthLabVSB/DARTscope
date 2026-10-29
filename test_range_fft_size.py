"""
Tests for the range FFT length (roadmap 9.2): never shorter than a chirp.

Run with:  python -m pytest test_range_fft_size.py -q
"""
import numpy as np
import pytest

from radar_processing import compute_range_doppler_map, compute_range_fft_cube, range_fft_size
from range_profile import range_profile_dbfs


def test_the_rule():
    assert range_fft_size(256) == 256
    assert range_fft_size(150) == 256
    assert range_fft_size(150, pad_pow2=False) == 150
    assert range_fft_size(384, 1024) == 1024          # zero padding is kept
    assert range_fft_size(384, 256) == 512            # truncation is refused
    assert range_fft_size(512, 256) == 512


def _chirps(n_samples, beats, frames=2, chirps=4, rx=2):
    n = np.arange(n_samples)
    sig = sum(np.exp(2j * np.pi * b * n / n_samples) for b in beats)
    return np.broadcast_to(sig, (frames, chirps, rx, n_samples)).astype(np.complex64).copy()


def test_a_long_chirp_keeps_its_resolution():
    """
    Two targets 3 range cells apart in a 512-sample chirp: resolved (Hann window) when
    all 512 samples are used; cut to 256 samples - what n_fft_range=256 used to do -
    they are 1.5 of the coarser cells apart and merge into one.
    """
    cube = _chirps(512, beats=(100.0, 103.0))
    spec, meta = compute_range_fft_cube(cube, {"dataFmt_adcFmt": "Complex1X"},
                                        n_fft_range=256, return_complex=True)
    assert spec.shape[-1] == 512
    p = np.abs(spec[0, 0, 0])
    peaks = [k for k in range(1, 511) if p[k] > p[k - 1] and p[k] > p[k + 1] and p[k] > 0.5 * p.max()]
    assert len(peaks) == 2
    cut = np.abs(np.fft.fft(cube[0, 0, 0, :256] * np.hanning(256)))
    peaks_cut = [k for k in range(1, 255) if cut[k] > cut[k - 1] and cut[k] > cut[k + 1] and cut[k] > 0.5 * cut.max()]
    assert len(peaks_cut) == 1


def test_the_range_doppler_map_and_range_profile_do_not_truncate_either():
    cube = _chirps(384, beats=(40.0,))
    rd, meta = compute_range_doppler_map(cube, {"dataFmt_adcFmt": "Complex1X",
                                                "digOutSampleRate_MHz": 10.0, "freqSlopeMHz_us": 60.0},
                                         n_fft_range=256)
    assert meta["n_fft_range"] == 512
    r, db = range_profile_dbfs(cube[0, :, 0, :], 10e6, 60e12, 256, 16)
    assert r.size == 512                       # complex samples: every bin is a range
