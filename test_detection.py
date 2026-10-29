"""
Tests for detection - CFAR on Range-Doppler maps.

Run with:  python -m pytest test_detection.py -q
"""
import numpy as np
import pytest

from detection import cfar_2d_ca, extract_peaks, range_only


def _map(rows, cols=128, targets=((2, 40, 100.0),), seed=0):
    """(Doppler, range) magnitudes: exponential noise of mean 1 and a few strong cells."""
    rng = np.random.default_rng(seed)
    x = rng.exponential(1.0, (rows, cols))
    for r, c, a in targets:
        x[r, c] = a
    return x


@pytest.mark.parametrize("method", ["average", "ordered"])
def test_a_full_map_is_tested_in_two_dimensions(method):
    x = _map(64, targets=((32, 60, 200.0),))
    det = cfar_2d_ca(x, guard_cells=2, training_cells=8, threshold_factor=12.0, method=method)
    assert det[32, 60]
    assert det.sum() <= 5               # noise stays below 12x its level (OS: its median)
    assert not det[:10].any() and not det[:, :10].any()     # no decision without a full window
    assert not range_only(64, 2, 8)


@pytest.mark.parametrize("method", ["average", "ordered"])
@pytest.mark.parametrize("rows", [4, 5])
def test_a_few_velocity_bins_are_tested_along_range(rows, method):
    """
    4 chirps per frame (the vital-signs settings) or Studio's 5 give 4-5 velocity bins:
    the 21-cell window never fitted along Doppler and the view never showed a detection.
    """
    x = _map(rows, targets=((rows // 2, 60, 200.0), (rows // 2, 90, 150.0)))
    assert range_only(rows, 2, 8)
    det = cfar_2d_ca(x, guard_cells=2, training_cells=8, threshold_factor=12.0, method=method)
    assert det[rows // 2, 60] and det[rows // 2, 90]
    assert det.sum() <= 4
    assert not det[:, :10].any() and not det[:, -10:].any()
    peaks = extract_peaks(det, x, min_distance=3)
    assert len(peaks) >= 2


def test_a_map_narrower_than_the_window_detects_nothing():
    assert not cfar_2d_ca(_map(4, cols=20, targets=((2, 10, 100.0),))).any()


def test_range_cells_can_be_resolution_cells():
    """
    A person is ~20 bins deep on a zero-padded range axis (80 samples on 256 points, 3.2
    bins a resolution cell). 2 guard and 8 training bins all lie inside them, the noise
    estimate is the person and nothing is detected; counted in resolution cells
    (7 and 26 bins) the training reaches past the person.
    """
    rng = np.random.default_rng(1)
    x = rng.exponential(1.0, (4, 256))
    x[:, 120:140] += 50.0
    plain = cfar_2d_ca(x, 2, 8, threshold_factor=6.0)
    scaled = cfar_2d_ca(x, 2, 8, threshold_factor=6.0, guard_cells_range=7, training_cells_range=26)
    assert not plain[:, 120:140].any()
    assert scaled[:, 125:135].all()
    assert not scaled[:, :33].any() and not scaled[:, -33:].any()     # no decision at the edges

