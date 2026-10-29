"""
Tests for iq_balance - IQ imbalance estimate and correction (roadmap 1.7).

Run with:  python -m pytest test_iq_balance.py -q
"""
import numpy as np
import pytest

import iq_balance
from capture_health import image_rejection_db

N, CHIRPS, FRAMES = 256, 16, 40


def imbalanced(x, gain_db, phase_deg):
    g, phi = 10 ** (gain_db / 20), np.deg2rad(phase_deg)
    mu, nu = (1 + g * np.exp(-1j * phi)) / 2, (1 - g * np.exp(1j * phi)) / 2
    return mu * x + nu * np.conj(x)


def scene(seed=0, cycles=(23.4, 61.7, 97.2), rx=2):
    """[frames, chirps, rx, samples]: a few targets with random phase per chirp, noise."""
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    x = np.zeros((FRAMES, CHIRPS, rx, N), complex)
    for i, c in enumerate(cycles):
        ph = np.exp(2j * np.pi * rng.random((FRAMES, CHIRPS, rx, 1)))
        x += (1.0 / (i + 1)) * ph * np.exp(2j * np.pi * c * n / N)
    x += 1e-3 * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x.astype(np.complex64)


def _irr(c, r):
    return image_rejection_db(c[:, :, r])


def test_the_image_method_recovers_gain_and_phase():
    y = scene()
    y[:, :, 0] = imbalanced(y[:, :, 0], 0.8, 5.0)
    y[:, :, 1] = imbalanced(y[:, :, 1], -0.4, -3.0)
    imb = iq_balance.estimate(y, "image")
    assert imb.gain_db == pytest.approx([0.8, -0.4], abs=0.05)
    assert imb.phase_deg == pytest.approx([5.0, -3.0], abs=0.3)
    fixed = iq_balance.apply(y, imb)
    for r in range(2):
        assert _irr(y, r) < 30 and _irr(fixed, r) > 50


def test_gram_schmidt_works_on_noise_like_content():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((FRAMES, CHIRPS, 1, N)) + 1j * rng.standard_normal((FRAMES, CHIRPS, 1, N)))
    y = imbalanced(x, 0.8, 5.0)
    imb = iq_balance.estimate(y, "gram-schmidt")
    assert imb.gain_db[0] == pytest.approx(0.8, abs=0.05)
    assert imb.phase_deg[0] == pytest.approx(5.0, abs=0.3)


def test_gram_schmidt_is_fooled_by_a_slow_target_and_the_image_method_is_not():
    """Why 'image' is the default: a strong target at 2.3 beat cycles per chirp."""
    y = scene(cycles=(2.3 + 4, 61.7), rx=1)              # just outside the excluded DC bins
    y = imbalanced(y, 0.3, 1.0)
    gs = iq_balance.estimate(y, "gram-schmidt")
    im = iq_balance.estimate(y, "image")
    assert abs(im.phase_deg[0] - 1.0) < 0.3
    assert abs(gs.phase_deg[0] - 1.0) > 3 * abs(im.phase_deg[0] - 1.0)


def test_correct_keeps_only_channels_it_does_not_harm():
    y = scene()
    y[:, :, 0] = imbalanced(y[:, :, 0], 0.8, 5.0)       # RX1 is balanced already
    out, imb, before, after = iq_balance.correct(y, "image")
    assert imb.use[0] and after[0] > before[0] + 15
    assert _irr(out, 0) > 45                            # corrected on all frames
    assert np.all(after[imb.use] >= before[imb.use] - 0.5)


def test_real_input_is_refused():
    with pytest.raises(ValueError):
        iq_balance.estimate(np.ones((2, 2, 1, 8)))
