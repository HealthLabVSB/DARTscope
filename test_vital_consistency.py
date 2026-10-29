"""
Consistency tests for the vital-signs chain.

These cover the class of bug where two tabs analyse the same recording with different
hidden assumptions - a hardcoded frame rate, a mislabelled phase method, or a silent
fallback - and then disagree for reasons unrelated to the signal.

Run with:  python -m pytest test_vital_consistency.py -q
"""
import numpy as np
import pytest

from hr_analysis import HRAnalysisLogic
from ai_vital_analysis import AIVitalAnalysis


def iq_arc(n=600, centre=(3.0, -2.0), radius=1.5, sweep_deg=140.0, noise=0.0, seed=0):
    """IQ samples on a circle that is offset from the origin (classic DC offset)."""
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(np.linspace(-sweep_deg / 2, sweep_deg / 2, n))
    x = centre[0] + radius * np.cos(ang)
    y = centre[1] + radius * np.sin(ang)
    if noise:
        x = x + rng.normal(0, noise, n)
        y = y + rng.normal(0, noise, n)
    return x + 1j * y, ang


def test_circle_fit_recovers_offset_centre():
    """A real circle fit removes the DC offset; mean-centering does not."""
    sig, true_ang = iq_arc()
    fitted = AIVitalAnalysis().extract_phase(sig, method="circlefit")

    # Phase about the true centre is the arc angle itself (up to an offset).
    fitted_span = np.ptp(fitted)
    true_span = np.ptp(true_ang)
    assert abs(fitted_span - true_span) < np.deg2rad(5), (
        f"circle fit should recover the {np.rad2deg(true_span):.0f} deg arc, "
        f"got {np.rad2deg(fitted_span):.0f} deg")

    # Plain mean-centering cannot: the centroid of an arc is not the circle centre.
    centroid = np.unwrap(np.angle(sig - np.mean(sig)))
    assert abs(np.ptp(centroid) - true_span) > np.deg2rad(15), (
        "test is meaningless if mean-centering already solves it")


def test_both_tabs_agree_on_circlefit():
    """HR Analysis and AI Vital must not compute different things under one name."""
    sig, _ = iq_arc(noise=0.01, seed=3)
    hr = HRAnalysisLogic().extract_phase_circle_fitting(sig)
    ai = AIVitalAnalysis().extract_phase(sig, method="circlefit")
    # Both are unwrapped phases of the same trajectory, so they may differ by a constant.
    assert np.allclose(hr - np.mean(hr), ai - np.mean(ai), atol=1e-6)


def test_circle_fit_survives_degenerate_input():
    """A straight line has no circle; the fallback must not raise."""
    t = np.linspace(0, 1, 200)
    line = (1 + t) + 1j * (2 + 0.5 * t)
    for phase in (HRAnalysisLogic().extract_phase_circle_fitting(line),
                  AIVitalAnalysis().extract_phase(line, method="circlefit")):
        assert phase.shape == line.shape
        assert np.all(np.isfinite(phase))


@pytest.mark.parametrize("fps", [10.0, 20.0, 30.0])
def test_hr_estimate_tracks_the_frame_rate(fps):
    """
    The same waveform sampled at a different frame rate must be reported at the right
    BPM - i.e. the estimator really uses the fps it is given, which is why a hardcoded
    fallback silently scales every result.
    """
    logic = HRAnalysisLogic()
    hr_true = 72.0
    n = int(fps * 30)
    t = np.arange(n) / fps
    sig = np.sin(2 * np.pi * (hr_true / 60.0) * t)
    res = logic.estimate_hr(sig, fps, "test", "test")
    assert abs(res["hr_bpm"] - hr_true) < 6.0, (
        f"at {fps} fps expected ~{hr_true} BPM, got {res['hr_bpm']:.1f}")


def test_wrong_fps_visibly_shifts_the_result():
    """Guards the assumption behind unifying the fps fallback."""
    logic = HRAnalysisLogic()
    fps_real, hr_true = 30.0, 72.0
    t = np.arange(int(fps_real * 30)) / fps_real
    sig = np.sin(2 * np.pi * (hr_true / 60.0) * t)
    good = logic.estimate_hr(sig, fps_real, "t", "t")["hr_bpm"]
    bad = logic.estimate_hr(sig, 15.0, "t", "t")["hr_bpm"]      # old silent fallback
    assert abs(good - hr_true) < 6.0
    assert abs(bad - hr_true) > 20.0, (
        f"a 2x wrong frame rate should clearly change the reported BPM; got {bad:.1f}")
