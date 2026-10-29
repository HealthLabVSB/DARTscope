"""
Tests for vital_bands - one set of breathing and heart bands for every vital tab (9.1).

Run with:  python -m pytest test_vital_bands.py -q
"""
import numpy as np
import pytest

import vital_bands


@pytest.fixture(autouse=True)
def _restore():
    before = vital_bands.current()
    yield
    vital_bands.set_bands(before.resp_bpm, before.heart_bpm)


def test_the_default_is_the_measured_one():
    b = vital_bands.current()
    assert b == vital_bands.PRESETS[vital_bands.DEFAULT_PRESET]
    assert b.heart_bpm == (50.0, 150.0) and b.resp_bpm == (5.0, 40.0)
    assert b.heart_hz == pytest.approx((50 / 60, 2.5))


def test_invalid_bands_are_refused_and_nothing_changes():
    before = vital_bands.current()
    for bad in [(120, 60), (0, 100), (60, 60), (50, 400)]:
        with pytest.raises(ValueError):
            vital_bands.set_bands(heart_bpm=bad)
    assert vital_bands.current() == before


def test_listeners_hear_every_change_and_presets_are_recognised():
    seen = []
    vital_bands.add_listener(seen.append)
    try:
        vital_bands.set_preset("Children, exercise")
        assert seen[-1] == vital_bands.PRESETS["Children, exercise"]
        assert vital_bands.preset_of() == "Children, exercise"
        vital_bands.set_bands(heart_bpm=(55, 140))
        assert vital_bands.preset_of() == vital_bands.CUSTOM
        n = len(seen)
        vital_bands.set_bands(heart_bpm=(55, 140))          # no change, no call
        assert len(seen) == n
    finally:
        vital_bands.remove_listener(seen.append)


def test_config_wins_over_the_shared_bands():
    vital_bands.set_bands(heart_bpm=(55, 140))
    assert vital_bands.from_config({}).heart_bpm == (55, 140)
    assert vital_bands.from_config({"heart_bpm": (40, 90)}).heart_bpm == (40, 90)
    assert vital_bands.from_config({"resp_bpm": (8, 30)}).resp_bpm == (8, 30)


def test_the_hr_tab_follows_the_shared_bands():
    from hr_analysis import HRAnalysisLogic
    logic = HRAnalysisLogic()
    vital_bands.add_listener(logic.apply_bands)
    try:
        vital_bands.set_bands(resp_bpm=(8, 30), heart_bpm=(60, 180))
        assert logic.filters["cardiac_hr_branch"]["low_hz"] == pytest.approx(1.0)
        assert logic.filters["cardiac_hr_branch"]["high_hz"] == pytest.approx(3.0)
        assert logic.filters["respiration_branch"]["high_hz"] == pytest.approx(0.5)
        assert logic.filters["cardiac_waveform_branch"]["low_hz"] == pytest.approx(1.0)
        # breathing validity follows the band too: 7 /min is outside 8-30
        fps = 20.0
        t = np.arange(int(60 * fps)) / fps
        slow = np.sin(2 * np.pi * (7 / 60) * t)
        assert logic.estimate_breathing_rate(slow, fps)["valid"] is False
    finally:
        vital_bands.remove_listener(logic.apply_bands)


def test_the_ai_tab_searches_the_shared_band():
    from ai_vital_analysis import AIVitalAnalysis
    an = AIVitalAnalysis()
    fps = 20.0
    t = np.arange(int(60 * fps)) / fps
    sig = 3 * np.sin(2 * np.pi * (15 / 60) * t) + 0.5 * np.sin(2 * np.pi * (132 / 60) * t)
    trend = an.estimate_trends(sig, fps, 20.0, 50.0)
    assert np.median([w["hr_bpm"] for w in trend]) == pytest.approx(132, abs=2)
    vital_bands.set_bands(heart_bpm=(48, 120))              # the old Vital Signs band
    trend = an.estimate_trends(sig, fps, 20.0, 50.0)
    assert all(w["hr_bpm"] <= 120 for w in trend)


def test_a_heart_above_120_is_found_on_the_vital_signs_tab_now():
    """With the old fixed 48-120 bpm band this rate could not be reported at all."""
    import vitalsigns as vs
    fps, dur, lam = 20.0, 60.0, 3e8 / 60.79e9
    t = np.arange(int(fps * dur)) / fps
    disp = 4e-3 * np.sin(2 * np.pi * 15 / 60 * t) + 0.3e-3 * np.sin(2 * np.pi * 132 / 60 * t)
    rng = np.random.default_rng(0)
    cube = (0.02 * (rng.standard_normal((t.size, 1, 4, 64))
                    + 1j * rng.standard_normal((t.size, 1, 4, 64)))).astype(np.complex64)
    cube[:, :, :, 20] += np.exp(1j * 4 * np.pi * disp / lam)[:, None, None]
    cfg = dict(auto_bin=False, bin_idx=20, roi_hw=1, roi_mode="Single bin", auto_rx=True,
               win_br=30.0, win_hr=15.0, win_step=1.0, phase_method="unwrap", dc_method="mean")
    r = vs.process_vital_signs_logic(cube, {"effectiveStartFreqGHz": 60.79}, fps,
                                     np.arange(64) * 0.05, cfg)
    assert r["hr_bpm"] == pytest.approx(132, abs=2)
