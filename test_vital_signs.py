"""
Ground-truth tests for the vital-signs chain.

A chest wall is synthesised moving at a known breathing and heart rate, and the real
processing chain has to recover both. Every test here corresponds to a defect that was
found by doing exactly this and getting the wrong answer back.

Run with:  python -m pytest test_vital_signs.py -q
"""
import numpy as np
import pytest
import scipy.signal as sg

import vitalsigns as vs
from hr_analysis import HRAnalysisLogic, MIN_PEAK_TO_AVG

BR_BPM, HR_BPM, DUR = 15.0, 72.0, 60.0
LAMBDA = 3e8 / 60.79e9          # IWR6843 at 60 GHz


def chest_cube(fps, r_bin=20, R=64, RX=4, noise=0.02, seed=0):
    """rfft_cube (F, chirps, RX, range) holding one breathing, beating chest."""
    rng = np.random.default_rng(seed)
    n = int(fps * DUR)
    t = np.arange(n) / fps
    disp = (4e-3 * np.sin(2 * np.pi * BR_BPM / 60 * t) +
            0.3e-3 * np.sin(2 * np.pi * HR_BPM / 60 * t))
    cube = (noise * (rng.standard_normal((n, 1, RX, R)) +
                     1j * rng.standard_normal((n, 1, RX, R)))).astype(np.complex64)
    cube[:, :, :, r_bin] += np.exp(1j * 4 * np.pi * disp / LAMBDA)[:, None, None]
    cube[:, :, :, r_bin + 8] += 0.8            # static clutter somewhere else
    return cube


def base_config(**over):
    cfg = dict(auto_bin=True, roi_hw=1, roi_mode="Single bin", auto_rx=True,
               clutter_removal=True, win_br=30.0, win_hr=15.0, win_step=1.0,
               conf_thresh=vs.DEFAULT_CONF_THRESH, phase_method='unwrap',
               dc_method='mean')
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("fps", [10.0, 15.0, 20.0, 30.0])
def test_rates_are_recovered_at_any_frame_rate(fps):
    r = vs.process_vital_signs_logic(chest_cube(fps), {'effectiveStartFreqGHz': 60.79},
                                     fps, np.arange(64) * 0.05, base_config())
    assert abs(r['br_bpm'] - BR_BPM) < 1.5
    assert abs(r['hr_bpm'] - HR_BPM) < 2.0


def test_auto_bin_follows_the_breathing_not_a_short_movement():
    """
    A person breathes at bin 20 for the whole minute; for 3 s something moves strongly at
    bin 40. On the energy of the whole recording the movement won; the median over 10 s
    windows takes the chest. Bins nearer than 0.3 m are never chosen.
    """
    fps = 20.0
    cube = chest_cube(fps)
    n = cube.shape[0]
    t = np.arange(n) / fps
    burst = (t >= 20) & (t < 23)
    swing = np.exp(1j * 2 * np.pi * 0.4 * t) * (1 + 3 * np.sin(2 * np.pi * 0.3 * t))
    cube[burst, :, :, 40] += (6.0 * swing[burst])[:, None, None]
    r_axis = np.arange(64) * 0.05
    rc = cube - cube.mean(axis=0, keepdims=True)
    f_br = (5 / 60, 40 / 60)
    assert vs.auto_bin(rc, fps, f_br, r_axis) == 20
    whole = vs.auto_bin(rc, fps, f_br, r_axis, window_s=DUR)        # the old rule
    assert whole == 40
    cube[:, :, :, 3] += (50.0 * swing)[:, None, None]                # 0.15 m: coupling
    assert vs.auto_bin(cube - cube.mean(axis=0, keepdims=True), fps, f_br, r_axis) == 20


def test_the_neighbour_with_drift_does_not_take_the_chest():
    """
    The chest breathes at bin 20; at bin 21 a stronger reflection drifts in amplitude
    without breathing. Picking the neighbour by the total power of the magnitude moved
    the analysis onto bin 21; the bin that breathes above its noise stays.
    """
    fps = 20.0
    cube = chest_cube(fps)
    n = cube.shape[0]
    t = np.arange(n) / fps
    cube[:, :, :, 21] += (3.0 * (1 + 0.8 * np.sin(2 * np.pi * 0.02 * t)))[:, None, None]
    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                     np.arange(64) * 0.05,
                                     base_config(roi_mode="Best bin in ROI", clutter_removal=False))
    assert r['best_bin'] == 20
    assert abs(r['br_bpm'] - BR_BPM) < 1.5


def test_every_receiver_counts_for_the_rates():
    """
    One receiver of four carries the chest, three only noise. Auto RX used to analyse the
    receiver with the most fluctuation - with noise that loud, a noisy one; the rates now
    come from all of them (breathing spectra summed, heart from the mean displacement).
    """
    fps = 20.0
    rng = np.random.default_rng(3)
    cube = chest_cube(fps, noise=0.02)
    cube[:, :, 1:, :] = (0.6 * (rng.standard_normal(cube[:, :, 1:, :].shape)
                                + 1j * rng.standard_normal(cube[:, :, 1:, :].shape))).astype(np.complex64)
    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                     np.arange(64) * 0.05, base_config(auto_bin=False, bin_idx=20,
                                                                       roi_mode="Single bin"))
    assert abs(r['br_bpm'] - BR_BPM) < 1.5
    rr_curr, rr_mean, _, _ = r['rr_stats']
    assert rr_mean is not None and abs(rr_mean - BR_BPM) < 1.5


def test_the_heart_comes_from_the_neighbouring_bins_too():
    """
    The chest spans three range bins, each with its own noise: the heart from the
    receivers of all three (heart_channels) is cleaner than from one bin's. A bin holding
    only noise - its unwrapped phase a random walk - stays out of the average.
    """
    fps = 20.0
    n = int(fps * DUR)
    rng = np.random.default_rng(7)
    t = np.arange(n) / fps
    heart = 0.03e-3 * np.sin(2 * np.pi * HR_BPM / 60 * t)
    # breathing of 2.405 rad (a zero of J0): the arc's mean is the circle's centre, so the
    # centring adds no harmonics of its own and the noise is what the average removes
    breath = 2.405 * LAMBDA / (4 * np.pi) * np.sin(2 * np.pi * BR_BPM / 60 * t)
    chest = np.exp(1j * 4 * np.pi * (breath + heart) / LAMBDA)
    sig = 0.15 * (rng.standard_normal((n, 4, 7)) + 1j * rng.standard_normal((n, 4, 7)))
    for b, a in ((2, 0.8), (3, 1.0), (4, 0.8)):
        sig[:, :, b] += a * chest[:, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, 4))[None, :]
    assert vs.arc_snr(sig[:, 0, 0]) < 1.2 < vs.ARC_SNR_MIN < vs.arc_snr(sig[:, 0, 2]) < vs.arc_snr(sig[:, 0, 3])
    three = vs.heart_channels(sig, 3, LAMBDA, half=3)
    one = vs.heart_channels(sig, 3, LAMBDA, half=0)
    assert three.shape == (12, n) and one.shape == (4, n)        # bins 2-4; 0, 1, 5, 6 out
    sos = sg.butter(4, [0.8, 2.5], 'bandpass', fs=fps, output='sos')

    def corr(d):
        return np.corrcoef(sg.sosfiltfilt(sos, d), heart)[0, 1]
    assert corr(three.mean(axis=0)) > corr(one.mean(axis=0)) + 0.03     # 0.93 against 0.87
    # the chain: "Best bin in ROI" takes the heart from the bin and its neighbours
    cube = sig[:, None, :, :].astype(np.complex64)
    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps, 0.3 + np.arange(7) * 0.05,
                                     base_config(auto_bin=False, bin_idx=3, roi_mode="Best bin in ROI",
                                                 clutter_removal=False, win_hr=30.0))
    assert r['bin_used'] == 3 and r['heart_channels'] == 12
    single = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps, 0.3 + np.arange(7) * 0.05,
                                          base_config(auto_bin=False, bin_idx=3, roi_mode="Single bin",
                                                      clutter_removal=False, win_hr=30.0))
    assert single['heart_channels'] == 4


@pytest.mark.parametrize("fps", [10.0, 15.0, 20.0])
def test_sliding_statistics_are_not_empty(fps):
    """
    The confidence used to be a share of the band energy, which shrinks as the FFT gets
    longer: a noiseless sinusoid scored 0.0976 against a threshold of 0.1, so every
    window was discarded and the tab's current/mean/min/max readouts stayed blank.
    """
    r = vs.process_vital_signs_logic(chest_cube(fps), {'effectiveStartFreqGHz': 60.79},
                                     fps, np.arange(64) * 0.05, base_config())
    rr_curr, rr_mean, _, _ = r['rr_stats']
    hr_curr, hr_mean, _, _ = r['hr_stats']
    assert rr_mean is not None and hr_mean is not None, "no window passed the threshold"
    assert abs(rr_mean - BR_BPM) < 1.5
    assert abs(hr_mean - HR_BPM) < 2.0


def test_confidence_does_not_depend_on_frame_rate():
    """
    The confidence used to be the peak's share of the band energy, which depends on how
    many FFT bins the band happens to cover - so it halved every time the FFT doubled
    (0.192 / 0.098 / 0.049 for one and the same sinusoid) and shrank as the frame rate
    dropped. That is what made the SAME 15 bpm breathing "valid" at 20 fps and
    "invalid" at 15 and 10 fps against one fixed threshold.

    Peak-to-average depends on the signal and on how long it was observed, not on the
    sampling rate.
    """
    confs = []
    for fps in (10.0, 15.0, 20.0, 30.0, 40.0):
        t = np.arange(int(30.0 * fps)) / fps
        _, conf, _, _ = vs.get_peak_conf(sg.detrend(np.sin(2 * np.pi * 1.2 * t)),
                                         fps, (0.8, 2.0))
        confs.append(conf)
    assert min(confs) > MIN_PEAK_TO_AVG, f"a clean sinusoid should qualify, got {confs}"
    assert max(confs) - min(confs) < 0.1, f"confidence moves with the frame rate: {confs}"


def test_longer_observation_is_more_confident():
    """Integrating for longer genuinely does separate the peak - that part is physics."""
    fps = 20.0
    confs = []
    for win_s in (15.0, 30.0, 60.0):
        t = np.arange(int(win_s * fps)) / fps
        _, conf, _, _ = vs.get_peak_conf(sg.detrend(np.sin(2 * np.pi * 1.2 * t)),
                                         fps, (0.8, 2.0))
        confs.append(conf)
    assert confs[0] < confs[1] < confs[2]


def test_noise_is_rejected():
    fps = 20.0
    rng = np.random.default_rng(3)
    _, conf, _, _ = vs.get_peak_conf(rng.standard_normal(int(30 * fps)), fps, (0.8, 2.0))
    assert conf < MIN_PEAK_TO_AVG, f"pure noise should not be confident, got {conf:.1f}"


def test_long_window_does_not_crash():
    """nfft was pinned at 4096; a 120 s window at 40 fps is 4800 samples and scipy raised."""
    fps = 40.0
    t = np.arange(int(120 * fps)) / fps
    f_p, _, _, _ = vs.get_peak_conf(np.sin(2 * np.pi * 0.25 * t), fps, (5 / 60, 40 / 60))
    assert abs(f_p * 60 - 15.0) < 1.0


def test_harmonic_suppression_is_matched_by_time():
    """
    Breathing windows (30 s) and heart windows (15 s) do not line up index by index.
    br_v[i] used to be paired with heart window i - 7.5 s away - and the heart series is
    longer, so its tail got no suppression at all.
    """
    fps = 20.0
    t = np.arange(int(DUR * fps)) / fps
    br = np.sin(2 * np.pi * 0.25 * t)
    rr_t, rr_v, _ = vs.sliding(br, fps, 30.0, 1.0, (5 / 60, 40 / 60))
    hr_t, _, _ = vs.sliding(br, fps, 15.0, 1.0, (0.8, 2.0), br_t=rr_t, br_v=rr_v)
    assert len(hr_t) > len(rr_t)                      # the mismatch that caused it
    # Every heart window must get a breathing value from its own point in time.
    for tc in hr_t:
        assert abs(float(np.interp(tc, rr_t, rr_v)) - BR_BPM) < 1.5


def test_the_bands_come_from_the_shared_setting_unless_the_config_names_them():
    """The old Resting checkbox did nothing at all; the bands are now shared (vital_bands)."""
    import vital_bands
    cube = chest_cube(20.0)
    args = ({'effectiveStartFreqGHz': 60.79}, 20.0, np.arange(64) * 0.05)
    shared = vs.process_vital_signs_logic(cube, *args, base_config())
    narrow = vs.process_vital_signs_logic(cube, *args, base_config(heart_bpm=(42, 102)))
    assert shared['f_heart_range'] == vital_bands.current().heart_hz
    assert narrow['f_heart_range'] == pytest.approx((0.7, 1.7))


def test_ica_is_reproducible():
    """FastICA starts from a random vector; unseeded it answered differently each run."""
    rng = np.random.default_rng(1)
    sigs = rng.standard_normal((600, 4))
    first = vs.ica_fusion(sigs, 20.0)
    for _ in range(3):
        assert np.allclose(vs.ica_fusion(sigs, 20.0), first)


# --- HR Analysis tab -------------------------------------------------------------

@pytest.mark.parametrize("fps", [10.0, 15.0, 20.0])
def test_cardiac_bands_stay_below_nyquist(fps):
    """
    The waveform branch reaches 8 Hz, which butter() cannot build below 16 fps - and
    15 fps is this application's own fallback. The error was swallowed and the RAW
    displacement was returned as if it had been filtered.
    """
    logic = HRAnalysisLogic()
    t = np.arange(int(60 * fps)) / fps
    raw = np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    for branch in ("cardiac_hr_branch", "cardiac_waveform_branch"):
        out = logic.apply_filter(raw, fps, logic.filters[branch])
        assert not np.allclose(out, raw), f"{branch} silently left the signal unfiltered"
        # breathing must actually be gone
        f, pxx = sg.welch(out, fs=fps, nperseg=min(len(out), 512))
        br_band = (f >= 0.15) & (f <= 0.4)
        hr_band = (f >= 1.0) & (f <= 1.4)
        assert pxx[hr_band].max() > pxx[br_band].max()


def test_quality_separates_signal_from_noise():
    """in-band/total on an already-bandpassed signal was ~1 for everything."""
    logic = HRAnalysisLogic()
    fps = 20.0
    t = np.arange(int(60 * fps)) / fps
    rng = np.random.default_rng(0)
    clean = logic.apply_filter(np.sin(2 * np.pi * 1.2 * t), fps,
                               logic.filters["cardiac_hr_branch"])
    noise = logic.apply_filter(rng.standard_normal(len(t)), fps,
                               logic.filters["cardiac_hr_branch"])
    good = logic.estimate_hr(clean, fps, "t", "t")
    bad = logic.estimate_hr(noise, fps, "t", "t")
    assert good["valid"] and not bad["valid"]
    assert good["quality"] > 4 * bad["quality"]


@pytest.mark.parametrize("fps", [10.0, 15.0, 20.0])
def test_breathing_validity_is_frame_rate_independent(fps):
    """The same 15 bpm breathing was 'valid' at 20 fps and 'invalid' at 15 and 10."""
    logic = HRAnalysisLogic()
    t = np.arange(int(60 * fps)) / fps
    resp = logic.apply_filter(np.sin(2 * np.pi * BR_BPM / 60 * t), fps,
                              logic.filters["respiration_branch"])
    br = logic.estimate_breathing_rate(resp, fps)
    assert br["valid"], f"clean breathing rejected at {fps} fps (conf {br['confidence']:.2f})"
    assert abs(br["br_bpm"] - BR_BPM) < 1.5


def test_quality_score_is_unit_independent():
    """1/std(diff(displacement)) changed the score when metres became millimetres."""
    logic = HRAnalysisLogic()
    fps = 20.0
    t = np.arange(int(60 * fps)) / fps
    base = 1e-3 * np.sin(2 * np.pi * 1.2 * t)
    scores = []
    for scale in (1.0, 1000.0):
        card = logic.apply_filter(base * scale, fps, logic.filters["cardiac_hr_branch"])
        scores.append(logic.compute_quality_metrics(card, fps, HR_BPM)["quality_score"])
    assert abs(scores[0] - scores[1]) < 1e-9


def test_defaults_survive_a_settings_change():
    """filters was a shallow copy, so an edit wrote straight into default_filters."""
    logic = HRAnalysisLogic()
    import vital_bands
    logic.set_filter_params("cardiac_hr_branch", {"low_hz": 0.5})
    assert logic.default_filters["cardiac_hr_branch"]["low_hz"] == vital_bands.current().heart_hz[0]
    assert logic.filters["cardiac_hr_branch"]["low_hz"] == 0.5


@pytest.mark.parametrize("truth", [48.0, 72.0, 96.0])
def test_wavelet_estimate_matches_truth(truth):
    """
    Used signal.cwt / signal.ricker, removed in SciPy 1.15, and mapped scale to
    frequency with 0.25/a instead of the wavelet's actual peak at 0.2251/a.
    """
    logic = HRAnalysisLogic()
    fps = 20.0
    t = np.arange(int(60 * fps)) / fps
    sig = logic.apply_filter(np.sin(2 * np.pi * truth / 60 * t), fps,
                             logic.filters["cardiac_hr_branch"])
    assert abs(logic.wavelet_hr_estimate(sig, fps)["hr_bpm"] - truth) < 2.5


@pytest.mark.parametrize("truth", [54.0, 72.0, 96.0])
def test_all_three_hr_estimators_agree_with_truth(truth):
    """
    The autocorrelation lag is an integer number of samples: at 20 fps the lags around
    72 bpm are 75.0, 70.6 and 66.7 bpm apart, so without interpolating the peak the
    method was quantised more coarsely than the 10 bpm consensus test that compares
    against it. Peak-to-peak counted every noise wiggle between beats (112 bpm for a
    clean 72 bpm signal) because it had no prominence requirement.
    """
    logic = HRAnalysisLogic()
    fps = 20.0
    t = np.arange(int(60 * fps)) / fps
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * truth / 60 * t) + 0.3 * rng.standard_normal(len(t))
    card = logic.apply_filter(sig, fps, logic.filters["cardiac_hr_branch"])
    m = logic.estimate_hr(card, fps, "t", "t")["hr_methods"]
    for name, value in m.items():
        assert abs(value - truth) < 4.0, f"{name} gave {value:.1f} for a {truth} bpm signal"


def test_trend_limiter_recovers_after_a_rejected_window():
    """
    The AI trend limiter kept comparing against the last ACCEPTED rate using an
    allowance of one step, no matter how much time had passed since - so one rejected
    window could lock the whole rest of the recording out.
    """
    from ai_vital_analysis import AIVitalAnalysis
    ai = AIVitalAnalysis()
    fps, win, overlap = 20.0, 10.0, 50.0
    t = np.arange(int(120 * fps)) / fps
    # heart rate steps from 60 to 90 bpm half way through
    inst = np.where(t < 60, 1.0, 1.5)
    phase = 2 * np.pi * np.cumsum(inst) / fps
    sig = np.sin(phase) + 0.6 * np.sin(2 * np.pi * 0.25 * t)
    out = ai.estimate_trends(sig, fps, win, overlap)
    late = [w for w in out if w['t_s'] > 80]
    assert any(w['valid'] for w in late), "the trend never recovered after the rate changed"


@pytest.mark.parametrize("fps", [20.0, 50.0, 100.0, 200.0])
def test_breathing_filter_is_stable_at_high_frame_rates(fps):
    """
    Breathing sits at a very low frequency relative to the sampling rate, and
    butter(..., (b, a)) loses its poles to round-off as that ratio grows: at 100 fps the
    0.083-0.667 Hz bandpass came back with poles at |z| = 1.0049 - outside the unit
    circle - and a 4 mm sine came out 2.7x too large; at 200 fps the output reached
    1e90. A 100 fps vital-signs capture is ordinary, so this was not a corner case.
    """
    n = int(60 * fps)
    t = np.arange(n) / fps
    amp = 4.0
    disp = amp * np.sin(2 * np.pi * BR_BPM / 60 * t)
    cube = np.zeros((n, 1, 1, 8), dtype=np.complex64)
    cube[:, 0, 0, 4] = np.exp(1j * 4 * np.pi * (disp * 1e-3) / LAMBDA)
    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                     np.arange(8) * 0.05,
                                     base_config(auto_bin=False, bin_idx=4,
                                                 roi_mode="Single bin",
                                                 clutter_removal=False,
                                                 win_br=min(30.0, 60.0),
                                                 win_hr=min(15.0, 30.0)))
    out_pp = float(np.ptp(r['sig_br'])) * 1e3            # metres -> mm
    assert np.isfinite(out_pp), "filter produced non-finite output"
    assert 0.5 * 2 * amp < out_pp < 2.0 * 2 * amp, \
        f"a {2*amp:.0f} mm peak-to-peak breath came out as {out_pp:.3g} mm at {fps} fps"


@pytest.mark.parametrize("fps", [20.0, 100.0, 200.0])
def test_hr_branch_filters_are_stable(fps):
    logic = HRAnalysisLogic()
    t = np.arange(int(60 * fps)) / fps
    x = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.sin(2 * np.pi * 1.2 * t)
    for branch in ("respiration_branch", "cardiac_hr_branch", "cardiac_waveform_branch"):
        y = logic.apply_filter(x, fps, logic.filters[branch])
        assert np.all(np.isfinite(y)), f"{branch} produced non-finite output at {fps} fps"
        assert np.ptp(y) < 10 * np.ptp(x), \
            f"{branch} amplified by {np.ptp(y)/np.ptp(x):.3g}x at {fps} fps"


def test_damaged_frames_are_detected_and_carried_through():
    """
    Where packets are lost the recorder writes zeros, and on a plot that is
    indistinguishable from a measurement. The mask has to reach the results so the
    waveform can be drawn dashed over those stretches.
    """
    fps = 100.0
    cube = chest_cube(fps, noise=0.02)
    cube[300:340] = 0                       # one burst of lost packets
    cube[1200:1215] = 0                     # and a shorter one
    mask = vs.detect_damaged_frames(cube)
    assert mask.sum() == 55
    assert vs.damaged_spans(mask) == [(300, 340), (1200, 1215)]

    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                     np.arange(64) * 0.05,
                                     base_config(damaged_frames=mask))
    assert r['damaged_frames'].sum() == 55
    assert abs(r['damaged_fraction'] - 55 / cube.shape[0]) < 1e-9


def test_a_clean_capture_is_not_flagged():
    """Nothing changes for a recording with no loss."""
    fps = 100.0
    cube = chest_cube(fps, noise=0.02)
    assert not vs.detect_damaged_frames(cube).any()
    r = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                     np.arange(64) * 0.05, base_config())
    assert r['damaged_fraction'] == 0.0
    assert abs(r['br_bpm'] - BR_BPM) < 1.5 and abs(r['hr_bpm'] - HR_BPM) < 2.0


def test_losses_do_not_change_the_measurement_path():
    """
    Marking is presentation only - the rates are computed exactly as before, so a
    capture is not silently reinterpreted because some frames were flagged.
    """
    fps = 100.0
    cube = chest_cube(fps, noise=0.02)
    plain = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                         np.arange(64) * 0.05, base_config())
    flagged = vs.process_vital_signs_logic(
        cube, {'effectiveStartFreqGHz': 60.79}, fps, np.arange(64) * 0.05,
        base_config(damaged_frames=np.zeros(cube.shape[0], bool)))
    assert plain['br_bpm'] == flagged['br_bpm']
    assert plain['hr_bpm'] == flagged['hr_bpm']
    marked = np.zeros(cube.shape[0], bool)
    marked[1000:1050] = True                    # frames flagged lost: still only a marking
    again = vs.process_vital_signs_logic(cube, {'effectiveStartFreqGHz': 60.79}, fps,
                                         np.arange(64) * 0.05, base_config(damaged_frames=marked))
    assert again['best_bin'] == plain['best_bin']
    assert again['br_bpm'] == plain['br_bpm'] and again['hr_bpm'] == plain['hr_bpm']


def test_damaged_spans_handles_the_edges():
    assert vs.damaged_spans(np.zeros(5, bool)) == []
    assert vs.damaged_spans(np.ones(5, bool)) == [(0, 5)]
    assert vs.damaged_spans(np.array([1, 0, 0, 1, 1], bool)) == [(0, 1), (3, 5)]
    assert vs.damaged_spans(np.array([], bool)) == []


# --- counting breaths one by one -------------------------------------------------------

UNEVEN_AT = [2.0, 7.0, 15.0, 19.0, 28.0, 34.0, 42.0, 47.0, 56.0]   # 9 breaths, 4-9 s apart


def uneven_breathing(fps, seconds=60.0, at=UNEVEN_AT, depth=0.004):
    """A chest that breathes 9 times a minute, unevenly: each breath a 3 s rise and fall."""
    t = np.arange(int(fps * seconds)) / fps
    x = np.zeros_like(t)
    for a in at:
        m = np.abs(t - a) < 1.5
        x[m] += depth * 0.5 * (1 + np.cos(np.pi * (t[m] - a) / 1.5))
    return t, x


def _in_band(x, fps, band):
    return sg.sosfiltfilt(sg.butter(4, band, btype="band", fs=fps, output="sos"), x)


def test_uneven_breaths_are_counted_as_a_person_counts_them():
    """
    The validation take of the cascade: a person counted 9 breaths in a minute, 3-9 s
    apart. Its spectrum has no clear peak (the strongest fell on the second harmonic,
    16 /min); counted one by one the breaths are all there.
    """
    fps, band = 100.0, (5 / 60, 40 / 60)
    t, x = uneven_breathing(fps)
    x = x + np.random.default_rng(0).normal(0, 2e-5, x.size)
    c = vs.count_breaths(_in_band(x, fps, band), fps, band,
                         vs.breathing_noise_rms(x, fps, band, 2.5))
    assert c["ok"] and c["n"] == 9
    assert np.allclose(c["times"], UNEVEN_AT, atol=0.5)
    assert c["rate"] == pytest.approx(60 / np.mean(np.diff(UNEVEN_AT)), abs=0.3)
    assert c["cv"] > 0.25                               # uneven, and said so
    assert c["snr"] > vs.BREATH_MIN_SNR


def test_band_passed_noise_is_not_breathing():
    fps, band = 100.0, (5 / 60, 40 / 60)
    x = np.random.default_rng(1).normal(0, 1e-4, int(60 * fps))
    c = vs.count_breaths(_in_band(x, fps, band), fps, band,
                         vs.breathing_noise_rms(x, fps, band, 2.5))
    assert not c["ok"]


def test_a_quiet_window_is_judged_by_the_swing_of_the_whole_recording():
    """Counted in a short stretch with no breath in it, the ripple is not a breath."""
    fps, band = 20.0, (5 / 60, 40 / 60)
    t, x = uneven_breathing(fps)
    x = _in_band(x + np.random.default_rng(2).normal(0, 2e-5, x.size), fps, band)
    whole = np.percentile(x, 95) - np.percentile(x, 5)
    quiet = x[int(20.5 * fps):int(26.5 * fps)]           # between the breaths at 19 and 28 s
    assert vs.count_breaths(quiet, fps, band, swing=whole)["n"] == 0
    assert vs.breathing_noise_rms(x, 2.0, band, 2.5) != vs.breathing_noise_rms(x, 2.0, band, 2.5)
