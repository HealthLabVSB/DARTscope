"""
Tests for the heart rate of a moving person (moving_hr.py, roadmap phase 12), on
synthetic captures (hr_synth.py) whose chest motion and heart rate are known, and the
two helpers it brought: the phase-centred range FFT (1.10) and the wavelength at the
middle of the ADC window (1.9).

Run with:  python -m pytest test_moving_hr.py -q
"""
import numpy as np
import pytest

import hr_synth
import moving_hr
import radar_processing as rp
import spectral
import vital_bands

FPS = 100.0


def run(motion, hr=72.0, seed=3, duration=40.0, **kw):
    person = hr_synth.Person(r0=1.2, hr_bpm=hr, motion=motion, duration_s=duration, seed=seed)
    cube, p = hr_synth.capture(person, duration_s=duration, fps=FPS, seed=seed)
    return person, moving_hr.analyse(cube, p, fps=FPS, **kw)


def displacement_error_mm(person, res):
    truth = person.chest_range(res.t_s)
    d = res.displacement_m - res.displacement_m.mean()
    return float(np.sqrt(np.mean((d - (truth - truth.mean())) ** 2)) * 1e3)


# --- 1.10 and 1.9 ---------------------------------------------------------------------------

@pytest.mark.parametrize("f_bins", [10.2, 10.5, 10.9, 11.4])
def test_centred_phase_is_the_same_across_the_main_lobe(f_bins):
    n, nfft = 64, 256
    x = np.exp(1j * (2 * np.pi * f_bins * np.arange(n) / n + 0.7))
    spec = spectral.centre_phase(np.fft.fft(x * np.hanning(n), nfft), n, nfft, full_spectrum=True)
    mid = 0.7 + np.pi * f_bins * (n - 1) / n             # the tone's phase mid-chirp
    k0 = int(round(f_bins * nfft / n))
    err = [np.angle(spec[k] * np.exp(-1j * mid)) for k in range(k0 - 4, k0 + 5)]
    assert np.max(np.abs(err)) < 1e-9


def test_centred_phase_of_a_slice_and_of_negative_bins():
    n, nfft = 32, 64
    x = np.exp(-1j * 2 * np.pi * 5.3 * np.arange(n) / n)  # negative frequency
    full = spectral.centre_phase(np.fft.fft(x * np.hanning(n), nfft), n, nfft, full_spectrum=True)
    part = spectral.centre_phase(np.fft.fft(x * np.hanning(n), nfft)[50:56], n, nfft, first_bin=50,
                                 full_spectrum=True)
    assert np.allclose(full[50:56], part)
    mid = -np.pi * 5.3 * (n - 1) / n
    assert abs(np.angle(full[nfft - 11] * np.exp(-1j * mid))) < 1e-9


def test_wavelength_at_the_middle_of_the_adc_window():
    p = {"effectiveStartFreqGHz": 60.0, "freqSlopeMHz_us": 100.0, "digOutSampleRate_MHz": 10.0,
         "numAdcSamples": 257}
    f_mid = 60e9 + 100e12 * 256 / (2 * 10e6)                 # +1.28 GHz
    assert rp.wavelength_mid_m(p) == pytest.approx(299_792_458.0 / f_mid)
    assert rp.wavelength_mid_m({"effectiveStartFreqGHz": 60.0}) == pytest.approx(rp._lambda_m(
        {"effectiveStartFreqGHz": 60.0}))


# --- the pieces ------------------------------------------------------------------------------

def test_the_tracker_unwraps_a_walker_across_frame_gaps():
    """0.8 m/s is 20 rad of phase from one frame to the next: only the chirps inside a
    frame and the prediction across the gap can follow that."""
    lam = 0.005
    t = np.arange(300)[:, None] * 0.01 + np.arange(4)[None, :] * 50e-6
    r = 1.0 + 0.8 * t + 0.005 * np.sin(2 * np.pi * 0.3 * t)
    z = np.exp(1j * 4 * np.pi * r / lam)
    rng = np.random.default_rng(0)
    z = z + 0.02 * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
    phase, rate, slips = moving_hr.track_phase(z, t, 0.02, lam)
    d = phase / (4 * np.pi / lam)
    err = (d - d[0, 0]) - (r - r[0, 0])
    assert np.max(np.abs(err)) < 0.2e-3
    assert np.median(rate / (4 * np.pi / lam)) == pytest.approx(0.8, abs=0.02)
    assert not slips.any()


def test_steps_left_by_dropouts_are_bridged():
    t = np.arange(1000) / FPS
    d = 0.004 * np.sin(2 * np.pi * 0.25 * t)
    bad = np.zeros(t.size, bool)
    bad[300:315] = True
    bad[600:612] = True
    d_broken = d.copy()
    d_broken[315:] += 0.0025                                 # a lost turn: lambda / 2
    d_broken[612:] -= 0.005
    fixed, n = moving_hr.stitch_steps(d_broken, FPS, bad, np.zeros(t.size, bool))
    assert n == 2
    ok = ~bad
    # the curve fitted across 120 ms of a breath leaves a fraction of a millimetre
    assert np.max(np.abs((fixed - d)[ok] - (fixed - d)[0])) < 0.3e-3


def test_the_trunk_bin_does_not_toggle():
    """Two reflections of the body, alternately a little stronger: the path stays."""
    rng = np.random.default_rng(1)
    e = rng.uniform(0.5, 1.0, (500, 30))
    e[:, 10] += 20 + 3 * np.sin(np.arange(500) / 3.0)
    e[:, 14] += 20 - 3 * np.sin(np.arange(500) / 3.0)
    path = moving_hr._track_bins(e, FPS, 0.5, dr_m=0.05)
    assert len(np.unique(path)) == 1


def test_the_trunk_bin_follows_a_walker():
    e = np.full((400, 60), 0.1)
    true = (10 + np.arange(400) * 0.08).astype(int)          # a bin every 12 frames
    e[np.arange(400), true] = 50.0
    path = moving_hr._track_bins(e, FPS, 0.1, dr_m=0.05)
    assert np.mean(np.abs(path - true)) < 1.5


def test_viterbi_follows_a_drifting_rate_through_noise():
    grid = np.arange(50, 150.1, 0.5)
    rng = np.random.default_rng(2)
    true = 70 + 10 * np.sin(np.arange(60) / 10.0)
    spec = rng.uniform(0, 1, (60, grid.size))
    for i, h in enumerate(true):
        spec[i] += 3 * np.exp(-0.5 * ((grid - h) / 2.0) ** 2)
    spec[10:14] = rng.uniform(0, 1, (4, grid.size))          # four windows with nothing
    path = moving_hr.viterbi(np.log(spec), grid, 1.0)
    assert np.mean(np.abs(grid[path] - true)) < 3.0


# --- the whole pipeline on synthetic captures ---------------------------------------------

@pytest.fixture(scope="module")
def rest():
    return run("rest", hr=lambda t: 70 + 10 * np.sin(2 * np.pi * t / 60.0))


def test_heart_rate_at_rest(rest):
    person, res = rest
    assert (res.motion == 0).all()
    assert res.coverage() > 0.9
    err = np.abs(res.hr_bpm - person.true_hr(res.win_t))
    assert err.mean() < 3.0
    assert displacement_error_mm(person, res) < 0.2
    assert np.nanmedian(res.br_bpm) == pytest.approx(15.0, abs=2.0)


def test_heart_rate_while_swaying():
    person, res = run("sway")
    assert (res.motion == 1).mean() > 0.8
    assert np.abs(res.hr_bpm - person.true_hr(res.win_t)).mean() < 3.0


def test_steps_are_found_and_notched_while_walking_in_place():
    person, res = run("walk_in_place", hr=72.0)
    assert np.nanmedian(res.cadence_hz) == pytest.approx(person.cadence, abs=0.05)
    assert displacement_error_mm(person, res) < 0.2
    assert np.abs(res.hr_bpm - 72.0)[res.valid].mean() < 5.0


def test_walking_is_tracked_but_gives_no_estimate_by_default():
    person, res = run("walk", hr=72.0)
    assert res.range_m.max() - res.range_m.min() > 1.5       # 2 m there and back
    assert displacement_error_mm(person, res) < 0.5
    assert (res.motion == 2).all() and not res.valid.any()
    _, res2 = run("walk", hr=72.0, allow_strong=True)
    assert res2.valid.mean() > 0.8
    assert np.abs(res2.hr_bpm - 72.0)[res2.valid].mean() < 5.0


def test_dropouts_do_not_leave_steps():
    person = hr_synth.Person(r0=1.2, hr_bpm=72.0, motion="rest", duration_s=40.0, seed=4)
    cube, p = hr_synth.capture(person, duration_s=40.0, fps=FPS, seed=4)
    for a in (500, 1300, 2100, 2900):
        cube[a:a + 12] = 0                                    # 120 ms lost, four times
    res = moving_hr.analyse(cube, p, fps=FPS)
    assert res.damaged.sum() == 48
    assert displacement_error_mm(person, res) < 0.3
    assert np.abs(res.hr_bpm - 72.0).mean() < 3.0


# --- reporting -------------------------------------------------------------------------------

def test_comparison_with_a_reference(rest):
    person, res = rest
    t = np.arange(0, 45, 0.5)
    stats = moving_hr.compare_with_reference(res, t, person.true_hr(t))
    assert stats["all"]["mae"] < 3.0 and stats["rest"]["n"] == stats["all"]["n"]
    lo, hi = stats["all"]["loa"]
    assert lo < stats["all"]["bias"] < hi
    assert stats["light"]["n"] == 0
    shifted = moving_hr.compare_with_reference(res, t, person.true_hr(t) + 10)
    assert shifted["all"]["bias"] == pytest.approx(stats["all"]["bias"] - 10, abs=0.5)


def test_window_table_and_description(rest):
    _, res = rest
    arr, header = res.window_table()
    assert arr.shape == (res.win_t.size, 9)
    assert header.split(",")[0] == "time_s" and "hr_bpm" in header
    assert "coverage" in moving_hr.describe(res)


def test_the_tool_reads_a_csv_reference(tmp_path):
    from tools import heart_rate
    f = tmp_path / "ecg.csv"
    f.write_text("time_s,hr_bpm\n0,70\n1,71\nbad,x\n2,72\n", encoding="utf-8")
    t, v = heart_rate.load_csv_reference(str(f))
    assert t.tolist() == [0, 1, 2] and v.tolist() == [70, 71, 72]
    g = tmp_path / "wrong.csv"
    g.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        heart_rate.load_csv_reference(str(g))


def test_a_still_trunk_moves_onto_the_bin_that_breathes():
    """
    The moving energy peaks on the strongest moving reflection, which for a seated person
    was a bin in front of the chest; at rest the path moves onto the neighbour that
    breathes most. A walking path is left alone.
    """
    fps, F, RX, G = 20.0, 1200, 4, 12
    t = np.arange(F) / fps
    rng = np.random.default_rng(0)
    xbar = 0.01 * (rng.standard_normal((F, RX, G)) + 1j * rng.standard_normal((F, RX, G)))
    xbar[:, :, 6] += np.exp(1j * 2.0 * np.sin(2 * np.pi * 0.25 * t))[:, None]      # the chest
    r = np.arange(G) * 0.05
    still = np.full(F, 5)
    assert np.all(moving_hr._rest_anchor(still, xbar, fps, r, np.zeros(F, bool)) == 6)
    walking = np.linspace(2, 9, F).round().astype(int)
    assert np.array_equal(moving_hr._rest_anchor(walking, xbar, fps, r, np.zeros(F, bool)), walking)


def test_the_heart_window_widens_while_the_person_is_still():
    """15 s windows at rest become 30 s about the same centre (shifted inside the
    recording at its ends); where anything within those 30 s moves, they stay."""
    fps, step = 10.0, 1.0
    wins = moving_hr._windows(int(90 * fps), fps, 15.0, step)
    motion = np.zeros(len(wins), int)
    motion[50:] = 1                                          # light motion from 57.5 s on
    out = moving_hr.rest_windows(wins, motion, 15.0, fps)
    full = int(moving_hr.REST_WIN_S * fps)
    mid = 20
    c = (wins[mid][0] + wins[mid][1]) // 2
    assert out[mid] == (c - full // 2, c - full // 2 + full)
    assert out[0] == (0, full)                               # the start: shifted inside
    assert all(out[i] == wins[i] for i in range(50, len(wins)))
    assert all(b - a == full for a, b in out[:28])
    assert out[28] == wins[28]                               # its 30 s reach the motion at 50 s
    assert moving_hr.rest_windows(wins[:5], motion[:5], 30.0, fps) == wins[:5]
    short = moving_hr._windows(int(20 * fps), fps, 15.0, step)
    assert moving_hr.rest_windows(short, np.zeros(len(short), int), 15.0, fps) == short


def test_the_reference_is_averaged_over_the_window_the_rate_is_from(rest):
    """A rate from a 30 s window is compared with the reference over those 30 s."""
    person, res = rest
    assert res.heart_win_s.shape == (res.win_t.size, 2)
    assert np.all(res.heart_win_s[:, 0] <= res.win_t) and np.all(res.heart_win_s[:, 1] >= res.win_t)
    assert np.all(np.diff(res.heart_win_s, axis=1) > 25)      # at rest: all of them 30 s
    t = np.arange(0, 45, 0.5)
    ramp = 60 + t                                            # the reference rises 1 bpm/s
    stats = moving_hr.compare_with_reference(res, t, ramp)
    e = [res.hr_bpm[i] - ramp[(t >= res.heart_win_s[i, 0]) & (t <= res.heart_win_s[i, 1])].mean()
         for i in np.flatnonzero(res.valid)]
    assert stats["all"]["n"] == len(e)
    assert stats["all"]["mae"] == pytest.approx(np.mean(np.abs(e)))


@pytest.mark.parametrize("hr", [60.0, 75.0])
def test_a_heart_on_a_breathing_harmonic_is_not_lost(hr):
    """
    Breathing at 15 /min puts its 4th and 5th harmonics on 60 and 75 bpm. Notched in the
    envelope too, where breathing leaves no harmonics, the heart there was lost and the
    track went to 2f (120 instead of 60 bpm).
    """
    person, res = run("rest", hr=hr, duration=45.0)
    ok = res.valid
    assert ok.mean() > 0.8
    assert np.median(np.abs(res.hr_bpm[ok] - person.true_hr(res.win_t[ok]))) < 3.0


def test_at_rest_the_heart_comes_from_every_receiver_of_the_trunk_bins():
    """The chest keeps the trunk's three bins of every receiver (chirps averaged); at rest
    their centred displacements, averaged, are what the heart is followed in."""
    person = hr_synth.Person(r0=1.2, hr_bpm=72.0, motion="rest", duration_s=30.0, seed=4)
    cube, p = hr_synth.capture(person, duration_s=30.0, fps=FPS, seed=4)
    chest = moving_hr.chest_signal(cube, p, fps=FPS)
    F = chest.z.shape[0]
    assert chest.bins.shape == (F, cube.shape[2], 3)
    d = moving_hr.rest_heart(chest, F)
    t = chest.t[:, 0]
    truth = person.chest_range(t)
    assert np.corrcoef(d, truth - truth.mean())[0, 1] > 0.95
    assert moving_hr.rest_heart(moving_hr.ChestSignal(chest.z, chest.t, FPS, chest.range_m, chest.damaged,
                                                      chest.lam_m, chest.noise), F) is None


# --- opt-in options of the T-MTT case study (defaults unchanged) -------------------------------

def test_breathing_harmonic_orders():
    assert moving_hr.breath_harmonic_orders(8.0, 3.0) == [2, 3, 4]
    assert moving_hr.breath_harmonic_orders(8.0, 3.0, "2-4") == [2, 3, 4]
    assert moving_hr.breath_harmonic_orders(8.0, 3.0, "comb") == list(range(2, 23))   # 3 Hz = 180 /min
    assert moving_hr.breath_harmonic_orders(float("nan"), 3.0, "comb") == []
    assert moving_hr.breath_harmonic_orders(8.0, 3.0, (5, 6)) == [5, 6]


def test_edge_index_counts_resolutions_to_the_nearer_band_edge():
    q_edge, q_band, q_harm = moving_hr.edge_index([56.0, 100.0, 144.0], (50.0, 150.0), 20.0, 30.0)
    assert q_band == pytest.approx([3.0, 25.0, 3.0])                 # 2 bpm per resolution in 30 s
    assert np.array_equal(q_edge, q_band)                             # harmonics off by default
    q_edge, _, q_harm = moving_hr.edge_index([81.0], (50.0, 150.0), 20.0, 30.0, harmonics="comb")
    assert q_harm[0] == pytest.approx(0.5) and q_edge[0] == pytest.approx(0.5)   # 4 x 20 = 80


def test_prewhitening_lifts_the_heart_over_the_breathing_tail():
    """Breathing 40 times the heart's amplitude: differentiated twice, the heart line wins
    the harmonic sum; prewhiten=0 is the default spectrum."""
    fs, dur = 100.0, 30.0
    t = np.arange(int(fs * dur)) / fs
    d = 4e-3 * np.sin(2 * np.pi * 0.12 * t) + 1e-4 * np.sin(2 * np.pi * 1.3 * t)
    bands = vital_bands.PRESETS["Adult"]
    wins = [(0, t.size)]
    g0, s0 = moving_hr.heart_spectra(d, fs, bands, wins, method="displacement")
    g1, s1 = moving_hr.heart_spectra(d, fs, bands, wins, method="displacement", prewhiten=0)
    assert np.allclose(s0, s1)
    g2, s2 = moving_hr.heart_spectra(d, fs, bands, wins, method="displacement", prewhiten=2)
    assert abs(g2[int(np.argmax(s2[0]))] - 78.0) <= 2.0
