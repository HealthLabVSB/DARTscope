"""
Tests for the Fastcheck (fastcheck.py): a synthetic recording - two walls, a breathing
chest, receiver noise - checked clean and then with each fault the tab is there to find.

Run with:  python -m pytest test_fastcheck.py -q
"""
import numpy as np
import pytest

import fastcheck

FPS = 20.0
PARAMS = {"digOutSampleRate_MHz": 10.0, "freqSlopeMHz_us": 60.0, "dataFmt_adcFmt": "Complex1X",
          "dataFmt_adcBits": 16, "effectiveStartFreqGHz": 60.25}
CHEST_BIN = 5                       # 128-point FFT: 0.195 m a bin, the chest at 0.98 m
BREATHS_HZ = 0.25                   # 15 /min


def make_cube(frames=900, chirps=4, rx=4, samples=128, breathing=True, seed=0, chest=None):
    """chest: the chest displacement per frame in metres (default 4 mm at 15 /min)."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / FPS
    s = np.arange(samples)
    lam = 3e8 / 60.25e9
    x = np.zeros((frames, chirps, rx, samples), np.complex128)
    for k, a in ((12, 0.05), (30, 0.02)):                     # two walls
        ph = rng.uniform(0, 2 * np.pi, rx)
        x += a * np.exp(1j * (2 * np.pi * k * s / samples))[None, None, None, :] \
            * np.exp(1j * ph)[None, None, :, None]
    if breathing:                                             # 4 mm of chest motion
        if chest is None:
            chest = 0.004 * np.sin(2 * np.pi * BREATHS_HZ * t)
        phi = 4 * np.pi * np.asarray(chest) / lam
        ph = rng.uniform(0, 2 * np.pi, rx)
        x += 0.03 * np.exp(1j * (2 * np.pi * CHEST_BIN * s / samples))[None, None, None, :] \
            * np.exp(1j * phi)[:, None, None, None] * np.exp(1j * ph)[None, None, :, None]
    x += (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) * 1e-3
    return x.astype(np.complex64)


def check(cube, **kw):
    return fastcheck.run("synthetic", PARAMS, cube, fps=FPS, **kw)


def finding(chk, name):
    found = [f for f in chk.findings if f.check == name]
    assert found, f"no '{name}' finding in {[f.check for f in chk.findings]}"
    return found[0]


@pytest.fixture(scope="module")
def clean():
    return check(make_cube())


# --- a clean recording -------------------------------------------------------------------

def test_clean_recording_has_no_integrity_problem(clean):
    assert not clean.damaged.any() and not clean.glitch.any()
    assert clean.clip_frac.max() == 0.0
    for name in ("zero-filled frames", "clipping", "frame glitches", "receivers over time"):
        assert finding(clean, name).severity == "ok", finding(clean, name)
    assert "complete frames 100.0 %" in clean.key_numbers()
    assert clean.rti_db.shape == (900, 65) and clean.range_m.size == 65


def test_the_breathing_chest_is_found(clean):
    assert clean.best_bin is not None
    assert abs(clean.best_bin - CHEST_BIN) <= 1
    good = clean.clear_rhythm()
    assert good.mean() > 0.8
    assert np.nanmedian(clean.br_rate[good]) == pytest.approx(60 * BREATHS_HZ, abs=1.0)
    f = finding(clean, "vital signs")
    assert "breathing at" in f.message and "15 /min" in f.message
    # the walls hold still: next to nothing in the breathing band there
    assert clean.br_ratio[12] < clean.br_ratio[clean.best_bin] / 100


def test_uneven_breathing_is_counted_not_lost():
    """
    9 breaths in a minute, 4-9 s apart (as the person on the cascade's validation take
    breathed): the spectrum of a 20 s window holds no clear peak, and Fastcheck found a
    rhythm in 2 of 21 windows and none for the recording. The breaths counted one by
    one give the rate and the rhythm.
    """
    from test_vital_signs import UNEVEN_AT, uneven_breathing
    _, x = uneven_breathing(FPS)
    chk = check(make_cube(frames=x.size, chest=x))
    assert chk.best_bin is not None and abs(chk.best_bin - CHEST_BIN) <= 1
    assert chk.notes["breaths"]["n"] == len(UNEVEN_AT)
    assert chk.clear_rhythm().mean() > 0.7
    msg = finding(chk, "vital signs").message
    assert "about 9 /min" in msg and "9 breaths counted in 60 s, irregular" in msg, msg


def test_no_breathing_is_said_so():
    chk = check(make_cube(breathing=False))
    msg = finding(chk, "vital signs").message
    assert "no slow motion" in msg or "no clear breathing" in msg, msg


def test_a_short_recording_is_not_judged_for_vital_signs():
    chk = check(make_cube(frames=40))
    assert chk.best_bin is None and chk.win_t.size == 0
    assert finding(chk, "vital signs").message.startswith("not judged")


# --- faults ------------------------------------------------------------------------------

def test_zero_filled_packets():
    cube = make_cube()
    cube[100:105] = 0                                        # a whole dropout ...
    cube[300].reshape(-1)[500:1300] = 0                      # ... and one lost packet
    chk = check(cube)
    assert np.flatnonzero(chk.damaged).tolist() == [100, 101, 102, 103, 104, 300]
    assert fastcheck.damaged_spans(chk.damaged) == [(100, 105), (300, 301)]
    assert fastcheck.longest_clean(chk.damaged) == 599       # frames 301-899
    f = finding(chk, "zero-filled frames")
    assert f.severity == "warning" and "2 dropout(s)" in f.message
    assert "dropouts 2" in chk.key_numbers()
    # the dropouts do not read as glitches, and the breathing survives them
    assert not chk.glitch.any()
    assert chk.best_bin is not None and abs(chk.best_bin - CHEST_BIN) <= 1


def test_breathing_survives_many_dropouts():
    """A recording that lost packets 40 times a minute (test_D lost 74 in 60 s): the
    phase is stitched across the gaps, so the rate stays and so does the rhythm."""
    cube = make_cube()
    rng = np.random.default_rng(7)
    for a in rng.choice(np.arange(20, 880), 30, replace=False):
        cube[a:a + int(rng.integers(1, 4))] = 0
    chk = check(cube)
    assert chk.damaged.sum() >= 30
    good = chk.clear_rhythm()
    assert good.mean() > 0.7
    assert np.nanmedian(chk.br_rate[good]) == pytest.approx(60 * BREATHS_HZ, abs=1.0)
    assert np.isfinite(chk.br_db).all() and np.nanmedian(chk.br_db) > 10


def test_the_heart_rate_of_a_seated_person():
    """A synthetic chest (hr_synth) breathing 15 /min with a heart at 72 bpm."""
    import hr_synth
    person = hr_synth.Person(r0=1.2, hr_bpm=72.0, motion="rest", duration_s=60.0, seed=2)
    cube, p = hr_synth.capture(person, duration_s=60.0, fps=100.0, seed=2)
    chk = fastcheck.run("synthetic heart", p, cube, fps=100.0)
    assert chk.hr_rate.size == chk.win_t.size and chk.hr_valid.mean() > 0.8
    assert np.median(chk.hr_rate[chk.hr_valid]) == pytest.approx(72.0, abs=3.0)
    assert "heart about 72 bpm" in finding(chk, "vital signs").message


def test_a_mostly_lost_recording_is_an_error():
    cube = make_cube()
    cube[::3] = 0
    assert finding(check(cube), "zero-filled frames").severity == "error"


def test_clipping():
    cube = make_cube()
    cube[50].reshape(-1)[:40] = 1.0 + 0j
    chk = check(cube)
    assert chk.clip_frac[50] > 0 and chk.clip_frac[49] == 0
    f = finding(chk, "clipping")
    assert f.severity == "warning" and "2.5 s" in f.message


def test_a_receiver_falling_silent():
    cube = make_cube()
    cube[600:, :, 2, :] *= 1e-4
    f = finding(check(cube), "receivers over time")
    assert f.severity == "warning" and "RX3" in f.message and "30.0 s" in f.message


def test_an_interference_burst_is_a_glitch():
    cube = make_cube()
    rng = np.random.default_rng(1)
    cube[400] += (0.05 * (rng.normal(size=cube[400].shape)
                          + 1j * rng.normal(size=cube[400].shape))).astype(np.complex64)
    chk = check(cube)
    assert chk.glitch[400]
    assert chk.glitch.sum() <= 2                             # the burst and the way back
    f = finding(chk, "frame glitches")
    assert f.severity == "warning" and "20.0 s" in f.message


def test_the_recorder_and_the_data_disagree():
    params = dict(PARAMS, dca_packets_zero_filled=0, dca_packets_received=100)
    cube = make_cube()
    cube[10:12] = 0
    chk = fastcheck.run("synthetic", params, cube, fps=FPS)
    assert finding(chk, "recorder vs data").severity == "warning"


# --- the rest ------------------------------------------------------------------------------

def test_real_samples():
    cube = make_cube().real.copy()
    chk = fastcheck.run("real", dict(PARAMS, dataFmt_adcFmt="Real"), cube, fps=FPS)
    assert chk.rti_db.shape == (900, 65)
    assert finding(chk, "zero-filled frames").severity == "ok"


def test_report_and_frame_table(clean):
    text = clean.report()
    assert text.splitlines()[1] == clean.verdict()
    assert clean.key_numbers() in text
    data, header = clean.frame_table()
    assert data.shape == (900, 6 + 4)
    assert header.split(",")[:3] == ["time_s", "zero_filled", "zero_fraction"]
    assert header.endswith("rx4_dbfs")


def test_progress_and_cancel():
    seen = []
    check(make_cube(frames=60), progress_cb=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1] == (60, 60)

    class Stop(Exception):
        pass

    def stop(done, total):
        raise Stop()
    with pytest.raises(Stop):
        check(make_cube(frames=60), progress_cb=stop)


def test_spans_and_clean_stretch():
    assert fastcheck.damaged_spans(np.zeros(5, bool)) == []
    assert fastcheck.longest_clean(np.zeros(5, bool)) == 5
    assert fastcheck.longest_clean(np.ones(5, bool)) == 0
    m = np.array([1, 0, 0, 1, 1, 0, 0, 0], bool)
    assert fastcheck.damaged_spans(m) == [(0, 1), (3, 5)]
    assert fastcheck.longest_clean(m) == 3


def test_rejects_a_cube_of_the_wrong_shape():
    with pytest.raises(ValueError):
        fastcheck.run("x", PARAMS, np.zeros((10, 4, 64)), fps=FPS)
