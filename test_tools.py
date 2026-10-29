"""
Tests for the command-line tools in tools/.

Everything here builds its own files in a temporary directory: the recordings the tools
were written against are hundreds of megabytes and are not in the repository, so a test
that needed them would simply not run.

Run with:  python -m pytest test_tools.py -q
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from tools import reference
from tools.pair_reference import overlap_s, pair, report
from tools.vital_range_scan import scan
from dataclasses import replace

from tools.decode_check import CaptureCheck, compare_profiles, problems


# --- reading a reference export --------------------------------------------------

def write_series(directory, name, start="10.07.2026 14:31:41", fs=128, n=256, date="10.07.2026"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name} - {date}.txt")
    rows = "".join(f"{date} 14:31:{41 + i // fs:02d},000; {i}\n" for i in range(n))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Signal Type: {name}_Type\nStart Time: {start}\nSample Rate: {fs}\n"
                f"Length: {n}\nUnit: \n\nData:\n{rows}")
    return path


def test_a_signal_header_gives_its_span(tmp_path):
    p = write_series(str(tmp_path), "Pleth", fs=128, n=1280)
    s = reference.read_series(p)
    assert s is not None
    assert s.name == "Pleth" and s.fs == 128 and s.samples == 1280
    assert s.duration_s == pytest.approx(10.0)
    assert s.end == s.start + timedelta(seconds=10)


def test_a_file_without_a_header_is_skipped(tmp_path):
    p = tmp_path / "notes - 10.07.2026.txt"
    p.write_text("just some notes\n", encoding="utf-8")
    assert reference.read_series(str(p)) is None


def test_a_session_collects_its_signals(tmp_path):
    d = tmp_path / "Subject - 10.07.2026"
    write_series(str(d), "Pleth", fs=128, n=1280)
    write_series(str(d), "Pulse", fs=4, n=40)
    ses = reference.read_session(str(d))
    assert ses is not None and len(ses.series) == 2
    assert ses.starts_agree()
    assert ses.find("pleth").fs == 128


def test_the_battery_channel_does_not_widen_a_session(tmp_path):
    """accu starts before the recording proper and must not move the session start."""
    d = tmp_path / "Subject - 10.07.2026"
    write_series(str(d), "Pleth", start="10.07.2026 14:31:41", fs=128, n=1280)
    write_series(str(d), "accu", start="10.07.2026 14:31:30", fs=1, n=20)
    ses = reference.read_session(str(d))
    assert ses.starts_agree()                       # accu is excluded from the check
    assert ses.start == datetime(2026, 7, 10, 14, 31, 41)


def test_disagreeing_starts_are_reported(tmp_path):
    d = tmp_path / "Subject - 10.07.2026"
    write_series(str(d), "Pleth", start="10.07.2026 14:31:41", fs=128, n=1280)
    write_series(str(d), "Pulse", start="10.07.2026 14:31:55", fs=4, n=40)
    assert not reference.read_session(str(d)).starts_agree()


def test_only_the_covered_window_is_read(tmp_path):
    p = write_series(str(tmp_path), "Pleth", fs=10, n=100)
    s = reference.read_series(p)
    whole = reference.read_values(s)
    window = reference.read_values(s, start=s.start + timedelta(seconds=2), duration_s=3.0)
    assert len(whole) == 100 and len(window) == 30
    assert window[0] == whole[20]


# --- pairing a capture with a session --------------------------------------------

class FakeCapture:
    """Just the attributes the pairing reads."""
    def __init__(self, stem, start, duration_s, source="DCA1000 recorder log"):
        self.stem, self.start, self.duration_s = stem, start, duration_s
        self.time_source = source

    @property
    def end(self):
        return self.start + timedelta(seconds=self.duration_s)


def make_session(tmp_path, name, start, minutes):
    d = tmp_path / name
    write_series(str(d), "Pleth", start=start.strftime("%d.%m.%Y %H:%M:%S"),
                 fs=1, n=int(minutes * 60))
    return reference.read_session(str(d))


def test_a_capture_inside_a_session_gets_its_offset(tmp_path):
    ses = make_session(tmp_path, "S1", datetime(2026, 7, 10, 12, 47, 30), 20)
    cap = FakeCapture("test_D1_K2", datetime(2026, 7, 10, 12, 48, 4), 62)
    got = pair([cap], [ses])[0]
    assert got.session == "S1"
    assert got.offset_into_session_s == pytest.approx(34.0)
    assert got.covered_pct == pytest.approx(100.0)


def test_a_capture_before_the_session_is_left_unpaired(tmp_path):
    ses = make_session(tmp_path, "S1", datetime(2026, 7, 10, 12, 47, 30), 20)
    cap = FakeCapture("test_D_K", datetime(2026, 7, 10, 12, 43, 20), 62)
    got = pair([cap], [ses])[0]
    assert got.session is None and got.covered_pct == 0.0
    assert overlap_s(cap, ses) == 0.0


def test_partial_coverage_is_reported_as_such(tmp_path):
    ses = make_session(tmp_path, "S1", datetime(2026, 7, 10, 12, 0, 0), 1)
    cap = FakeCapture("half", datetime(2026, 7, 10, 12, 0, 30), 60)
    got = pair([cap], [ses])[0]
    assert 40.0 < got.covered_pct < 60.0
    assert "Partially covered" in report([got], [ses])


def test_the_session_with_the_most_overlap_wins(tmp_path):
    early = make_session(tmp_path, "early", datetime(2026, 7, 10, 12, 0, 0), 2)
    late = make_session(tmp_path, "late", datetime(2026, 7, 10, 12, 1, 30), 10)
    cap = FakeCapture("c", datetime(2026, 7, 10, 12, 1, 40), 60)
    assert pair([cap], [early, late])[0].session == "late"


def test_a_capture_dated_from_the_api_log_is_flagged(tmp_path):
    ses = make_session(tmp_path, "S1", datetime(2026, 7, 10, 12, 0, 0), 10)
    cap = FakeCapture("c", datetime(2026, 7, 10, 12, 1, 0), 60,
                      source="API log (first command)")
    assert "API log" in report(pair([cap], [ses]), [ses])


# --- scanning range bins for a person --------------------------------------------

def synthetic_slow(fps=20.0, seconds=90.0, n_bins=40, person_bin=12,
                   breaths_per_min=15.0, wavelength=0.005, seed=0):
    """
    A scene with one breathing target and one strong still reflector.

    The wall is given the larger amplitude on purpose: that is what makes a power- or
    stability-based score pick it.
    """
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.arange(n) / fps
    cube = (rng.normal(0, 0.02, (n, 1, n_bins))
            + 1j * rng.normal(0, 0.02, (n, 1, n_bins))).astype(np.complex128)
    chest = 0.004 * np.sin(2 * np.pi * (breaths_per_min / 60.0) * t)
    for b, spread in ((person_bin - 1, 0.5), (person_bin, 1.0), (person_bin + 1, 0.5)):
        cube[:, 0, b] += spread * np.exp(1j * 4 * np.pi * chest / wavelength)
    cube[:, 0, 30] += 4.0                       # a wall: bigger, and it never moves
    return cube, np.arange(n_bins) * 0.05, fps, wavelength


def test_the_scan_finds_the_breathing_target_not_the_wall():
    cube, range_m, fps, lam = synthetic_slow()
    rows = scan(cube, range_m, fps, lam, min_range=0.2, max_range=1.9)
    assert rows, "nothing passed the power floor"
    best = rows[0]
    assert abs(best.range_m - 0.60) < 0.08         # bin 12 at 0.05 m per bin
    assert abs(best.resp_per_min - 15.0) < 1.5
    assert best.neighbours_agreeing >= 1           # a torso spans several bins
    assert abs(rows[0].range_m - 1.50) > 0.2       # not the wall at bin 30


def test_the_wall_scores_below_the_person():
    cube, range_m, fps, lam = synthetic_slow()
    rows = {round(r.range_m, 2): r for r in
            scan(cube, range_m, fps, lam, min_range=0.2, max_range=1.9)}
    assert rows[1.50].score < rows[0.60].score


def test_a_still_reflector_with_a_faint_ripple_does_not_beat_an_uneven_breather():
    """
    The cascade validation take: a person breathing 9 times a minute, unevenly, spreads
    the breathing over the band (concentration 0.02-0.04), and a still reflector at
    2.15 m with a faint regular ripple scored 0.09 and won. It moved 0.004 mm.
    """
    from test_vital_signs import uneven_breathing
    from tools.vital_range_scan import MIN_BREATH_MM
    fps, lam = 20.0, 0.005
    _, chest = uneven_breathing(fps)
    n = chest.size
    rng = np.random.default_rng(3)
    cube = (rng.normal(0, 0.02, (n, 1, 40)) + 1j * rng.normal(0, 0.02, (n, 1, 40)))
    for b, spread in ((11, 0.5), (12, 1.0), (13, 0.5)):
        cube[:, 0, b] += spread * np.exp(1j * 4 * np.pi * chest / lam)
    ripple = 1e-5 * np.sin(2 * np.pi * 0.25 * np.arange(n) / fps)     # 0.01 mm at 15 /min
    cube[:, 0, 30] += 4.0 * np.exp(1j * 4 * np.pi * ripple / lam)
    rows = scan(cube, np.arange(40) * 0.05, fps, lam, min_range=0.2, max_range=1.9)
    by_range = {round(r.range_m, 2): r for r in rows}
    person, still = by_range[0.60], by_range[1.50]
    # scored on concentration and agreement alone, the reflector won
    assert (still.resp_concentration * (1 + still.neighbours_agreeing)
            > person.resp_concentration * (1 + person.neighbours_agreeing))
    assert abs(rows[0].range_m - 0.60) < 0.08
    assert rows[0].breaths == 9 and rows[0].counted_per_min == pytest.approx(8.9, abs=0.3)
    assert still.resp_displacement_mm < MIN_BREATH_MM and still.score == 0.0


def test_the_bin_that_moves_most_outscores_a_weaker_copy_of_it():
    """A reflection of the chest further out breathes the same breaths, 20 dB weaker."""
    fps, lam = 20.0, 0.005
    n = int(60 * fps)
    chest = 0.004 * np.sin(2 * np.pi * 0.25 * np.arange(n) / fps)
    rng = np.random.default_rng(4)
    cube = (rng.normal(0, 0.002, (n, 1, 40)) + 1j * rng.normal(0, 0.002, (n, 1, 40)))
    for b0, a in ((12, 1.0), (26, 0.1)):
        for b, spread in ((b0 - 1, 0.5), (b0, 1.0), (b0 + 1, 0.5)):
            cube[:, 0, b] += a * spread * np.exp(1j * 4 * np.pi * chest / lam)
    rows = {round(r.range_m, 2): r for r in
            scan(cube, np.arange(40) * 0.05, fps, lam, min_range=0.2, max_range=1.9,
                 power_floor_db=-40.0)}
    assert rows[1.30].motion_db == pytest.approx(-20.0, abs=1.0)
    assert rows[0.60].score > 5 * rows[1.30].score


def test_bins_below_the_power_floor_are_not_scanned():
    cube, range_m, fps, lam = synthetic_slow()
    strict = scan(cube, range_m, fps, lam, min_range=0.2, max_range=1.9,
                  power_floor_db=-1.0)
    assert len(strict) < len(scan(cube, range_m, fps, lam, 0.2, 1.9))


def test_an_empty_range_window_scans_nothing():
    cube, range_m, fps, lam = synthetic_slow()
    assert scan(cube, range_m, fps, lam, min_range=3.0, max_range=3.0) == []


# --- comparing captures of one scene ---------------------------------------------

def test_matching_profiles_correlate_and_a_broken_one_stands_out():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 129)
    profiles = {f"good{i}": base + rng.normal(0, 0.05, 129) for i in range(3)}
    lowest, worst = compare_profiles(profiles)
    assert lowest > 0.95

    profiles["broken"] = base[::-1]
    lowest, worst = compare_profiles(profiles)
    assert lowest < 0.5
    assert "broken" in worst[0][1:]


def test_a_single_profile_cannot_be_compared():
    lowest, worst = compare_profiles({"only": np.arange(129.0)})
    assert np.isnan(lowest) and worst == []


def test_a_real_container_is_not_flagged_for_its_file_size():
    """
    A REAL capture written into a complex-shaped container holds twice the int16 the
    declared geometry needs, and the decode halves it again. Flagging that as suspect
    would call every such recording broken; the frame count is what tells them apart.
    """
    ok = CaptureCheck(stem="Real_1lane", status="OK", size_ratio=2.0,
                      frames_declared=250, frames_decoded=249)
    assert problems(ok) == []

    # the fault before the container fix: 499 frames at 25 fps is 20 s, from 12 s
    wrong = replace(ok, frames_decoded=499, fps=25.0, recorded_s=12.0)
    assert len(problems(wrong)) == 1
    msg = problems(wrong)[0]
    assert "499 frames" in msg and "wrong shape" in msg and "2.00x" in msg


def test_a_broken_cube_is_reported():
    base = CaptureCheck(stem="x", status="OK", frames_declared=100, frames_decoded=100)
    assert problems(replace(base, non_finite=7))[0].startswith("7 non-finite")
    assert problems(replace(base, all_zero=True)) == ["the whole cube is zero"]
    assert "2 RX channel(s)" in problems(replace(base, dead_rx=2))[0]


# --- finding the raw data beside a log ------------------------------------------

@pytest.mark.parametrize("bin_name", ["cap_Raw_0.bin", "cap_0.bin", "cap.bin"])
def test_every_naming_of_the_raw_file_is_found(tmp_path, bin_name):
    """mmWave Studio writes cap_Raw_0.bin, the DCA1000 command line writes cap.bin."""
    from tools.captures import first_bin_for
    log = tmp_path / "cap_LogFile.txt"
    log.write_text("")
    (tmp_path / bin_name).write_bytes(b"\x00\x00")
    assert first_bin_for(str(log)) == str(tmp_path / bin_name)


def test_the_studio_name_wins_when_both_exist(tmp_path):
    from tools.captures import first_bin_for
    log = tmp_path / "cap_LogFile.txt"
    log.write_text("")
    for name in ("cap.bin", "cap_Raw_0.bin"):
        (tmp_path / name).write_bytes(b"\x00\x00")
    assert first_bin_for(str(log)).endswith("cap_Raw_0.bin")


def test_a_log_without_data_has_no_binary(tmp_path):
    from tools.captures import first_bin_for
    log = tmp_path / "cap_LogFile.txt"
    log.write_text("")
    (tmp_path / "cap_short.bin").write_bytes(b"\x00\x00")   # a user's trimmed copy
    assert first_bin_for(str(log)) is None

# --- the Vital Signs view against a somnograph (tools.vitals_check) -------------------

def synthetic_pleth(fs=128.0, dur=90.0, hr_bpm=72.0, resp=None, seed=0):
    """
    A finger PPG: one pulse per beat at hr_bpm, and modulations of its baseline, its pulse
    amplitude and its beat rate: resp = {"riiv": bpm, "riav": bpm, "rifv": bpm}.
    """
    rng = np.random.default_rng(seed)
    resp = resp or {}
    t = np.arange(int(dur * fs)) / fs

    def mod(name, depth):
        f = resp.get(name)
        return depth * np.sin(2 * np.pi * f / 60.0 * t) if f else 0.0 * t
    rate = hr_bpm + mod("rifv", 4.0)                         # bpm, beat to beat
    phase = np.cumsum(rate / 60.0) / fs
    pulse = np.maximum(np.sin(2 * np.pi * phase), 0) ** 3
    v = (1.0 + mod("riav", 0.25)) * pulse + mod("riiv", 0.6)
    return t, v + 0.01 * rng.standard_normal(t.size)


def test_ppg_beats_give_the_heart_rate():
    from tools.vitals_check import Reference, ppg_beats
    fs = 128.0
    t, v = synthetic_pleth(fs, hr_bpm=72.0)
    pk, amp = ppg_beats(v, fs)
    ref = Reference("s", 90.0, t[pk])
    assert ref.hr_beats(10, 80) == pytest.approx(72.0, abs=1.0)


def test_breathing_counts_only_when_the_three_surrogates_agree():
    """
    Breathing at 15 /min in the baseline, the pulse amplitude and the beat rate: the
    reference is 15. The Mayer wave at 6 /min in the baseline and the beat rate while the
    amplitude follows the chest at 17: no reference - two agreeing are not enough.
    """
    from tools.vitals_check import Reference, breathing_surrogates
    fs = 128.0
    t, v = synthetic_pleth(fs, resp={"riiv": 15, "riav": 15, "rifv": 15})
    ref = Reference("s", 90.0, np.zeros(0), resp=breathing_surrogates(v, fs))
    rate, rates = ref.breathing(10, 80)
    assert rate == pytest.approx(15.0, abs=1.0)
    t, v = synthetic_pleth(fs, resp={"riiv": 6, "riav": 17, "rifv": 6})
    ref = Reference("s", 90.0, np.zeros(0), resp=breathing_surrogates(v, fs))
    rate, rates = ref.breathing(10, 80)
    assert np.isnan(rate)
    assert rates["riiv"] == pytest.approx(6.0, abs=1.0) and rates["riav"] == pytest.approx(17.0, abs=1.0)


def test_windows_are_compared_over_their_own_span():
    from tools.vitals_check import Reference, compare
    ref = Reference("s", 60.0, np.arange(0, 60, 60 / 75.0))       # 75 bpm beats
    ref.pulse_t, ref.pulse = np.arange(0, 60, 0.25), np.full(240, 76.0)
    ref.resp = {}
    res = {"conf_th": 6.0, "hr_times": np.array([15.0, 30.0, 45.0]),
           "hr_values": np.array([75.0, 80.0, 70.0]), "hr_confs": np.array([8.0, 8.0, 2.0]),
           "rr_times": np.array([]), "rr_values": np.array([]), "rr_confs": np.array([])}
    out = compare(res, ref, win_br=30.0, win_hr=30.0)
    assert out["hr_pulse"]["n"] == 3
    assert out["hr_pulse"]["mae"] == pytest.approx((1 + 4 + 6) / 3)
    assert out["hr_beats"]["mae"] == pytest.approx((0 + 5 + 5) / 3, abs=0.2)
    assert out["hr_coverage"] == pytest.approx(2 / 3)
