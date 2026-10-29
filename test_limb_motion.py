"""
Tests for limb_motion - swings of legs and arms and their maximum speed (roadmap phase 10).

The captures are synthetic (limb_synth.py): a torso and limbs made of many point
scatterers, turned into raw ADC samples, so the whole chain from the range FFT on is
exercised. Positive velocity is away from the radar.

Run with:  python -m pytest test_limb_motion.py -q
"""
import contextlib
import io

import numpy as np
import pytest

import antenna_config as ac
import capture_params
import clutter
import limb_motion as lm
import limb_synth as ls
import radar_processing as rp


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


LEG = dict(t_start=0.4, duration=0.4, v_peak=3.0, amp=0.3)       # tip 3 m/s faster than the torso
ARM = dict(t_start=1.1, duration=0.3, v_peak=-2.0, amp=0.2)      # tip 2 m/s towards the radar


def _walker_params(**kw):
    """1 TX, 50 us chirps, frames back to back: v_max 24.9 m/s, continuous slow time."""
    return ls.params(ntx=1, loops=32, idle_us=10, ramp_us=40, n_samples=64, **kw)


@pytest.fixture(scope="module")
def walker():
    """A person walking away at 1 m/s who swings a leg forward, then an arm back."""
    p = _walker_params()
    cube = _quiet(ls.capture, ls.Body(r0=2.0, v_torso=1.0, swings=[LEG, ARM]), p, 1.6)
    return p, cube


@pytest.fixture(scope="module")
def walker_result(walker):
    p, cube = walker
    return _quiet(lm.analyse, cube, p)


def _one(swings, direction):
    found = [s for s in swings if s.direction == direction]
    assert len(found) == 1, swings
    return found[0]


# --- time axes and the shared slow time ----------------------------------------------------

def test_chirp_times_and_the_slow_time_axis():
    p = ls.params(ntx=3, loops=4, idle_us=10, ramp_us=40, frame_ms=5.0)
    t = capture_params.chirp_times(p, 2, 12)
    assert t.shape == (2, 12)
    assert t[0, 1] - t[0, 0] == pytest.approx(50e-6)
    assert t[1, 0] == pytest.approx(5e-3)
    ax = capture_params.slow_time_axis(p, 2, 4, 150e-6, offset_s=50e-6, t0_s=10.0)
    assert ax[1, 3] == pytest.approx(10.0 + 5e-3 + 50e-6 + 3 * 150e-6)
    # a cube read with only the second transmitter's chirps (Chirp: TX:TX2)
    pf = dict(p, cube_chirp_filtered=True, rd_selected_positions=[1], rd_pattern_len_original=3)
    assert capture_params.chirp_times(pf, 1, 4)[0] == pytest.approx(np.array([1, 4, 7, 10]) * 50e-6)


def test_frames_without_a_period_are_taken_back_to_back():
    p = ls.params(ntx=1, loops=8)
    p['framePeriodicity_ms_selected'] = None
    p['frameConfig'] = {**p['frameConfig'], 'framePeriodicity_ms': None}
    ax = _quiet(capture_params.slow_time_axis, p, 3, 8, 50e-6)
    assert np.allclose(np.diff(ax.reshape(-1)), 50e-6)


def test_the_spectrogram_has_the_velocity_axis_of_the_range_doppler_map():
    p = ls.params(ntx=3, loops=16, idle_us=100, ramp_us=40, n_samples=64)
    rng = np.random.default_rng(0)
    cube = (rng.standard_normal((4, 48, 4, 64)) + 1j * rng.standard_normal((4, 48, 4, 64))
            ).astype(np.complex64)
    _, meta = _quiet(rp.compute_range_doppler_map, cube, p, n_fft_range=64)
    x, interval, info = rp.doppler_slow_time(np.fft.fft(cube, axis=-1), p)
    assert interval == pytest.approx(meta['chirp_interval_tdm_s'])
    assert info['positions'] == [0] and info['pattern_len'] == 3 and x.shape[-1] == 16
    spec = _quiet(lm.spectrogram, cube, p, gate="fixed", bins=(2, 20))
    assert spec.v_max == pytest.approx(-meta['velocity_axis'][0])


@pytest.mark.parametrize("frame_ms, loops, layout", [(None, 32, "continuous"),
                                                     (40.0, 512, "frame"),
                                                     (20.0, 32, "per-frame")])
def test_the_layout_follows_the_frame_timing(frame_ms, loops, layout):
    p = ls.params(ntx=1, loops=loops, idle_us=10, ramp_us=40, n_samples=32, frame_ms=frame_ms)
    rng = np.random.default_rng(1)
    shape = (6, loops, 2, 32)
    cube = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    spec = _quiet(lm.spectrogram, cube, p, gate="fixed", bins=(2, 10), t0_s=30.0)
    assert spec.layout == layout
    assert np.all(np.diff(spec.times_s) > 0) and spec.times_s[0] >= 30.0
    assert spec.power.shape == (len(spec.times_s), len(spec.velocity_m_s))
    assert bool(spec.meta["warnings"]) == (layout == "per-frame")


# --- swings and their speed ----------------------------------------------------------------

def test_the_leg_swing_and_its_maximum_speed(walker_result):
    sw = _one(walker_result.swings, "away")
    assert sw.t_peak_s == pytest.approx(0.6, abs=0.05)
    assert sw.v_peak_m_s == pytest.approx(4.0, abs=0.2)          # measured +0.1 m/s
    assert sw.v_rel_m_s == pytest.approx(3.0, abs=0.25)
    assert 0.1 < sw.duration_s < 0.45
    assert not sw.unfolded and not sw.saturated
    assert walker_result.summary["max_swing_speed_m_s"] == pytest.approx(4.0, abs=0.2)


def test_an_arm_swinging_back_through_zero_velocity(walker_result):
    """The arm's band crosses 0 m/s, where clutter suppression leaves a notch."""
    sw = _one(walker_result.swings, "towards")
    assert sw.t_peak_s == pytest.approx(1.25, abs=0.05)
    assert sw.v_peak_m_s == pytest.approx(-1.0, abs=0.2)
    assert walker_result.summary["max_swing_towards_m_s"] == pytest.approx(-1.0, abs=0.2)


def test_the_torso_line_and_a_quiet_envelope(walker_result):
    env = walker_result.env
    assert np.median(env.torso_m_s) == pytest.approx(1.0, abs=0.05)
    quiet = (env.times_s > 0.1) & (env.times_s < 0.35)
    assert np.nanmax(env.relative_up[quiet]) < 0.4             # the window's skirt, no more
    assert np.nanmax(env.relative_down[quiet]) < 0.4


def test_half_power_edges_are_less_biased_than_the_noise_threshold(walker):
    p, cube = walker
    spec = _quiet(lm.spectrogram, cube, p)
    k = int(np.argmin(np.abs(spec.times_s - 0.6)))
    half = lm.envelopes(spec, method="half-power").upper_m_s[k - 2:k + 3].max()
    thr = lm.envelopes(spec, method="threshold").upper_m_s[k - 2:k + 3].max()
    assert abs(half - 4.0) < 0.2 < thr - 4.0


def test_noise_alone_gives_no_swing():
    p = _walker_params()
    cube = _quiet(ls.capture, ls.Body(torso_amp=0.0), p, 1.2)
    res = _quiet(lm.analyse, cube, p)
    assert res.swings == []


def test_a_body_standing_still_is_measured_from_zero():
    """Clutter suppression removes the still torso; the swing is still relative to 0."""
    p = _walker_params()
    cube = _quiet(ls.capture, ls.Body(r0=2.0, v_torso=0.0, swings=[LEG]), p, 1.2)
    res = _quiet(lm.analyse, cube, p)
    sw = _one(res.swings, "away")
    assert sw.v_peak_m_s == pytest.approx(3.0, abs=0.2)
    assert abs(sw.torso_m_s) < 0.1


def test_a_swing_faster_than_v_max_is_unfolded():
    """3 TX x 140 us like the IWR6843 AoA captures: v_max 2.96 m/s, the foot reaches 4."""
    p = ls.params(ntx=3, loops=32, idle_us=100, ramp_us=40, n_samples=64)
    cube = _quiet(ls.capture, ls.Body(r0=2.0, v_torso=1.0, swings=[LEG]), p, 1.2)
    res = _quiet(lm.analyse, cube, p, heights=False)
    assert res.spec.v_max == pytest.approx(2.96, abs=0.02)
    sw = _one(res.swings, "away")
    assert sw.v_peak_m_s == pytest.approx(4.0, abs=0.25)
    assert sw.unfolded and not sw.saturated


def test_a_body_wider_than_the_velocity_axis_is_flagged():
    p = ls.params(ntx=3, loops=32, idle_us=100, ramp_us=40, n_samples=64)
    both = [dict(LEG, v_peak=3.5, amp=0.4), dict(LEG, v_peak=-3.5, amp=0.4)]
    cube = _quiet(ls.capture, ls.Body(r0=2.0, swings=both), p, 1.2)
    res = _quiet(lm.analyse, cube, p, heights=False)
    assert res.summary["saturated_fraction"] > 0.05
    # no number without the flag: a swing next to the saturated stretch says so
    assert all(s.saturated for s in res.swings)


@pytest.mark.parametrize("method", [m for m in clutter.METHODS])
def test_every_clutter_method_finds_the_leg_and_invents_nothing(walker, method):
    p, cube = walker
    q = dict(p, clutter_method=method,
             clutter_alpha=clutter.alpha_from_time_constant(5.0, 32 * 50e-6))
    res = _quiet(lm.analyse, cube, q, heights=False)
    sw = _one(res.swings, "away")
    assert sw.v_peak_m_s == pytest.approx(4.0, abs=0.25)
    assert all(abs(s.v_peak_m_s) < 5.0 for s in res.swings)


@pytest.mark.parametrize("method", ["mean", "mti3"])
def test_frames_with_gaps_give_one_spectrum_each(method):
    """MTI colours the noise (+12 dB at v_max for mti3): no false swings from it."""
    p = dict(ls.params(ntx=1, loops=128, idle_us=10, ramp_us=40, n_samples=64, frame_ms=20.0),
             clutter_method=method)
    cube = _quiet(ls.capture, ls.Body(r0=2.0, v_torso=1.0, swings=[LEG]), p, 1.2)
    res = _quiet(lm.analyse, cube, p, heights=False)
    assert res.spec.layout == "per-frame"
    sw = _one(res.swings, "away")
    assert sw.v_peak_m_s == pytest.approx(4.0, abs=0.35)       # 0.39 m/s resolution here
    assert all(abs(s.v_peak_m_s) < 5.0 for s in res.swings)


def test_the_height_of_the_fastest_point_tells_a_leg_from_an_arm():
    """IWR6843AOP, 3 TX x 4 RX: the foot low in front, the hand at the height of the radar."""
    p = ls.params(ntx=3, loops=32, idle_us=100, ramp_us=40, n_samples=64,
                  antenna=ac.AOP_DEFAULT)
    leg = dict(LEG, el_deg=-20.0)
    arm = dict(ARM, v_peak=-2.5, amp=0.3, el_deg=8.0)
    cube = _quiet(ls.capture, ls.Body(r0=2.0, v_torso=1.0, swings=[leg, arm]), p, 1.6)
    res = _quiet(lm.analyse, cube, p, radar_height_m=1.0)
    s_leg, s_arm = _one(res.swings, "away"), _one(res.swings, "towards")
    assert s_leg.elevation_deg == pytest.approx(-20.0, abs=3.0)
    assert s_arm.elevation_deg == pytest.approx(8.0, abs=3.0)
    assert (s_leg.limb, s_arm.limb) == ("leg", "arm")
    assert s_leg.height_m < 0.6 < s_arm.height_m


def test_no_height_without_a_vertical_array(walker_result):
    assert all(np.isnan(s.elevation_deg) and s.limb == "" for s in walker_result.swings)


# --- motion capture reference ---------------------------------------------------------------

def test_radial_velocity_of_a_point_passing_the_radar():
    t = np.arange(0.0, 2.0, 0.01)
    x0, y0, vx = -1.5, 2.0, 1.2
    xyz = np.column_stack([x0 + vx * t, np.full_like(t, y0), np.zeros_like(t)])
    d = np.hypot(x0 + vx * t, y0)
    expected = (x0 + vx * t) * vx / d
    v = lm.radial_velocity_from_positions(t, xyz, [0.0, 0.0, 0.0])
    assert np.max(np.abs(v - expected)[5:-5]) < 1e-3
    # uneven sampling falls back to np.gradient
    tu = np.sort(np.random.default_rng(2).uniform(0, 2, 300))
    xyzu = np.column_stack([x0 + vx * tu, np.full_like(tu, y0), np.zeros_like(tu)])
    vu = lm.radial_velocity_from_positions(tu, xyzu, [0, 0, 0])
    assert np.median(np.abs(vu - (x0 + vx * tu) * vx / np.hypot(x0 + vx * tu, y0))) < 5e-3


def test_swings_against_the_joint_tip_from_motion_capture(walker_result):
    t = np.arange(0.0, 1.6, 0.01)                                # a 100 Hz mocap stream
    tip = 2.0 + 1.0 * t + ls.swing_displacement(t, **{k: LEG[k] for k in ("t_start", "duration", "v_peak")}) \
        + ls.swing_displacement(t, **{k: ARM[k] for k in ("t_start", "duration", "v_peak")})
    ref_v = lm.radial_velocity_from_positions(t, np.column_stack([tip, 0 * t, 0 * t]), [0, 0, 0])
    cmp = lm.compare_with_reference(walker_result.swings, t, ref_v)
    assert cmp["n"] == 2
    assert cmp["mae_m_s"] < 0.2
    shifted = lm.compare_with_reference(walker_result.swings, t - 0.2, ref_v, offset_s=0.2)
    assert shifted["mae_m_s"] == pytest.approx(cmp["mae_m_s"], abs=1e-9)


# --- detection logic and exports ------------------------------------------------------------

def _envelope(times, upper, lower=None):
    n = len(times)
    lower = np.zeros(n) if lower is None else lower
    return lm.Envelopes(times_s=times, torso_m_s=np.zeros(n), upper_m_s=upper,
                        lower_m_s=lower, saturated=np.zeros(n, bool), floor=np.ones(n),
                        threshold_db=8.0, method="half-power", v_max=10.0)


def test_detection_hysteresis_merging_and_duration():
    t = np.arange(0.0, 4.0, 0.005)
    up = np.full_like(t, 0.2)                                     # the body's own width
    up += np.where((t > 0.5) & (t < 0.8), 3.0 * np.sin(np.pi * (t - 0.5) / 0.3) ** 2, 0.0)
    up[(t > 0.64) & (t < 0.655)] = 0.2                            # a fade in the middle
    up[(t > 2.0) & (t < 2.01)] = 3.0                              # a 10 ms blip
    up += np.where((t > 3.0) & (t < 3.9), 2.5, 0.0)               # held for 0.9 s
    sw = lm.detect_swings(_envelope(t, up), max_duration=0.8)
    assert len(sw) == 1                                           # not the blip, not the hold
    assert sw[0].t_peak_s == pytest.approx(0.65, abs=0.03)
    assert sw[0].v_peak_m_s == pytest.approx(3.2, abs=0.05)
    assert 0.15 < sw[0].duration_s < 0.3


def test_exports_and_report(walker_result):
    rows = lm.swing_rows(walker_result.swings)
    assert rows and list(rows[0]) == list(lm.SWING_FIELDS)
    arr, header = lm.envelope_table(walker_result.env)
    assert arr.shape == (len(walker_result.env.times_s), 5) and header.startswith("time_s,")
    img, extent = lm.display_image(walker_result.spec, max_columns=100)
    assert img.shape[0] == len(walker_result.spec.velocity_m_s) and img.shape[1] <= 101
    assert extent[0] < walker_result.spec.times_s[0] < walker_result.spec.times_s[-1] < extent[1]
    text = lm.describe(walker_result)
    assert "Max speed of swings" in text and "away" in text


# --- the command-line tool ------------------------------------------------------------------

def _tip(t):
    """Range of the synthetic walker's limb tips (only one limb swings at a time)."""
    parts = [ls.swing_displacement(t, s["t_start"], s["duration"], s["v_peak"]) for s in (LEG, ARM)]
    return 2.0 + 1.0 * t + parts[0] + parts[1]


def test_the_command_line_tool(tmp_path, monkeypatch, walker, capsys):
    import csv
    from tools import limb_speed

    p, cube = walker

    class Cap:
        stem, params = "synthetic", p

    monkeypatch.setattr(limb_speed, "load", lambda path: (Cap, cube))
    t = np.arange(0.0, 1.6, 0.01)
    ref = tmp_path / "mocap.csv"
    with open(ref, "w", encoding="utf-8") as f:            # long format, millimetres
        f.write("time_s,joint,x,y,z\n")
        for ti, ri in zip(t, _tip(t)):
            f.write(f"{ti:.3f},ankle,{ri * 1000:.3f},0,0\n{ti:.3f},head,0,0,1700\n")
    swings, env = tmp_path / "swings.csv", tmp_path / "env.csv"
    rc = limb_speed.main(["walk_Raw_0.bin", "--csv", str(swings), "--envelopes", str(env),
                          "--reference", str(ref), "--radar-pos", "0,0,0", "--joint", "ankle"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Max speed of swings" in out and "2 swings matched" in out
    with open(swings, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["direction"] for r in rows} == {"away", "towards"}
    assert np.loadtxt(env, delimiter=",", skiprows=1).shape[1] == 5
    # a segment: times stay in recording time
    assert limb_speed.main(["walk_Raw_0.bin", "--from", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "from 1.0 s" in out and "towards" in out and "   away" not in out


def test_reference_formats(tmp_path):
    import json
    from tools import limb_speed

    wide = tmp_path / "wide.csv"
    wide.write_text("time,wrist_x,wrist_y,wrist_z\n0.0,1000,0,0\n0.1,1100,0,0\n", encoding="utf-8")
    t, xyz = limb_speed.load_reference(str(wide), "wrist")
    assert t.tolist() == [0.0, 0.1] and xyz[1].tolist() == [1.1, 0.0, 0.0]
    js = tmp_path / "rec.json"                              # RemoteCaptury example recorder
    js.write_text(json.dumps({"poses": {"3": [
        {"timestamp": 2_000_000, "root_position": [0, 0, 500]},
        {"timestamp": 1_000_000, "root_position": [0, 0, 400]}]}}), encoding="utf-8")
    t, xyz = limb_speed.load_reference(str(js), time_scale=1e-6)
    assert t.tolist() == [1.0, 2.0] and xyz[:, 2].tolist() == [0.4, 0.5]
