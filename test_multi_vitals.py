"""
Tests for the Multi Vitals view (multi_vitals.py, multi_vitals_gui.py): synthetic rooms
(hr_synth.crowd_capture - people breathing and beating at known rates and positions,
seen by an IWR6843AOP with 3 TX, walls) have to come out with everyone in the right place
and their own rates; and the pieces - the sidelobe test, the tracker, the steering of a
chirp sent on two TX at once - each on their own.

Run with:  python -m pytest test_multi_vitals.py -q
"""
import numpy as np
import pytest

import hr_synth
import multi_vitals as mv

FPS = 20.0


def seated(*specs, seconds=60.0, seed=0):
    """specs: (x, y, breaths per min, bpm[, amplitude]) of seated people."""
    people = [hr_synth.Occupant(x=s[0], y=s[1], breath_hz=s[2] / 60.0, hr_bpm=s[3],
                                amp=(s[4] if len(s) > 4 else 1.0), seed=i + 1)
              for i, s in enumerate(specs)]
    return hr_synth.crowd_capture(people, duration_s=seconds, fps=FPS, seed=seed)


def where(p):
    return float(np.median(p.range_m)), float(np.median(p.angle_deg))


def match(res, truth):
    """For every true (x, y, br, hr), the reported person nearest to it (or None)."""
    out = []
    for x, y, br, hr, *_ in truth:
        best = min(res.people, key=lambda p: np.hypot(np.median(p.x_m) - x, np.median(p.y_m) - y),
                   default=None)
        out.append(best)
    return out


ROOM = [(-0.8, 1.2, 12.0, 62.0), (0.9, 1.6, 18.0, 80.0), (0.0, 2.6, 15.0, 70.0)]


@pytest.fixture(scope="module")
def room():
    return seated(*ROOM)


@pytest.mark.parametrize("method", ["motion", "breathing"])
@pytest.mark.parametrize("beam", ["bartlett", "capon"])
def test_three_seated_people_each_with_their_own_rates(room, method, beam):
    cube, p = room
    res = mv.analyse(cube, p, fps=FPS, method=method, beam=beam, r_max=4.0)
    real = [q for q in res.people if q.echo_of is None]
    assert len(real) == 3, res.summary()
    for q, (x, y, br, hr) in zip(match(res, ROOM), ROOM):
        r, a = where(q)
        assert r == pytest.approx(np.hypot(x, y), abs=0.15)
        assert a == pytest.approx(np.rad2deg(np.arctan2(x, y)), abs=4.0)
        got_br, got_hr = q.median_rates()
        assert got_br == pytest.approx(br, abs=1.0)
        assert got_hr == pytest.approx(hr, abs=3.0)
        assert not q.moving.any() and len(q.stays) == 1


def test_range_only_needs_no_angles(room):
    """People at different distances without angles; two closer than Min separation are one."""
    cube, p = room
    res = mv.analyse(cube, p, fps=FPS, method="range", r_max=4.0)
    assert res.angles_deg.size == 1
    ranges = sorted(round(where(q)[0], 1) for q in res.people)
    assert ranges == [1.4, 2.6]                       # 1.44 and 1.84 m are 0.4 m apart: one
    near = min(res.people, key=lambda q: where(q)[0])
    assert near.median_rates()[0] == pytest.approx(12.0, abs=1.0)


def test_capon_finds_the_weaker_of_two_at_one_range():
    """
    A person 12 dB weaker 30 deg beside another at the same range lies inside the main
    lobe of the AOP's four azimuth positions: the Bartlett map has one peak, Capon two -
    each with its own rates.
    """
    a = np.deg2rad(15)
    cube, p = seated((-1.8 * np.sin(a), 1.8 * np.cos(a), 12.0, 60.0, 4.0),
                     (1.8 * np.sin(a), 1.8 * np.cos(a), 20.0, 84.0))
    bart = mv.analyse(cube, p, fps=FPS, beam="bartlett", r_max=4.0)
    capon = mv.analyse(cube, p, fps=FPS, beam="capon", r_max=4.0)
    assert len([q for q in bart.people if q.echo_of is None]) == 1
    real = sorted([q for q in capon.people if q.echo_of is None], key=lambda q: where(q)[1])
    assert len(real) == 2, capon.summary()
    assert [round(q.median_rates()[0]) for q in real] == [12, 20]
    assert real[1].median_rates()[1] == pytest.approx(84.0, abs=3.0)


def test_a_person_who_walks_in_and_sits_down():
    def path(t):
        k = np.clip(np.asarray(t, float) / 4.0, 0, 1)        # 1.95 m in 4 s, then seated
        return 1.75 * (1 - k) + 0.5 * k, 3.0 * (1 - k) + 1.5 * k
    people = [hr_synth.Occupant(path=path, breath_hz=0.25, hr_bpm=75, seed=4),
              hr_synth.Occupant(x=-1.0, y=2.2, breath_hz=0.18, hr_bpm=58, seed=5)]
    cube, p = hr_synth.crowd_capture(people, duration_s=60, fps=FPS, seed=1)
    res = mv.analyse(cube, p, fps=FPS, method="motion", r_max=4.0)
    walker = min(res.people, key=lambda q: np.hypot(q.x_m[-1] - 0.5, q.y_m[-1] - 1.5))
    first = walker.blocks[0] * mv.BLOCK_S
    assert first < 3.0                                     # found while still walking
    assert walker.moving[:6].all() and not walker.moving[-20:].any()
    assert walker.stays and walker.stays[0][0] / FPS > 3.0  # measured once seated
    assert walker.median_rates()[0] == pytest.approx(15.0, abs=1.0)
    assert walker.median_rates()[1] == pytest.approx(75.0, abs=3.0)
    # on the map: moving while walking, rates once the windows hold the seated person
    b0 = int(walker.blocks[2])
    assert walker.at(b0, b0 * mv.BLOCK_S, mv.BLOCK_S)["moving"]
    late = walker.at(int(walker.blocks[-1]), 55.0, mv.BLOCK_S)
    assert late["br"] == pytest.approx(15.0, abs=1.0)


def test_a_reflection_is_marked_as_an_echo():
    """A wall behind a person reflects them: the same breathing, further away and weaker."""
    person = hr_synth.Occupant(x=0.3, y=1.5, breath_hz=0.25, hr_bpm=70, seed=1)
    mirror = hr_synth.Occupant(x=0.3, y=1.5, breath_hz=0.25, hr_bpm=70, seed=1, amp=0.5)
    # the same chest seen along a longer path from another direction
    mirror.position = lambda t: (np.full_like(np.asarray(t, float), -1.2),
                                 np.full_like(np.asarray(t, float), 2.8))
    cube, p = hr_synth.crowd_capture([person, mirror], duration_s=60, fps=FPS, seed=2)
    res = mv.analyse(cube, p, fps=FPS, r_max=4.0)
    assert len(res.people) == 2
    near, far = sorted(res.people, key=lambda q: where(q)[0])
    assert far.echo_of == near.id and near.echo_of is None
    assert far.echo_corr > 0.95
    assert "echo" in res.summary()


def test_a_sidelobe_is_not_a_second_person(room, monkeypatch):
    cube, p = room
    arr = mv.build_array(cube, p, fps=FPS, r_max=4.0)
    leak = mv.pattern(arr)
    A = arr.angles_deg.size
    snr = np.zeros((arr.range_m.size, A))
    j = int(np.argmin(np.abs(arr.angles_deg - 0)))
    snr[8] = 10 * np.log10(1e4 * leak[:, j] + 1)          # one target and its sidelobes
    dets = mv.detections(snr, arr.range_m, arr.angles_deg, 12.0, leak=leak)
    assert len(dets) == 1 and dets[0][1] == pytest.approx(0.0)
    assert len(mv.detections(snr, arr.range_m, arr.angles_deg, 12.0)) == 1  # they hardly stand out of their range
    monkeypatch.setattr(mv, "PROMINENCE_DB", -np.inf)
    assert len(mv.detections(snr, arr.range_m, arr.angles_deg, 12.0)) > 1   # without either test
    assert len(mv.detections(snr, arr.range_m, arr.angles_deg, 12.0, leak=leak)) == 1


def test_a_ring_around_a_walker_is_no_people():
    """Capon spreads a walker's motion evenly around their range ring 12 dB below them: its
    ripples are no peaks of their own."""
    ang = np.arange(-75.0, 75.5, 1.0)
    rng = np.arange(0, 4, 0.08)
    snr = np.zeros((rng.size, ang.size))
    r = int(np.argmin(np.abs(rng - 1.9)))
    snr[r] = 33 + 0.3 * np.sin(np.arange(ang.size) * 0.7)           # the ring, rippling
    snr[r, 70:75] = [40, 45, 46, 45, 40]                              # the walker
    snr[30, 2] = 40                                                   # a peak at -73 deg
    dets = mv.detections(snr, rng, ang, 12.0)
    assert len(dets) == 1 and dets[0][1] == pytest.approx(-3.0)


def test_a_track_broken_at_a_corner_is_joined():
    """A walker missed for a second continues as the same person."""
    dets = []
    for b in range(40):
        x = -1.0 + 0.05 * b
        dets.append([] if 15 <= b < 17 else [(np.hypot(x, 2.0), np.rad2deg(np.arctan2(x, 2.0)), 20.0)])
    # gaps longer than the tracker bridges, shorter than LINK_GAP_S
    for b in range(15, 18):
        dets[b] = []
    tracks = mv.track(dets)
    assert len(tracks) == 1 and tracks[0].id == 1
    assert len(tracks[0].rows) == 40


def test_two_people_who_never_meet_stay_two():
    dets = [[(1.5, -30.0, 20.0), (2.5, 30.0, 18.0)] for _ in range(30)]
    tracks = mv.track(dets)
    assert len(tracks) == 2
    assert {round(np.hypot(t.rows[-1][1], t.rows[-1][2]), 1) for t in tracks} == {1.5, 2.5}


def test_a_chirp_on_two_tx_is_the_sum_of_both():
    """Mask 5 (TX1 and TX3 at once, as the square/rectangle datasets send it)."""
    import antenna_config as ac
    p = {"antenna_config": ac.get_config(ac.AOP_DEFAULT)}
    both, slots, _, note = mv._steering(p, 4, [2, 5], mv.ANGLES_DEG)
    single, _, _, _ = mv._steering(p, 4, [2, 1, 4], mv.ANGLES_DEG)
    assert slots == [0, 1] and "several TX" in note
    # the same centre in both arrays shifts every element by one phase per angle: compare
    # the TX1 + TX3 slot with the sum of the TX1 and TX3 slots up to that common phase
    s = single[:, 4:8] + single[:, 8:12]
    ratio = both[:, 4:8] / s
    assert np.allclose(np.abs(ratio), 1.0, atol=1e-4)
    assert np.allclose(ratio / ratio[:, :1], 1.0, atol=1e-4)


def test_the_cascade_without_calibration_falls_back_to_range(room):
    cube, p = room
    q = dict(p, cascade=True, antenna_config=None, platform="", chip_hint="AWR2243")
    res = mv.analyse(cube, q, fps=FPS, method="motion", r_max=4.0)
    assert res.method == "range" and res.angles_deg.size == 1
    assert any("cascade" in n for n in res.notes)


def test_nobody_in_an_empty_room():
    cube, p = hr_synth.crowd_capture([], duration_s=30, fps=FPS, seed=3)
    for method in mv.METHODS:
        res = mv.analyse(cube, p, fps=FPS, method=method, r_max=4.0)
        assert res.people == [], (method, res.summary())
        assert res.summary().startswith("no person")


def test_the_tab_runs_draws_and_follows_the_frame(tk_root, room, monkeypatch):
    import tkinter as tk
    import multi_vitals_gui as g
    cube, p = room
    top = tk.Toplevel(tk_root)
    try:
        tab = g.MultiVitalsTab(top, lambda: (cube, p, FPS, 0.0, "room"))
        tab.run()
        assert tab.result is not None and "3 person" in tab.summary.cget("text")
        tab.show_frame(600)
        texts = [lab.get_text() for _, _, _, lab in tab._artists["dots"].values()]
        assert len(texts) == 3 and all("BR " in s and "HR " in s for s in texts)
        data, header = tab.positions_for_export()
        assert header.startswith("id,time_s") and data.shape[1] == header.count(",") + 1
        tab.show_echoes.set(True)
        tab.redraw()
    finally:
        top.destroy()


@pytest.mark.parametrize("speed, apart", [(1.0, 0.6), (1.4, 0.6), (0.6, 0.0)])
def test_two_walkers_crossing_keep_their_tracks(speed, apart):
    """
    Over 1.5 s of frames two people crossing at 1 m/s 0.6 m apart were smeared into
    three tracks jumping between them; the 0.5 s map places each where they are.
    """
    a = lambda t: (np.asarray(t, float) * speed - 2.0, np.full_like(np.asarray(t, float), 2.0))
    b = lambda t: (2.0 - np.asarray(t, float) * speed, np.full_like(np.asarray(t, float), 2.0 + apart))
    cube, p = hr_synth.crowd_capture([hr_synth.Occupant(path=a, seed=1), hr_synth.Occupant(path=b, seed=2)],
                                     duration_s=4.0 / speed, fps=FPS, seed=0)
    res = mv.analyse(cube, p, fps=FPS, r_max=4.0)
    assert len(res.people) == 2, res.summary()
    for q in res.people:                         # each crosses the whole way, in its own lane
        assert abs(q.x_m[-1] - q.x_m[0]) > 2.8
        assert np.ptp(q.y_m) < 0.3
        assert q.moving.mean() > 0.5


def test_a_track_that_jumps_onto_someone_else_is_cut():
    """
    A weak target 0.85 m behind a seated person jumped onto them in one block and took
    over their detections; cut at the jump, the seated person stays one track.
    """
    seat = (1.0, 0.0, 30.0)
    ghost = (1.76, 0.0, 15.0)
    dets = [[seat] for _ in range(40)]
    for b in range(10, 20):
        dets[b] = [seat, ghost]
    dets[20] = [(1.25, 0.0, 15.0)]                 # the ghost on its way, the person missed
    tracks = mv.track(dets)
    seated = [t for t in tracks if abs(np.median([np.hypot(r[1], r[2]) for r in t.rows]) - 1.0) < 0.1]
    assert len(seated) == 1
    assert seated[0].rows[0][0] == 0 and seated[0].rows[-1][0] == 39


def test_one_place_is_one_person():
    """
    Two tracks staying at the same spot one after the other are one person - a person
    who leaves for half a minute and sits down again, or a reflection that drifted onto
    a seated person and followed them after their own track ended.
    """
    def away(t):
        t = np.asarray(t, float)
        gone = (t > 20) & (t < 50)
        return np.where(gone, 5.5, 0.6), np.where(gone, 5.5, 1.5)
    people = [hr_synth.Occupant(path=away, breath_hz=0.25, hr_bpm=75, seed=3),
              hr_synth.Occupant(x=-1.0, y=2.2, breath_hz=0.18, hr_bpm=58, seed=4)]
    cube, p = hr_synth.crowd_capture(people, duration_s=80, fps=FPS, seed=1)
    res = mv.analyse(cube, p, fps=FPS, r_max=4.0)
    assert len(res.people) == 2, res.summary()
    back = min(res.people, key=lambda q: np.median(q.range_m))
    assert len(back.stays) == 2                       # before leaving and after coming back
    assert back.stays[0][1] / FPS <= 21 and back.stays[1][0] / FPS >= 49
    assert back.median_rates()[1] == pytest.approx(75.0, abs=3.0)


def test_two_people_near_the_cascade_are_two_with_exact_paths():
    """The cascade's 18.5 wavelengths: its far field begins at 2.7 m. With plane waves a
    person seated 1.1 m in front showed at four angles 35 deg apart and a second one at
    2 m, 12 dB weaker, was lost under their smear."""
    import imaging_synth as isy
    import antenna_config as ac
    p = isy.params(ntx=1, loops=8, n_samples=128, fs_msps=5, slope_mhz_us=100, f0_ghz=77.0,
                   antenna=ac.CASCADE_CONFIG, masks=[1, 2, 4, 8, 16, 32], idle_us=7, ramp_us=33,
                   frame_ms=20)

    def person(r, az, hz, ph=0.0):
        a = np.deg2rad(az)
        x, y = r * np.sin(a), r * np.cos(a)
        return isy.Point(x, y, 0.0, swing=(0.005 * x / r, 0.005 * y / r, 0), swing_hz=hz, swing_phase=ph)

    cube = isy.capture([person(1.11, 14, 0.25), person(2.0, -26, 0.2, 1.0)], p, frames=750, seed=3, noise=0.003)
    res = mv.analyse(cube, p, fps=50.0, method="motion")
    real = [q for q in res.people if q.echo_of is None]
    assert len(real) == 2 and "exact paths" in res.notes[0]
    got = sorted((float(np.median(q.range_m)), float(np.median(q.angle_deg))) for q in real)
    assert abs(got[0][0] - 1.11) < 0.1 and abs(got[0][1] - 14) < 3
    assert abs(got[1][0] - 2.0) < 0.1 and abs(got[1][1] + 26) < 3


def test_a_weaker_target_behind_a_person_is_their_echo():
    b = np.arange(60)
    mk = lambda i, x, y, snr: mv.Person(id=i, blocks=b, x_m=np.full(60, x), y_m=np.full(60, y),
                                        detected=np.ones(60, bool), moving=np.zeros(60, bool), snr_db=snr)
    person, behind, aside, other = mk(1, 0.0, 1.0, 40), mk(2, 0.3, 2.5, 25), mk(3, -1.2, 2.5, 25), mk(4, 1.5, 1.5, 38)
    mv._mark_ghosts([behind, person, aside, other])
    assert behind.echo_of == 1                      # 15 dB weaker, 1.5 m behind, 7 deg off
    assert aside.echo_of is None and other.echo_of is None   # 26 deg off; as strong
