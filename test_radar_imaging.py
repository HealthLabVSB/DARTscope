"""
Tests for the Imaging view (radar_imaging.py, radar_imaging_gui.py): synthetic scenes
(imaging_synth - point reflectors and walking bodies at known positions and velocities,
every chirp and every element's path exact) have to come out where they are, as fast as
they move: still reflectors, movers beyond one TX slot's velocity limit, two at one range
and speed, people sitting and breathing (micro-motion) and a wall's echo of one of them,
the cascade's 48 channels near and far, walkers crossing each other's path; and nothing
where nothing is.

Run with:  python -m pytest test_radar_imaging.py -q
"""
import io
import contextlib

import numpy as np
import pytest

import antenna_config as ac
import imaging_synth as isy
import radar_imaging as ri


def run(points, p, frames, **settings):
    cube = isy.capture(points, p, frames=frames, seed=3)
    with contextlib.redirect_stdout(io.StringIO()):
        return ri.analyse(cube, p, fps=1000.0 / p["framePeriodicity_ms_selected"],
                          settings=ri.Settings(**settings)), cube


def truth(pt, p, frame):
    """(r, v, az, el, x, y, z) of a Point in the middle of a frame's chirps."""
    tc = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    t = frame * p["framePeriodicity_ms_selected"] * 1e-3 + \
        (p["frameConfig"]["numLoops"] * p["uniqueChirps_effective"] - 1) / 2 * tc
    x, y, z = pt.x + pt.vx * t, pt.y + pt.vy * t, pt.z + pt.vz * t
    r = np.sqrt(x * x + y * y + z * z)
    return dict(r=r, v=(x * pt.vx + y * pt.vy + z * pt.vz) / r, az=np.rad2deg(np.arctan2(x, y)),
                el=np.rad2deg(np.arcsin(z / r)), x=x, y=y, z=z)


def nearest(q, t):
    d = np.sqrt((q["x_m"] - t["x"]) ** 2 + (q["y_m"] - t["y"]) ** 2 + (q["z_m"] - t["z"]) ** 2)
    k = int(np.argmin(d))
    return q[k], float(d[k])


AOP = dict(ntx=3, loops=64, n_samples=128, fs_msps=10, slope_mhz_us=60, idle_us=60, ramp_us=40)


def polar(r, az, el, **kw):
    a, e = np.deg2rad(az), np.deg2rad(el)
    return isy.Point(r * np.cos(e) * np.sin(a), r * np.cos(e) * np.cos(a), r * np.sin(e), **kw)


def test_still_reflectors_all_over_the_field_of_view():
    p = isy.params(ntx=3, loops=32, n_samples=128, fs_msps=10, slope_mhz_us=60)
    pts = [polar(r, a, e, amp=1.5) for r, a, e in
           ((1.2, -45, 10), (2.0, 0, 0), (2.7, 30, -20), (3.5, -15, 25), (4.2, 55, 5), (5.0, -60, -10))]
    res, _ = run(pts, p, 4, mode="still")
    for f in range(res.frames):
        q = res.frame(f)
        assert q.size == len(pts)                   # walls next to each other hide none
        for pt in pts:
            hit, d = nearest(q, truth(pt, p, f))
            assert d < 0.03
            assert hit["kind"] == ri.KIND_STILL


def test_movers_beyond_one_slots_velocity_limit_are_unfolded():
    p = isy.params(**AOP)                           # one slot's limit: 4.15 m/s
    pts = [isy.Point(-1.0, 2.0, 0.2, vy=0.5), isy.Point(0.5, 3.0, -0.3, vx=0.4, vy=-1.5),
           isy.Point(1.5, 2.5, 0.0, vy=5.5), isy.Point(-0.8, 5.2, 0.4, vx=0.3, vy=-6.5)]
    res, _ = run(pts, p, 6, mode="moving")
    for f in range(res.frames):
        q = res.frame(f)
        assert q.size == len(pts)
        for pt in pts:
            t = truth(pt, p, f)
            hit, d = nearest(q, t)
            assert d < 0.03 and abs(hit["velocity_m_s"] - t["v"]) < 0.1
    off, _ = run(pts, p, 6, mode="moving", unfold=False)
    fast = [truth(pt, p, 3) for pt in pts[2:]]
    q = off.frame(3)
    assert all(np.min(np.abs(q["velocity_m_s"] - t["v"])) > 1.0 for t in fast)


def test_a_slow_mover_is_never_unfolded_to_a_racing_speed():
    p = isy.params(**AOP)
    res, _ = run([isy.Point(0.3, 2.0, 0.0, vy=0.8, amp=0.2)], p, 4, mode="moving")
    assert res.points.size and np.all(np.abs(res.points["velocity_m_s"]) < 1.5)


@pytest.mark.parametrize("sep", [30, 45])
def test_two_at_one_range_and_speed_are_two_points(sep):
    p = isy.params(**AOP)
    a = np.deg2rad(sep / 2)
    pts = [isy.Point(-3 * np.sin(a), 3 * np.cos(a), 0, vy=1.0), isy.Point(3 * np.sin(a), 3 * np.cos(a), 0, vy=1.0)]
    res, _ = run(pts, p, 3, mode="moving")
    q = res.frame(1)
    assert q.size == 2
    for pt in pts:
        hit, d = nearest(q, truth(pt, p, 1))
        assert d < 0.05


def test_a_chirp_on_two_tx_is_the_sum_of_both():
    p = isy.params(ntx=3, loops=64, masks=[2, 5], idle_us=60, ramp_us=40)
    pts = [isy.Point(-1.0, 2.0, 0.2, vy=0.5), isy.Point(1.5, 2.5, 0.0, vy=3.0)]
    res, _ = run(pts, p, 3, mode="moving")
    for pt in pts:
        hit, d = nearest(res.frame(1), truth(pt, p, 1))
        assert d < 0.03


@pytest.mark.parametrize("az, el, r", [(0, 0, 3.5), (6.5, 0, 3.5), (6.5, 0, 8.0), (-30, -10, 3.5), (40, 20, 1.5)])
def test_the_cascade_near_and_far(az, el, r):
    """Between azimuth grid points, in elevation (one row 0.8 wavelengths up) and nearer than
    its far field (2.7 m): plane waves put an echo between two grid points 2.5 deg too high."""
    p = isy.params(ntx=1, loops=16, n_samples=128, fs_msps=10, slope_mhz_us=60, f0_ghz=77.0,
                   antenna=ac.CASCADE_CONFIG, masks=[1, 2, 4, 8, 16, 32], idle_us=10, ramp_us=40)
    a = np.deg2rad(az)
    pt = polar(r, az, el, vx=0.8 * np.sin(a), vy=0.8 * np.cos(a))
    res, _ = run([pt], p, 2, mode="moving")
    q = res.frame(0)
    assert q.size == 1
    t = truth(pt, p, 0)
    assert abs(q["azimuth_deg"][0] - t["az"]) < 0.1 and abs(q["elevation_deg"][0] - t["el"]) < 0.3
    assert abs(q["range_m"][0] - t["r"]) < 0.01 and q["coherence"][0] > 0.99


def seated_people(echo=False):
    pts = []
    for (x, y, z), hz in (((-0.8, 1.5, 0.0), 0.25), ((1.0, 2.2, -0.2), 0.2), ((0.1, 3.2, 0.1), 0.3)):
        r = np.sqrt(x * x + y * y + z * z)
        pts.append(isy.Point(x, y, z, swing=(0.004 * x / r, 0.004 * y / r, 0.004 * z / r), swing_hz=hz,
                             swing_phase=hz * 10))
    walls = [isy.Point(-2.0, 4.0, 0.3, amp=3.0), isy.Point(1.8, 3.6, -0.4, amp=3.0)]
    ghost = isy.Point(1.6, 3.6, 0.0, swing=(2 * 0.004 * 1.6 / 3.94, 2 * 0.004 * 3.6 / 3.94, 0), swing_hz=0.25,
                      swing_phase=2.5)
    return pts, walls + ([ghost] if echo else [])


SEATED = dict(ntx=3, loops=4, n_samples=128, fs_msps=10, slope_mhz_us=60, idle_us=10, ramp_us=40, frame_ms=50)


def test_people_sitting_still_are_found_by_their_breathing_and_walls_are_not():
    p = isy.params(**SEATED)
    people, walls = seated_people()
    res, _ = run(people + walls, p, 40, mode="people")
    for f in (20, 30, 39):
        q = res.frame(f)
        assert q.size == 3 and np.all(q["kind"] == ri.KIND_MICRO)
        for pt in people:
            assert nearest(q, truth(pt, p, f))[1] < 0.03
    assert len(res.objects(39)) == 3


def test_a_wall_reflecting_a_breathing_person_is_marked_as_an_echo():
    p = isy.params(**SEATED)
    people, others = seated_people(echo=True)
    res, _ = run(people + others, p, 240, mode="people")
    late = res.points[res.points["frame"] >= 100]
    ghost = late[np.hypot(late["x_m"] - 1.6, late["y_m"] - 3.6) < 0.4]
    assert ghost.size > 100 and np.all(ghost["kind"] == ri.KIND_ECHO)
    real = late[late["kind"] == ri.KIND_MICRO]
    assert real.size >= 3 * 130
    assert all(o.get("echo_of", -1) < 0 for o in res.objects(200))


def test_nothing_where_nothing_is():
    p = isy.params(**AOP)
    res, _ = run([], p, 8, mode="all")
    assert res.points.size == 0 and not any(res.objects(f) for f in range(8))


def test_walkers_crossing_keep_their_tracks():
    p = isy.params(**AOP, frame_ms=50)
    a = isy.body(-1.5, 2.6, 1.0, 0.3, seed=1)
    b = isy.body(1.5, 3.4, -1.0, -0.3, seed=2)
    res, _ = run(a + b, p, 60, mode="moving")
    long = {k: v for k, v in res.trails.items() if v.shape[0] > 40}
    assert len(long) == 2
    ends = sorted((tuple(np.round(v[-1, 1:3], 1)) for v in long.values()))
    assert np.hypot(ends[0][0] + 1.5, ends[0][1] - 2.5) < 0.4      # b, after 3 s
    assert np.hypot(ends[1][0] - 1.5, ends[1][1] - 3.5) < 0.4      # a


def test_one_walker_is_one_object():
    p = isy.params(**AOP, frame_ms=50)
    res, _ = run(isy.body(-1.5, 3.0, 1.0, 0.2, seed=1), p, 40, mode="people")
    assert len(res.trails) == 1
    assert np.mean([len(res.objects(f)) for f in range(10, 40)]) > 0.9


def test_radar_height_and_tilt():
    p = isy.params(ntx=3, loops=32, n_samples=128, fs_msps=10, slope_mhz_us=60)
    pt = isy.Point(0.0, 2.0, 0.0, amp=1.5)
    res, _ = run([pt], p, 1, mode="still", height_m=1.2, tilt_deg=10)
    q = res.frame(0)
    # pointed 10 deg down: what the radar sees straight ahead is below its height
    assert abs(q["z_m"][0] - (1.2 - 2.0 * np.sin(np.deg2rad(10)))) < 0.03
    assert abs(q["y_m"][0] - 2.0 * np.cos(np.deg2rad(10))) < 0.03


def test_frame_images_look_where_the_points_are():
    p = isy.params(**SEATED)
    people, _ = seated_people()
    cube = isy.capture(people[:1], p, frames=30, seed=3)
    t = truth(people[0], p, 25)
    for mode in ("micro", "still", "all"):
        im = ri.frame_images(cube, p, 25, ri.Settings(mode=mode), fps=20)
        r, a = np.unravel_index(np.argmax(im["bev"]), im["bev"].shape)
        assert abs(im["az_deg"][a] - t["az"]) < 1.5 and abs(im["range_m"][r] - t["r"]) < 0.15
        e, a = np.unravel_index(np.argmax(im["front"]), im["front"].shape)
        assert abs(im["front_az_deg"][a] - t["az"]) < 1.5


def test_the_cascade_without_a_calibration_has_no_angles():
    p = isy.params(ntx=1, antenna=ac.CASCADE_CONFIG, masks=[1, 2, 4, 8, 16, 32])
    p = dict(p, cascade=True)
    p.pop("antenna_config", None)
    assert ri.array_for(p, 8) is None


def test_bad_settings_are_refused():
    p = isy.params(**AOP)
    with pytest.raises(ValueError):
        ri.analyse(np.zeros((1, 192, 4, 128), np.complex64), p, fps=20, settings=ri.Settings(mode="sideways"))


def test_the_tab_runs_draws_and_follows_the_frame(tk_root, tmp_path):
    import tkinter as tk
    import radar_imaging_gui as g
    p = isy.params(**AOP, frame_ms=50)
    pts = isy.body(-1.5, 2.6, 1.0, 0.3, seed=1) + [isy.Point(2.0, 4.0, 0.3, amp=2.0)]
    cube = isy.capture(pts, p, frames=30, seed=3)
    top = tk.Toplevel(tk_root)
    try:
        tab = g.ImagingTab(top, lambda: (cube, p, 20.0, 0.0, "walker"))
        tab.mode.set("Everything")
        tab.height.set("1.0")
        tab.run()
        assert tab.result is not None and "points in 30 frames" in tab.summary.cget("text")
        for i, page in enumerate(g.PAGES):
            tab.viz_nb.select(i)
            tab.show_frame(20)
            assert tab._drawn.get(page) == 20
        for by in g.COLOR_BY:
            tab.color_by.set(by)
            tab.trail.set("3")
            tab.redraw()
        data, header = tab.points_for_export()
        assert header.startswith("time_s,frame") and data.shape[1] == header.count(",") + 1
        g.write_ply(str(tmp_path / "c.ply"), tab.result.points)
        text = (tmp_path / "c.ply").read_text()
        assert f"element vertex {tab.result.points.size}" in text
        tab.clear("gone")
        assert tab.result is None
    finally:
        top.destroy()


def _tracks(tracks: dict, fps: float = 20.0):
    """Objects of every frame and trails for tracks {id: (frames, x, y, snr)}."""
    F = max(int(fs.max()) for fs, *_ in tracks.values()) + 1
    by_frame = [[] for _ in range(F)]
    trails = {}
    for tid, (fs, xs, ys, snr) in tracks.items():
        trails[tid] = np.column_stack([fs, xs, ys, np.zeros(len(fs))]).astype(float)
        for f, x, y in zip(fs, xs, ys):
            by_frame[int(f)].append({"track": tid, "x_m": float(x), "y_m": float(y), "z_m": 0.0, "n": 5,
                                     "snr_db": snr, "echo_of": -1, "velocity_m_s": 0.0, "size_m": (0.3, 0.3, 0.3)})
    return by_frame, trails


def test_the_people_layer_one_track_a_person_no_reflections_no_blips():
    fps, f = 20.0, np.arange(200)
    one = lambda v: np.full(200, v)
    by_frame, trails = _tracks({1: (f, one(0.0), one(2.0), 30.0),          # someone 2 m away
                                2: (f, one(0.2), one(2.1), 25.0),          # a second track on their body
                                3: (f, one(0.1), one(3.2), 18.0),          # their reflection, 12 dB weaker, behind
                                4: (f[:30], one(-1.5)[:30], one(1.5)[:30], 28.0),   # 1.5 s of something
                                5: (f, one(1.5), one(1.5), 22.0)}, fps)    # something else, 8 dB weaker
    none = np.zeros(0, dtype=[(k, "f4") for k in ri.POINT_FIELDS])
    persons = ri.people(none, by_frame, trails, fps)
    assert persons == {1, 5} and 2 not in trails
    assert all(sum(o["person"] for o in objs) == 2 for objs in by_frame)
    # Multi Vitals found only the first: the other is no person, nor a weak moment on the reflection
    import copy
    mine = copy.deepcopy(by_frame)
    mine[100][0].update(x_m=0.1, y_m=3.2, snr_db=15.0)
    kept = ri.confirm_people(persons, mine, trails, fps, [(f / fps, one(0.0), one(2.0))])
    assert kept == {1}
    assert not any(o["person"] for o in mine[100]) and mine[101][0]["person"]
    # Multi Vitals lost them (found a reflection instead): the strongest stays the person
    kept = ri.confirm_people(persons, by_frame, trails, fps, [(f / fps, one(1.5), one(1.5))])
    assert 1 in kept and all(objs[0]["person"] for objs in by_frame)


def test_the_tab_shows_the_people_and_the_rest_only_when_asked(tk_root):
    import tkinter as tk
    import radar_imaging_gui as g
    p = isy.params(**AOP, frame_ms=50)
    pts = isy.body(-1.5, 2.6, 0.6, 0.2, seed=1) + [isy.Point(2.0, 4.0, 0.3, vy=0.2, amp=0.3)]
    cube = isy.capture(pts, p, frames=80, seed=3)
    top = tk.Toplevel(tk_root)
    try:
        tab = g.ImagingTab(top, lambda: (cube, p, 20.0, 0.0, "walker"))
        tab.run()
        res = tab.result
        assert res.person_tracks and tab.layered() and "people:" in tab.summary.cget("text")
        f = 70
        q, rest = tab.shown_points(f), tab.shown_points(f, others=True)
        assert q.size and rest.size == 0
        objs = res.objects(f, people_only=True)
        assert len(objs) == 1 and all(np.hypot(q["x_m"] - objs[0]["x_m"], q["y_m"] - objs[0]["y_m"]) < 1.0)
        tab.show_others.set(True)
        tab.redraw()
        both = tab.shown_points(f).size + tab.shown_points(f, others=True).size
        assert both >= q.size and tab.shown_points(f).size == q.size
        for i, page in enumerate(g.PAGES):
            tab.viz_nb.select(i)
            tab.show_frame(f)
            assert tab._drawn.get(page) == f
    finally:
        top.destroy()
